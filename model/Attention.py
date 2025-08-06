import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Dict, Tuple
try:
    from .config             import ByteModelConfig
    # from .utils.KVCache      import KVCache
    from .Position_Embedding import ByteDynamicRoPE
except:
    from config             import ByteModelConfig
    # from utils.KVCache      import KVCache
    from Position_Embedding import ByteDynamicRoPE

class ByteMultiHeadSelfAttention(nn.Module):
    """
    多头自注意力机制模块，支持长序列、FlashAttention、KV缓存、因果掩码、Dropout、混合精度与张量并行。

    Attributes:
        tp_size            (int)            : 张量并行组的大小。
        tp_rank            (int)            : 当前设备在并行组中的rank。
        tp_group           (int)            : 张量并行使用的通信组。
        num_heads          (int)            : 总注意力头数。
        num_kv_heads       (int)            : 用于Key/Value的注意力头数。
        num_rep            (int)            : Key/Value头的重复倍数。
        window_size        (int)            : 滑动窗口大小，用于局部注意力。
        num_local_heads    (int)            : 本地注意力头数（头数 / 并行组数）。
        num_local_kv_heads (int)            : 本地Key/Value头数。
        head_dim           (int)            : 每个头的维度。
        scale              (Tensor)         : 缩放系数 = 1 / sqrt(head_dim)。
        embed_dim          (int)            : 总的嵌入维度。
        use_flash          (bool)           : 是否使用FlashAttention。
        W_q/W_k/W_v        (nn.Linear)      : Q/K/V投影矩阵。
        W_o                (nn.Linear)      : 输出投影矩阵。
        rotary_emb         (ByteDynamicRoPE): 动态位置编码模块。
        attn_dropout       (Dropout)        : 注意力Dropout。
        resid_dropout      (Dropout)        : 残差Dropout。
    """

    def __init__(
        self,
        args: ByteModelConfig,
        layer_id: Optional[int] = None,
        num_layers: Optional[int] = None
    ):
        super().__init__()

        # ===== 张量并行配置 =====
        self.tp_size  = max(1, args.tensor_parallel_size)
        self.tp_rank  = dist.get_rank() if dist.is_initialized() else 0
        self.tp_group = args.tensor_parallel_group

        # ===== 注意力头配置 =====
        # 根据是否指定n_kv_heads，确定用于键(key)和值(value)的头的数量。
        self.num_heads    = args.num_attention_heads
        self.num_kv_heads = args.num_kv_heads or args.num_attention_heads
        # 重复次数，用于扩展键和值的尺寸。
        self.num_rep      = self.num_heads // self.num_kv_heads
        # 窗口大小
        self.window_size  = args.attention_window_size or 0

        # 验证头数可被并行组整除
        assert self.num_heads % self.tp_size    == 0, "num_heads必须能被tp_size整除"
        assert self.num_kv_heads % self.tp_size == 0, "num_kv_heads必须能被tp_size整除"

        # ---------- 张量并行 ----------
        # 计算头数，等于总头数除以模型并行处理大小。
        self.num_local_heads    = self.num_heads // self.tp_size
        # 本地键值头数，等于键值头数除以模型并行处理大小。
        self.num_local_kv_heads = self.num_kv_heads // self.tp_size
        # 每个头的维度，等于模型维度除以头的总数。
        self.head_dim  = args.model_dim // self.num_heads
        self.scale     = torch.rsqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        self.embed_dim = args.model_dim

        # ---- 特性开关 ----
        self.use_flash = args.use_flash_attention

        # ---- KV Cache ----

        # ===== 投影层 =====
        # Q/K/V投影层（按头维度切分）
        self.W_q = nn.Linear(self.embed_dim, self.num_local_heads * self.head_dim, bias=False)
        self.W_k = nn.Linear(self.embed_dim, self.num_local_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(self.embed_dim, self.num_local_kv_heads * self.head_dim, bias=False)

        # 输出投影层（按特征维度切分），将多头注意力的拼接结果映射回embed_dim维度
        self.output_dim_per_partition = self.embed_dim // self.tp_size
        self.W_o = nn.Linear(self.output_dim_per_partition, self.embed_dim, bias=False)

        # ===== RoPE位置编码 =====
        self.rotary_emb = ByteDynamicRoPE(
            dim         = self.head_dim,
            base_theta  = args.base_theta,
            ntk_alpha   = args.ntk_alpha,
            max_seq_len = args.max_seq_len
        )

        # ===== 正则化 =====
        # 注意力权重dropout，防止过拟合
        self.attn_dropout  = nn.Dropout(args.attention_dropout_prob)
        # 残差路径dropout，也叫DropPath，用于深层模型正则
        self.resid_dropout = nn.Dropout(args.residual_dropout_prob)

        # ===== 权重初始化 =====
        # 权重初始化())如果给定num_layers用于缩放)
        if num_layers is not None:
            self._init_weights(num_layers)

    def _init_weights(self, num_layers: int):
        """
        权重初始化，按照 √(2 * num_layers) 缩放。
        """
        std = 0.02 / math.sqrt(2 * num_layers)
        for lin in (self.W_q, self.W_k, self.W_v):
            nn.init.normal_(lin.weight, mean=0.0, std=std)

        # 输出层特殊初始化
        nn.init.normal_(self.W_o.weight, mean=0.0, std=std * self.tp_size)

    def _repeat_kv(self, kv: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        重复 Key/Value 以匹配 Query 的头数())Grouped-Query Attention)

        输入:
            kv: Tensor[batch_size, seq_len, num_kv_heads, head_dim] - 原始的 KV 张量
            n_rep: int - 每个 KV 需要复制的次数，使其头数与 Q 对齐

        返回:
            Tensor[batch_size, seq_len, num_kv_heads * n_rep, head_dim] - 重复后的 KV 张量
        """
        if n_rep == 1:
            return kv

        assert kv.dim() == 4, f"kv 应为 4 维 [batch_size, seq_len, num_kv_heads, head_dim]，但得到 {kv.shape}"
        batch_size, seq_len, num_kv_heads, head_dim = kv.shape

        # 新形状: [batch_size, seq_len, num_kv_heads, n_rep, head_dim] → reshape 到 [batch_size, seq_len, num_kv_heads * n_rep, head_dim]
        kv = kv.unsqueeze(3)                     # [batch_size, seq_len, num_kv_heads, 1, head_dim]
        kv = kv.expand(batch_size, seq_len, num_kv_heads, n_rep, head_dim)    # [batch_size, seq_len, num_kv_heads, n_rep, head_dim]
        kv = kv.reshape(batch_size, seq_len, num_kv_heads * n_rep, head_dim)  # [batch_size, seq_len, num_heads, head_dim]

        return kv.contiguous()  # 确保内存连续，避免后续错误

    def _build_causal_mask(self, seq_len: int, device: torch.device = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        生成因果掩码(Causal Mask)
        确保位置i只能关注位置j<=i的token，防止信息泄露

        支持两种模式:
          1. 标准因果掩码 (window_size <= 0 或 seq_len <= window_size)
          2. 滑动窗口掩码 (window_size > 0 且 seq_len > window_size)

        参数:
        :param seq_len: 当前序列长度(含填充token)
        :param device: 输出张量设备(与输入数据保持一致)
        :param dtype: 输出数据类型(通常与注意力分数类型一致)

        Return:
        mask: 形状为(1, 1, seq_len, seq_len)的掩码张量
        """
        min_val = torch.finfo(dtype).min

        # === 情况1: 使用标准因果掩码 ===
        if self.window_size <= 0 or seq_len <= self.window_size:
            # 创建上三角矩阵())不含主对角线)
            # 对角线下方())j<=i)为0，上方())j>i)为1
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
                diagonal=1
            )

            # 转换为目标数据类型：需要屏蔽的位置设为负无穷，其他为0
            # 注意：部分框架())如FlashAttention)要求bool类型，此处提供通用实现
            mask = mask.to(dtype)  # 转换为目标dtype())通常是float)

            # 对需要屏蔽的位置())j>i)设置极大负值
            # 使softmax后概率接近0
            mask = mask.masked_fill(mask == 1, min_val)

            # 添加必要的维度：适配多头注意力机制
            # (seq_len, seq_len) -> (1, 1, seq_len, seq_len)
            # 支持广播至 [batch_size, num_heads, seq_len, seq_len]
            mask = mask.unsqueeze(0).unsqueeze(0)

            return mask

        # === 情况2: 使用滑动窗口掩码 ===
        # 创建位置索引矩阵
        rows = torch.arange(seq_len, device=device).view(-1, 1)
        cols = torch.arange(seq_len, device=device).view(1, -1)
        
        # 计算位置距离
        dist = rows - cols
        
        # 创建掩码条件:
        # 1. 未来位置 (cols > rows) -> 屏蔽
        # 2. 距离超过窗口 (dist >= window_size) -> 屏蔽
        mask = (cols > rows) | (dist >= self.window_size)
        
        # 转换为加法掩码
        mask = torch.zeros((seq_len, seq_len), device=device, dtype=dtype)
        mask = mask.masked_fill(mask, min_val)

        # 添加必要的维度：适配多头注意力机制
        # (seq_len, seq_len) -> (1, 1, seq_len, seq_len)
        # 支持广播至 [batch_size, num_heads, seq_len, seq_len]
        mask = mask.unsqueeze(0).unsqueeze(0)

        return mask
    
    def _adjust_padding_mask(self, padding_mask: torch.Tensor, seq_len: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        """
        统一调整 padding 掩码的形状，适配后续 attention 操作。
        若未提供，则返回 None。

        参数:
            additive_mask: [B, 1, 1, T] or [B, T]
            seq_len: 当前序列长度
            device: 当前计算设备
            dtype: 当前计算数据类型（如 float32）

        返回:
            调整后的掩码张量: [B, 1, 1, T]
        """
        if padding_mask is None:
            return None

        # 若 shape 为 [B, T]，转换为 [B, 1, 1, T]
        if padding_mask.dim() == 2:
            padding_mask = padding_mask[:, None, None, :]

        # 确保形状正确
        assert padding_mask.shape[-1] == seq_len, f"padding mask最后一维应与seq_len一致，但为 {padding_mask.shape[-1]} != {seq_len}"

        return padding_mask.to(dtype=dtype, device=device)

    def _merge_masks(self, causal_mask: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        合并因果掩码和 padding 掩码，返回最终用于 attention 的掩码。

        参数:
        :param causal_mask: Tensor[1, 1, T, T]，下三角因果掩码
        :param padding_mask: Tensor[B, 1, 1, T]，padding mask

        返回:
            合并后的掩码: Tensor[B, 1, T, T]
        """
        if padding_mask is None:
            return causal_mask

        # 将 padding mask 扩展到 query 维度： [B, 1, 1, T] -> [B, 1, T, T]
        # 即：每个 query token 都对所有 key 应用 padding 屏蔽
        padding_mask = padding_mask.expand(-1, -1, causal_mask.size(-2), -1)  # [B, 1, T, T]

        # 广播加法合并两个 mask，注意类型必须一致
        attn_mask = causal_mask + padding_mask

        return attn_mask

    def forward(
        self,
        x: torch.Tensor,                               # 输入张量，形状 [B, T, embed_dim]
        padding_mask: torch.Tensor = None,            # 可选Padding掩码，形状 [B, 1, 1, T]
    ) -> torch.Tensor:
        # ===== 1. 张量并行输入切分 =====
        if self.tp_size > 1:
            # 特征维度切分 (embed_dim -> [embed_dim // tp_size])
            x = x.chunk(self.tp_size, dim=-1)[self.tp_rank]
        batch_size, seq_len, _ = x.shape
        device  = x.device

        # 获取数据类型dtype
        param_dtype   = x.dtype                        # fp16 / bf16 / fp32
        compute_dtype = torch.float32                  # 统一本层计算精度

        # ===== 2. QKV 投影 & 拆分 ===== 
        # 线性变换得到QKV，形状 [B, T, embed_dim]
        # [B,T,E] -> [B,T,3E] 再 chunk 也可，但这里单独调用
        q = self.W_q(x)  # [B, T, local_heads * head_dim]
        k = self.W_k(x)  # [B, T, local_kv_heads * head_dim]
        v = self.W_v(x)  # [B, T, local_kv_heads * head_dim]

        # reshape成多头格式 [B, T, H, head_dim]
        q = q.view(batch_size, seq_len, self.num_local_heads, self.head_dim)       # [B,T,H,dh]
        # KV 做 num_kv_heads 拆分后 repeat 到 num_heads
        k = k.view(batch_size, seq_len, self.num_local_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_local_kv_heads, self.head_dim)

        # ===== 3. Rotary Position Embedding =====
        q, k = self.rotary_emb.apply_rotary(q, k)

        # ===== 4. 重复k,v以匹配 =====
        k = self._repeat_kv(k, self.num_rep)
        v = self._repeat_kv(v, self.num_rep)

        # ===== 5. 转成 FlashAttention 要求的维度 ===== 
        # 调整维度为FlashAttention所需 [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # ===== 6. 构建并自动调整 additive_mask 长度，和 KV 缓存长度同步 ===== 
        padding_mask = self._adjust_padding_mask(padding_mask, seq_len, device, compute_dtype)
        
        # ===== 7. 构建基础因果掩码 ===== 
        causal_mask = self._build_causal_mask(seq_len, device, compute_dtype)
        
        # ===== 8. 合并因果掩码和padding掩码 ===== 
        attn_mask = self._merge_masks(causal_mask, padding_mask)

        # ===== 9. 构建 Mask & Attention ===== 
        if self.use_flash:
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0
            )
        else:
            # 经典 Attention 实现
            attn_scores  = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn_scores  = attn_scores + attn_mask
            attn_weights = F.softmax(attn_scores, dim=-1, dtype=compute_dtype)
            attn_weights = self.attn_dropout(attn_weights)
            attn_out     = torch.matmul(attn_weights, v)


        # ===== 10. OutPut：拼回 & 输出投影 & Residual Dropout ===== 
        # [B, H, T, D] → [B, T, H, D]
        attn_out = attn_out.transpose(1, 2).contiguous()
        # 多头拼接，形状恢复到 [B, T, embed_dim]
        attn_out = attn_out.view(batch_size, seq_len, self.embed_dim)
        # 输出线性映射
        attn_out = self.W_o(attn_out)
        
        # ===== 11. 张量并行输出聚合 =====
        if self.tp_size > 1:
            # 异步all-reduce通信
            handle = dist.all_reduce(
                attn_out, 
                op=dist.ReduceOp.SUM,
                group=self.tp_group,
                async_op=True
            )
            
            # 等待通信完成
            handle.wait()

        # ===== 12. 残差dropout连接并返回 ===== 
        attn_out = self.resid_dropout(attn_out)

        return attn_out

if __name__ == "__main__":
    args = ByteModelConfig(
        model_dim=128,                # 嵌入维度 E
        num_attention_heads=8,        # 多头注意力 H
        num_kv_heads=4,               # KV头数
        ntk_alpha=1.0,
        base_theta=10000,
        attention_dropout_prob=0.1,
        residual_dropout_prob=0.1,
        max_seq_len=512,
        tensor_parallel_size=1,        # 模型并行大小
        attention_window_size=4096,
        use_flash_attention=False,    # 关闭FlashAttention，便于调试
    )
    
    # ===== 无KVCache测试 =====
    print("BaseTest without KVCache...")
    # 创建Attention层
    attention = ByteMultiHeadSelfAttention(args)
    
    # 创建测试输入
    batch_size = 2
    seq_len = 16
    x = torch.randn(batch_size, seq_len, args.model_dim)
    
    # 前向传播
    with torch.no_grad():
        y = attention(x)

    print(f"Input shape : {x.shape}") # [batch_size, seq_len, model_dim]
    print(f"Output shape: {y.shape}") # [batch_size, seq_len, model_dim]
    print("Without KVCache test completed.")

    # ===== 启动KVCache测试 ====
    # print("\nTesting KVCache...")
    # # 创建 KVCache 实例
    # kv_cache = KVCache(
    #     num_layers=1,  # 添加必需的层数参数())测试用1层)
    #     num_heads=args.num_kv_heads,  # 使用正确的参数名 num_heads
    #     head_dim=args.model_dim // args.num_attention_heads,
    #     max_seq_len=args.max_seq_len,
    #     device=x.device
    # )

    # attn_kvcache = ByteMultiHeadSelfAttention(args, kv_cache=kv_cache)
    # print(attn_kvcache)
    
    # # 模拟自回归生成过程
    # for i in range(seq_len):
    #     # 每次处理一个 token
    #     x_step = x[:, i:i+1, :]
        
    #     # 前向传播())使用缓存)
    #     with torch.no_grad():
    #         y_step, kv_cache = attn_kvcache(x_step)
        
    #     print(f"Step {i+1}: Input shape {x_step.shape}, Output shape {y_step.shape}")
    
    # print("KVCache test completed.")
    
