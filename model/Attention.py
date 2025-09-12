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
    from .utils.KVCache      import ByteKVCache
    from .Position_Embedding import ByteDynamicRoPE
except:
    from config             import ByteModelConfig
    from utils.KVCache      import ByteKVCache
    from Position_Embedding import ByteDynamicRoPE

class ByteMultiHeadSelfAttention(nn.Module):
    """
    多头自注意力机制模块，支持：
    - 张量并行（Tensor Parallelism）
    - KV缓存（KV Cache）用于高效自回归推理
    - FlashAttention加速计算
    - 动态RoPE位置编码
    - 分组查询注意力（Grouped Query Attention）
    - 滑动窗口注意力（Sliding Window Attention）

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
        """
        多头自注意力层初始化
        
        参数:
            args: ByteModelConfig - 模型配置参数
            layer_id: int - 当前层ID（用于KV缓存索引）
            num_layers: int - 总层数（用于权重初始化）
        """
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

        # ===== 特性开关 =====
        self.use_flash = args.use_flash_attention

        # ===== KV Cache =====
        self.use_kvcache: bool       = args.use_kvcache
        self.layer_id: Optional[int] = layer_id  # 供外部按层索引到相应缓存分区
        self.max_cache_len: int      = args.max_seq_len

        # ===== 投影层 =====
        # Q/K/V投影层（按头维度切分）
        # 查询投影（输出维度: 本地头数 * 头维度）
        self.W_q = nn.Linear(self.embed_dim, self.num_local_heads * self.head_dim, bias=False)
        # 键投影（输出维度: 本地键/值头数 * 头维度）
        self.W_k = nn.Linear(self.embed_dim, self.num_local_kv_heads * self.head_dim, bias=False)
        # 值投影（输出维度: 本地键/值头数 * 头维度）
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

    def init_kv_cache(
        self, 
        batch_size: int, 
        device: torch.device,
        args: ByteModelConfig = None
    ) -> ByteKVCache:
        """
        初始化适用于当前层的KV缓存
        
        参数:
            batch_size: int - 批大小
            device: torch.device - 缓存设备
            args: ByteModelConfig - 模型配置（可选）
            
        返回:
            ByteKVCache - 初始化后的KV缓存实例
        """
        return ByteKVCache(
            num_layers    = args.cache_layers,
            num_heads     = self.num_local_kv_heads,
            head_dim      = self.head_dim,
            max_seq_len   = self.max_cache_len,
            batch_size    = batch_size,
            dtype         = args.cache_dtype,
            device        = device,
            memory_format = torch.contiguous_format
        )

    def _init_weights(self, num_layers: int):
        """
        权重初始化，按照 √(2 * num_layers) 缩放。
        参数:
            num_layers: int - 总层数
        """
        # 标准差 = 0.02 / sqrt(2 * num_layers)
        std = 0.02 / math.sqrt(2 * num_layers)

        # 初始化Q/K/V投影权重
        for lin in (self.W_q, self.W_k, self.W_v):
            nn.init.normal_(lin.weight, mean=0.0, std=std)

        # 输出层特殊初始化（考虑张量并行）
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
        # 在第四个维度（头的维度前）添加一个新的维度
        kv = kv[:, :, :, None, :]                                             # [batch_size, seq_len, num_kv_heads, 1, head_dim]
        # 将新添加的维度扩展到n_rep大小，实现重复的效果
        kv = kv.expand(batch_size, seq_len, num_kv_heads, n_rep, head_dim)    # [batch_size, seq_len, num_kv_heads, n_rep, head_dim]
        # 重新塑形，合并键/值对头的数量和重复次数的维度
        kv = kv.reshape(batch_size, seq_len, num_kv_heads * n_rep, head_dim)  # [batch_size, seq_len, num_heads, head_dim]

        return kv.contiguous()  # 确保内存连续，避免后续错误

    def _build_causal_mask(
        self, 
        seq_len: int, 
        device: torch.device = None, 
        dtype: torch.dtype = torch.float32,
        offset: int = 0
    ) -> torch.Tensor:
        """
        生成因果掩码(Causal Mask)(支持缓存偏移)
        确保位置i只能关注位置j<=i的token，防止信息泄露

        支持两种模式:
          1. 标准因果掩码 (window_size <= 0 或 seq_len <= window_size)
          2. 滑动窗口掩码 (window_size > 0 且 seq_len > window_size)

        参数:
            seq_len(int): 当前序列长度(含填充token)
            device(torch.device): 输出张量设备(与输入数据保持一致)
            dtype(torch.dtype): 输出数据类型(通常与注意力分数类型一致)
            offset(int): 偏移量（即已缓存的 token 数，增量推理时使用）

        返回:
            mask(Tensor): 因果掩码 [1, 1, seq_len, total_len]（total_len = offset + seq_len）
        """
        min_val = -1e9                    # 掩码最小值（用于softmax前）
        total_len = seq_len + offset      # 序列总长度（已缓存 + 当前）

        # === 情况1: 使用标准因果掩码 ===
        if self.window_size <= 0 or total_len <= self.window_size:
            # 创建上三角矩阵())不含主对角线)
            # 对角线下方())j<=i)为0，上方())j>i)为1
            mask = torch.triu(
                torch.ones(total_len, total_len, device=device, dtype=torch.bool),
                diagonal=1
            )

            # 转换为目标数据类型：需要屏蔽的位置设为负无穷，其他为0
            # 注意：部分框架())如FlashAttention)要求bool类型，此处提供通用实现
            mask = mask.to(dtype)  # 转换为目标dtype())通常是float)

            # 对需要屏蔽的位置())j>i)设置极大负值
            # 使softmax后概率接近0
            mask = mask.masked_fill(mask == 1, min_val)

            # 裁剪出当前 step 需要的部分（行对应当前 step）
            # offset:total_len 是当前 step 的行
            # 0:total_len 是所有可关注的列（缓存 + 当前）
            mask = mask[offset:total_len, :total_len]

            # 添加必要的维度：适配多头注意力机制
            # (seq_len, seq_len) -> (1, 1, seq_len, seq_len)
            # 支持广播至 [batch_size, num_heads, seq_len, seq_len]
            mask = mask.unsqueeze(0).unsqueeze(0)

            return mask

        # === 情况2: 使用滑动窗口掩码 ===
        # 创建位置索引矩阵
        rows = torch.arange(offset, total_len, device=device).view(-1, 1) # [seq_len, 1]
        cols = torch.arange(0, total_len, device=device).view(1, -1) # [1, seq_len]
        
        # 计算相对位置距离(i - j)
        dist = rows - cols
        
        # 创建掩码条件:
        # 1. 因果性：只允许关注过去位置 (j <= i)
        # 2. 局部性：只允许关注 [i - window_size + 1, i] 范围内的位置
        causal_cond = (cols <= rows)  # j <= i
        window_cond = (dist < self.window_size)  # i - j < window_size
        
        # 有效区域 = 因果性 AND 局部性
        valid_mask = causal_cond & window_cond

        # 反转逻辑：有效区域为0，无效区域为min_val
        mask_matrix = torch.full(
            (seq_len, total_len), 
            fill_value=min_val, 
            device=device, 
            dtype=dtype
        )
        mask_matrix.masked_fill_(valid_mask, 0)  # 有效区域置0

        # 添加必要的维度：适配多头注意力机制
        # (seq_len, seq_len) -> (1, 1, seq_len, seq_len)
        # 支持广播至 [batch_size, num_heads, seq_len, seq_len]
        mask = mask_matrix.unsqueeze(0).unsqueeze(0)

        return mask
    
    def forward(
        self,
        x: torch.Tensor,                                # 输入张量，形状 [B, T, embed_dim]
        kv_cache: Optional[ByteKVCache] = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: Tensor - 输入序列 [batch_size, seq_len, embed_dim]
            kv_cache: ByteKVCache - KV缓存（可选）
            
        返回:
            output: Tensor - 注意力输出 [B, T, E]
            meta: Dict - 缓存元信息（当使用缓存时）或 None
        """
        # ===== 1. 张量并行输入切分 =====
        if self.tp_size > 1:
            # 特征维度切分 (embed_dim -> [embed_dim // tp_size])
            x = x.chunk(self.tp_size, dim=-1)[self.tp_rank]
        batch_size, seq_len, _ = x.shape
        device                 = x.device

        # 获取数据类型dtype
        param_dtype   = x.dtype                        # fp16 / bf16 / fp32
        compute_dtype = torch.float32                  # 统一本层计算精度

        # ===== 2. 读取 KVCache 缓存 =====
        cache_len = 0              # 缓存长度
        if self.use_kvcache and kv_cache is not None:
            # 从缓存中获取例是KV [B, kv_h, T_cache, D]
            cache_len  = kv_cache.current_seq_len(self.layer_id)

        # ===== 3. QKV 投影 & 拆分 ===== 
        # 线性变换得到QKV，形状 [B, T, embed_dim]
        # [B,T,E] -> [B,T,3E] 再 chunk 也可，但这里单独调用
        q = self.W_q(x)  # 查询投影 [B, T, local_heads * head_dim]
        k = self.W_k(x)  # 键投影   [B, T, local_kv_heads * head_dim]
        v = self.W_v(x)  # 值投影   [B, T, local_kv_heads * head_dim]

        # reshape成多头格式 [B, T, H, head_dim]
        q = q.view(batch_size, seq_len, self.num_local_heads, self.head_dim)       # [B,T,H,dh]
        # KV 做 num_kv_heads 拆分后 repeat 到 num_heads
        k = k.view(batch_size, seq_len, self.num_local_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_local_kv_heads, self.head_dim)

        # ===== 4. Rotary Position Embedding =====
        q, k = self.rotary_emb.apply_rotary(q, k, cache_len)

        # ===== 5. 保存用于 cache 的未 repeat kv（形状 [B, kv_H, T, D]） =====
        k_kv = k.transpose(1,2).contiguous()  # [B, kv_H, T, D]
        v_kv = v.transpose(1,2).contiguous()  # [B, kv_H, T, D]

        # ===== 6. 重复k,v以匹配 =====
        k = self._repeat_kv(k, self.num_rep)  # [B, T, H, D]
        v = self._repeat_kv(v, self.num_rep)  # [B, T, H, D]

        # ===== 7. 转成 FlashAttention 要求的维度 ===== 
        # 调整维度为FlashAttention所需 [B, H, T, D]
        q = q.transpose(1, 2)     # [B, H, T, D]
        k_cur = k.transpose(1, 2) # [B, H, T, D]
        v_cur = v.transpose(1, 2) # [B, H, T, D]

        # ===== 8. 拼接历史KV缓存值 =====
        if self.use_kvcache and kv_cache is not None:
            k_cache, v_cache = kv_cache.get_kv(self.layer_id)
            # 检查缓存是否为空
            if k_cache.numel() > 0 and v_cache.numel() > 0:
                # 确保 device/dtype 正确
                k_cache = k_cache.to(device=device, dtype=param_dtype)
                v_cache = v_cache.to(device=device, dtype=param_dtype)

                # repeat_interleave 以匹配 query 的头数 (num_rep)
                k_cache = k_cache[:, :, None, :, :].expand(-1, -1, self.num_rep, -1, -1)
                k_cache = k_cache.reshape(k_cache.shape[0], -1, *k_cache.shape[3:])

                v_cache = v_cache[:, :, None, :, :].expand(-1, -1, self.num_rep, -1, -1)
                v_cache = v_cache.reshape(v_cache.shape[0], -1, *v_cache.shape[3:])
                # 拼接
                k_full = torch.cat([k_cache, k_cur], dim=-2)  # [B, H, T, D]
                v_full = torch.cat([v_cache, v_cur], dim=-2)
            else:
                # 缓存为空时直接使用当前KV
                k_full = k_cur
                v_full = v_cur
        else:
            # 没有使用缓存的情况
            k_full = k_cur
            v_full = v_cur

        # ===== 9. 构建因果掩码 ===== 
        causal_mask = self._build_causal_mask(seq_len, device, compute_dtype, cache_len)

        # ===== 10. 构建 Mask & Attention ===== 
        if self.use_flash:
            # 使用FlashAttention（高效实现）
            attn_out = F.scaled_dot_product_attention(
                query     = q,         # [B, H, T, D]
                key       = k_full,    # [B, H, T_total, D]
                value     = v_full,    # [B, H, T_total, D]
                attn_mask = causal_mask, # [B, 1, T, T_total]
                dropout_p = self.attn_dropout.p if self.training else 0.0
            )
        else:
            # 标准注意力实现
            # 计算注意力分数 [B, H, T, T_total]
            attn_scores  = torch.matmul(q, k_full.transpose(-2, -1)) * self.scale
            
            # 应用掩码
            attn_scores  = attn_scores + causal_mask
            
            # 计算注意力权重
            attn_weights = F.softmax(attn_scores, dim=-1, dtype=compute_dtype)
            attn_weights = self.attn_dropout(attn_weights)
            attn_weights = attn_weights.to(v_full.dtype)
            
            # 加权求和
            attn_out     = torch.matmul(attn_weights, v_full)


        # ===== 13. 拼回多头维度 -> local partition 的 feature 形状 ===== 
        # [B, H, T, D] → [B, T, H, D]
        attn_out = attn_out.transpose(1, 2).contiguous()
        # [B, T, num_heads * head_dim]  -> [B, T, embed_dim]
        local_out_dim = self.head_dim * self.num_local_heads
        # 多头拼接，形状恢复到 [B, T, embed_dim]
        attn_out_local = attn_out.view(batch_size, seq_len, local_out_dim)
        
        # ===== 14. 张量并行输出聚合 =====
        if self.tp_size > 1:
            # 创建收集列表（每个设备一个）
            gather_list = [torch.zeros_like(attn_out_local) for _ in range(self.tp_size)]
            # 全收集操作（跨设备） dist.all_gather 会在每个进程把所有 partition 的片段收集到 gather_list
            dist.all_gather(gather_list, attn_out_local.contiguous(), group=self.tp_group)
            # 沿特征维度拼接 [B, T, E]
            attn_out = torch.cat(gather_list, dim=-1)  # [B, T, embed_dim]
        else:
            attn_out = attn_out_local  # already full

        # ===== 15 输出投影 & residual dropout ===== 
        attn_out = self.W_o(attn_out)  # [B, T, embed_dim]
        attn_out = self.resid_dropout(attn_out)

        # ===== 16. 写入 KVCache（写入未 repeat 的 kv_kv/v_kv）并返回 meta =====
        if self.use_kvcache and kv_cache is not None:
            # kv_cache.append_batch expects [B, num_heads_kv_local, L_block, head_dim]
            # k_kv/v_kv 是 [B, kv_H_local, T_cur, D]
            kv_cache.append_batch(self.layer_id, k_kv, v_kv)
            # 返回缓存元信息
            meta = {"past_len": cache_len, "new_len": cache_len + seq_len}
            return attn_out, meta

        return attn_out, None

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
        use_kvcache=True,
        cache_layers=1,
        cache_dtype=torch.float32
    )
    
    # ===== 1. 无KVCache测试 =====
    print("BaseTest without KVCache...")
    # 创建Attention层
    attention = ByteMultiHeadSelfAttention(args, layer_id=0, num_layers=1)
    
    # 创建测试输入
    batch_size = 2
    seq_len = 16
    x = torch.randn(batch_size, seq_len, args.model_dim)
    
    # 前向传播
    with torch.no_grad():
        y, _ = attention(x)

    print(f"Input shape : {x.shape}") # [batch_size, seq_len, model_dim]
    print(f"Output shape: {y.shape}") # [batch_size, seq_len, model_dim]
    print("Without KVCache test completed.")

    # ===== 2. 初始化KV缓存 =====
    batch_size = 2
    device = torch.device("cpu")
    kv_cache = attention.init_kv_cache(batch_size, device, args)
    print(f"KVCache initialized. Capacity: {kv_cache.capacity()}")

    # ===== 3. 处理提示序列 =====
    prompt_len = 16
    x_prompt = torch.randn(batch_size, prompt_len, args.model_dim, device=device)
    print(f"\nProcessing prompt (length={prompt_len})...")
    
    with torch.no_grad():
        output_prompt, meta = attention(x_prompt, kv_cache=kv_cache)
    
    print(f"Output shape: {output_prompt.shape}")
    print(f"Cache length after prompt: {kv_cache.current_seq_len(0)}")
    
    # ===== 4. 自回归生成（单步模式） =====
    print("\nStarting autoregressive generation (step-by-step)...")
    
    for step in range(3):  # 生成3个token
        # 使用上一步输出的最后一个token作为输入
        x_next = output_prompt[:, -1:, :] if step == 0 else output_step[:, -1:, :]
        
        with torch.no_grad():
            output_step, meta = attention(x_next, kv_cache=kv_cache)
        
        cache_len = kv_cache.current_seq_len(0)
        print(f"Step {step+1}: Generated token | Output shape: {output_step.shape} | Cache length: {cache_len}")
    
    # ===== 5. 自回归生成（批量模式） =====
    print("\nStarting autoregressive generation (batch mode)...")
    
    # 生成4个token的批量
    gen_len = 4
    x_batch = torch.randn(batch_size, gen_len, args.model_dim, device=device)
    
    with torch.no_grad():
        output_batch, meta = attention(x_batch, kv_cache=kv_cache)
    
    cache_len = kv_cache.current_seq_len(0)
    print(f"Generated {gen_len} tokens | Output shape: {output_batch.shape} | Cache length: {cache_len}")
    
    # ===== 6. 测试缓存管理功能 =====
    print("\nTesting cache management features...")
    
    # 测试缓存重排序（用于束搜索）
    new_order = torch.tensor([1, 0], device=device)  # 交换两个样本的顺序
    kv_cache.reorder(new_order)
    print("Cache reordered for beam search")
    
    # 测试缓存剪枝
    new_max_len = 20
    kv_cache.prune(new_max_len)
    print(f"Cache pruned to {new_max_len} tokens")
    
    # 测试缓存状态保存/加载
    cache_state = kv_cache.state_dict()
    print(f"Cache state saved (size: {len(cache_state['layers'])} layers)")
    
    # 创建新缓存并加载状态
    new_kv_cache = attention.init_kv_cache(batch_size, device, args)
    new_kv_cache.load_state_dict(cache_state)
    print(f"Cache state loaded to new instance. Cache length: {new_kv_cache.current_seq_len(0)}")
    
    # 测试缓存清空
    kv_cache.clear()
    print(f"Cache cleared. Current length: {kv_cache.current_seq_len(0)}")
    
    print("\nAll KVCache tests completed successfully!")
    
