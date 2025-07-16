import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing             import Optional, Dict, Tuple
try:
    from .config             import ByteModelConfig
    from .utils.KVCache      import KVCache
    from .Position_Embedding import XPosRotaryEmbedding
except:
    from config             import ByteModelConfig
    from utils.KVCache      import KVCache
    from Position_Embedding import XPosRotaryEmbedding

class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力机制实现，支持：
      - XPos Rotary 位置编码（稳定长序列）
      - FlashAttention 优化（PyTorch >= 2.1）
      - KV缓存增量推理（支持Sliding Window）
      - 因果掩码（可选）
      - Dropout及残差DropPath正则化
      - 混合精度及张量并行
    
    形状约定
    输入：[B, T, E ] 
    输出：[B, T, E]
    -----------
    B: 批量大小，batch_size
    T: 序列长度，token长度=seq_len
    E: 嵌入维度，embed_dim

    参数：
    -----------
    args: ByteModelConfig，模型配置参数
    kv_cache: 可选，KV缓存实例，支持增量推理加速
    layer_id: 可选，当前层ID，用于KV缓存索引
    num_layers: 可选，当前层数，用于权重缩放
    """

    def __init__(
        self,
        args: ByteModelConfig,
        kv_cache: Optional["KVCache"] = None,
        layer_id: Optional[int] = None,
        num_layers: Optional[int] = None
    ):
        super().__init__()

        # ---------- 头部参数 ----------
        # 根据是否指定n_kv_heads，确定用于键（key）和值（value）的头的数量。
        self.num_kv_heads = args.num_kv_heads or args.num_attention_heads
        assert args.num_attention_heads % args.num_kv_heads == 0, "num_attention_heads 必须能被 num_kv_heads 整除"
        self.embed_dim = args.model_dim
        self.num_heads = args.num_attention_heads

        # ---------- 张量并行 ----------
        self.model_parallel_size = max(1, args.model_parallel_size)
        # 计算头数，等于总头数除以模型并行处理大小。
        self.num_local_heads = args.num_attention_heads // self.model_parallel_size
        # 本地键值头数，等于键值头数除以模型并行处理大小。
        self.num_local_kv_heads = args.num_kv_heads // self.model_parallel_size
        # 重复次数，用于扩展键和值的尺寸。
        self.num_rep = self.num_heads // self.num_local_kv_heads
        # 每个头的维度，等于模型维度除以头的总数。
        self.head_dim = args.model_dim // self.num_attention_heads
        self.scale = torch.rsqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        # ---- 模型并行通信组初始化 ----
        if self.model_parallel_size > 1:
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            # 为张量并行单独建子组（同 rank % mp == gid）
            world = dist.get_world_size()
            ranks  = [r for r in range(world)
                      if r % self.model_parallel_size == dist.get_rank() % self.model_parallel_size]
            self.mp_group = dist.new_group(ranks=ranks, backend="nccl")
            self.mp_world_size = len(ranks)
            self.mp_rank = ranks.index(dist.get_rank())
        else:
            self.mp_group = None
            self.mp_world_size = 1
            self.mp_rank = 0

        # ---- 特性开关 ----
        self.use_flash = args.use_flash_attention
        self.causal = args.use_causal

        # ---- KV Cache ----
        self.kv_cache = kv_cache
        self.layer_id = layer_id

        # ---- 四个独立线性层 ----
        # 生成Q,K,V权重矩阵
        self.W_q = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=False)
        self.W_k = nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        # 输出投影层，将多头注意力的拼接结果映射回embed_dim维度
        self.W_o = nn.Linear(self.embed_dim , self.embed_dim, bias=False)

        # ---- XPos Rotary ----
        # XPos Rotary 位置编码模块
        self.rotary = XPosRotaryEmbedding(
            head_dim=self.head_dim,
            scale_base=args.xpos_scale_base,
            theta=args.xpos_rope_theta,
        )

        # ---- Dropout ----
        # 注意力权重dropout，防止过拟合
        self.attn_dropout = nn.Dropout(args.attention_dropout_prob)
        # 残差路径dropout，也叫DropPath，用于深层模型正则
        self.resid_dropout = nn.Dropout(args.residual_dropout_prob)

        # ---- 全局因果 Mask 缓存 ----
        # 缓存bool型因果mask，避免重复计算（FlashAttention用）
        # 一次性生成最大因果 Mask，shape = [max_seq_len, max_seq_len]
        max_len = args.max_seq_len
        full = torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("full_causal_mask", full, persistent=False)# 可选剪枝 mask
        self.head_mask = torch.ones(self.num_local_heads, dtype=torch.bool)

        # ---- 权重初始化 ----
        # 权重初始化（如果给定num_layers用于缩放）
        if num_layers is not None:
            self._init_weights(num_layers)

        # ----- 量化标志 ----
        self.quantized = False

    def _init_weights(self, num_layers: int):
        """
        权重初始化，按照 √(2 * num_layers) 缩放，
        提升深层Transformer训练稳定性（参考DeepNet论文）。
        """
        std = 0.02 / math.sqrt(2 * num_layers)
        for lin in (self.W_q, self.W_k, self.W_v, self.W_o):
            nn.init.normal_(lin.weight, mean=0.0, std=std)

    def prune_heads(self, heads: list[int]):
        """剪掉指定头，将其 mask 设为 False"""
        mask = self.head_mask.clone()
        mask[heads] = False
        self.head_mask = mask

    def quantize(self):
        """动态量化线性层"""
        if not self.quantized:
            self.W_q = torch.quantization.quantize_dynamic(self.W_q, {nn.Linear}, dtype=torch.qint8)
            self.W_k = torch.quantization.quantize_dynamic(self.W_k, {nn.Linear}, dtype=torch.qint8)
            self.W_v = torch.quantization.quantize_dynamic(self.W_v, {nn.Linear}, dtype=torch.qint8)
            self.W_o = torch.quantization.quantize_dynamic(self.W_o, {nn.Linear}, dtype=torch.qint8)
            self.quantized = True

    def _build_attention_mask(self, T: int, Tk: int, additive_mask: Optional[torch.Tensor]):
        """
        构建注意力掩码：
        - 因果掩码保证当前token只能attend过去及当前token
        - padding掩码加法屏蔽无效token
        - 支持动态长度Tk > max_seq_len的情况（动态生成因果掩码）

        参数：
          T: query序列长度
          Tk: key序列长度（可能包含缓存）
          additive_mask: padding掩码，形状 [B, 1, 1, Tk]，float或bool

        返回：
          如果使用FlashAttention，返回形状 [B, 1, T, Tk] bool掩码
          否则返回形状 [T, Tk] 的 float掩码，True处为 -inf，用于scores相加
        """

        if Tk <= self.full_causal_mask.size(1):
            # 从缓存的预先生成最大因果掩码中切片
            causal_sq = self.full_causal_mask[:T, :Tk]  # [T,Tk] bool
        else:
            # 超过缓存最大长度，动态生成上三角掩码
            causal_sq = torch.triu(
                torch.ones(T, Tk, dtype=torch.bool, device=self.full_causal_mask.device),
                diagonal=1,
            )  # [T,Tk]

        if self.use_flash:
            # FlashAttention期望bool掩码，扩展batch和head维度
            flash_mask = causal_sq.unsqueeze(0).unsqueeze(1)  # [1,1,T,Tk]
            if additive_mask is not None:
                pad_bool = additive_mask.to(torch.bool)       # [B,1,1,Tk]
                flash_mask = flash_mask | pad_bool             # 广播或逻辑或
            return flash_mask

        else:
            # 经典实现，掩码加负无穷屏蔽无效token
            # 初始化为全0 float掩码
            inf_mask = torch.zeros((T, Tk), device=self.full_causal_mask.device, dtype=torch.float32)
            # 上三角True的位置赋值 -inf，屏蔽未来token
            inf_mask = inf_mask.masked_fill(causal_sq, float('-inf'))
            # 扩展维度以匹配scores形状 [1,1,T,Tk]，方便广播
            inf_mask = inf_mask.unsqueeze(0).unsqueeze(0)  # [1,1,T,Tk]

            if additive_mask is not None:
                # additive_mask形状 [B,1,1,Tk]，广播相加
                inf_mask = inf_mask + additive_mask

            return inf_mask

    def forward(
        self,
        x: torch.Tensor,                               # 输入张量，形状 [B, T, embed_dim]
        additive_mask: Optional[torch.Tensor] = None,  # 可选padding掩码，形状 [B, 1, 1, T_k]
    ) -> torch.Tensor:
        B, T, _ = x.shape
        device  = x.device

        # —— 1. KVCache 设备同步 —— 
        if self.kv_cache is not None:
            # 确保 cache 在同一设备
            self.kv_cache.to(device)
        
        # —— 2. 获取历史缓存长度，默认为0（无缓存或训练阶段）——
        past_len = 0
        if self.kv_cache is not None and self.layer_id is not None:
            past_len = self.kv_cache.layer_length(self.layer_id)  # 历史缓存长度，保证位置编码连续

        # —— 3. 获取数据类型dtype ——
        param_dtype   = x.dtype                        # fp16 / bf16 / fp32
        compute_dtype = torch.float32                  # 统一本层计算精度

        # —— 4. QKV 投影 & 拆分 —— 
        # 线性变换得到QKV，形状 [B, T, embed_dim]
        # [B,T,E] -> [B,T,3E] 再 chunk 也可，但这里单独调用
        q = self.W_q(x).to(param_dtype)
        k = self.W_k(x).to(param_dtype)
        v = self.W_v(x).to(param_dtype)

        # reshape成多头格式 [B, T, H, head_dim]
        q = q.view(B, T, self.num_local_heads, self.head_dim)       # [B,T,H,dh]
        # KV 做 num_kv_heads 拆分后 repeat 到 num_heads
        k = k.view(B, T, self.num_local_kv_heads, self.head_dim)
        v = v.view(B, T, self.num_local_kv_heads, self.head_dim)
        # 如果num_local_kv_heads不等于num_local_heads，重复k,v以匹配
        if self.num_local_kv_heads != self.num_local_heads:
            repeat_times = self.num_local_heads // self.num_local_kv_heads
            # 在 head 维度上平铺
            k = k.repeat_interleave(repeat_times, dim=2)
            v = v.repeat_interleave(repeat_times, dim=2)

        # —— 4. Rotary Position Embedding —— 
        # 4.1 XPos Rotary 位置编码，生成长度为 (past_len + T) 的 cos, sin, scale，支持缓存历史拼接
        cos_full, sin_full, scale_full = self.rotary._get_cos_sin_scale(past_len + T, device, compute_dtype)
        # 4.2 只取当前输入位置对应的编码（即从past_len开始的切片）
        cos = cos_full[past_len : past_len + T]   # [T, head_dim]
        sin = sin_full[past_len : past_len + T]
        scale = scale_full[past_len : past_len + T]

        # 4.3 扩展维度用于广播
        cos = cos[None, :, None, :]   # [1, T, 1, head_dim]
        sin = sin[None, :, None, :]
        scale = scale[None, :, None, :]

        # Q先乘scale，旋转编码后乘cos，后半部分旋转后乘sin
        q = q * scale
        q = q * cos + self.rotary._rotate_half(q) * sin

        # K先除scale，旋转编码后乘cos，后半部分旋转后乘sin
        k = k / scale
        k = k * cos + self.rotary._rotate_half(k) * sin

        # —— 5. KV 缓存（增量推理） —— 
        # 增量推理时从KV缓存获取过去缓存，拼接当前KV
        if self.kv_cache is not None and self.layer_id is not None:
            # 将当前KV写入缓存（自动滑动窗口）
            self.kv_cache.append(self.layer_id, k, v)
            # 再获取缓存中完整 KV，避免重复拼接
            k_cat, v_cat = self.kv_cache.get(self.layer_id)  # [B, Tp, H, D]
        else:
            # 训练或无缓存推理，直接使用当前KV
            k_cat, v_cat = k, v

        Tk = k_cat.size(1)  # 拼接后键值长度

        # —— 6. 转成 FlashAttention 要求的维度 ——
        # 调整维度为FlashAttention所需 [B, H, T, D]
        q = q.transpose(1, 2)       # [B, H, T, D]
        k_cat = k_cat.transpose(1, 2)
        v_cat = v_cat.transpose(1, 2)

        # —— 7. 构建 Mask & Attention ——
        # 构建因果mask及padding mask
        if self.use_flash:
            attn_mask = self._build_attention_mask(T, Tk, additive_mask)
            attn_out = F.scaled_dot_product_attention(
                q, k_cat, v_cat,
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=self.causal
            )
        else:
            # 经典 Attention 实现，支持增量推理
            scores = ( q.to(compute_dtype) @ k_cat.to(compute_dtype).transpose(-1,-2) ) * self.scale.to(compute_dtype)  # [B,H,T,Tk]
            if additive_mask is not None:
                scores = scores + additive_mask.to(compute_dtype)
            if self.causal:
                inf_mask = self._build_attention_mask(T, Tk, additive_mask)
                scores = scores + inf_mask.to(compute_dtype)
            probs    = F.softmax(scores, dim=-1)
            probs    = self.attn_dropout(probs).to(param_dtype)
            attn_out = probs @ v_cat.to(param_dtype)  # [B,H,T,dh]


        # —— 8. OutPut：拼回 & 输出投影 & Residual Dropout ——
        # [B, H, T, D] → [B, T, H, D]
        attn_out = attn_out.transpose(1, 2).contiguous()
        # 多头拼接，形状恢复到 [B, T, embed_dim]
        attn_out = attn_out.view(B, T, self.embed_dim)
        # 输出线性映射
        attn_out = self.W_o(attn_out)

        # —— 9. 张量并行 All‑Reduce 汇聚 ——
        if self.mp_world_size > 1:
            dist.all_reduce(attn_out, op=dist.ReduceOp.SUM, group=self.mp_group)
            attn_out = attn_out / self.mp_world_size
        
        # —— 10. 残差dropout连接并返回 ——
        attn_out = self.resid_dropout(attn_out)

        return attn_out

if __name__ == "__main__":
    args = ByteModelConfig(
        model_dim=128,                # 嵌入维度 E
        num_attention_heads=8,        # 多头注意力 H
        num_kv_heads=4,               # KV头数
        xpos_scale_base=512,          # XPos参数
        xpos_rope_theta=10000,
        attention_dropout_prob=0.1,
        residual_dropout_prob=0.1,
        max_seq_len=512,
        model_parallel_size=1,        # 模型并行大小
        use_flash_attention=False,    # 关闭FlashAttention，便于调试
        use_causal=True
    )
    
    # ===== 无KVCache测试 =====
    print("BaseTest without KVCache...")
    # 创建Attention层
    attention = MultiHeadSelfAttention(args)
    
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
    print("\nTesting KVCache...")
    # 创建 KVCache 实例
    kv_cache = KVCache(
        num_layers=1,  # 添加必需的层数参数（测试用1层）
        num_heads=args.num_kv_heads,  # 使用正确的参数名 num_heads
        head_dim=args.model_dim // args.num_attention_heads,
        max_seq_len=args.max_seq_len,
        device=x.device
    )

    attn_kvcache = MultiHeadSelfAttention(args, kv_cache=kv_cache)
    
    # 模拟自回归生成过程
    for i in range(seq_len):
        # 每次处理一个 token
        x_step = x[:, i:i+1, :]
        
        # 前向传播（使用缓存）
        with torch.no_grad():
            y_step, kv_cache = attn_kvcache(x_step)
        
        print(f"Step {i+1}: Input shape {x_step.shape}, Output shape {y_step.shape}")
    
    print("KVCache test completed.")
    
