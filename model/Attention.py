import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing             import Optional, Dict, Tuple
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
        # 根据是否指定n_kv_heads，确定用于键（key）和值（value）的头的数量。
        self.num_kv_heads = args.num_kv_heads or args.num_attention_heads
        # 基本维度
        assert args.num_attention_heads % args.num_kv_heads == 0, "num_attention_heads 必须能被 num_kv_heads 整除"

        self.embed_dim = args.model_dim
        self.num_heads = args.num_attention_heads
        # 计算头数，等于总头数除以模型并行处理大小。
        self.num_local_heads = args.num_attention_heads // args.model_parallel_size
        # 本地键值头数，等于键值头数除以模型并行处理大小。
        self.num_local_kv_heads = args.num_kv_heads // args.model_parallel_size
        # 重复次数，用于扩展键和值的尺寸。
        self.num_rep = self.num_heads // self.num_local_kv_heads
        # 每个头的维度，等于模型维度除以头的总数。
        self.head_dim = args.model_dim // args.num_attention_heads
        self.scale = torch.rsqrt(torch.tensor(self.head_dim, dtype=torch.float32))

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
        self._rotary_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        # ---- Dropout ----
        # 注意力权重dropout，防止过拟合
        self.attn_dropout = nn.Dropout(args.attention_dropout_prob)
        # 残差路径dropout，也叫DropPath，用于深层模型正则
        self.resid_dropout = nn.Dropout(args.residual_dropout_prob)

        # ---- 全局因果 Mask 缓存 ----
        # 缓存bool型因果mask，避免重复计算（FlashAttention用）
        # 一次性生成最大因果 Mask，shape = [max_seq_len, max_seq_len]
        max_len = args.max_position_embeddings
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

    def _get_rotary(self, seq_len: int, device, dtype):
        """缓存或计算 cos, sin, scale"""
        if seq_len not in self._rotary_cache:
            cos, sin = self.rotary._compute_cos_sin(seq_len, device, dtype)
            scale = self.rotary._compute_xpos_scale(seq_len, device, dtype)
            self._rotary_cache[seq_len] = (cos, sin, scale)
        return self._rotary_cache[seq_len]

    def _build_attention_mask(
        self,
        T: int,
        Tk: int,
        additive_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        返回 attn_mask：
         - FlashAttention 分支需 [B,1,T,Tk] bool
         - 经典分支需 [T,Tk] float(-inf) + additive_mask
        """
        # 1. 切片因果 Mask
        causal_sq = self.full_causal_mask[:T, :Tk]  # [T,Tk]

        if self.use_flash:
            # 转为 [1,1,T,Tk]
            flash_mask = causal_sq.unsqueeze(0).unsqueeze(1)  # bool
            if additive_mask is not None:
                pad_bool = additive_mask.to(torch.bool)      # [B,1,1,Tk]
                # 广播 OR，得到 [B,1,T,Tk]
                flash_mask = pad_bool if flash_mask is None else (flash_mask | pad_bool)
            return flash_mask

        else:
            # float mask: -inf where True
            inf_mask = causal_sq.to(additive_mask.dtype if additive_mask is not None else torch.float32)
            inf_mask = inf_mask.masked_fill(causal_sq, float("-inf"))  # [T,Tk]
            if additive_mask is not None:
                # additive_mask: [B,1,1,Tk], 经典路径直接加到 scores 上即可
                return inf_mask
            return inf_mask

    def forward(
        self,
        x: torch.Tensor,                      # 输入张量，形状 [B, T, embed_dim]
        additive_mask: Optional[torch.Tensor] = None,  # 可选padding掩码，形状 [B, 1, 1, T_k]
    ) -> torch.Tensor:
        B, T, _ = x.shape
        device = x.device
        dtype = x.dtype

        # —— 1. KVCache 设备同步 —— 
        if self.kv_cache is not None:
            # 确保 cache 在同一设备
            self.kv_cache.to(device)

        # —— 2. QKV 投影 & 拆分 —— 
        # 线性变换得到QKV，形状 [B, T, embed_dim]
        # [B,T,E] -> [B,T,3E] 再 chunk 也可，但这里单独调用
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # reshape成多头格式 [B, T, H, head_dim]
        q = q.view(B, T, self.num_local_heads, self.head_dim)       # [B,T,H,dh]
        # KV 做 num_kv_heads 拆分后 repeat 到 num_heads
        k = k.view(B, T, self.num_local_kv_heads, self.head_dim)
        v = v.view(B, T, self.num_local_kv_heads, self.head_dim)
        if self.num_local_kv_heads != self.num_local_heads:
            repeat_times = self.num_local_heads // self.num_local_kv_heads
            # 在 head 维度上平铺
            k = k.repeat_interleave(repeat_times, dim=2)
            v = v.repeat_interleave(repeat_times, dim=2)

        # —— 3. Rotary Position Embedding —— 
        # XPos Rotary 位置编码，稳定长序列建模
        cos, sin, scale = self._get_rotary(T, device, dtype)
        q = q * scale[None,:,None,:]
        q = q * cos[None,:,None,:] + self.rotary._rotate_half(q * scale[None,:,None,:]) * sin[None,:,None,:]
        k = k / scale[None,:,None,:]
        k = k * cos[None,:,None,:] + self.rotary._rotate_half(k) * sin[None,:,None,:]

        # —— 4. KV 缓存（增量推理） —— 
        # 增量推理时从KV缓存获取过去缓存，拼接当前KV
        if self.kv_cache is not None and self.layer_id is not None:
            past_len = self.kv_cache.layer_length(self.layer_id) 
            if past_len > 0:
                past_k, past_v = self.kv_cache.get(self.layer_id)  # [B, Tp, H, D]
                k_cat = torch.cat([past_k, k], dim=1)              # 拼接历史和当前K
                v_cat = torch.cat([past_v, v], dim=1)              # 拼接历史和当前V
            else:
                k_cat, v_cat = k, v

            # 将当前KV写入缓存（自动滑动窗口）
            self.kv_cache.append(self.layer_id, k, v)
        else:
            # 训练或无缓存推理，直接使用当前KV
            k_cat, v_cat = k, v

        Tk = k_cat.size(1)  # 拼接后键值长度

        # —— 5. 转成 FlashAttention 要求的维度 ——
        # 调整维度为FlashAttention所需 [B, H, T, D]
        q = q.transpose(1, 2)       # [B, H, T, D]
        k_cat = k_cat.transpose(1, 2)
        v_cat = v_cat.transpose(1, 2)

        # —— 6. 构建 Mask & Attention ——
        # 构建因果mask及padding mask
        if self.use_flash and q.is_cuda:
            attn_mask = self._build_attention_mask(T, Tk, additive_mask)
            attn_out = F.scaled_dot_product_attention(
                q, k_cat, v_cat,
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
            )
        else:
            scores = (q @ k_cat.transpose(-1,-2)) * self.scale  # [B,H,T,Tk]
            if additive_mask is not None:
                scores = scores + additive_mask
            if self.causal:
                inf_mask = self._build_attention_mask(T, Tk, additive_mask)
                scores = scores + inf_mask
            probs = F.softmax(scores, dim=-1)
            probs = self.attn_dropout(probs)
            attn_out = probs @ v_cat  # [B,H,T,dh]


        # —— 7. OutPut：拼回 & 输出投影 & Residual Dropout ——
        # [B, H, T, D] → [B, T, H, D]
        attn_out = attn_out.transpose(1, 2).contiguous()
        # 多头拼接，形状恢复到 [B, T, embed_dim]
        attn_out = attn_out.view(B, T, self.embed_dim)
        # 输出线性映射
        attn_out = self.W_o(attn_out)
        # 残差dropout
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
        max_position_embeddings=512,
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
        max_seq_len=args.max_position_embeddings,
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
    
