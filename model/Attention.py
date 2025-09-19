import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from typing import Optional

try:
    from .config import StellarByteModelArgs
    from .PositionEmbedding import StellarByteRotaryPositionEmbedding
except:
    from config import StellarByteModelArgs

class StellarByteAttention(nn.Module):
    def __init__(self, args: StellarByteModelArgs):
        super().__init__()
        self.num_kv_heads = args.num_heads if args.num_kv_heads is None else args.num_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.num_heads // model_parallel_size
        self.n_local_kv_heads = self.num_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            self.enabled_flash_attn = args.enabled_flash_attn
        else:
            self.enabled_flash_attn = False
            print("FlashAttention is not available, using regular attention")

        self.rope = StellarByteRotaryPositionEmbedding(
            args.dim,
            args.max_seq_len,
            args.rope_theta,
        )

        init_method = nn.init.xavier_uniform_

        self.wq = ColumnParallelLinear(
            args.dim,
            args.num_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=init_method,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.num_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=init_method,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.num_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=init_method,
        )
        self.wo = RowParallelLinear(
            args.num_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=init_method,
        )

        # 延迟缓存初始化，减少内存使用
        self.register_buffer('cache_k', torch.zeros(
            args.max_batch_size,
            args.max_seq_len,
            self.n_local_kv_heads,
            self.head_dim,
        ), persistent=False)
        
        self.register_buffer('cache_v', torch.zeros(
            args.max_batch_size,
            args.max_seq_len,
            self.n_local_kv_heads,
            self.head_dim,
        ), persistent=False)
        
        self.cache_initialized = False

        self.register_buffer('causal_mask', None, persistent=False)
        self.current_mask_size = 0

        self.attn_dropout = nn.Dropout(args.attention_dropout)
        self.resid_dropout = nn.Dropout(args.resid_dropout)

    def _initialize_cache(self, device, dtype):
        """延迟初始化缓存，减少初始内存使用"""
        if not self.cache_initialized:
            self.cache_k = self.cache_k.to(device=device, dtype=dtype)
            self.cache_v = self.cache_v.to(device=device, dtype=dtype)
            self.cache_initialized = True# 预分配掩码缓冲区

    def _create_causal_mask(self, seq_len: int, start_pos: int, device: torch.device) -> torch.Tensor:
        """更新或创建因果掩码，使用缓冲区避免重复分配"""
        total_len = start_pos + seq_len
        
        # 如果已有足够大的掩码，直接使用
        if (self.causal_mask is not None and 
            self.causal_mask.size(2) >= seq_len and 
            self.causal_mask.size(3) >= total_len):
            # 返回适当切片
            return self.causal_mask[:, :, :seq_len, :total_len]
        
        # 否则创建新掩码并缓存
        self.current_mask_size = max(self.current_mask_size, total_len)
        
        # 创建完整的因果掩码
        mask = torch.full((total_len, total_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        
        # 注册为缓冲区以便在不同设备间正确移动
        self.register_buffer('causal_mask', mask.unsqueeze(0).unsqueeze(0), persistent=False)
        
        return self.causal_mask[:, :, :seq_len, :total_len]

    def _compute_attention(self, xq: torch.Tensor, keys: torch.Tensor, 
                          values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """计算注意力权重和输出"""
        
        # 计算注意力分数
        scores = torch.matmul(xq, keys.transpose(2, 3)) * self.scale
        
        # 应用掩码
        scores = scores + mask
            
        # 计算注意力权重
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(xq)
        
        # 应用注意力dropout
        attn_weights = self.attn_dropout(attn_weights)

        # 应用注意力权重到值上
        output = torch.matmul(attn_weights, values)
        
        return output.transpose(1, 2).contiguous()

    def _compute_attention_flash(self, xq: torch.Tensor, keys: torch.Tensor, 
                          values: torch.Tensor) -> torch.Tensor:
        """计算注意力权重和输出"""
        # FlashAttention期望输入形状: (batch_size, seq_len, n_heads, head_dim)
        # 并且会自动处理因果掩码
        output = F.scaled_dot_product_attention(
            xq,  # (bs, n_heads, seq_len, head_dim) -> (bs, seq_len, n_heads, head_dim)
            keys,
            values,
            attn_mask=None,
            is_causal=True,
            softmax_scale=self.scale,
            dropout_p=self.dropout if self.training else 0.0
        )
        return output.transpose(1, 2).contiguous()  # 转置回 (bs, n_heads, seq_len, head_dim)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
    ):
        batch_size, seqlen, _ = x.shape
        device, dtype = x.device, x.dtype
        
        # 延迟初始化缓存
        self._initialize_cache(device, dtype)

        # 并行计算QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # 重塑张量形状
        xq = xq.view(batch_size, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(batch_size, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seqlen, self.n_local_kv_heads, self.head_dim)

        # 应用旋转位置编码
        xq, xk = self.rope(xq, xk, freqs_cis=freqs_cis)

        # 更新缓存
        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)
        self.cache_k[:batch_size, start_pos : start_pos + seqlen] = xk
        self.cache_v[:batch_size, start_pos : start_pos + seqlen] = xv

        # 获取完整的键值缓存
        keys = self.cache_k[:batch_size, : start_pos + seqlen]
        values = self.cache_v[:batch_size, : start_pos + seqlen]

        # 重复k/v头以匹配查询头的数量
        keys = self.rope.repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = self.rope.repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        # 创建因果掩码
        mask = self._create_causal_mask(seqlen, start_pos, xq.device)

        # 转置以获得正确的形状 (bs, n_heads, seq_len, head_dim)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算注意力
        if self.enabled_flash_attn:
            output = self._compute_attention_flash(xq, keys, values, mask)
        else:
            output = self._compute_attention(xq, keys, values, mask)  # (bs, n_local_heads, seqlen, cache_len + seqlen)

        # 重塑并应用输出投影
        output = output.view(batch_size, seqlen, -1)

        # 应用残差dropout
        output = self.resid_dropout(output)
        
        # 应用输出投影
        output = self.wo(output)

        return output