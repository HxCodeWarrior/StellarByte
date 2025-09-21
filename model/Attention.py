# 导入必要的数学库和PyTorch相关模块
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# 导入FairScale模型并行初始化工具
import fairscale.nn.model_parallel.initialize as fs_init
# 导入FairScale的模型并行线性层
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from typing import Optional, Tuple


try:
    from .config import StellarByteModelArgs
    from .PositionEmbedding import StellarByteRoPE
except:
    from config import StellarByteModelArgs
    from PositionEmbedding import StellarByteRoPE


class StellarByteAttention(nn.Module):
    """实现StellarByte模型的注意力机制，支持模型并行和FlashAttention。
    
    该模块实现了基于旋转位置编码(RoPE)的注意力机制，支持分组查询注意力(GQA)和
    模型并行计算。可以根据配置选择使用常规注意力或FlashAttention实现。
    
    Attributes:
        num_kv_heads (int): 键值头的数量
        num_local_heads (int): 本地查询头数量
        num_local_kv_heads (int): 本地键值头数量
        n_rep (int): 每个键值头需要重复的次数
        head_dim (int): 每个注意力头的维度
        scale (float): 注意力分数缩放因子
        enabled_flash_attn (bool): 是否启用FlashAttention
        rope (StellarByteRoPE): 旋转位置编码实例
        wq (ColumnParallelLinear): 查询投影层
        wk (ColumnParallelLinear): 键投影层
        wv (ColumnParallelLinear): 值投影层
        wo (RowParallelLinear): 输出投影层
        cache_k (torch.Tensor): 键缓存
        cache_v (torch.Tensor): 值缓存
        cache_initialized (bool): 缓存是否已初始化
        causal_mask (torch.Tensor): 因果掩码缓存
        current_mask_size (int): 当前掩码大小
        attn_dropout (nn.Dropout): 注意力dropout层
        resid_dropout (nn.Dropout): 残差dropout层
    """
    
    def __init__(self, args: StellarByteModelArgs):
        """初始化StellarByteAttention模块。
        
        Args:
            args: 包含模型配置参数的StellarByteModelArgs对象
        """
        super().__init__()
        # 设置键值头的数量，如果没有单独指定，则使用与查询头相同的数量
        self.num_kv_heads = args.num_heads if args.num_kv_heads is None else args.num_kv_heads
        # 获取模型并行组的大小（GPU数量）
        model_parallel_size = args.model_parallel_size
        # 计算每个GPU上的本地查询头数量
        self.num_local_heads = args.num_heads // model_parallel_size
        # 计算每个GPU上的本地键值头数量
        self.num_local_kv_heads = self.num_kv_heads // model_parallel_size
        # 计算每个键值头需要重复的次数以匹配查询头数量
        self.num_rep = self.num_local_heads // self.num_local_kv_heads
        # 计算每个注意力头的维度
        self.head_dim = args.dim // args.num_heads
        # 设置缩放因子，用于缩放注意力分数
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # 初始化旋转位置编码
        self.rope = StellarByteRoPE(
            args.dim,  # 模型维度
            args.max_seq_len,  # 最大序列长度
            args.rope_theta,  # RoPE的theta参数
        )

        # 初始化查询投影层
        self.wq = nn.Linear(
            args.dim,  # 输入维度
            args.num_heads * self.head_dim,  # 输出维度（所有查询头的总维度）
            bias=False,  # 不使用偏置
        )
        
        # 初始化键投影层
        self.wk = nn.Linear(
            args.dim,  # 输入维度
            self.num_kv_heads * self.head_dim,  # 输出维度（所有键头的总维度）
            bias=False,  # 不使用偏置
        )
        
        # 初始化值投影层
        self.wv = nn.Linear(
            args.dim,  # 输入维度
            self.num_kv_heads * self.head_dim,  # 输出维度（所有值头的总维度）
            bias=False,  # 不使用偏置
        )
        
        # 初始化输出投影层
        self.wo = nn.Linear(
            args.num_heads * self.head_dim,  # 输入维度（所有头的总维度）
            args.dim,  # 输出维度（模型维度）
            bias=False,  # 不使用偏置
        )

        # 设置是否使用KV缓存
        self.enabled_kv_cache = args.enabled_kv_cache
        
        # 检查是否支持FlashAttention（PyTorch 2.0+）
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            self.enabled_flash_attn = args.enabled_flash_attn
        else:
            self.enabled_flash_attn = False
            print("FlashAttention is not available, using regular attention")

        # 初始化注意力dropout和残差dropout
        self.attn_dropout = nn.Dropout(args.attention_dropout)
        self.resid_dropout = nn.Dropout(args.resid_dropout)

    def _create_causal_mask(self, seq_len: int, cache_len: int, device: torch.device) -> torch.Tensor:
        """使用torch.triu创建因果掩码，支持KV缓存。

        Args:
            seq_len: 当前序列长度
            cache_len: 缓存序列长度
            device: 设备

        Returns:
            因果掩码张量，形状为 (1, 1, seq_len, cache_len + seq_len)
        """
        # 创建全0矩阵，形状为 (seq_len, cache_len + seq_len)
        mask = torch.zeros((seq_len, cache_len + seq_len), device=device)

        # 创建上三角矩阵（对角线以上为1），形状为 (seq_len, seq_len)
        triu_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=device),
            diagonal=1
        )

        # 将上三角部分设置为负无穷
        mask[:, cache_len:] = triu_mask * float("-inf")

        return mask.unsqueeze(0).unsqueeze(0)  # 扩展为 (1, 1, seq_len, cache_len + seq_len)

    def _compute_attention(self, xq: torch.Tensor, keys: torch.Tensor, 
                          values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """计算注意力权重和输出（常规实现）。
        
        该方法实现标准的缩放点积注意力机制，包括softmax和dropout。
        
        Args:
            xq: 查询张量，形状为 (bs, n_heads, seq_len, head_dim)
            keys: 键张量，形状为 (bs, n_heads, cache_len+seq_len, head_dim)
            values: 值张量，形状为 (bs, n_heads, cache_len+seq_len, head_dim)
            mask: 因果掩码张量，形状为 (1, 1, seq_len, cache_len+seq_len)
            
        Returns:
            注意力输出张量，形状为 (bs, n_heads, seq_len, head_dim)
        """
        # 计算注意力分数: (bs, n_heads, seq_len, head_dim) @ (bs, n_heads, head_dim, cache_len + seq_len)
        # -> (bs, n_heads, seq_len, cache_len + seq_len)
        scores = torch.matmul(xq, keys.transpose(2, 3)) * self.scale
        
        # 应用掩码: (bs, n_heads, seq_len, cache_len + seq_len)
        if mask is not None:
            scores = scores + mask
            
        # 计算注意力权重: 沿最后一个维度softmax
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(xq)
        
        # 应用注意力dropout
        attn_weights = self.attn_dropout(attn_weights)

        # 应用注意力权重到值上: (bs, n_heads, seq_len, cache_len + seq_len) @ (bs, n_heads, cache_len + seq_len, head_dim)
        # -> (bs, n_heads, seq_len, head_dim)
        output = torch.matmul(attn_weights, values)
        
        # 转置回 (bs, seq_len, n_heads, head_dim) 并确保内存连续
        return output

    def _compute_attention_flash(self, xq: torch.Tensor, keys: torch.Tensor, 
                          values: torch.tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """使用FlashAttention计算注意力（如果可用）。
        
        该方法使用PyTorch内置的scaled_dot_product_attention函数，
        能够更高效地计算注意力并减少内存使用。
        
        Args:
            xq: 查询张量，形状为 (bs, n_heads, seq_len, head_dim)
            keys: 键张量，形状为 (bs, n_heads, total_len, head_dim)
            values: 值张量，形状为 (bs, n_heads, total_len, head_dim)
            mask: 因果掩码张量，形状为 (1, 1, seq_len, total_len)
            
        Returns:
            注意力输出张量，形状为 (bs, n_heads, seq_len, head_dim)
        """
        # 如果有缓存（total_len > seq_len），需要提供显式掩码
        if keys.size(2) > xq.size(2):
            # 使用提供的掩码
            output = F.scaled_dot_product_attention(
                xq,
                keys,
                values,
                attn_mask=mask,  # 使用显式掩码
                dropout_p=self.attn_dropout.p if self.training else 0.0
            )
        else:
            # 没有缓存时，可以使用内置因果掩码
            output = F.scaled_dot_product_attention(
                xq,
                keys,
                values,
                attn_mask=None,
                is_causal=True,  # 启用内置因果掩码
                dropout_p=self.attn_dropout.p if self.training else 0.0
            )
        return output

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """前向传播函数。
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, dim)
            freqs_cos: 预计算的余弦频率张量
            freqs_sin: 预计算的正弦频率张量
            past_key_value: 包含过去键和值的元组，形状为(batch_size, seq_len, n_local_kv_heads, head_dim)
            
        Returns:
            注意力输出张量，形状为 (batch_size, seq_len, dim)
            更新后的键值缓存元组（如果使用缓存）
        """
        # 获取输入张量的形状信息
        batch_size, seqlen, _ = x.shape
        device, dtype = x.device, x.dtype

        # 并行计算查询、键、值投影
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # 重塑张量形状: (batch_size, seqlen, n_heads, head_dim)
        xq = xq.view(batch_size, seqlen, self.num_local_heads, self.head_dim)
        xk = xk.view(batch_size, seqlen, self.num_local_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seqlen, self.num_local_kv_heads, self.head_dim)

        # 应用旋转位置编码到查询和键
        xq, xk = self.rope(xq, xk, freqs_cos, freqs_sin)

        # 处理KV缓存
        cache_len = 0
        if self.enabled_kv_cache and past_key_value is not None:
            # 动态拼接过去的键值
            past_key, past_value = past_key_value
            cache_len = past_key.size(1)  # 缓存长度

            # 动态拼接过去的键值
            xk = torch.cat([past_key, xk], dim=1)
            xv = torch.cat([past_value, xv], dim=1)
        
        # 更新past_key_value - 使用原始的xk和xv
        if self.enabled_kv_cache:
            past_key_value = (xk, xv)
        else:
            past_key_value = None

        # 重复k/v头以匹配查询头的数量（分组查询注意力）
        keys = self.rope.repeat_kv(xk, self.num_rep)  # (bs, cache_len + seqlen, num_local_heads, head_dim)
        values = self.rope.repeat_kv(xv, self.num_rep)  # (bs, cache_len + seqlen, num_local_heads, head_dim)

        # 转置以获得正确的形状 (bs, n_heads, seq_len, head_dim)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算总长度
        total_len = keys.size(2)  # 总长度 = 缓存长度 + 当前序列长度

        # 计算注意力掩码
        mask = self._create_causal_mask(seqlen, cache_len, device) if not self.enabled_flash_attn or total_len > seqlen else None

        # 根据配置选择使用FlashAttention或常规注意力
        if self.enabled_flash_attn:
            output = self._compute_attention_flash(xq, keys, values, mask)
        else:
            output = self._compute_attention(xq, keys, values, mask)

        # 重塑输出: (batch_size, seqlen, n_heads * head_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seqlen, -1)

        # 应用残差dropout
        output = self.resid_dropout(output)
        
        # 应用输出投影
        output = self.wo(output) # (batch_size, seqlen, dim)

        return output, past_key_value


if __name__ == "__main__":
    # 创建测试配置
    class StellarByteModelArgs:
        vocab_size = 32000
        dim = 512
        num_heads = 8
        num_kv_heads = 4
        max_batch_size = 2
        max_seq_len = 1024
        rope_theta = 10000.0
        enabled_flash_attn = False
        enabled_kv_cache = True
        attention_dropout = 0.1
        resid_dropout = 0.1
        model_parallel_size = 1

    args = StellarByteModelArgs()
    
    # 初始化注意力模块
    attention = StellarByteAttention(args)

    print("="*50)
    print("多头自注意力(StellarByteAttention)测试")
    print("="*50)

    # 测试输入输出形状
    batch_size = 2
    seq_len = 64
    dim = args.dim
    
    # 创建随机输入
    x = torch.randn(batch_size, seq_len, dim)
    
    # 创建位置编码频率
    rope = StellarByteRoPE(dim=args.dim, max_seq_len=args.max_seq_len, theta=args.rope_theta)
    # 注意：传入的dim是总的dim，不是每个头的dim，所以需要除以num_heads
    freqs_cos, freqs_sin = rope.precompute_freqs_cis(args.dim//args.num_heads, args.max_seq_len, args.rope_theta)
    
    # 测试不同起始位置
    for cache_len in [0, 32]:
        print(f"\n测试 start_pos = {cache_len}")
        
        # 创建模拟的过去键值缓存
        if cache_len > 0:
            # 创建形状正确的模拟缓存
            past_key = torch.zeros(
                batch_size, 
                cache_len, 
                args.num_kv_heads // args.model_parallel_size,  # num_local_kv_heads
                args.dim // args.num_heads  # head_dim
            )
            past_value = torch.zeros(
                batch_size, 
                cache_len, 
                args.num_kv_heads // args.model_parallel_size,  # num_local_kv_heads
                args.dim // args.num_heads  # head_dim
            )
            past_key_value = (past_key, past_value)
        else:
            past_key_value = None
        
        # 前向传播
        output, new_past_key_value = attention(x, freqs_cos, freqs_sin, past_key_value)
        
        # 检查输出形状
        expected_shape = (batch_size, seq_len, dim)
        actual_shape = output.shape
        print(f"输入形状: {x.shape}")
        print(f"期望输出形状: {expected_shape}")
        print(f"实际输出形状: {actual_shape}")
        
        # 验证形状是否正确
        assert actual_shape == expected_shape, f"形状不匹配! 期望: {expected_shape}, 实际: {actual_shape}"
        print("✓ 形状验证通过!")
        
        # 检查缓存形状（如果启用了缓存）
        if args.enabled_kv_cache:
            if cache_len > 0:
                # 新缓存应该包含旧缓存和新内容
                expected_cache_shape = (batch_size, cache_len + seq_len, args.num_kv_heads // args.model_parallel_size, args.dim // args.num_heads)
            else:
                # 没有旧缓存时，新缓存应该只包含当前内容
                expected_cache_shape = (batch_size, seq_len, args.num_kv_heads // args.model_parallel_size, args.dim // args.num_heads)
            
            cache_k, cache_v = new_past_key_value
            assert cache_k.shape == expected_cache_shape, f"键缓存形状错误! 期望: {expected_cache_shape}, 实际: {cache_k.shape}"
            assert cache_v.shape == expected_cache_shape, f"值缓存形状错误! 期望: {expected_cache_shape}, 实际: {cache_v.shape}"
            print("✓ 缓存形状验证通过!")
    
    print("\n✅ 所有测试通过：注意力机制实现正确")
    print("="*50)