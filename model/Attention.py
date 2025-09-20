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
from typing import Optional


try:
    from .config import StellarByteModelArgs
    from .PositionEmbedding import StellarByteRoPE
except:
    from config import StellarByteModelArgs


class StellarByteAttention(nn.Module):
    """实现StellarByte模型的注意力机制，支持模型并行和FlashAttention。
    
    该模块实现了基于旋转位置编码(RoPE)的注意力机制，支持分组查询注意力(GQA)和
    模型并行计算。可以根据配置选择使用常规注意力或FlashAttention实现。
    
    Attributes:
        num_kv_heads (int): 键值头的数量
        n_local_heads (int): 本地查询头数量
        n_local_kv_heads (int): 本地键值头数量
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
        model_parallel_size = fs_init.get_model_parallel_world_size()
        # 计算每个GPU上的本地查询头数量
        self.n_local_heads = args.num_heads // model_parallel_size
        # 计算每个GPU上的本地键值头数量
        self.n_local_kv_heads = self.num_kv_heads // model_parallel_size
        # 计算每个键值头需要重复的次数以匹配查询头数量
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 计算每个注意力头的维度
        self.head_dim = args.dim // args.num_heads
        # 设置缩放因子，用于缩放注意力分数
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # 检查是否支持FlashAttention（PyTorch 2.0+）
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            self.enabled_flash_attn = args.enabled_flash_attn
        else:
            self.enabled_flash_attn = False
            print("FlashAttention is not available, using regular attention")

        # 初始化旋转位置编码
        self.rope = StellarByteRoPE(
            args.dim,  # 模型维度
            args.max_seq_len,  # 最大序列长度
            args.rope_theta,  # RoPE的theta参数
        )

        # 使用Xavier均匀分布初始化权重
        init_method = nn.init.xavier_uniform_

        # 初始化查询投影层（列并行）
        self.wq = ColumnParallelLinear(
            args.dim,  # 输入维度
            args.num_heads * self.head_dim,  # 输出维度（所有查询头的总维度）
            bias=False,  # 不使用偏置
            gather_output=False,  # 不收集输出（保持并行）
            init_method=init_method,  # 初始化方法
        )
        
        # 初始化键投影层（列并行）
        self.wk = ColumnParallelLinear(
            args.dim,  # 输入维度
            self.num_kv_heads * self.head_dim,  # 输出维度（所有键头的总维度）
            bias=False,  # 不使用偏置
            gather_output=False,  # 不收集输出
            init_method=init_method,  # 初始化方法
        )
        
        # 初始化值投影层（列并行）
        self.wv = ColumnParallelLinear(
            args.dim,  # 输入维度
            self.num_kv_heads * self.head_dim,  # 输出维度（所有值头的总维度）
            bias=False,  # 不使用偏置
            gather_output=False,  # 不收集输出
            init_method=init_method,  # 初始化方法
        )
        
        # 初始化输出投影层（行并行）
        self.wo = RowParallelLinear(
            args.num_heads * self.head_dim,  # 输入维度（所有头的总维度）
            args.dim,  # 输出维度（模型维度）
            bias=False,  # 不使用偏置
            input_is_parallel=True,  # 输入已经是并行的
            init_method=init_method,  # 初始化方法
        )

        # 设置是否使用KV缓存
        self.enabled_kv_cache = args.enabled_kv_cache

        # 只有在使用KV缓存时才初始化缓存
        if self.enabled_kv_cache:
            # 延迟缓存初始化，减少内存使用
            # 注册键缓存缓冲区（不持久化，不保存到检查点）
            self.register_buffer('cache_k', torch.zeros(
                args.max_batch_size,  # 最大批处理大小
                args.max_seq_len,  # 最大序列长度
                self.n_local_kv_heads,  # 本地键头数量
                self.head_dim,  # 头维度
            ), persistent=False)

            # 注册值缓存缓冲区
            self.register_buffer('cache_v', torch.zeros(
                args.max_batch_size,  # 最大批处理大小
                args.max_seq_len,  # 最大序列长度
                self.n_local_kv_heads,  # 本地值头数量
                self.head_dim,  # 头维度
            ), persistent=False)
        
            # 标记缓存是否已初始化
            self.cache_initialized = False
        else:
            # 不使用缓存时，设置为None
            self.register_buffer('cache_k', None, persistent=False)
            self.register_buffer('cache_v', None, persistent=False)
            self.cache_initialized = True  # 设置为已初始化，避免后续检查

        # 注册因果掩码缓冲区
        self.register_buffer('causal_mask', None, persistent=False)
        # 跟踪当前掩码大小
        self.current_mask_size = 0

        # 初始化注意力dropout和残差dropout
        self.attn_dropout = nn.Dropout(args.attention_dropout)
        self.resid_dropout = nn.Dropout(args.resid_dropout)

    def _initialize_cache(self, device, dtype):
        """延迟初始化缓存，减少初始内存使用。
        
        该方法在第一次前向传播时被调用，将缓存移动到正确的设备和数据类型。
        
        Args:
            device: 目标设备
            dtype: 目标数据类型
        """
        if not self.cache_initialized:
            # 将缓存移动到指定设备和数据类型
            self.cache_k = self.cache_k.to(device=device, dtype=dtype)
            self.cache_v = self.cache_v.to(device=device, dtype=dtype)
            self.cache_initialized = True  # 标记缓存已初始化

    def reset_cache(self) -> None:
        """重置KV缓存。
        
        该方法用于在需要时重置缓存状态，例如在处理新序列时。
        """
        if self.enabled_kv_cache and self.cache_initialized:
            # 将缓存重置为零
            self.cache_k.zero_()
            self.cache_v.zero_()

    def _create_causal_mask(self, seq_len: int, start_pos: int, device: torch.device) -> torch.Tensor:
        """更新或创建因果掩码，使用缓冲区避免重复分配。
        
        该方法创建或复用因果掩码，防止模型关注未来的位置信息。
        
        Args:
            seq_len: 当前序列长度
            start_pos: 当前序列在缓存中的起始位置
            device: 掩码所在的设备
            
        Returns:
            适当大小的因果掩码张量
        """
        # 如果不使用缓存，则总长度就是当前序列长度
        if not self.enabled_kv_cache:
            total_len = seq_len
        else:
            # 计算总长度（当前位置 + 序列长度）
            total_len = start_pos + seq_len
        
        # 如果已有足够大的掩码，直接使用
        if (self.causal_mask is not None and 
            self.causal_mask.size(2) >= seq_len and 
            self.causal_mask.size(3) >= total_len):
            # 返回适当切片
            return self.causal_mask[:, :, :seq_len, :total_len]
        
        # 否则创建新掩码并缓存
        self.current_mask_size = max(self.current_mask_size, total_len)
        
        # 创建完整的因果掩码（上三角为负无穷）
        mask = torch.full((total_len, total_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)  # 保留主对角线上方的元素
        
        # 注册为缓冲区以便在不同设备间正确移动
        # 添加批次和头维度 (1, 1, seq_len, total_len)
        self.register_buffer('causal_mask', mask.unsqueeze(0).unsqueeze(0), persistent=False)
        
        # 返回适当大小的切片
        return self.causal_mask[:, :, :seq_len, :total_len]

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
        scores = scores + mask
            
        # 计算注意力权重: 沿最后一个维度softmax
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(xq)
        
        # 应用注意力dropout
        attn_weights = self.attn_dropout(attn_weights)

        # 应用注意力权重到值上: (bs, n_heads, seq_len, cache_len + seq_len) @ (bs, n_heads, cache_len + seq_len, head_dim)
        # -> (bs, n_heads, seq_len, head_dim)
        output = torch.matmul(attn_weights, values)
        
        # 转置回 (bs, seq_len, n_heads, head_dim) 并确保内存连续
        return output.transpose(1, 2).contiguous()

    def _compute_attention_flash(self, xq: torch.Tensor, keys: torch.Tensor, 
                          values: torch.Tensor) -> torch.Tensor:
        """使用FlashAttention计算注意力（如果可用）。
        
        该方法使用PyTorch内置的scaled_dot_product_attention函数，
        能够更高效地计算注意力并减少内存使用。
        
        Args:
            xq: 查询张量，形状为 (bs, n_heads, seq_len, head_dim)
            keys: 键张量，形状为 (bs, n_heads, cache_len+seq_len, head_dim)
            values: 值张量，形状为 (bs, n_heads, cache_len+seq_len, head_dim)
            
        Returns:
            注意力输出张量，形状为 (bs, n_heads, seq_len, head_dim)
        """
        # FlashAttention期望输入形状: (batch_size, seq_len, n_heads, head_dim)
        # 并且会自动处理因果掩码
        output = F.scaled_dot_product_attention(
            xq,
            keys,
            values,
            attn_mask=None,  # FlashAttention自动处理因果掩码
            is_causal=True,  # 启用因果掩码
            dropout_p=self.attn_dropout.p if self.training else 0.0  # 训练时使用dropout
        )
        # 转置回 (bs, n_heads, seq_len, head_dim) 并确保内存连续
        return output.transpose(1, 2).contiguous()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
    ):
        """前向传播函数。
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, dim)
            start_pos: 当前序列在缓存中的起始位置
            freqs_cis: 预先计算的频率张量，用于旋转位置编码
            
        Returns:
            注意力输出张量，形状为 (batch_size, seq_len, dim)
        """
        # 获取输入张量的形状信息
        batch_size, seqlen, _ = x.shape
        device, dtype = x.device, x.dtype
        
        # 延迟初始化缓存（减少初始内存使用）
        if self.enabled_kv_cache:
            self._initialize_cache(device, dtype)

        # 并行计算查询、键、值投影
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # 重塑张量形状: (batch_size, seqlen, n_heads, head_dim)
        xq = xq.view(batch_size, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(batch_size, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seqlen, self.n_local_kv_heads, self.head_dim)

        # 应用旋转位置编码到查询和键
        xq, xk = self.rope(xq, xk, freqs_cis=freqs_cis)

        # 处理KV缓存
        if self.enabled_kv_cache:
            # 确保缓存与当前张量在同一设备上
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)
            
            # 更新键值缓存（将当前序列插入缓存中的指定位置）
            self.cache_k[:batch_size, start_pos : start_pos + seqlen] = xk
            self.cache_v[:batch_size, start_pos : start_pos + seqlen] = xv

            # 获取完整的键值缓存（从开始到当前位置+序列长度）
            keys = self.cache_k[:batch_size, : start_pos + seqlen]
            values = self.cache_v[:batch_size, : start_pos + seqlen]
        else:
            # 不使用缓存，直接使用当前键值
            keys, values = xk, xv

        # 重复k/v头以匹配查询头的数量（分组查询注意力）
        keys = self.rope.repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = self.rope.repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        # 创建因果掩码，防止关注未来信息
        mask = self._create_causal_mask(seqlen, start_pos, xq.device)

        # 转置以获得正确的形状 (bs, n_heads, seq_len, head_dim)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # 根据配置选择使用FlashAttention或常规注意力
        if self.enabled_flash_attn:
            # 对于FlashAttention，需要确定是否使用因果掩码
            is_causal = self.enabled_kv_cache or (start_pos == 0)  # 如果使用缓存或从头开始，则使用因果掩码
            output = self._compute_attention_flash(xq, keys, values)
        else:
            output = self._compute_attention(xq, keys, values, mask)

        # 重塑输出: (batch_size, seqlen, n_heads * head_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seqlen, -1)

        # 应用残差dropout
        output = self.resid_dropout(output)
        
        # 应用输出投影
        output = self.wo(output)

        return output  # (batch_size, seqlen, dim)