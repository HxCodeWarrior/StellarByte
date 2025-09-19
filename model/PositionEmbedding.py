import torch
import torch.nn as nn
from typing import Tuple, Optional

class StellarByteRotaryPositionEmbedding(nn.Module):
    """
    旋转位置编码 (Rotary Position Embedding, RoPE) 实现类
    
    RoPE 是一种通过旋转矩阵将位置信息编码到Transformer的查询和键向量中的方法。
    它能够使模型感知token的绝对位置和相对位置信息，同时具有良好的外推性。
    
    参考文献: 
    RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)
    """
    
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        """
        初始化旋转位置编码
        
        参数:
            dim: 特征维度（通常是注意力头的维度）
            max_seq_len: 支持的最大序列长度
            theta: 频率计算的基数，控制波长范围（默认10000.0）
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # 预计算频率矩阵（复数形式）
        self.register_buffer('freqs_cis', self.precompute_freqs_cis(dim, max_seq_len, theta))
        
    @staticmethod
    def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
        """
        预计算旋转位置编码所需的复数频率向量
        
        参数:
            dim: 特征维度
            end: 序列最大长度
            theta: 频率计算的基数
            
        返回:
            freqs_cis: 形状为 (end, dim//2) 的复数张量
        """
        # 计算频率向量: [1/(theta^(0/dim)), 1/(theta^(2/dim)), ..., 1/(theta^((dim-2)/dim))]
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        
        # 创建位置索引 [0, 1, 2, ..., end-1]
        t = torch.arange(end, device=freqs.device, dtype=torch.float32)
        
        # 计算外积: 位置索引 × 频率向量
        # 结果形状: (end, dim//2)，每个元素表示位置pos在维度i上的旋转角度
        freqs = torch.outer(t, freqs)
        
        # 将角度转换为复数形式: e^(i * angle) = cos(angle) + i * sin(angle)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # 复数形式
        
        return freqs_cis
    
    def reshape_for_broadcast(self, freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        调整频率张量的形状以便与查询/键张量进行广播
        
        参数:
            freqs_cis: 预计算的频率张量
            x: 查询或键张量
            
        返回:
            重塑后的频率张量，形状适合与x进行广播
        """
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        
        # 创建新形状: 保持序列长度和头维度不变，其他维度设置为1
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        
        return freqs_cis.view(*shape)
    
    def forward(self, xq: torch.Tensor, xk: torch.Tensor, start_pos: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将旋转位置编码应用于查询和键张量
        
        参数:
            xq: 查询张量，形状通常为 (batch_size, seq_len, n_heads, head_dim)
            xk: 键张量，形状同上
            start_pos: 起始位置，用于处理超过预计算长度的序列
            
        返回:
            旋转后的查询和键张量
        """
        seq_len = xq.shape[1]
        
        # 如果序列长度超过预计算长度，动态扩展频率矩阵
        if start_pos + seq_len > self.freqs_cis.shape[0]:
            self.freqs_cis = self.precompute_freqs_cis(
                self.dim, start_pos + seq_len, self.theta
            ).to(xq.device)
        
        # 获取当前序列对应的频率切片
        freqs_cis = self.freqs_cis[start_pos:start_pos + seq_len]
        
        # 将查询和键转换为复数形式
        # 将最后维度分成实部和虚部两部分，然后转换为复数
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        
        # 调整频率张量形状以匹配查询张量
        freqs_cis = self.reshape_for_broadcast(freqs_cis, xq_)
        
        # 应用旋转: 复数乘法相当于旋转操作
        # 每个复数对(x+iy)乘以(cosθ + isinθ)实现旋转
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        
        # 返回与输入相同数据类型的旋转后的查询和键
        return xq_out.type_as(xq), xk_out.type_as(xk)
    
    def repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        重复键值头，用于分组查询注意力机制
        
        参数:
            x: 键或值张量，形状为 (bs, seq_len, n_kv_heads, head_dim)
            n_rep: 重复次数
            
        返回:
            重复后的张量，形状为 (bs, seq_len, n_kv_heads * n_rep, head_dim)
        """
        bs, slen, n_kv_heads, head_dim = x.shape
        
        if n_rep == 1:
            return x
        
        # 通过添加新维度并扩展来重复键值头
        return (
            x[:, :, :, None, :]
            .expand(bs, slen, n_kv_heads, n_rep, head_dim)
            .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
        )


# 使用示例
if __name__ == "__main__":
    # 初始化参数
    batch_size = 1
    seq_len = 128
    n_heads = 8
    head_dim = 64
    
    # 创建RoPE实例
    rope = StellarByteRotaryPositionEmbedding(dim=head_dim, max_seq_len=512)
    
    # 创建随机查询和键张量
    xq = torch.randn(batch_size, seq_len, n_heads, head_dim)
    xk = torch.randn(batch_size, seq_len, n_heads, head_dim)
    
    # 应用旋转位置编码
    xq_rotated, xk_rotated = rope(xq, xk)
    
    print("原始查询形状:", xq.shape)
    print("旋转后查询形状:", xq_rotated.shape)
    print("旋转位置编码应用成功!")