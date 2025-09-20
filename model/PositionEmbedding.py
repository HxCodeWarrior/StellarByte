import torch
import torch.nn as nn
from typing import Tuple, Optional

class StellarByteRoPE(nn.Module):
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
        
        # 计算余弦和正弦分量
        freqs_cos = torch.cos(freqs)
        freqs_sin = torch.sin(freqs)
        
        return freqs_cos, freqs_sin
    
    def reshape_for_broadcast(self, freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        调整频率张量的形状以便与查询/键张量进行广播
        
        参数:
            freqs_cis: 预计算的频率张量
            x: 查询或键张量
            
        返回:
            重塑后的频率张量，形状适合与x进行广播
        """
        # 获取x的维度数
        ndim = x.ndim
        
        # 断言，确保1在x的维度范围内
        assert 0 <= 1 < ndim
        
        # 断言，确保freqs_cis的形状与x的第二维和最后一维相同
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        
        # 构造一个新的形状，除了第二维和最后一维，其他维度都为1，这样做是为了能够将freqs_cis与x进行广播操作
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        
        # 将freqs_cis调整为新的形状，并返回
        return freqs_cis.view(shape)
    
    def forward(self, xq: torch.Tensor, xk: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将旋转位置编码应用于查询和键张量
        
        参数:
            xq: 查询张量，形状通常为 (batch_size, seq_len, n_heads, head_dim)
            xk: 键张量，形状同上
            freqs_cos: 预计算的余弦频率张量
            freqs_sin: 预计算的正弦频率张量
            
        返回:
            旋转后的查询和键张量
        """
        seq_len = xq.shape[1]
        
        # 将查询和键张量转换为浮点数，并重塑形状以分离实部和虚部
        xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
        xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)
        
        # 重新塑形频率张量以进行广播
        freqs_cos = self.reshape_for_broadcast(freqs_cos[:seq_len], xq_r)
        freqs_sin = self.reshape_for_broadcast(freqs_sin[:seq_len], xq_r)
        
        # 应用旋转，分别计算旋转后的实部和虚部
        xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
        xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
        xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
        xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos
        
        # 将最后两个维度合并，并还原为原始张量的形状
        xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
        xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)
        
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
    import torch
    import torch.nn as nn
    
    # 初始化参数
    batch_size = 2
    seq_len = 128
    n_heads = 8
    head_dim = 64
    max_seq_len = 512
    
    # 创建RoPE实例
    rope = StellarByteRoPE(dim=head_dim, max_seq_len=max_seq_len)
    
    # 预计算频率向量
    freqs_cos, freqs_sin = rope.precompute_freqs_cis(
        dim=head_dim, 
        end=max_seq_len, 
        theta=rope.theta
    )
    
    print("="*50)
    print("旋转位置编码(RoPE)测试")
    print("="*50)
    print(f"预计算频率向量形状:")
    print(f"  freqs_cos: {freqs_cos.shape} (应为: ({max_seq_len}, {head_dim//2}))")
    print(f"  freqs_sin: {freqs_sin.shape} (应为: ({max_seq_len}, {head_dim//2}))")
    
    # 创建随机查询和键张量
    xq = torch.randn(batch_size, seq_len, n_heads, head_dim)
    xk = torch.randn(batch_size, seq_len, n_heads, head_dim)
    
    print("\n原始查询形状:", xq.shape)
    print("原始键形状:", xk.shape)
    
    # 应用旋转位置编码
    xq_rotated, xk_rotated = rope(xq, xk, freqs_cos, freqs_sin)
    
    print("\n旋转后查询形状:", xq_rotated.shape)
    print("旋转后键形状:", xk_rotated.shape)
    
    # 形状验证
    assert xq_rotated.shape == xq.shape, "查询旋转后形状不匹配!"
    assert xk_rotated.shape == xk.shape, "键旋转后形状不匹配!"
    
    # 基本旋转效果验证
    # 创建两个位置不同的相同向量
    vec = torch.randn(1, 1, 1, head_dim).repeat(1, 2, 1, 1)
    vec[:, 1, :, :] = vec[:, 0, :, :]  # 复制相同向量
    
    # 应用旋转位置编码
    vec_rotated, _ = rope(vec, vec, freqs_cos, freqs_sin)
    
    # 验证位置0和位置1的向量不同
    diff = torch.norm(vec_rotated[0, 0] - vec_rotated[0, 1])
    print(f"\n位置差异验证: 相同向量在不同位置旋转后的差异 = {diff.item():.4f} (应大于0)")
    assert diff > 1e-6, "旋转位置编码未产生位置差异!"
    
    print("\n✅ 所有测试通过：旋转位置编码实现正确")
    print("="*50)
