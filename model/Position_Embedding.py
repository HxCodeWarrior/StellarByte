import torch
import math

# ByteDynamicRoPE: 动态旋转位置编码模块
# 支持动态频率缩放（NTK-RoPE）与工业级缓存机制，用于长序列Transformer模型的Q/K编码
class ByteDynamicRoPE:
    def __init__(
        self, 
        dim: int, 
        base_theta: float = 10000.0, 
        ntk_alpha: float = 1.0,
        max_seq_len: int = 2048, 
        device=None
    ):
        """
        构造函数：

        :param dim: attention head 维度(应为偶数)
        :param base_theta: 控制频率衰减的基数(一般为 10000.0)
        :param ntk_alpha: NTK动态缩放因子(>1时启用动态NTK)
        :param max_seq_len: 最大序列长度(用于缓存 sin/cos 表)
        :param device: 计算设备(默认优先使用 CUDA)
        """
        assert dim % 2 == 0, "RoPE 位置编码要求 dim 必须是偶数"

        self.dim = dim
        self.base_theta = base_theta
        self.ntk_alpha = ntk_alpha
        self.max_seq_len = max_seq_len
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 预计算基础频率
        self._compute_base_freq()

        # 预计算静态频率，缓存下来用于旋转编码
        self._build_cos_sin_table(max_seq_len)

    def _compute_base_freq(self):
        """频率计算"""
        dim_half = self.dim // 2
        # 公式：theta_j = base_theta^{-2j/dim}
        theta = self.base_theta ** (-2 * torch.arange(0, dim_half, 1) / self.dim)
        self.base_freq = theta.to(self.device)
        
    def _ntk_scale_factor(self, seq_len: int) -> float:
        """动态NTK缩放因子"""
        if seq_len <= self.max_seq_len or self.ntk_alpha <= 0:
            return 1.0
        
        # 公式：(L_curr/L_train)^(dim/(dim-2))
        ratio = seq_len / self.max_seq_len
        exponent = self.dim / (self.dim - 2)
        return self.ntk_alpha * (ratio ** exponent)

    def _build_cos_sin_table(self, seq_len: int):
        """
        构建 cos/sin 表，用于后续旋转编码。
        """
        # NTK 动态缩放
        scale = self._ntk_scale_factor(seq_len)
        inv_freq = self.base_freq / scale

        # 生成每个位置的索引(0 ~ seq_len - 1)
        t = torch.arange(seq_len, device=self.device).float()  # [seq_len]

        # 计算位置与频率的乘积：位置 * 频率(广播成二维矩阵)
        freqs = torch.outer(t, inv_freq)  # [seq_len, dim_half]

        # 预计算 cos 和 sin 值，缓存下来(RoPE 的核心)
        self.cos_cached = freqs.cos()  # [seq_len, dim/2]
        self.sin_cached = freqs.sin()  # [seq_len, dim/2]

        # 预扩展维度用于广播 [1, max_seq_len, 1, dim/2]
        self.cos_cached = self.cos_cached[None, :, None, :]
        self.sin_cached = self.sin_cached[None, :, None, :]

    def _apply_rotary_half(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        将 cos/sin 应用于向量 x 的旋转部分(即最后一维)

        :param x: [..., head_dim]
        :param cos: [1, seq_len, 1, dim/2]
        :param sin: [1, seq_len, 1, dim/2]

        :return: 旋转后向量，形状与 x 相同[q_rot, k_rot]
        """
        # 将x的最后一维拆分为复数对：x = [x0, x1, x2, x3, ...] -> 
        # x_complex = [[x0, x1], [x2, x3], ...]
        x_complex = x.float().view(*x.shape[:-1], -1, 2)  # [..., dim/2, 2]
        
        # 分离实部和虚部
        x_real = x_complex[..., 0]  # [..., dim/2]
        x_imag = x_complex[..., 1]  # [..., dim/2]
        
        # 复数旋转：(x_real + i*x_imag) * (cos + i*sin)
        # = (x_real*cos - x_imag*sin) + i*(x_real*sin + x_imag*cos)
        x_rot_real = x_real * cos - x_imag * sin
        x_rot_imag = x_real * sin + x_imag * cos
        
        # 重组旋转后向量
        x_rotated = torch.stack([x_rot_real, x_rot_imag], dim=-1)  # [..., dim/2, 2]
        return x_rotated.flatten(-2, -1).type_as(x)  # [..., dim]

    def apply_rotary(self, q: torch.Tensor, k: torch.Tensor, seq_offset: int = 0):
        """
        同时处理 Q 和 K 的旋转位置编码

        :param q: Q 向量 [batch, seq_len, num_heads, head_dim]
        :param k: K 向量 [batch, seq_len, num_heads, head_dim]
        :param seq_offset: 位置偏移量

        :return: (旋转后的Q, 旋转后的K)
        """
        seq_len = q.shape[1]
        head_dim = q.shape[-1]
        assert head_dim == self.dim, "Head dim 必须和 RoPE 初始化时一致"

        # 构建并缓存 cos/sin 编码表
        if seq_len + seq_offset > self.max_seq_len:
            self._build_cos_sin_table(seq_len + seq_offset)

        # 从缓存中提取当前序列位置对应的 cos/sin
        cos = self.cos_cached[:, seq_offset:seq_offset+seq_len, :, :]  # [1, seq_len, 1, dim/2]
        sin = self.sin_cached[:, seq_offset:seq_offset+seq_len, :, :]  # [1, seq_len, 1, dim/2]

        # 统一应用旋转逻辑
        q_rotated = self._apply_rotary_half(q, cos, sin)
        k_rotated = self._apply_rotary_half(k, cos, sin)
        
        return q_rotated, k_rotated

if __name__ == '__main__':
    dim = 128
    batch_size = 2
    seq_len = 16
    num_heads = 8
    ntk_alpha = 1.0
    rope = ByteDynamicRoPE(dim=dim, ntk_alpha=ntk_alpha)

    q = torch.randn(batch_size, seq_len, num_heads, dim) 
    k = torch.randn(batch_size, seq_len, num_heads, dim)

    print(f"Q shape : {q.shape}\nK shape : {k.shape}")
