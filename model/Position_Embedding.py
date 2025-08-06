import torch
import math

class ByteDynamicRoPE:
    """ByteDynamicRoPE 动态旋转位置编码模块。

    支持长序列的 Rotary Position Embedding（RoPE），引入了 NTK-RoPE 的动态频率缩放机制，
    并配有工业级的缓存以加速长序列 Q/K 编码。

    Attributes:
        dim (int): 注意力头维度，必须为偶数。
        base_theta (float): 控制频率衰减的基数，默认值为 10000。
        ntk_alpha (float): NTK 动态缩放因子，大于 1 时启用动态频率缩放。
        max_seq_len (int): 默认最大序列长度，用于构建和缓存 sin/cos 表。
        device (torch.device): 模块运算使用的设备（CPU/GPU）。
    """

    def __init__(
        self, 
        dim: int, 
        base_theta: float = 10000.0, 
        ntk_alpha: float = 1.0,
        max_seq_len: int = 2048, 
        device=None
    ):
        """初始化 ByteDynamicRoPE 模块。

        Args:
            dim (int): 每个注意力头的维度，必须为偶数。
            base_theta (float, optional): 控制频率衰减的基数。默认为 10000.0。
            ntk_alpha (float, optional): NTK 动态缩放系数，通常 >1 启用动态 NTK-RoPE。默认为 1.0。
            max_seq_len (int, optional): 初始最大序列长度，用于缓存频率表。默认为 2048。
            device (torch.device, optional): 使用的计算设备。默认自动选择 CUDA。
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
        """计算 RoPE 基础频率表，用于构建 sin/cos 表"""
        dim_half = self.dim // 2
        # 公式：theta_j = base_theta^{-2j/dim}
        theta = self.base_theta ** (-2 * torch.arange(0, dim_half, 1) / self.dim)
        self.base_freq = theta.to(self.device)
        
    def _ntk_scale_factor(self, seq_len: int) -> float:
        """计算动态 NTK 缩放因子。

        Args:
            seq_len (int): 当前输入序列长度。

        Returns:
            float: 动态频率缩放因子，若不启用 NTK 或序列较短，则为 1.0。
        """
        if seq_len <= self.max_seq_len or self.ntk_alpha <= 0:
            return 1.0
        
        # 公式：(L_curr/L_train)^(dim/(dim-2))
        ratio = seq_len / self.max_seq_len
        exponent = self.dim / (self.dim - 2)
        return self.ntk_alpha * (ratio ** exponent)

    def _build_cos_sin_table(self, seq_len: int):
        """构建并缓存给定序列长度的 cos/sin 表。

        Args:
            seq_len (int): 要构建的位置编码表的序列长度。
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
        """对张量 x 应用旋转位置编码。

        Args:
            x (torch.Tensor): 输入张量 [..., head_dim]。
            cos (torch.Tensor): 对应位置的 cos 编码表 [1, seq_len, 1, dim/2]。
            sin (torch.Tensor): 对应位置的 sin 编码表 [1, seq_len, 1, dim/2]。

        Returns:
            torch.Tensor: 应用旋转位置编码后的张量，形状同 x。
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
        """对 Q/K 向量应用 RoPE 编码。

        Args:
            q (torch.Tensor): 查询向量 Q，形状为 [batch, seq_len, num_heads, head_dim]。
            k (torch.Tensor): 键向量 K，形状为 [batch, seq_len, num_heads, head_dim]。
            seq_offset (int, optional): 当前序列在整个上下文中的偏移位置。默认值为 0。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 旋转位置编码后的 (Q, K)。
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
