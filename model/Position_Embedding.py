import torch
import torch.nn as nn
from typing import Tuple

class XPosRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, scale_base: float = 512, theta: float = 10000.0):
        super().__init__()
        """
        XPos改进版RoPE
        Args:
            dim: 每个head的维度（应为偶数）
            scale_base: 控制比例缩放因子，默认512是XPos推荐值
            theta: 控制频率基数
        """
        assert head_dim % 2 == 0, "Dimension must be even for rotary embedding."
        self.dim = head_dim
        self.scale_base = scale_base
        self.theta = theta

        # 原始 RoPE 的频率
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _compute_cos_sin(self, seq_len: int, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device=device, dtype=dtype))  # [seq_len, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]
        return torch.cos(emb), torch.sin(emb)

    def _compute_xpos_scale(self, seq_len: int, device, dtype):
        """
        XPos: 构造可学习的缩放因子，使 RoPE 更稳定
        """
        pos = (torch.arange(seq_len, device=device, dtype=dtype) - seq_len // 2) / self.scale_base
        pos = pos.unsqueeze(-1)  # [seq_len, 1]
        scale = torch.exp(pos * (torch.arange(0, self.dim, 2, device=device, dtype=dtype) / self.dim))  # [seq_len, dim/2]
        scale = torch.cat([scale, scale], dim=-1)  # [seq_len, dim]
        return scale

    def _rotate_half(self, x):
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, xq: torch.Tensor, xk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xq, xk: [batch, seq_len, n_head, head_dim]
        Returns:
            xq_rot, xk_rot: 同形状，加入了XPos嵌入
        """
        B, S, H, D = xq.shape
        device, dtype = xq.device, xq.dtype
        if D != self.dim:
            raise ValueError(f"head_dim mismatch: got {D}, expected {self.dim}")

        cos, sin = self._compute_cos_sin(S, device, dtype)  # [S, D]
        scale = self._compute_xpos_scale(S, device, dtype)  # [S, D]

        # broadcasting: [1, S, 1, D]
        cos, sin, scale = cos[None, :, None, :], sin[None, :, None, :], scale[None, :, None, :]

        xq_scaled = xq * scale
        xk_scaled = xk / scale  # 注意这里是除以scale，是XPos的核心稳定性设计

        xq_out = xq_scaled * cos + self._rotate_half(xq_scaled) * sin
        xk_out = xk_scaled * cos + self._rotate_half(xk_scaled) * sin

        return xq_out, xk_out

if __name__ == "__main__":
    from config import ByteModelConfig
    args = ByteModelConfig()

    xq = torch.randn(1, 50, 6, args.model_dim) # bs, seq_len, dim//n_head, n_head_dim
    xk = torch.randn(1, 50, 6, args.model_dim) # bs, seq_len, dim//n_head, n_head_dim

    rotary = XPosRotaryEmbedding(args.model_dim, args.xpos_scale_base, args.xpos_rope_theta)
    xq_rot, xk_rot = rotary(xq, xk)
    print(xq_rot.shape, xk_rot.shape)   # torch.Size([1, 50, 6, 768]) torch.Size([1, 50, 6, 768])
