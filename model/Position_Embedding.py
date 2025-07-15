import torch
import torch.nn as nn
from typing import Tuple, Optional

class RotaryCache:
    """
    缓存 cos/sin/scale 三个张量，按 (device, dtype, dim, theta, scale_base) 为键，支持动态扩容
    """
    _store = {}

    @classmethod
    def get(
        cls,
        device: torch.device,
        dtype: torch.dtype,
        seq_len: int,
        dim: int,
        theta: float,
        scale_base: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (device, dtype, dim, theta, scale_base)
        if key not in cls._store or cls._store[key][0].size(0) < seq_len:
            t = torch.arange(seq_len, device=device, dtype=dtype)
            inv_freq = 1.0 / (theta ** (
                torch.arange(0, dim, 2, device=device, dtype=dtype) / dim
            ))
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]
            cos = torch.cos(emb)
            sin = torch.sin(emb)

            pos = (t - seq_len // 2) / scale_base
            log_s = pos.unsqueeze(-1) * (
                torch.arange(0, dim, 2, device=device, dtype=dtype) / dim
            )
            log_s.clamp_(min=-12.0, max=12.0)
            scale_half = torch.exp(log_s)
            scale = torch.cat([scale_half, scale_half], dim=-1)

            cls._store[key] = (cos, sin, scale)

        cos, sin, scale = cls._store[key]
        # 保证返回张量 dtype 与请求一致
        if cos.dtype != dtype:
            cos, sin, scale = [t.to(dtype) for t in (cos, sin, scale)]
        return cos[:seq_len], sin[:seq_len], scale[:seq_len]



class XPosRotaryEmbedding(nn.Module):
    def __init__(self, 
                 head_dim: int, 
                 scale_base: float = 512, 
                 theta: float = 10000.0,
                 learnable_scale: bool = True,
                 extrapolation: Optional[str] = None
    ):
        super().__init__()
        """
        XPos
        Args:
            dim: 每个head的维度（应为偶数）
            scale_base: 控制比例缩放因子，默认512是XPos推荐值
            theta: 控制频率基数
            learnable_scale: 是否使用可学习的缩放因子
            extrapolation: 用于控制频率的线性插值方式
        """
        assert head_dim % 2 == 0, "Dimension must be even for rotary embedding."
        self.dim = head_dim
        self.scale_base = scale_base
        self.theta = theta
        self.learnable_scale = learnable_scale
        self.extrapolation = extrapolation

        # 原始 RoPE 的频率
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 可学习的log_scale_weight参数，初始化为线性分布
        if self.learnable_scale:
            init_scale = torch.arange(0, head_dim // 2).float() / (head_dim // 2)
            self.log_scale_weight = nn.Parameter(init_scale)  # 形状 (dim/2,)
        else:
            self.register_buffer("log_scale_weight", torch.arange(0, head_dim // 2).float() / (head_dim // 2))

    def _get_cos_sin_scale(self, seq_len: int, device, dtype):
        # 使用缓存
        cos, sin, scale = RotaryCache.get(device, dtype, seq_len, self.dim, self.theta, self.scale_base)
        # 如果使用learnable scale，需要覆盖scale
        if self.learnable_scale:
            scale = self._compute_xpos_scale(seq_len, device, dtype)
        return cos, sin, scale

    def _compute_xpos_scale(self, seq_len: int, device, dtype):
        """
        XPos: 构造可学习的缩放因子，使 RoPE 更稳定
        """
        pos = (torch.arange(seq_len, device=device, dtype=dtype) - seq_len // 2) / self.scale_base
        pos = pos.unsqueeze(-1)  # [seq_len, 1]
        # 利用log_scale_weight作为缩放权重，广播至seq_len
        scale = torch.exp(pos * self.log_scale_weight.unsqueeze(0))  # [seq_len, dim/2]
        scale = torch.cat([scale, scale], dim=-1)  # [seq_len, dim]
        return scale

    @staticmethod
    def _rotate_half(x):
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

        cos, sin, scale = self._get_cos_sin_scale(S, device, dtype)  # [S, D]

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
