import torch
import torch.nn as nn
import torch.jit as jit


@jit.script
def _rms_norm(x: torch.Tensor,
              weight: torch.Tensor,
              eps: float) -> torch.Tensor:
    """
    Fused RMS‑Norm kernel (TorchScript).
    • 先将输入提升到 float32 进行平方与求均值，避免半精度溢出 / 精度损失
    • clamp_min(eps) ⟹ 比 “+ eps” 更稳健，彻底杜绝 0 或极小分母
    • 不使用任何条件分支，方便 JIT / TorchDynamo 图优化
    """
    rms = torch.mean(x.to(torch.float32).pow(2), dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(rms.clamp_min(eps))
    # 把 inv_rms 和 weight 都 cast 到 x.dtype
    inv_rms = inv_rms.to(x.dtype)
    weight   = weight.to(x.dtype)
    # cast 回原 dtype，再点乘权重
    return (x * inv_rms.to(x.dtype)) * weight


class RMSNorm(nn.Module):
    """
    RMSNorm（Root‑Mean‑Square Layer Norm）

    Args
    ----
    dim : int
        最后一个维度大小
    eps : float, optional
        数值稳定常数，默认 1e-6
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        # γ（可学习缩放因子）
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = torch.tensor(eps, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:                # noqa: D401
        # 主逻辑完全委托给 TorchScript 函数，JIT 可原地融合 mul/rsqrt
        return _rms_norm(x, self.weight, self.eps)

if __name__ == "__main__":
    from config import ByteModelConfig

    args = ByteModelConfig()
    norm = RMSNorm(args.model_dim, args.layer_norm_eps)
    x = torch.randn(1, 512, args.model_dim)
    output = norm(x)
    print(output.shape) # torch.Size([1, 512, args.model_dim])