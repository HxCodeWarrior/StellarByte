import torch
import torch.nn as nn
import torch.jit as jit

@jit.script
def _rms_norm(x: torch.Tensor,
              weight: torch.Tensor,
              eps: float) -> torch.Tensor:
    """
    使用 TorchScript 加速的 RMSNorm 核心函数。

    RMSNorm（Root Mean Square Layer Normalization）是 LayerNorm 的一种变体，
    其不减去均值，仅通过均方根值进行归一化，更适用于自回归模型中。

    本函数为 JIT-fused 核心计算逻辑，无条件分支以利于图优化。

    Args:
        x (torch.Tensor): 输入张量，形状为 [..., dim]
        weight (torch.Tensor): 可学习的缩放因子，形状为 [dim]
        eps (float): 数值稳定项，防止除以 0（或极小值）

    Returns:
        torch.Tensor: 归一化后的输出张量，形状同 x
    """
    # Step 1: 计算 x 的平方均值（均方）
    # 先将 x 转为 float32，避免半精度溢出或精度损失
    rms = torch.mean((x * x).float(), dim=-1, keepdim=True)

    # Step 2: 对均方值加稳定项并开平方的倒数（即：1 / sqrt(mean(x^2))）
    # clamp_min 比 "+ eps" 更稳健，可避免除以接近 0 的值
    inv_rms = torch.rsqrt(rms.clamp_min(eps))

    # Step 3: 将 inv_rms 和 weight 转回 x 的数据类型（如 float16/bfloat16）
    inv_rms = inv_rms.to(x.dtype)
    weight  = weight.to(x.dtype)

    # Step 4: 应用 RMSNorm：先乘以 inv_rms 再乘以可学习参数 weight
    return (x * inv_rms) * weight


class ByteRMSNorm(nn.Module):
    """
    ByteRMSNorm 模块：一种高效的 RMSNorm 实现，用于替代 LayerNorm。

    RMSNorm 仅基于均方根缩放向量，不减均值，具有更快的推理速度与更低的数值不稳定性。

    Attributes:
        weight (torch.nn.Parameter): 可学习缩放参数 γ，形状为 [dim]
        eps (torch.Tensor): 数值稳定项 ε，保存在 buffer 中供推理使用

    Args:
        dim (int): 输入张量的最后一个维度大小
        eps (float, optional): 数值稳定常数，默认为 1e-6
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """
        构造 ByteRMSNorm 层。

        Args:
            dim (int): 输入特征维度
            eps (float, optional): 数值稳定项，防止除以零。默认值为 1e-6。
        """
        super().__init__()

        # 创建可学习参数 γ，初始化为全 1，控制每个维度的缩放比例
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

        # 注册 ε 为 buffer（非参数），确保推理阶段一致性与可持久化
        self.register_buffer("eps", torch.tensor(eps, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """
        前向传播，应用 RMSNorm 归一化。

        Args:
            x (torch.Tensor): 输入张量，形状为 [..., dim]

        Returns:
            torch.Tensor: 归一化输出张量，形状与输入相同
        """
        # 直接调用 TorchScript 加速函数完成归一化
        return _rms_norm(x, self.weight, self.eps)


if __name__ == "__main__":
    from config import ByteModelConfig

    args = ByteModelConfig()
    norm = ByteRMSNorm(128, args.layer_norm_eps)
    x = torch.randn(2, 16, 128)
    output = norm(x)
    print(f'Input shape: {x.shape}\nOutPut shape: {output.shape}') # torch.Size([2, 16, 128)