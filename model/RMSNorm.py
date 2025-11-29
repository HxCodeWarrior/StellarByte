import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMSNorm 实现（轻量归一化）。

    该归一化仅基于均方根（root mean square），与 LayerNorm 不同，
    不减去均值，仅按标准差缩放，参数量更小（只有可学习的 scale）。
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        """初始化 RMSNorm。

        Args:
            dim: 归一化的维度（隐藏层大小）。
            eps: 防止除零的小常数。
        """
        super().__init__()
        # eps 防止数值不稳定
        self.eps = eps
        # 可学习缩放参数，初始为 1 向量
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """计算 RMS 归一化（不包含学习缩放）。

        步骤：
        1. 计算最后一维的均方（mean of squares）
        2. 开平方并加 eps（防止除零）后取倒数
        3. 与原输入相乘实现归一化
        """
        # x.pow(2).mean(-1, keepdim=True)：求最后一维每个位置的均方
        # torch.rsqrt：求倒平方根，等价于 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向计算：先做 RMS 归一化再乘以可学习缩放参数。

        这里将输入转换为 float 做计算以提升数值稳定性，最后再转换回原始 dtype。
        """
        # 先将输入转为 float（fp32）计算，提高稳定性
        normalized = self._norm(x.float())
        # 乘以可学习权重，并转换回输入的 dtype
        return self.weight * normalized.type_as(x)

if __name__ == '__main__':
    # 测试 RMSNorm 前向传播
    rms_norm = RMSNorm(dim=512)
    x = torch.randn(1, 512)  # 模拟一个批量大小为 1，维度为 512 的输入
    output = rms_norm(x)
    print(output.shape)  # 期望输出：torch.Size([1, 512])