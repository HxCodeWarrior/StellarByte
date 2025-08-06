import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .RMSNorm import ByteRMSNorm
except ImportError:
    from RMSNorm import ByteRMSNorm

class ByteMLP(nn.Module):
    """
    Gated ByteMLP 模块（门控多层感知机）。

    该模块是 Transformer 中 FeedForward 层的改进版本，使用门控机制（如 GEGLU）
    以及高效的 RMSNorm 替代传统 LayerNorm，提升数值稳定性与速度。

    Attributes:
        w13 (nn.Linear): 输入维度到 2 倍隐藏层维度的线性映射（用于门控结构）
        w2 (nn.Linear): 从隐藏层还原回输出维度的线性映射
        norm (ByteRMSNorm): RMSNorm 层，用于归一化输入
        dropout (nn.Dropout): Dropout 层，用于防止过拟合

    Args:
        dim (int): 输入和输出的维度
        hidden_dim (int, optional): 隐藏层维度；若为 None，将自动按 dim 推算并对齐
        multiple_of (int, optional): 隐藏层维度对齐倍数（默认 256）
        dropout (float, optional): dropout 概率（默认 0.1）
        eps (float, optional): RMSNorm 中的数值稳定常数（默认 1e-6）
        bias (bool, optional): 是否在线性层中使用偏置（默认 False）
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = None,
        multiple_of: int = 256,
        dropout: float = 0.1,
        eps: float = 1e-6,
        bias: bool = False,
    ):
        super().__init__()

        # 若未指定 hidden_dim，则按 (8/3)*dim 向上取整为 multiple_of 的倍数
        if hidden_dim is None:
            hidden_dim = int(8 * dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # 将输入投影为 2 倍 hidden_dim（用于门控机制：一半为值分支，一半为门控分支）
        self.w13 = nn.Linear(dim, hidden_dim * 2, bias=bias)

        # 输出投影层，将 hidden_dim 降回 dim
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)

        # 使用 RMSNorm 进行归一化，更快且数值稳定
        self.norm = ByteRMSNorm(dim, eps)

        # Dropout 层，控制过拟合
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。

        Args:
            x (torch.Tensor): 输入张量，形状为 [batch_size, seq_len, dim]

        Returns:
            torch.Tensor: 输出张量，形状为 [batch_size, seq_len, dim]
        """
        # 残差连接输入备份
        residual = x

        # Step 1: 输入归一化处理
        x = self.norm(x)

        # Step 2: 一次线性变换输出两份：值分支 + 门控分支（GEGLU结构）
        x_proj = self.w13(x)  # [B, T, 2 * hidden_dim]

        # Step 3: 沿最后一个维度均分为两部分
        x_gate, x_value = x_proj.chunk(2, dim=-1)

        # Step 4: 对值分支使用 SiLU 激活函数（替代 ReLU，平滑且性能好）
        x_value = F.silu(x_value)

        # Step 5: 对门控分支使用 Sigmoid 激活，输出 (0,1) 区间的门控权重
        x_gate = torch.sigmoid(x_gate)

        # Step 6: 门控机制：逐元素乘，控制信息流通强度
        x = x_value * x_gate  # [B, T, hidden_dim]

        # Step 7: 输出映射回原维度
        x = self.w2(x)  # [B, T, dim]

        # Step 8: Dropout 处理防止过拟合
        x = self.dropout(x)

        # Step 9: 残差连接
        output = x + residual

        return output

if __name__ == '__main__':
    from config import ByteModelConfig

    # 初始化模型配置（假设 config 提供了基本结构参数）
    args = ByteModelConfig(
        model_dim=128,                # 输入维度
        num_attention_heads=8,        # 多头注意力数（未使用于此 MLP）
        dim_multiplier=4,             # 隐藏层维度放大倍率（未直接用到）
        residual_dropout_prob=0.1,    # Dropout 概率
    )

    # 创建 ByteMLP 实例
    ByteMLP = ByteMLP(args.model_dim)

    # 构造输入张量 [batch=2, seq_len=16, dim=128]
    x = torch.randn(2, 16, args.model_dim)

    # 前向传播
    output = ByteMLP(x)

    # 打印输入输出维度
    print("Input shape : {}".format(x.shape))      # torch.Size([2, 16, 128])
    print("Output shape: {}".format(output.shape)) # torch.Size([2, 16, 128])
