import torch
import torch.nn as nn
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear
)
from typing import Optional

try:
    from .config import StellarByteModelArgs
    from .RMSNorm import StellarByteRMSNorm
except ImportError:
    from config import StellarByteModelArgs
    from RMSNorm import StellarByteRMSNorm

class StellarByteFeedForward(nn.Module):
    """
    门控多层感知机(Gated MLP)模块，Transformer中前馈网络的改进实现。

    核心特点:
    - 使用门控机制(GEGLU)增强非线性表达能力
    - 采用RMSNorm替代LayerNorm提升计算效率和稳定性
    - 支持隐藏层维度自动对齐优化

    属性:
        w1 (nn.Linear): 输入投影层，输出值分支（GEGLU结构）
        w2 (nn.Linear): 输出投影层，将隐藏层映射回原始维度
        w3 (nn.Linear): 输出投影层，输出门控权重
        norm (StellarByteRMSNorm): RMS归一化层
        dropout (nn.Dropout): 随机失活层防止过拟合

    参数:
        dim (int): 输入/输出特征维度
        hidden_dim (int, optional): 隐藏层维度，未指定时自动计算
        multiple_of (int, optional): 隐藏层维度对齐基数(默认256)
        dropout (float, optional): 随机失活概率(默认0.1)
        eps (float, optional): RMSNorm的数值稳定常数(默认1e-6)
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        dropout: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()

        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # 门控分支线性层 (输入dim -> 隐藏层hidden_dim)
        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        ) 

        # 值分支线性层 (输入dim -> 隐藏层hidden_dim)
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

        # 输出投影层 (隐藏层hidden_dim -> 输出dim)
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )

        # 使用 RMSNorm 进行归一化，更快且数值稳定
        self.norm = StellarByteRMSNorm(dim, eps)

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

        # Step 2: 线性变换输出两份：值分支 + 门控分支（GEGLU结构）
        x_gate  = self.w1(x)  # [B, T, hidden_dim]
        x_value = self.w3(x)   # [B, T, hidden_dim]

        # Step 3: 对值分支使用 SiLU 激活函数（替代 ReLU，平滑且性能好）
        x_value = F.silu(x_value)

        # Step 4: 对门控分支使用 Sigmoid 激活，输出 (0,1) 区间的门控权重
        x_gate = torch.sigmoid(x_gate)

        # Step 5: 门控机制：逐元素乘，控制信息流通强度
        x = x_value * x_gate  # [B, T, hidden_dim]

        # Step 6: 输出映射回原维度
        x = self.w2(x)  # [B, T, dim]

        # Step 7: Dropout 处理防止过拟合
        x = self.dropout(x)

        # Step 8: 残差连接
        output = x + residual

        return output

if __name__ == '__main__':
    from config import StellarByteModelArgs

    # 初始化模型配置（假设 config 提供了基本结构参数）
    args = StellarByteModelArgs(
        dim=128,
        hidden_dim=256,
        multiple_of=256,
        ffn_dim_multiplier=1,
        num_heads=8,
        dim_multiplier=4,
    )

    # 创建 ByteMLP 实例，传递所有必需参数
    mlp = StellarByteFeedForward(
        dim=args.dim,
        hidden_dim=args.hidden_dim,
        multiple_of=args.multiple_of,
        ffn_dim_multiplier=args.ffn_dim_multiplier,
        dropout=0.1,
        eps=1e-6
    )

    # 构造输入张量 [batch=2, seq_len=16, dim=128]
    x = torch.randn(2, 16, args.dim)

    # 前向传播
    output = StellarByteFeedForward(x)

    # 打印输入输出维度
    print("Input shape : {}".format(x.shape))      # torch.Size([2, 16, 128])
    print("Output shape: {}".format(output.shape)) # torch.Size([2, 16, 128])
