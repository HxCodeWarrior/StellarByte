import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .RMSNorm import ByteRMSNorm
except ImportError:
    from RMSNorm import ByteRMSNorm

class ByteMLP(nn.Module):
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
        norm (ByteRMSNorm): RMS归一化层
        dropout (nn.Dropout): 随机失活层防止过拟合

    参数:
        dim (int): 输入/输出特征维度
        hidden_dim (int, optional): 隐藏层维度，未指定时自动计算
        multiple_of (int, optional): 隐藏层维度对齐基数(默认256)
        dropout (float, optional): 随机失活概率(默认0.1)
        eps (float, optional): RMSNorm的数值稳定常数(默认1e-6)
        bias (bool, optional): 线性层是否使用偏置(默认False)
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
            # 基础计算：约为原始维度的3倍 (8/3 ≈ 2.666)
            hidden_dim = int(8 * dim / 3)
            # 向上对齐到multiple_of的倍数（确保硬件友好）
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # 门控投影层：dim -> 2*hidden_dim
        # 输出将拆分为值分支和门控分支
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)  # gate/激活分支
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)  # value/线性分支

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
