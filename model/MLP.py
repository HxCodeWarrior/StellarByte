import torch
import torch.nn as nn
import torch.nn.functional as F

class ByteMLP(nn.Module):
    def __init__(
        self, 
        dim: int, 
        hidden_dim: int = None, 
        multiple_of: int = 256, 
        dropout: float = 0.1, 
        bias: bool = False
    ):
        """
        门控多层感知机模块 (Gated ByteMLP)
        
        参数:
        dim: 输入/输出特征的维度
        hidden_dim: 隐藏层维度（可选，默认自动计算）
        multiple_of: 隐藏层维度的对齐基数（确保维度是此值的倍数）
        dropout: Dropout概率（默认0.1）
        bias: 是否在线性层使用偏置（默认False）
        """
        super().__init__()
        
        # 自动计算隐藏层维度（若未指定）
        if hidden_dim is None:
            # 计算基础隐藏层维度（约为输入维度的8/3倍）
            hidden_dim = int(8 * dim / 3)
            # 将维度对齐到最近的multiple_of倍数（向上取整）
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # 定义线性变换层:
        # w1: 输入层 → 隐藏层
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        # w2: 隐藏层 → 输出层（恢复原始维度）
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        # w3: 输入层 → 门控层（与w1输出同维度）
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)

        # 激活函数: SiLU (Swish) 激活函数
        self.act = nn.SiLU()
        # Dropout层: 防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播过程
        
        参数:
        x: 输入张量 [batch_size, seq_len, dim]
        
        返回:
        处理后的张量 [batch_size, seq_len, dim]
        """
        # 路径1: 线性变换 + 激活函数
        a = self.act(self.w1(x))  # [batch_size, seq_len, hidden_dim]
        
        # 路径2: 线性变换（无激活函数）
        b = self.w3(x)  # [batch_size, seq_len, hidden_dim]
        
        # 门控机制: 逐元素相乘 (a * b)
        x = a * b  # [batch_size, seq_len, hidden_dim]
        
        # 降维回原始维度
        x = self.w2(x)  # [batch_size, seq_len, dim]
        
        # 应用Dropout后返回结果
        return self.dropout(x)

if __name__ == '__main__':
    from config import ByteModelConfig

    args = ByteModelConfig(
        model_dim=128,                # 嵌入维度 E
        num_attention_heads=8,        # 多头注意力 H
        dim_multiplier=4,             # 隐藏层维度的对齐基数
        residual_dropout_prob=0.1,    # 残差连接dropout率
    )

    ByteMLP = ByteMLP(args.model_dim)

    x = torch.randn(2, 16, args.model_dim)

    output = ByteMLP(x)
    print("Input shape : {}".format(x.shape))      # [batch_size, seq_len, model_dim]
    print("Output shape: {}".format(output.shape)) # [batch_size, seq_len, model_dim
