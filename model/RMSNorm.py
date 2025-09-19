import torch
import torch.nn as nn

class StellarByteRMSNorm(torch.nn.Module):
    """实现RMSNorm（Root Mean Square Normalization）模块"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        初始化RMSNorm层
        
        参数:
            dim: 输入特征的维度
            eps: 防止除以零的小常数，默认值为1e-6
        """
        # 调用父类Module的初始化方法
        super().__init__()
        # 设置一个极小值eps，用于数值稳定性，防止除以零的情况
        self.eps = eps
        # 创建可学习参数weight，初始化为全1向量，维度与输入特征维度相同
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        执行RMS归一化操作
        
        参数:
            x: 输入张量
            
        返回:
            归一化后的张量
        """
        # 计算RMS值：对输入x的平方沿最后一个维度求均值，加上eps防止除零，然后取平方根倒数
        # 最后将结果与原始输入x相乘，实现归一化
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x: 输入张量
            
        返回:
            经过RMSNorm处理后的张量
        """
        # 1. 将输入转换为float类型进行归一化计算，然后再转换回原始数据类型
        # 2. 应用_norm方法进行RMS归一化
        output = self._norm(x.float()).type_as(x)
        # 3. 使用可学习权重weight对归一化结果进行缩放
        return output * self.weight


if __name__ == "__main__":
    from config import StellarByteModelArgs

    args = StellarByteModelArgs()
    norm = StellarByteRMSNorm(128, args.layer_norm_eps)
    x = torch.randn(2, 16, 128)
    output = norm(x)
    print(f'Input shape: {x.shape}\nOutPut shape: {output.shape}') # torch.Size([2, 16, 128)