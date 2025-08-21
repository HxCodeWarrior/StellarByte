import torch
import torch.nn as nn
import torch.distributed as dist
import math
from typing import Optional

try:
    from .config import ByteModelConfig
except ImportError:
    from config import ByteModelConfig

class ByteEmbedding(nn.Module):
    """支持张量并行的分布式词嵌入层。
    
    核心特性：
    1. 模型维度切分：将完整嵌入矩阵按特征维度切分到多个设备
    2. 方差稳定：输出乘以√(model_dim)保持数值稳定性
    3. 权重共享：提供接口与输出投影层共享权重
    
    设计原理：
    - 在张量并行环境中，每个设备仅存储完整词嵌入矩阵的一部分特征
    - 前向传播时，每个设备处理输入token并生成部分嵌入向量
    - 最终输出需通过AllReduce操作拼接为完整向量（由后续模块处理）
    
    参数:
        args (ByteModelConfig): 模型配置对象
    """
    
    def __init__(self, args: ByteModelConfig):
        super().__init__()
        self.args = args
        
        # ================= 张量并行配置 =================
        # 获取并行组大小（单卡运行时默认为1）
        self.tp_size = max(1, args.tensor_parallel_size)
        # 当前设备在并行组中的排名
        self.tp_rank = dist.get_rank() if dist.is_initialized() else 0
        # 分布式通信组
        self.tp_group = args.tensor_parallel_group
        
        # 计算每个设备分得的特征维度
        # 示例：model_dim=512, tp_size=4 → 每个设备128维
        self.embed_dim_per_partition = args.model_dim // self.tp_size

        # ================= 嵌入层初始化 =================
        # 关键：每个设备只创建部分特征的嵌入矩阵
        # - num_embeddings: 完整词表大小（所有设备相同）
        # - embedding_dim: 分配到的局部特征维度
        self.embed_tokens  = nn.Embedding(
            num_embeddings = args.vocab_size, 
            embedding_dim  = self.embed_dim_per_partition,
            dtype  = torch.float32,
        )

    def get_weight(self):
        """获取当前设备的嵌入权重（用于权重共享）。
        
        典型应用场景：
        - 输出投影层共享词嵌入权重
        - 在张量并行中，每个设备提供自己的部分权重
        
        返回:
            nn.Parameter: 形状为 [vocab_size, embed_dim_per_partition] 的权重矩阵
        """
        return self.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """前向传播逻辑。
        
        处理流程：
        1. 局部嵌入查找：获取当前设备负责的特征切片
        2. 方差缩放：乘以√(局部维度)保持数值稳定
        
        注意：
        - 输出需在后续通过AllReduce拼接完整向量
        
        参数:
            input_ids (torch.Tensor): 输入token ID矩阵 [batch_size, seq_len]
        
        返回:
            torch.Tensor: 局部嵌入向量 [batch_size, seq_len, embed_dim_per_partition]
        """
        # 1. 嵌入查找（仅当前设备负责的特征维度）
        # 输入: [B, T] → 输出: [B, T, D_local]
        x = self.embed_tokens(input_ids)  # [B, T, D_per_tp]
        
        # 2. 缩放嵌入保持方差稳定
        # 原始Transformer设计：乘以√(model_dim)
        # 张量并行调整：乘以√(局部维度)
        scaling_factor = math.sqrt(self.embed_dim_per_partition)
        x = x * scaling_factor
        
        return x

if __name__ == '__main__':
    # 测试用例
    config = ByteModelConfig(
        vocab_size = 1000,
        model_dim  = 128,
        tensor_parallel_size  = 1,
        tensor_parallel_group = None
    )
    embedding_layer = ByteEmbedding(config)

    batch_size = 2
    seq_len    = 16
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, seq_len), dtype=torch.long)

    # 前向计算
    output = embedding_layer(input_ids)
    
    print(f"Input_ids: {input_ids}")
    print(f'Input_ids shape: {input_ids.shape}')
    print(f"Embedding shape: {output.shape}")  # 期望: [2, 16, 128]
    print("输出示例 (第一个样本第一个token的向量前5维):", output[0,0,:5])