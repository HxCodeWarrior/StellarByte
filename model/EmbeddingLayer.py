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
    """支持张量并行的词嵌入层
    
    特性：
    1. 按特征维度切分嵌入矩阵
    2. 输出乘以sqrt(model_dim)保持方差稳定
    3. 支持权重共享（与输出投影层）
    
    Args:
        args: 模型配置
    """
    
    def __init__(self, args: ByteModelConfig):
        super().__init__()
        self.args = args
        
        # 张量并行配置
        self.tp_size = max(1, args.tensor_parallel_size)
        self.tp_rank = dist.get_rank() if dist.is_initialized() else 0
        self.tp_group = args.tensor_parallel_group
        
        # 计算每个分区的嵌入维度
        self.embed_dim_per_partition = args.model_dim // self.tp_size

        # 嵌入层 - 每个设备存储完整词表的部分特征
        self.embed_tokens  = nn.Embedding(
            num_embeddings = args.vocab_size, 
            embedding_dim  = self.embed_dim_per_partition,
            dtype  = torch.float32,
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """Xavier正态分布初始化嵌入权重"""
        nn.init.normal_(
            self.embed_tokens.weight, 
            mean=0.0, 
            std=0.02  # GPT系列标准初始化
        )
    
    def get_weight(self):
        """权重共享接口"""
        return self.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            input_ids: 输入token IDs [batch_size, seq_len]
        
        Returns:
            嵌入向量 [batch_size, seq_len, embed_dim_per_partition]
        """
        # 嵌入查找
        x = self.embed_tokens(input_ids)  # [B, T, D_per_tp]
        
        # 缩放嵌入保持方差稳定
        x = x * math.sqrt(self.embed_dim_per_partition)
        
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
    print(f"Embedding shape: {output.shape}")  # 期望: [2, 16, 128]
    print("输出示例 (第一个样本第一个token的向量前5维):", output[0,0,:5])