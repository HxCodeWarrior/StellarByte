"""
==========================
#   StellarByte 模型库
==========================
高性能Transformer语言模型实现
"""

__version__ = "0.1.0"

# 导出主要模块
from .Model import ByteModel
from .config import ByteModelConfig
from .EmbeddingLayer import ByteEmbedding
from .Attention import ByteMultiHeadSelfAttention
from .DecoderLayer import ByteDecoderLayer
from .MLP import ByteMLP
from .RMSNorm import ByteRMSNorm
from .Position_Embedding import ByteDynamicRoPE

# 导出工具模块
from .utils.KVCache import ByteKVCache
from .utils.LoRA import LoRALinear

__all__ = [
    "ByteModel",
    "ByteModelConfig",
    "ByteEmbedding",
    "ByteMultiHeadSelfAttention",
    "ByteDecoderLayer",
    "ByteMLP",
    "ByteRMSNorm",
    "ByteDynamicRoPE",
    "ByteKVCache",
    "LoRALinear",
]