"""
==========================
#   StellarByte 模型库
==========================
高性能Transformer语言模型实现
"""

__version__ = "0.1.0"

# 导出主要模块
from .Model import StellarByteModel
from .config import StellarByteModelArgs
from .RMSNorm import StellarByteRMSNorm
from .PositionEmbedding import StellarByteRoPE
from .Attention import StellarByteAttention
from .FeedForward import StellarByteFeedForward
from .MoE import StellarByteMOEFeedForward
from .TransformerBlock import StellarByteBlock
from .Model import StellarByteModel, StellarByteForCausalLM

# 导出工具模块
from .utils.KVCache import ByteKVCache
from .utils.LoRA import LoRALinear

__all__ = [
    "StellarByteModel",
    "StellarByteForCausalLM",
    "StellarByteBlock",
    "StellarByteModelArgs",
    "StellarByteRMSNorm",
    "StellarByteRoPE",
    "StellarByteAttention",
    "StellarByteFeedForward",
    "StellarByteMOEFeedForward",
    "ByteKVCache",
    "LoRALinear",
]