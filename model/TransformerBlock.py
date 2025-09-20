import torch
import torch.nn as nn
from typing import Optional,Dict


try:
    from .config import StellarByteModelArgs
    from .RMSNorm import StellarByteRMSNorm
    from .Attention import StellarByteAttention
    from .FeedForward import StellarByteMLP
    from .MoE import StellarByteMOEFeedForward
except:
    from config import StellarByteModelArgs
    from RMSNorm import StellarByteRMSNorm
    from Attention import StellarByteAttention
    from FeedForward import StellarByteMLP
    from MoE import StellarByteMOEFeedForward

class StellarByteBlock(nn.Module):
    def __init__(self, layer_id: int, args: StellarByteModelArgs):
        super().__init__()
        self.num_heads = args.num_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.num_heads
        self.attention = StellarByteAttention(args)
        self.enabled_moe = args.enabled_moe
        if args.enabled_moe:
            self.feed_forward = StellarByteMOEFeedForward(args)
        else:
            self.feed_forward = StellarByteMLP(
                dim=args.dim,
                hidden_dim=4 * args.dim,
                multiple_of=args.multiple_of,
                ffn_dim_multiplier=args.ffn_dim_multiplier,
            )
        self.layer_id = layer_id
        self.attention_norm = StellarByteRMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = StellarByteRMSNorm(args.dim, eps=args.norm_eps)

        # 存储MoE相关的统计信息
        self.moe_aux_loss = None
        self.moe_expert_usage = None

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor
    ):
        # 注意力层
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis)
        
        # 前馈层 (可能是MoE或普通MLP)
        ffn_input = self.ffn_norm(h)
        ffn_output = self.feed_forward(ffn_input)

        # 如果是MoE，保存辅助损失和专家使用情况
        if self.enabled_moe:
            self.moe_aux_loss = getattr(self.feed_forward, 'aux_loss', None)
            self.moe_expert_usage = self.feed_forward.get_expert_usage()
        
        out = h + ffn_output
        return out
    
    def get_moe_aux_loss(self) -> Optional[torch.Tensor]:
        """获取MoE辅助损失"""
        return self.moe_aux_loss

    def get_moe_expert_usage(self) -> Optional[Dict[int, int]]:
        """获取MoE专家使用统计"""
        return self.moe_expert_usage