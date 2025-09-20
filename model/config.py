import torch
from transformers import PretrainedConfig
from typing import Optional

class StellarByteModelArgs(PretrainedConfig):
    model_type = "StellarByte_CausalLM"

    def __init__(
        self,
        vocab_size: int = 32768,
        dim: int = 4096,
        num_layers: int = 32,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        multiple_of: int = 256,  # make SwiGLU hidden layer size     multiple of large power of 2
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: float = 1e-5,
        rope_theta: float = 500000,
        max_batch_size: int = 32,
        max_seq_len: int = 2048,
        enabled_flash_attn: bool = False,
        enabled_kv_cache: bool = False,
        atttention_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        ffn_dropout: float = 0.0,

        enabled_moe: bool = False,
        num_experts_per_tok: int = 2,
        num_routed_experts: int = 4,
        num_shared_experts: int = 1,
        scoring_func: str = 'softmax',
        aux_loss_alpha: float = 0.1,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.enabled_flash_attn = enabled_flash_attn
        self.enabled_kv_cache = enabled_kv_cache
        self.attention_dropout = atttention_dropout
        self.resid_dropout = resid_dropout
        self.ffn_dropout = ffn_dropout
        
        # ========== MoELayer ==========
        self.enabled_moe = enabled_moe,
        self.num_experts_per_tok = num_experts_per_tok,
        self.num_routed_experts = num_routed_experts,
        self.num_shared_experts = num_shared_experts,
        self.scoring_func = scoring_func,
        self.aux_loss_alpha = aux_loss_alpha,
        self.seq_aux = seq_aux,
        self.norm_topk_prob = norm_topk_prob,
        
        super().__init__(
            **kwargs
        )