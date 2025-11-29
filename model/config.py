"""
StellarByte 配置模块

用于定义模型的超参数与可选特性（例如 MoE）。
"""

from transformers import PretrainedConfig


class StellarByteConfig(PretrainedConfig):
    """StellarByte 模型的配置类。

    该类继承自 HuggingFace 的 PretrainedConfig，用于保存和加载模型超参数。
    所有参数在 __init__ 函数中定义并赋默认值。
    """

    model_type = "stellarbytecasualmodel"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: float = 1e6,
        inference_rope_scaling: bool = False,
        flash_attn: bool = True,
        # MoE 相关配置（仅当 use_moe=True 时生效）
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.1,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        """初始化 StellarByte 配置。

        参数说明：
        - dropout: dropout 比例
        - hidden_size: 模型隐藏维度
        - num_attention_heads: 注意力头数
        - max_position_embeddings: 最大位置编码长度
        - use_moe: 是否启用 MoE
        其他参数与原 MiniMindConfig 保持相近，仅改名为 StellarByte 前缀。
        """
        super().__init__(**kwargs)
        # 基本超参
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        # RoPE 外推缩放相关字典（若启用 inference_rope_scaling）
        self.rope_scaling = {
            "beta_fast": 4,
            "beta_slow": 1,
            "factor": 4,
            "original_max_position_embeddings": 2048,
            "type": "yarn",
        } if self.inference_rope_scaling else None

        # 是否使用 Flash Attention（若 PyTorch 支持）
        self.flash_attn = flash_attn

        # MoE 相关超参
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
