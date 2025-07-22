import torch
from transformers import PretrainedConfig

class ByteModelConfig(PretrainedConfig):
    model_type = "stellarbyte_model"

    def __init__(
        self,
        vocab_size: int = 32000,              # 词汇表大小
        model_dim: int = 768,                 # 模型维度
        num_layers: int = 12,                 # Transformer层数
        num_attention_heads: int = 16,        # 多头注意力头数
        num_kv_heads: int = 8,                # 多头KV注意力头数（默认为num_heads，可做头分离）
        hidden_dim: int = None,               # 隐藏层维度
        dim_multiplier: int = 4,              # 隐藏层维度的对齐基数
        max_seq_len: int = 2048,              # 最大序列长度
        drop_path_prob: float = 0.0,          # DropPath残差连接dropout率
        hidden_dropout_prob: float = 0.1,     # 隐藏层dropout率
        attention_dropout_prob: float = 0.1,  # 注意力dropout率
        residual_dropout_prob: float = 0.1,   # 残差连接dropout率
        layer_norm_eps: float = 1e-5,         # 层归一化epsilon值
        initializer_range: float = 0.02,      # 初始化参数范围
        tie_word_embeddings: bool = False,    # 是否绑定输入输出词嵌入
        xpos_rope_theta: float = 10000.0,     # XPos位置编码theta参数
        xpos_scale_base: float = 512,         # XPos比例缩放因子
        use_flash_attention: bool = False,    # 是否使用FlashAttention
        use_cache: bool = True,               # 是否使用KV缓存加速推理
        key_cache_dtype: torch.dtype = torch.float16,  # KV缓存中Key的精度
        value_cache_dtype: torch.dtype = torch.float16,  # KV缓存中Value的精度
        model_parallel_size: int = 1,         # 模型并行大小
        tensor_parallel_size: int = 1,        # 张量并行大小
        tensor_parallel_rank: int = 0,        # 张量并行rank
        use_causal: bool = False,             # 是否使用因果遮罩（GPT类模型通常为True）
        layerscale_init : float = 1e-5,       # 层尺度初始化值
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.hidden_dim = hidden_dim if hidden_dim is not None else 4 * model_dim
        self.dim_multiplier = dim_multiplier
        self.max_seq_len = max_seq_len
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range

        # ======== Initializer =========
        self.layerscale_init = layerscale_init

        # ========== Dropout ===========
        self.residual_dropout_prob = residual_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.drop_path_prob = drop_path_prob

        # ========= Attention ==========
        self.use_causal = use_causal
        self.use_flash_attention = use_flash_attention
        

        # ============ XPos ============
        self.xpos_rope_theta = xpos_rope_theta
        self.xpos_scale_base = xpos_scale_base

        # ========== KV Cache ==========
        self.use_cache = use_cache
        self.key_cache_dtype = key_cache_dtype
        self.value_cache_dtype = value_cache_dtype
        self.tensor_parallel_size = tensor_parallel_size
        self.tensor_parallel_rank = tensor_parallel_rank

        # ========== Parallel ===========
        self.model_parallel_size = model_parallel_size
        
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )