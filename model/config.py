import torch
from transformers import PretrainedConfig

class ByteModelConfig(PretrainedConfig):
    model_type = "stellarbyte_model"

    def __init__(
        self,
        vocab_size: int = 32768,              # 词汇表大小
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
        base_theta: float = 10000.0,          # 位置编码theta参数
        ntk_alpha: float = 1.0,               # 位置编码NTK-alpha参数
        use_flash_attention: bool = False,    # 是否使用FlashAttention
        use_kvcache: bool = True,             # 是否使用KV缓存加速推理
        cache_dtype: torch.dtype = torch.float16,  # KV缓存中Key的精度
        attention_window_size: int = 0,        # 注意力窗口大小
        parallel_residual: bool = True,        # 串并行残差
        tensor_parallel_size: int = 1,         # 张量并行大小
        tensor_parallel_group: int = 0,        # 张量并行rank
        layerscale_init : float = 1e-5,        # 层尺度初始化值
        initializer_range: float = 0.02,       # 权重初始化范围
        moe_enabled: bool = False,             # 是否使用MoE
        moe_num_experts: int = 2,              # 专家数量
        moe_k: int = 1,                        # 每个专家选择的token数
        moe_capacity_factor: float = 1.25,     # 专家容量因子
        moe_loss_coefficient: float = 0.01,    # 专家损失系数
        moe_world_size: int = 1,               # MoE并行大小
        moe_rank: int = 0,                     # MoE并行rank
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

        # ======== Initializer =========
        self.initializer_range = initializer_range
        self.layerscale_init = layerscale_init

        # ========== Dropout ===========
        self.residual_dropout_prob = residual_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.drop_path_prob = drop_path_prob

        # ========= Attention ==========
        self.use_flash_attention = use_flash_attention
        self.attention_window_size = attention_window_size
        
        # ========== MoELayer ==========
        self.moe_enabled = moe_enabled
        self.moe_num_experts = moe_num_experts
        self.moe_k = moe_k
        self.moe_capacity_factor = moe_capacity_factor
        self.moe_loss_coefficient = moe_loss_coefficient
        self.world_size = moe_world_size
        self.rank = moe_rank

        # ======== Dynamic RoPE ========
        self.base_theta = base_theta
        self.ntk_alpha = ntk_alpha

        # ========== KV Cache ==========
        self.use_kvcache = use_kvcache
        self.cache_dtype = cache_dtype

        # ========== Parallel ===========
        self.parallel_residual = parallel_residual
        self.tensor_parallel_size = tensor_parallel_size
        self.tensor_parallel_group = tensor_parallel_group
        
        super().__init__(
            **kwargs
        )