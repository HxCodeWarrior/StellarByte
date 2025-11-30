"""
StellarByte 配置模块

用于定义模型的超参数与可选特性）
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
        # ==================== 模型基本超参数 ====================
        vocab_size: int = 64000,
        hidden_size: int = 512,
        num_hidden_layers: int = 8,
        hidden_act: str = "silu",
        intermediate_size: int = None,
        
        # ==================== 多头注意力层配置 ====================
        num_attention_heads: int = 8,
        num_key_value_heads: int = 2,
        flash_attn: bool = True,
        max_position_embeddings: int = 32768,
        
        # ==================== RoPE位置编码配置 ====================
        rope_theta: float = 1e6,
        inference_rope_scaling: bool = False,
        
        # ==================== 正则化配置 ====================
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-05,
        
        # ==================== MoE混合专家配置 ====================
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 8,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        
        # ==================== KV缓存配置 ====================
        use_cache: bool = True,
        dtype: str = "bfloat16",

        # ==================== 初始化配置 ====================
        initializer_range: float = 0.02,
        verbose_init: bool = False,
        
        # ==================== 其他超参数配置 ====================
        pad_token_id: int = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        
        **kwargs,
    ):
        """初始化 StellarByte 配置。
        
        Args:
            # ==================== 模型基本超参数 ====================
            vocab_size (int): 词汇表大小，决定模型能够识别的token数量
            hidden_size (int): 隐藏层维度，表示每个隐藏状态的维度大小
            num_hidden_layers (int): Transformer层数，决定模型的深度
            hidden_act (str): 隐藏层激活函数，如'silu'、'gelu'、'relu'等
            intermediate_size (int): 前馈神经网络中间层维度，若为None则自动计算
            
            # ==================== 多头注意力层配置 ====================
            num_attention_heads (int): 注意力头数量，用于多头注意力机制
            num_key_value_heads (int): Key-Value头数量，用于分组查询注意力(GQA)
            flash_attn (bool): 是否启用Flash Attention优化，提升训练和推理效率
            max_position_embeddings (int): 最大序列长度，决定模型能处理的最大token数
            
            # ==================== RoPE位置编码配置 ====================
            rope_theta (float): RoPE旋转位置编码的基数，影响位置编码的频率分布
            inference_rope_scaling (bool): 是否在推理时启用RoPE外推缩放，支持更长序列
            
            # ==================== 正则化配置 ====================
            dropout (float): Dropout比率，用于防止过拟合，0.0表示不使用dropout
            rms_norm_eps (float): RMS归一化的小常数，防止除零错误
            
            # ==================== MoE混合专家配置 ====================
            use_moe (bool): 是否启用混合专家架构，启用后可大幅增加模型参数但不增加计算量
            num_experts_per_tok (int): 每个token选择的路由专家数量
            n_routed_experts (int): 路由专家总数，每个专家是一个独立的前馈网络
            n_shared_experts (int): 共享专家数量，所有token都会经过的共享前馈网络
            scoring_func (str): 专家选择评分函数，'softmax'或'sigmoid'
            aux_loss_alpha (float): 辅助损失系数，用于平衡专家负载均衡
            seq_aux (bool): 是否在序列级别计算辅助损失
            norm_topk_prob (bool): 是否对top-k专家概率进行归一化

            # ==================== KV缓存配置 ====================
            use_cache (bool): 是否使用KV缓存，推理时可显著提升速度
            dtype (str): 模型参数的数据类型，'bfloat16'或'float32'

            # ==================== 初始化配置 ====================
            initializer_range (float): 参数初始化范围，用于权重矩阵的初始化
            
            # ==================== 其他超参数配置 ====================
            pad_token_id (int): 填充token的ID，用于序列填充
            bos_token_id (int): 序列开始token的ID
            eos_token_id (int): 序列结束token的ID
            tie_word_embeddings (bool): 是否绑定输入和输出词嵌入权重
        """
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )
        
        # ==================== 模型基本超参数 ====================
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_act = hidden_act
        
        # 自动计算中间层维度（如果未指定）
        if intermediate_size is None:
            # 使用经典比例：hidden_size * 8/3 四舍五入到128的倍数
            self.intermediate_size = int((hidden_size * 8 / 3) / 128) * 128
        else:
            self.intermediate_size = intermediate_size
        
        # ==================== 多头注意力层配置 ====================
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.flash_attn = flash_attn
        self.max_position_embeddings = max_position_embeddings
        
        # ==================== RoPE位置编码配置 ====================
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        
        # RoPE外推缩放配置（仅在启用时设置）
        if self.inference_rope_scaling:
            self.rope_scaling = {
                "beta_fast": 4,      # 快速衰减系数，控制高频分量的衰减速度
                "beta_slow": 1,      # 慢速衰减系数，控制低频分量的衰减速度
                "factor": 4,         # 缩放因子，决定外推的长度倍数
                "original_max_position_embeddings": 2048,  # 原始训练的最大序列长度
                "type": "yarn",      # 外推方法类型，YARN是一种先进的RoPE外推技术
            }
        else:
            self.rope_scaling = None
        
        # ==================== 正则化配置 ====================
        self.dropout = dropout
        self.rms_norm_eps = rms_norm_eps
        
        # ==================== MoE混合专家配置 ====================
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        
        # ==================== KV缓存配置 ====================
        self.use_cache = use_cache
        self.dtype = dtype

        # ==================== 初始化配置 ====================
        self.initializer_range = initializer_range
        self.verbose_init = verbose_init
        
        # ==================== 计算派生参数 ====================
        # 计算每个注意力头的维度
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        # 验证参数配置的有效性
        self._validate_parameters()
    
    def _validate_parameters(self):
        """验证参数配置的有效性"""
        
        # 验证基础架构参数
        assert self.hidden_size % self.num_attention_heads == 0, (
            f"hidden_size({self.hidden_size})必须能被num_attention_heads({self.num_attention_heads})整除"
        )
        
        assert self.num_attention_heads % self.num_key_value_heads == 0, (
            f"num_attention_heads({self.num_attention_heads})必须能被num_key_value_heads({self.num_key_value_heads})整除"
        )
        
        # 验证MoE参数
        if self.use_moe:
            assert self.n_routed_experts > 0, "路由专家数量必须大于0"
            assert self.num_experts_per_tok > 0, "每token专家数必须大于0"
            assert self.num_experts_per_tok <= self.n_routed_experts, (
                f"每token专家数({self.num_experts_per_tok})不能超过总专家数({self.n_routed_experts})"
            )
            assert self.scoring_func in ["softmax", "sigmoid"], (
                f"不支持的评分函数: {self.scoring_func}"
            )
    
    def __str__(self):
        """返回配置的字符串表示"""
        moe_info = ""
        if self.use_moe:
            moe_info = (
                f"\n  MoE配置:"
                f"\n    - 专家总数: {self.n_routed_experts} (路由) + {self.n_shared_experts} (共享)"
                f"\n    - 每token专家数: {self.num_experts_per_tok}"
                f"\n    - 辅助损失系数: {self.aux_loss_alpha}"
            )
        
        rope_info = ""
        if self.inference_rope_scaling:
            rope_info = (
                f"\n  RoPE外推:"
                f"\n    - 类型: yarn"
                f"\n    - 缩放因子: {self.rope_scaling['factor']}"
                f"\n    - 原始长度: {self.rope_scaling['original_max_position_embeddings']}"
            )
        
        return (
            f"StellarByteConfig:"
            f"\n 基础架构:"
            f"\n    - 词汇表: {self.vocab_size}"
            f"\n    - 隐藏层: {self.hidden_size}"
            f"\n    - 层数: {self.num_hidden_layers}"
            f"\n    - 注意力头: {self.num_attention_heads} (KV头: {self.num_key_value_heads})"
            f"\n    - 序列长度: {self.max_position_embeddings}"
            f"\n    - 中间层: {self.intermediate_size}"
            f"{moe_info}"
            f"{rope_info}"
            f"\n 优化特性:"
            f"\n    - Flash Attention: {self.flash_attn}"
            f"\n    - RoPE theta: {self.rope_theta}"
            f"\n    - 激活函数: {self.hidden_act}"
            f"\n 正则化:"
            f"\n    - Dropout: {self.dropout}"
            f"\n    - RMSNorm eps: {self.rms_norm_eps}"
        )