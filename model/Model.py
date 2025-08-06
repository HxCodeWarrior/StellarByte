import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Tuple, Union

try:
    from .EmbeddingLayer     import ByteEmbedding
    from .DecoderLayer       import ByteDecoderLayer
    from .RMSNorm            import ByteRMSNorm
    from .config             import ByteModelConfig
except ImportError:
    from EmbeddingLayer     import ByteEmbedding
    from DecoderLayer       import ByteDecoderLayer
    from RMSNorm            import ByteRMSNorm
    from config             import ByteModelConfig

class ByteTransformer(PreTrainedModel):
    config_class = ByteModelConfig
    last_loss    = Optional[torch.Tensor]

    def __init__(self, args: ByteModelConfig = None):
        super().__init__(args)
        # 初始化模型参数
        self.args = args
        # 词汇表大小
        self.vocab_size = args.vocab_size
        # 模型层数
        self.num_layers = args.num_layers

        # 词嵌入层
        self.token_embedding = ByteEmbedding(args)
        # Dropout层
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        # Decoder层
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.num_layers):
            self.layers.append(
                ByteDecoderLayer(args=args, layer_id=layer_id)
            )
        # 归一化层
        self.norm = ByteRMSNorm(dim=args.model_dim, eps=args.layer_norm_eps)
        # 输出层
        self.output = nn.Linear(args.model_dim, args.vocab_size, bias=False)

        # 词嵌入层权重雨输出层权重共享
        self.token_embedding.weight = self.output.weight

        # 初始化权重
        self.apply(self._init_weights)
        # 残差投影层特殊缩放初始化

        # 初始化最后一次向前传播的损失属性
        self.last_loss = None
        self.OUT = CausalLMOutputWithPast()
        self._no_split_models = [name for name, _ in self.named_modules()]  # 不分割的模块列表
    
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            # 线性层：Xavier初始化
            std = self.args.initializer_range
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
        elif isinstance(module, nn.Embedding):
            # 嵌入层：正态分布初始化
            nn.init.normal_(module.weight, mean=0.0, std=self.args.initializer_range)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # 获取输入张量的形状信息
        _batch_size, seq_len = input_ids.shape

        # 词嵌入
        hidden_states = self.token_embedding(input_ids) # [B, T, D] 或 [B, T, D/tp]

        # Dropout
        hidden_states = self.dropout(hidden_states)
        
        # Decoder层
        for layer in self.layers:
            hidden_states = layer(hidden_states, padding_mask)

        # 归一化
        hidden_states = self.norm(hidden_states)

        # 输出
        logits = self.output(hidden_states)

        # 损失计算
        loss = None
        if labels is not None:
            # 移位标签和预测
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # 计算交叉熵损失
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,  # 忽略特殊标签
                reduction='mean'
            )
            self.last_loss = loss.detach()
        else:
            # 推理时，只对最后一个位置的输出进行向前传播计算
            logits = self.output(hidden_states[:, [-1], :])
            self.last_loss = None

        # 设置输出
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('last_loss', self.last_loss)

        return self.OUT
    
    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_seq_len: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        eos_token_id: int = None,
        **kwargs
    ):
        pass

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    args = ByteModelConfig(
        dim=1024,
        n_layers=18,
    )
    # 实例化Model
    model = ByteTransformer(args=args)
    # 计算model的全部参数
    num_params = sum(p.numel() for p in model.parameters())
    print(f'LLM总参数量：{num_params / 1e6:.3f} 百万')

    prompt = "你好呀，今天吃什么呢？你过得怎么样嘞？"
    text = f"{tokenizer.bos_token}{prompt}{tokenizer.eos_token}"
    print(f"Input text: {text}")

    input_id = tokenizer(text).data['input_ids']
    print("input_ids :", input_id)
    print("dcode_str :", tokenizer.decode(input_id))

    X = torch.tensor(input_id[:-1]).unsqueeze(0)
    Y = torch.tensor(input_id[1:]).unsqueeze(0)
    print("X shape :", X.shape)
    print("Y shape :", Y.shape)

    # 将输入张量传入模型
    output = model(X, Y)