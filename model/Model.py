import math
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

class ByteModel(PreTrainedModel):
    config_class = ByteModelConfig
    loss         = Optional[torch.Tensor]

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
        for pn, p in self.named_parameters():
            if pn.endswith('w2.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * args.num_layers))

        # 初始化最后一次向前传播的损失属性
        self.loss = None
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
        attention_mask: Optional[torch.Tensor] = None,
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
            hidden_states = layer(hidden_states, attention_mask)

        # 归一化
        hidden_states = self.norm(hidden_states)

        # 输出
        logits = self.output(hidden_states)

        # 损失计算
        loss = None
        if labels is not None:
            # 计算交叉熵损失
            loss = F.cross_entropy(
                logits.view(-1,logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,  # 忽略特殊标签
                reduction='mean'
            )
            self.loss = loss
        else:
            # 推理时，只对最后一个位置的输出进行向前传播计算
            logits = self.output(hidden_states[:, [-1], :])
            self.loss = None

        # 设置输出
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('loss', self.loss)

        return self.OUT
    
    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_seq_len: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        repetition_context: int = 512,
        eos_token_id: int = None,
        return_logits: bool = False,
        **kwargs
    ):
        """
        自回归文本生成函数
        
        Args:
            input_ids: 初始输入序列 [batch_size, seq_len]
            max_seq_len: 生成的最大序列长度（包括输入长度）
            temperature: 采样温度（>0）
            top_k: 只考虑概率最高的k个token（0表示禁用）
            repetition_penalty: 重复惩罚因子（>1）
            repetition_context: 应用重复惩罚的上下文长度（0表示禁用）
            eos_token_id: 结束符ID（遇到时停止生成）
            return_logits: 是否返回每一步的logits
        
        Return:
            生成的完整序列 [batch_size, new_seq_len]
            如果return_logits=True，同时返回logits [batch_size, gen_len, vocab_size]
        """
        # 准备工作
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        batch_size = input_ids.shape[0]

        # 结束符处理逻辑
        if eos_token_id is None:
            eos_token_id = -1  # 无效ID

        # 预采样参数校验
        temperature = max(temperature, 1e-5)  # 防止除零
        top_k = max(top_k, 0)
        repetition_penalty = max(repetition_penalty, 1.0)

        # 存储生成的 token（初始为输入）
        generated = input_ids.clone()
        all_logits = [] if return_logits else None

        for _ in range(max_seq_len):
            # 模型前向传播
            output = self.forward(generated)
            
            # 获取当前logits
            logits = output.logits     # [B, 1, vocab_size]
            logits = logits[:, -1, :]  # 取出最后一个token位置的logits [B, vocab_size]

            # 应用重复惩罚
            logits = self.repetition_penalty(
                logits, 
                generated,
                penalty=repetition_penalty,
                context_size=repetition_context
            )

            # 采样下一个token
            next_token = self.sample_next_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            # 更新活动样本的生成序列
            generated = torch.cat([generated, next_token], dim=1)

            # 记录logits（如果需要）
            if return_logits:
                full_logits = torch.full((batch_size, logits.shape[-1]), float('-inf'), device=device)
                full_logits[output] = logits
                all_logits.append(full_logits)

            # 检查是否遇到eos_token_id
            if eos_token_id is not None:
                if (next_token == eos_token_id).all():
                    break

        # 输出生成结果
        if return_logits:
            all_logits = torch.stack(all_logits, dim=1)  # [B, T, vocab_size]
            return generated, all_logits
        else:
            return generated
    

    def sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """
        采样下一个token

        Args:
            logits: [batch_size, vocab_size] 当前步的logits
            temperature: 采样温度
            top_k: 保留概率最高的k个token
            top_p: 保留累计概率达到p的最小token集合

        Returns:
            next_token: [batch_size, 1] 下一个token ID
        """
        # 温度调节
        if temperature != 1.0:
            logits = logits / temperature

        # Top-K过滤
        if top_k > 0:
            topk_values, _ = torch.topk(logits, top_k, dim=-1)
            min_topk = topk_values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_topk, torch.full_like(logits, float('-inf')), logits)

        # Top-p过滤
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # 创建移除掩码
            removal_mask = cumulative_probs > top_p
            removal_mask[:, 1:] = removal_mask[:, :-1].clone()
            removal_mask[:, 0] = False

            # 应用掩码
            removal_mask = removal_mask.scatter(-1, sorted_indices, removal_mask)
            logits = logits.masked_fill(removal_mask, float('-inf'))

        # 概率采样
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def repetition_penalty(
        self,
        logits: torch.Tensor,
        generated: torch.Tensor,
        penalty: float = 1.2,
        context_size: int = 512
    ) -> torch.Tensor:
        """
        应用重复惩罚机制，降低已出现token的概率

        Args:
            logits: [batch_size, vocab_size] 当前步的logits
            generated: [batch_size, seq_len] 已生成的序列
            penalty: 重复惩罚因子(>1)
            context_size: 考虑惩罚的上下文长度

        Returns:
            应用惩罚后的logits
        """
        if penalty <= 1.0:
            return logits

        batch_size, vocab_size = logits.shape
        device = logits.device

        # 仅考虑最近的上下文
        recent_tokens = generated[:, -context_size:]

        for b in range(batch_size):
            # 获取当前样本的独特token
            unique_tokens = torch.unique(recent_tokens[b])

            # 对重复token应用惩罚
            logits[b, unique_tokens] = logits[b, unique_tokens] / penalty

        return logits


    def model_info(self) -> dict:
        """
        返回模型基础信息字典，包含：
        - 总参数量
        - 可训练参数量
        - 模型层数
        - 隐藏层维度
        - 注意力头数
        - 词汇表大小
        - 模型配置摘要
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "num_layers": self.num_layers,
            "hidden_size": self.args.model_dim,
            "num_attention_heads": self.args.num_attention_heads,
            "vocab_size": self.vocab_size,
            "config_summary": {
                "model_dim": self.args.model_dim,
                "num_layers": self.args.num_layers,
                "max_seq_len": self.args.max_seq_len,
                "hidden_dropout": self.args.hidden_dropout_prob,
                "attention_dropout": self.args.attention_dropout_prob,
                "residual_dropout_prob": self.args.residual_dropout_prob,
                "layer_norm_eps": self.args.layer_norm_eps,
                "rope_theta": self.args.base_theta,
                "ntk_alpha": self.args.ntk_alpha,
            }
        }

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    args = ByteModelConfig(
        dim=1024,
        num_layers=18,
    )
    # 实例化Model
    model = ByteModel(args=args)
    # 计算model的全部参数
    model_info = model.model_info()
    print(f'模型基础信息：{model_info}')

    prompt = "你好呀，今天吃什么呢？你过得怎么样嘞？"
    text = f"{tokenizer.bos_token}{prompt}{tokenizer.eos_token}"
    print(f"Input text: {text}")

    input_ids = tokenizer(text).data['input_ids']
    print("input_ids :", input_ids)
    print("dcode_str :", tokenizer.decode(input_ids))

    X = torch.tensor(input_ids[:-1]).unsqueeze(0)
    Y = torch.tensor(input_ids[1:]).unsqueeze(0)
    print("X shape :", X.shape)
    print("Y shape :", Y.shape)

    # 将输入张量传入模型
    output = model(X, Y)

    # 自回归文本生成
    input_ids = torch.tensor([tokenizer.encode("你好呀，今天吃什么呢？")]).to(model.device)
    generated = model.generate(
        input_ids, 
        max_seq_len=100, 
        temperature=0.8, 
        top_k=30, 
        top_p=0.8,
        repetition_penalty=1.2,
        repetition_context=512,
        eos_token_id=tokenizer.eos_token_id
    )
    print("输出结果：", tokenizer.decode(generated[0]))
