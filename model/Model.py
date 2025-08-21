import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Tuple, Union

try:
    from .utils.KVCache      import ByteKVCache
    from .EmbeddingLayer     import ByteEmbedding
    from .DecoderLayer       import ByteDecoderLayer
    from .RMSNorm            import ByteRMSNorm
    from .config             import ByteModelConfig
except ImportError:
    from utils.KVCache      import ByteKVCache
    from EmbeddingLayer     import ByteEmbedding
    from DecoderLayer       import ByteDecoderLayer
    from RMSNorm            import ByteRMSNorm
    from config             import ByteModelConfig

class ByteModel(PreTrainedModel):
    """基于Transformer架构的自回归语言模型。
    
    核心组件：
    - 词嵌入层 (ByteEmbedding)
    - 多层解码器 (ByteDecoderLayer)
    - 输出投影层 (Linear)
    - 自回归文本生成功能 (generate)
    
    特性：
    1. 支持张量并行和分布式训练
    2. 实现高效KV缓存加速推理
    3. 完整的文本生成功能（温度采样、Top-K/P等）
    4. 权重共享（词嵌入层与输出层）
    5. 残差投影层特殊初始化
    
    参数:
        args (ByteModelConfig): 模型配置对象
    """
    config_class = ByteModelConfig        # Hugging Face模型配置类
    loss         = Optional[torch.Tensor] # 训练损失存储

    def __init__(self, args: ByteModelConfig = None):
        super().__init__(args)
        # 初始化模型参数
        self.args = args
        # 词汇表大小
        self.vocab_size = args.vocab_size
        # 模型层数
        self.num_layers = args.num_layers

        # ================= 模型核心组件 =================
        # 词嵌入层
        self.token_embedding = ByteEmbedding(args)
        # Dropout层
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        # Decoder层堆叠
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.num_layers):
            self.layers.append(
                ByteDecoderLayer(args=args, layer_id=layer_id)
            )
        # 归一化层
        self.norm = ByteRMSNorm(dim=args.model_dim, eps=args.layer_norm_eps)
        # 输出层
        self.output = nn.Linear(args.model_dim, args.vocab_size, bias=False)

         # ================ 权重共享 ================
        # 词嵌入层权重雨输出层权重共享
        self.token_embedding.weight = self.output.weight

        # ================ 权重初始化 ================
        # 初始化权重
        self.apply(self._init_weights)
        # 残差投影层特殊缩放初始化
        for pn, p in self.named_parameters():
            if pn.endswith('w2.weight'):
                # 缩放初始化：标准差与层数成反比
                std = 0.02 / math.sqrt(2 * args.num_layers)
                torch.nn.init.normal_(p, mean=0.0, std=std)

        # 初始化MoE辅助损失zux_loss
        self.aux_loss = torch.tensor(0.0, requires_grad=True)
        # 初始化最后一次向前传播的损失属性
        self.loss = None
        # Hugging Face兼容输出格式
        self.OUT = CausalLMOutputWithPast()
        # 分布式训练相关（完整模型不分割）
        self._no_split_models = [name for name, _ in self.named_modules()]  # 不分割的模块列表
    
    def _init_weights(self, module):
        """模块权重初始化策略。
        
        根据模块类型应用不同初始化：
        - Linear层：Xavier正态分布
        - Embedding层：标准正态分布
        
        Args:
            module (nn.Module): 待初始化的神经网络模块
        """
        std = self.args.initializer_range
        if isinstance(module, nn.Linear):
            # 线性层：Xavier初始化
            nn.init.normal_(module.weight, mean=0.0, std=std)
            # 偏置项：零初始化
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
        elif isinstance(module, nn.Embedding):
            # 嵌入层：正态分布初始化
            nn.init.normal_(module.weight, mean=0.0, std=std)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[ByteKVCache] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """前向传播逻辑。
        
        处理流程：
        1. 词嵌入
        2. Dropout
        3. 多层解码器处理
        4. 最终层归一化
        5. 输出投影
        6. 损失计算（训练时）
        
        参数:
            input_ids: 输入token ID矩阵 [batch_size, seq_len]
            labels: 目标token ID矩阵（用于训练）[batch_size, seq_len]
            attention_mask: 注意力掩码（防止关注填充token）
            kv_cache: KV缓存对象（加速自回归生成）
        
        返回:
            CausalLMOutputWithPast: 包含logits和loss的输出对象
        """

        # 获取输入张量的形状信息
        _batch_size, seq_len = input_ids.shape

        # 1. 词嵌入
        hidden_states = self.token_embedding(input_ids) # [B, T, D] 或 [B, T, D/tp]

        # 2. Dropout
        hidden_states = self.dropout(hidden_states)
        
        # 3. 确保辅助损失设备一致性
        self.aux_loss = self.aux_loss.to(hidden_states.device)

        # 4. Decoder层处理
        for layer in self.layers:
            hidden_states, aux_loss = layer(
                hidden_states, 
                attention_mask, 
                kv_cache
            )

            # 累加MoE辅助损失
            if aux_loss is not None:
                self.aux_loss += aux_loss * self.args.moe_loss_coefficient

        # 5. 归一化
        hidden_states = self.norm(hidden_states)

        # 6. 输出投影
        logits = self.output(hidden_states)

        # 7. 损失计算
        loss = None
        if labels is not None:
            # 计算交叉熵损失
            loss = F.cross_entropy(
                logits.view(-1,logits.size(-1)),  # 展平为 [batch*seq_len, vocab_size]
                labels.view(-1),                  # 展平为 [batch*seq_len]
                ignore_index=-100,                # 忽略特殊标签（如填充token）
                reduction='mean'                  # 批内平均损失
            )
            # 已有损失：交叉熵损失、MoE辅助损失
            # 总损失 = 交叉熵损失
            self.loss = loss
        else:
            # 推理时，只对最后一个位置的输出进行向前传播计算
            logits = self.output(hidden_states[:, [-1], :]) # 保留最后一个token的输出
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
        """自回归文本生成函数
        
        生成策略：
        1. 使用KV缓存加速自回归生成
        2. 支持多种采样技术（温度、Top-K、Top-p）
        3. 重复惩罚机制避免重复生成

        参数:
            input_ids: 初始输入序列 [batch_size, seq_len]
            max_seq_len: 生成的最大序列长度（包括输入长度）
            temperature: 采样温度（>0）
            top_k: 只考虑概率最高的k个token（0表示禁用）
            repetition_penalty: 重复惩罚因子（>1）
            repetition_context: 应用重复惩罚的上下文长度（0表示禁用）
            eos_token_id: 结束符ID（遇到时停止生成）
            return_logits: 是否返回每一步的logits
        
        返回:
            生成的完整序列 [batch_size, new_seq_len]
            如果return_logits=True，同时返回logits [batch_size, gen_len, vocab_size]
        """
        # =============== 准备工作 ===============
        device     = next(self.parameters()).device # 模型所在设备
        input_ids  = input_ids.to(device)           # 确保输入在正确设备
        batch_size = input_ids.shape[0]             # 批大小

        # 结束符处理逻辑
        if eos_token_id is None:
            eos_token_id = -1  # 无效ID

        # 预采样参数校验
        temperature        = max(temperature, 1e-5)       # 防止除零
        top_k              = max(top_k, 0)                # 确保非负  
        repetition_penalty = max(repetition_penalty, 1.0) # 确保≥1

        # 计算本地KV头数（考虑张量并行和分组查询注意力）
        head_dim = self.args.model_dim // self.args.num_attention_heads
        # 计算本地KV头数（考虑张量并行和分组查询注意力）
        num_kv_heads = self.args.num_kv_heads or self.args.num_attention_heads
        num_local_kv_heads = num_kv_heads // max(1, self.args.tensor_parallel_size)

        # 创建并清空缓存
        kv_cache = ByteKVCache(
            num_layers  = self.args.num_layers,
            num_heads   = num_local_kv_heads,
            head_dim    = head_dim,
            max_seq_len = max_seq_len,
            batch_size  = batch_size,
            dtype       = self.output.weight.dtype,
            device      = device
        )

        # =============== 生成循环 ===============
        generated  = input_ids.clone()             # 存储生成的 token（初始为输入）
        all_logits = [] if return_logits else None # logits记录器

        # 先跑一次全量前向，把 prompt 全部写入缓存
        _ = self.forward(input_ids=generated, kv_cache=kv_cache)

        for _ in range(max_seq_len - input_ids.size(1)):
            # 1. 模型前向传播
            output = self.forward(
                input_ids=generated[:, -1:],  # 仅最后1个token作为输入
                kv_cache=kv_cache
            )
            
            # 2. 获取当前logits
            logits = output.logits     # [B, 1, vocab_size]
            logits = logits[:, -1, :]  # 取出最后一个token位置的logits [B, vocab_size]

            # 3. 应用重复惩罚（降低已出现token概率）
            logits = self.repetition_penalty(
                logits, 
                generated,
                penalty=repetition_penalty,
                context_size=repetition_context
            )

            # 4. 采样下一个token
            next_token = self.sample_next_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            # 5. 更新活动样本的生成序列
            generated = torch.cat([generated, next_token], dim=1)

            # 6. 记录logits（如果需要）
            if return_logits:
                all_logits.append(logits.unsqueeze(1))  # [B, 1, V]

            # 7. 检查是否遇到eos_token_id
            if eos_token_id is not None:
                if (next_token == eos_token_id).all():
                    break

        # =============== 输出生成结果 ===============
        if return_logits:
            # 拼接所有logits: [B, gen_len, V]
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
        """基于logits采样下一个token。
        
        支持技术：
        - 温度缩放 (Temperature scaling)
        - Top-K采样
        - Top-p（核采样）
        
        参数:
            logits: 原始预测logits [batch_size, vocab_size]
            temperature: 采样温度（>0）
            top_k: 保留概率最高的k个token（0=禁用）
            top_p: 保留累计概率达p的最小token集合（1.0=禁用）
        
        返回:
            next_token: 采样结果 [batch_size, 1]
        """
        # 温度调节
        if temperature != 1.0:
            logits = logits / temperature

        # Top-K过滤（保留概率最高的k个token）
        if top_k > 0:
            # 获取第k大的值作为阈值
            topk_values, _ = torch.topk(logits, top_k, dim=-1)
            min_topk       = topk_values[:, -1].unsqueeze(-1)
            # 低于阈值的设为负无穷
            logits = torch.where(
                logits < min_topk, 
                torch.full_like(logits, float('-inf')), 
                logits
            )

        # Top-p过滤（核采样）
        if top_p < 1.0:
            # 降序排序logits
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            # 计算累积概率
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # 创建移除掩码（累计概率>top_p的token）
            removal_mask = cumulative_probs > top_p
            # 保留第一个超过阈值的token（右移掩码）
            removal_mask[:, 1:] = removal_mask[:, :-1].clone()
            removal_mask[:, 0] = False

            # 恢复原始索引顺序
            removal_mask = removal_mask.scatter(-1, sorted_indices, removal_mask)
            # 应用掩码（将不需要的token设为负无穷）
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

        参数:
            logits: [batch_size, vocab_size] 当前步的logits
            generated: [batch_size, seq_len] 已生成的序列
            penalty: 重复惩罚因子(>1)
            context_size: 考虑惩罚的上下文长度

        返回:
            应用惩罚后的logits
        """
        # 无惩罚情况快速返回
        if penalty <= 1.0:
            return logits

        batch_size, vocab_size = logits.shape
        device = logits.device

        # 仅考虑最近的上下文（节省计算）
        recent_tokens = generated[:, -context_size:]

        # 逐样本处理
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
    print(f'Output: {output}')

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
