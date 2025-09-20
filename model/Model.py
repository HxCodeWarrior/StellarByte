import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional,List,Tuple,Union
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    VocabParallelEmbedding,
)
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from .config import StellarByteModelArgs
    from .RMSNorm import StellarByteRMSNorm
    from .PositionEmbedding import StellarByteRotaryPositionEmbedding
    from .TransformerBlock import StellarByteBlock
    from .MoE import StellarByteMOEFeedForward
except:
    from config import StellarByteModelArgs
    from RMSNorm import StellarByteRMSNorm
    from PositionEmbedding import StellarByteRotaryPositionEmbedding
    from TransformerBlock import StellarByteBlock
    from MoE import StellarByteMOEFeedForward

class StellarByteModel(nn.Module):
    def __init__(self, params: StellarByteModelArgs):
        # 调用父类初始化方法
        super().__init__()
        # 保存模型参数
        self.params = params
        # 设置词汇表大小
        self.vocab_size = params.vocab_size
        # 设置网络层数
        self.n_layers = params.n_layers
        # 初始化
        init_method = nn.init.xavier_uniform_

        # 初始化词嵌入层，使用并行嵌入提高效率
        self.tok_embeddings = VocabParallelEmbedding(
            params.vocab_size,  # 词汇表大小
            params.dim,  # 嵌入维度
            init_method=init_method  # 初始化方法
        )

        # 初始化Transformer层列表
        self.layers = torch.nn.ModuleList()
        # 循环创建每一层Transformer块
        for layer_id in range(params.n_layers):
            self.layers.append(StellarByteBlock(layer_id, params))

        # 初始化RMS归一化层
        self.norm = StellarByteRMSNorm(params.dim, eps=params.norm_eps)
        
        # 初始化输出线性层，使用列并行提高效率
        self.output = ColumnParallelLinear(
            params.dim,  # 输入维度
            params.vocab_size,  # 输出维度（词汇表大小）
            bias=False,  # 不使用偏置
            init_method=init_method  # 初始化方法
        )

        # 预计算旋转位置编码的频率矩阵
        self.freqs_cis = StellarByteRotaryPositionEmbedding.precompute_freqs_cis(
            params.dim // params.n_heads,  # 每个头的维度
            params.max_seq_len * 2,  # 最大序列长度（乘以2可能是为了缓存）
            params.rope_theta,  # RoPE的theta参数
        )

    def forward(self, tokens: torch.Tensor, start_pos: int, targets: Optional[torch.Tensor] = None):
        # 获取输入tokens的批次大小和序列长度
        _bsz, seqlen = tokens.shape
        # 将tokens转换为词嵌入向量
        h = self.tok_embeddings(tokens)
        # 将位置编码频率矩阵移动到与嵌入相同的设备
        self.freqs_cis = self.freqs_cis.to(h.device)
        # 获取从start_pos开始到start_pos+seqlen的位置编码
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        # 逐层处理输入
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis)
        # 应用层归一化
        h = self.norm(h)
        # 通过输出层获取预测结果，并转换为float类型
        output = self.output(h).float()
        
        # 根据是否提供targets决定输出计算方式
        if targets is not None:
            # 训练模式：计算全部位置的logits和损失
            logits = self.output(h)
            # 计算交叉熵损失，忽略索引0（padding），保持每个位置的损失值
            last_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1), 
                ignore_index=0, 
                reduction='none'
            )
        else:
            # 推理模式：只计算最后一个位置的输出
            logits = self.output(h[:, [-1], :]) 
            last_loss = None

        # 计算所有MoE层的辅助损失之和（用于负载平衡）
        aux_loss = sum(
            layer.feed_forward.aux_loss  # 获取每层的辅助损失
            for layer in self.layers  # 遍历所有层
            if isinstance(layer.feed_forward, StellarByteMOEFeedForward)  # 只处理MoE层
        )
        # 返回输出和辅助损失
        return output, logits, last_loss, aux_loss

class StellarByteForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = StellarByteModelArgs

    def __init__(self, config: StellarByteModelArgs = None):
        self.config = config or StellarByteModelArgs()
        super().__init__(self.config)
        self.model = StellarByteModel(self.config)
        self.OUT = CausalLMOutputWithPast()
    
    def forward(self, tokens: torch.Tensor, start_pos: int, targets: Optional[torch.Tensor] = None, **args):
        output, logits, last_loss, aux_loss = self.model(tokens, start_pos, targets, **args)
        self.OUT.__setitem__('output', output)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('last_loss', last_loss)
        self.OUT.__setitem__('aux_loss', aux_loss)
        return self.OUT
