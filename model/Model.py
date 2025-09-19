import torch
import torch.nn as nn
from typing import Optional,List,Tuple,Union
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    VocabParallelEmbedding,
)
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
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

    # 使用推理模式（不计算梯度，提高推理速度）
    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        # 获取输入tokens的批次大小和序列长度
        _bsz, seqlen = tokens.shape
        # 将tokens转换为词嵌入向量
        h = self.tok_embeddings(tokens)
        # 将位置编码频率矩阵移动到与嵌入相同的设备
        self.freqs_cis = self.freqs_cis.to(h.device)
        # 获取从start_pos开始到start_pos+seqlen的位置编码
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        # 初始化注意力掩码为None
        mask = None
        # 如果序列长度大于1，需要创建注意力掩码
        if seqlen > 1:
            # 创建一个全为负无穷的矩阵，用于掩码
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            # 将上三角部分（不包括对角线）设置为负无穷，防止看到未来信息
            mask = torch.triu(mask, diagonal=1)

            # 当使用键值缓存时，我们只计算新序列的注意力分数
            # 因此分数矩阵的大小为(seqlen, cache_len + seqlen)
            # 掩码条目(i, j)对于j > cache_len + i被屏蔽，因为行i对应的是token cache_len + i
            # 水平拼接零矩阵和上三角掩码矩阵
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)  # 确保掩码与h的数据类型一致

        # 逐层处理输入
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        # 应用层归一化
        h = self.norm(h)
        # 通过输出层获取预测结果，并转换为float类型
        output = self.output(h).float()
        
        # 计算所有MoE层的辅助损失之和（用于负载平衡）
        aux_loss = sum(
            layer.feed_forward.aux_loss  # 获取每层的辅助损失
            for layer in self.layers  # 遍历所有层
            if isinstance(layer.feed_forward, StellarByteMOEFeedForward)  # 只处理MoE层
        )
        # 返回输出和辅助损失
        return output, aux_loss


class StellarByteForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = StellarByteModelArgs

    def __init__(self, config: StellarByteModelArgs = None):
        self.config = config or StellarByteModelArgs()
        super().__init__(self.config)
        self.model = StellarByteModelArgs(self.config)
        self.OUT = CausalLMOutputWithPast()

    def forward():
        pass