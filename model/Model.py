import math
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
    from .PositionEmbedding import StellarByteRoPE
    from .TransformerBlock import StellarByteBlock
    from .MoE import StellarByteMOEFeedForward
except:
    from config import StellarByteModelArgs
    from RMSNorm import StellarByteRMSNorm
    from PositionEmbedding import StellarByteRoPE
    from TransformerBlock import StellarByteBlock
    from MoE import StellarByteMOEFeedForward

class StellarByteModel(nn.Module):
    config_class = StellarByteModelArgs

    def __init__(self, args: StellarByteModelArgs):
        # 调用父类初始化方法
        super().__init__()
        # 保存模型参数
        self.args = args
        # 设置词汇表大小
        self.vocab_size = args.vocab_size
        # 设置网络层数
        self.num_layers = args.num_layers

        # 初始化词嵌入层，使用并行嵌入提高效率
        self.tok_embeddings = nn.Embedding(
            args.vocab_size,  # 词汇表大小
            args.dim,  # 嵌入维度
        )

        # 初始化Transformer层列表
        self.layers = nn.ModuleList()
        # 循环创建每一层Transformer块
        for layer_id in range(args.num_layers):
            self.layers.append(StellarByteBlock(layer_id, args))

        # 初始化RMS归一化层
        self.norm = StellarByteRMSNorm(args.dim, eps=args.norm_eps)
        
        # 初始化输出线性层，使用列并行提高效率
        self.output = nn.Linear(
            args.dim,  # 输入维度
            args.vocab_size,  # 输出维度（词汇表大小）
            bias=False,  # 不使用偏置
        )

        # 预计算旋转位置编码的频率矩阵
        freqs_cos, freqs_sin = StellarByteRoPE.precompute_freqs_cis(
            args.dim // args.num_heads,  # 每个头的维度
            args.max_seq_len,  # 最大序列长度
            args.rope_theta,  # RoPE的theta参数
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # 初始化所有权重
        self.apply(self._init_weights)
        # 对残差投影进行特殊的缩放初始化
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * args.num_layers))

        self.OUT = CausalLMOutputWithPast()

    def _init_weights(self, module):
        # 初始化权重的函数
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, 
                input_ids: torch.Tensor, 
                labels: Optional[torch.Tensor] = None, 
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None):
        # 获取输入input_ids的批次大小和序列长度
        _bsz, seqlen = input_ids.shape

        # 处理KV缓存
        if past_key_values is not None:
            # 使用过去的KV缓存
            cache_len = past_key_values[0][0].shape[2] if past_key_values[0] is not None else 0
        else:
            # 没有过去的KV缓存
            cache_len = 0
            past_key_values = [None] * len(self.layers)

        # 将input_ids转换为词嵌入向量
        hidden_states = self.tok_embeddings(input_ids)

        # 将位置编码频率矩阵移动到与嵌入相同的设备
        self.freqs_cos = self.freqs_cos.to(hidden_states.device)
        self.freqs_sin = self.freqs_sin.to(hidden_states.device)

        # 当前序列的位置编码切片
        freqs_cos = self.freqs_cos[cache_len:cache_len+seqlen]
        freqs_sin = self.freqs_sin[cache_len:cache_len+seqlen]

        # 存储每层的KV缓存
        presents = []

        # 逐层处理输入
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            # 前向传播
            hidden_states, present = layer(
                hidden_states, 
                freqs_cos, 
                freqs_sin, 
                past_key_value
            )
            
            # 存储该层的新KV缓存
            presents.append(present)

        # 应用层归一化
        hidden_states = self.norm(hidden_states)

        # 通过输出层获取预测结果，并转换为float类型
        output = self.output(hidden_states).float()
        
        # 根据是否提供labels决定输出计算方式
        if labels is not None:
            # 训练模式：计算全部位置的logits和损失
            logits = self.output(hidden_states)
            # 计算交叉熵损失，忽略索引0（padding），保持每个位置的损失值
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                labels.view(-1), 
                ignore_index=-100, 
                reduction='none'
            )
        else:
            # 推理模式：只计算最后一个位置的输出
            logits = self.output(hidden_states[:, [-1], :]) 
            loss = None

        # 计算所有MoE层的辅助损失之和（用于负载平衡）
        aux_loss = sum(
            layer.feed_forward.aux_loss  # 获取每层的辅助损失
            for layer in self.layers  # 遍历所有层
            if isinstance(layer.feed_forward, StellarByteMOEFeedForward)  # 只处理MoE层
        )

        # 返回输出和辅助损失
        self.OUT.__setitem__('output', output)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('loss', loss)
        self.OUT.__setitem__('aux_loss', aux_loss)
        return self.OUT


if __name__ == "__main__":
    print("="*50)
    print("StellarByteBlock 形状验证测试")
    print("="*50)

    # 设置测试参数
    class StellarByteModelArgs:
        vocab_size=32768
        dim=768
        num_layers=32
        num_heads=32
        num_kv_heads=None
        multiple_of=256
        ffn_dim_multiplier=None
        norm_eps=1e-5
        rope_theta=500000
        max_batch_size=32
        max_seq_len=2048
        enabled_flash_attn=False
        enabled_kv_cache=False
        attention_dropout=0.0
        resid_dropout=0.0
        ffn_dropout=0.0
        rms_norm_eps=1e-6

        enabled_moe=False
        num_experts_per_tok=2
        num_routed_experts=4
        num_shared_experts=1
        scoring_func='softmax'
        aux_loss_alpha=0.1
        seq_aux=True
        norm_topk_prob=True
        gating_dim=768

        model_parallel_size: int = 1

    args = StellarByteModelArgs()
    
    # 创建模型实例
    model = StellarByteModel(args)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params:,}")
    
    # 测试1: 训练模式（带labels）
    print("\n=== 测试1: 训练模式 ===")
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, args.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, args.vocab_size, (batch_size, seq_len))
    
    # 前向传播
    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
    
    # 验证输出形状
    print(f"输入形状: {input_ids.shape}")
    print(f"目标形状: {labels.shape}")
    print(f"Logits形状: {outputs.logits.shape}")
    print(f"Loss值: {outputs.loss.item() if outputs.loss is not None else 'None'}")
    print(f"Aux Loss值: {outputs.aux_loss if hasattr(outputs, 'aux_loss') else 'None'}")
    
    # 验证输出形状是否符合预期
    assert outputs.logits.shape == (batch_size, seq_len, args.vocab_size), \
        f"Logits形状错误: 期望({batch_size}, {seq_len}, {args.vocab_size}), 实际{outputs.logits.shape}"
    
    # 测试2: 推理模式（不带labels）
    print("\n=== 测试2: 推理模式 ===")
    with torch.no_grad():
        outputs = model(input_ids)
    
    print(f"输入形状: {input_ids.shape}")
    print(f"Logits形状: {outputs.logits.shape}")
    print(f"Loss值: {outputs.loss.item() if outputs.loss is not None else 'None'}")
    
    # 验证输出形状是否符合预期
    assert outputs.logits.shape == (batch_size, 1, args.vocab_size), \
        f"Logits形状错误: 期望({batch_size}, 1, {args.vocab_size}), 实际{outputs.logits.shape}"
    
    # 测试3: 使用KV缓存的推理
    print("\n=== 测试3: 使用KV缓存的推理 ===")
    past_key_values = None
    for i in range(seq_len):
        # 逐步输入token
        step_input = input_ids[:, i:i+1]
        
        with torch.no_grad():
            outputs = model(step_input, past_key_values=past_key_values)
        
        past_key_values = outputs.past_key_values
        print(f"步骤 {i+1}: 输入形状 {step_input.shape}, 输出形状 {outputs.logits.shape}")
        
        # 验证每一步的输出形状
        assert outputs.logits.shape == (batch_size, 1, args.vocab_size), \
            f"步骤 {i+1} Logits形状错误: 期望({batch_size}, 1, {args.vocab_size}), 实际{outputs.logits.shape}"
    
    print("\n✅ 所有测试通过! 模型输入输出形状正确。")