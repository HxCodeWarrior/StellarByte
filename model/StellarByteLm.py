"""
StellarByte 的 CausalLM 封装：继承 PreTrainedModel 与 GenerationMixin，提供生成接口。

该模块将主模型包装，并实现 lm_head（与 embedding 权重共享）。
"""

from typing import Optional, List, Tuple, Union
import torch
from torch import nn
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from config import StellarByteConfig
from StellarByteModel import StellarByteModel


class StellarByteForCausalLM(PreTrainedModel, GenerationMixin):
    """StellarByte 因果语言模型包装器，提供训练与生成接口。"""

    config_class = StellarByteConfig

    def __init__(self, config: StellarByteConfig = None):
        """初始化：构建主模型与 lm_head，并做权重共享。"""
        self.config = config or StellarByteConfig()
        super().__init__(self.config)
        # 主模型
        self.model = StellarByteModel(self.config)
        # lm_head（将隐藏态投影到 vocab 空间），无 bias
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        
        # 权重绑定
        if config.tie_word_embeddings:
            # 权重共享（embedding 与 lm_head 共享同一权重）
            self.model.embed_tokens.weight = self.lm_head.weight

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **args,
    ) -> CausalLMOutputWithPast:
        """前向计算并返回 CausalLMOutputWithPast。

        logits_to_keep 支持在训练过程中只计算部分 logits 以节省显存（可选）。
        """
        # 调用主模型获取隐藏态和 presents
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args,
        )

        # logits_to_keep 可以是 int 或 tensor 的 slice
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # 只对指定位置计算 logits（通常用于训练中节省内存）
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        output = CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
        # 将 aux_loss 附加到输出对象上，便于上层读取
        output.aux_loss = aux_loss
        return output

# 测试 StellarByteForCausalLM 模块
if __name__ == "__main__":
    print("=== 测试 StellarByteForCausalLM 模块 ===")
    
    # 使用相同的配置
    config = StellarByteConfig(
        vocab_size=32000,
        hidden_size=512,
        num_attention_heads=8,
        num_key_value_heads=4,
        num_hidden_layers=4,
        max_position_embeddings=2048,
        dropout=0.1,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_scaling=None,
        intermediate_size=1024,
        hidden_act="silu",
        use_moe=True,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        scoring_func="softmax",
        aux_loss_alpha=0.01,
        seq_aux=False,
        norm_topk_prob=True,
        flash_attn=False,
        verbose_init=False
    )
    
    # 初始化语言模型
    lm_model = StellarByteForCausalLM(config)
    
    # 创建输入
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    print("--- 测试完整 LM 前向 ---")
    output = lm_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True
    )
    
    print(f"输入 tokens 形状: {input_ids.shape}")
    print(f"输出 logits 形状: {output.logits.shape}")
    print(f"Past KV 数量: {len(output.past_key_values)}")
    print(f"隐藏状态形状: {output.hidden_states.shape}")
    print(f"辅助损失: {output.aux_loss}")
    
    # 测试权重共享
    print("--- 测试权重共享 ---")
    print(f"Embedding 和 LM head 权重是否相同: {torch.allclose(lm_model.model.embed_tokens.weight, lm_model.lm_head.weight)}")
    
    # 测试部分 logits 计算（用于训练时节省内存）
    print("--- 测试部分 logits 计算 ---")
    partial_output = lm_model(
        input_ids=input_ids,
        logits_to_keep=4  # 只计算最后4个位置的logits
    )
    print(f"部分 logits 形状: {partial_output.logits.shape}")
    
    # 测试自回归生成
    print("--- 测试自回归生成 ---")
    # 模拟第一步
    first_output = lm_model(
        input_ids=input_ids[:, :1],
        use_cache=True
    )
    print(f"第一步 logits 形状: {first_output.logits.shape}")
    
    # 模拟第二步（使用cache）
    second_output = lm_model(
        input_ids=input_ids[:, 1:2],
        past_key_values=first_output.past_key_values,
        use_cache=True
    )
    print(f"第二步 logits 形状: {second_output.logits.shape}")
    
    print("StellarByteForCausalLM 模块测试通过!")
