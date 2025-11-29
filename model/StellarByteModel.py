"""
StellarByte 主模型定义：Embedding + N x Block + final RMSNorm。

模型支持 KV cache、RoPE（预计算 cos/sin）以及 MoE 的辅助损失汇总。
"""

from typing import Optional, List, Tuple
import torch
from torch import nn
from config import StellarByteConfig
from RoPE import precompute_freqs_cis
from StellarByteBlock import StellarByteBlock


class StellarByteModel(nn.Module):
    """主模型类，包含 embedding 层与若干 Transformer blocks。"""

    def __init__(self, config: StellarByteConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers

        # token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # dropout
        self.dropout = nn.Dropout(config.dropout)
        # Transformer 层列表
        self.layers = nn.ModuleList([StellarByteBlock(l, config) for l in range(self.num_hidden_layers)])
        # 最后层 RMSNorm
        from RMSNorm import RMSNorm

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 预计算 RoPE 的 cos/sin 矩阵（按 head_dim 传入）
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        # 注册为 buffer（不参与训练但随模型移动到设备）
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        """主前向函数，返回 hidden_states、presents 与 aux_loss（若 MoE 存在）。"""
        batch_size, seq_length = input_ids.shape
        # 处理 past_key_values 的兼容性
        if hasattr(past_key_values, "layers"):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        # 计算 start_pos（基于 past key 的长度）
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # embedding + dropout
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 构造位置嵌入（cos, sin）切片
        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + seq_length],
            self.freqs_sin[start_pos : start_pos + seq_length],
        )

        presents = []
        # 逐层计算
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)

        # 最终归一化
        hidden_states = self.norm(hidden_states)

        # 汇总所有 MoE 层的辅助损失（若存在）
        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.layers
            if hasattr(layer.mlp, "aux_loss")
        )

        return hidden_states, presents, aux_loss

# 测试 StellarByteModel 模块
if __name__ == "__main__":
    print("=== 测试 StellarByteModel 模块 ===")
    
    # 使用相同的配置
    class TestConfig:
        vocab_size = 32000
        hidden_size = 512
        num_attention_heads = 8
        num_key_value_heads = 4
        num_hidden_layers = 4
        max_position_embeddings = 2048
        dropout = 0.1
        rms_norm_eps = 1e-6
        rope_theta = 10000.0
        rope_scaling = None
        intermediate_size = 1024
        hidden_act = "silu"
        use_moe = True
        n_routed_experts = 4
        num_experts_per_tok = 2
        n_shared_experts = 1
        scoring_func = "softmax"
        aux_loss_alpha = 0.01
        seq_aux = False
        norm_topk_prob = True
        flash_attn = False
    
    config = TestConfig()
    
    # 初始化模型
    model = StellarByteModel(config)
    
    # 创建输入
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    print("--- 测试完整模型前向 ---")
    hidden_states, presents, aux_loss = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True
    )
    
    print(f"输入 tokens 形状: {input_ids.shape}")
    print(f"输出隐藏状态形状: {hidden_states.shape}")
    print(f"KV cache 数量: {len(presents)}")
    print(f"每个 KV cache 形状: k={presents[0][0].shape}, v={presents[0][1].shape}")
    print(f"辅助损失: {aux_loss}")
    
    # 测试自回归推理（逐步生成）
    print("--- 测试自回归推理 ---")
    # 模拟第一步
    first_output, first_presents, _ = model(
        input_ids=input_ids[:, :1],  # 只输入第一个token
        use_cache=True
    )
    print(f"第一步输出形状: {first_output.shape}")
    
    # 模拟第二步（使用第一步的cache）
    second_output, second_presents, _ = model(
        input_ids=input_ids[:, 1:2],  # 第二个token
        past_key_values=first_presents,
        use_cache=True
    )
    print(f"第二步输出形状: {second_output.shape}")
    print(f"第二步 KV cache 形状: k={second_presents[0][0].shape}, v={second_presents[0][1].shape}")
    
    print("StellarByteModel 模块测试通过!\n")
