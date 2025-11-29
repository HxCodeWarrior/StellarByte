"""
StellarByte 单层 Block（组合 Attention + RMSNorm + FFN / MoE）。

每一行都有中文注释，接口与原始 MiniMindBlock 保持兼容。
"""

import torch
from torch import nn
from RMSNorm import RMSNorm
from Attention import MultiHeadAttention
from FFN import FeedForward, MOEFeedForward
from config import StellarByteConfig


class StellarByteBlock(nn.Module):
    """单层 Transformer Block，包含 attention 与 MLP（或 MoE）。"""

    def __init__(self, layer_id: int, config: StellarByteConfig):
        """初始化 Block。

        Args:
            layer_id: 层的序号（用于日志/调试）
            config: StellarByteConfig 配置
        """
        super().__init__()
        self.layer_id = layer_id
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads

        # 自注意力子模块
        self.self_attn = MultiHeadAttention(config)
        # pre-norm（输入归一化）
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # attention 后的层归一化
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # MLP：根据配置选择普通 FFN 或 MoE FFN
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        """Block 前向：

        1. pre-norm -> attention -> residual add
        2. post-attention norm -> ffn/moe -> residual add

        返回新的 hidden_states 与 present_key_value（若 use_cache=True）。
        """
        # 保存残差用于后续相加
        residual = hidden_states
        # attention 输入先做 RMSNorm
        attn_input = self.input_layernorm(hidden_states)
        # 调用自注意力模块（会返回 present kv，当 use_cache=True 时）
        attn_output, present_key_value = self.self_attn(
            attn_input,
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )
        # 添加残差
        hidden_states = residual + attn_output

        # 将 attention 后的结果做 post-attn norm 并送入 MLP/MoE
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value

# 测试 StellarByteBlock 模块
if __name__ == "__main__":
    print("=== 测试 StellarByteBlock 模块 ===")
    
    # 创建统一的测试配置
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
        flash_attn = False
        # FFN 配置
        intermediate_size = 1024
        hidden_act = "silu"
        # MoE 配置
        use_moe = False
        n_routed_experts = 4
        num_experts_per_tok = 2
        n_shared_experts = 1
        scoring_func = "softmax"
        aux_loss_alpha = 0.01
        seq_aux = False
        norm_topk_prob = True
    
    config = TestConfig()
    
    # 测试普通 Block
    print("--- 测试普通 Block ---")
    block = StellarByteBlock(layer_id=0, config=config)
    
    # 创建输入数据
    batch_size, seq_len = 2, 16
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # 预计算位置编码（与主模型一致）
    from RoPE import precompute_freqs_cis
    freqs_cos, freqs_sin = precompute_freqs_cis(
        dim=config.hidden_size // config.num_attention_heads,
        end=config.max_position_embeddings,
        rope_base=config.rope_theta,
        rope_scaling=config.rope_scaling,
    )
    position_embeddings = (freqs_cos[:seq_len], freqs_sin[:seq_len])
    
    # 前向传播
    output, present_kv = block(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        use_cache=True
    )
    
    print(f"Block 输入形状: {hidden_states.shape}")
    print(f"Block 输出形状: {output.shape}")
    print(f"KV cache 形状: k={present_kv[0].shape}, v={present_kv[1].shape}")
    
    # 测试带 past_key_values
    print("--- 测试带 KV cache 的 Block ---")
    output_with_cache, new_present_kv = block(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        past_key_value=present_kv,
        use_cache=True
    )
    print(f"带 cache 输出形状: {output_with_cache.shape}")
    print(f"新 KV cache 形状: k={new_present_kv[0].shape}, v={new_present_kv[1].shape}")
    
    # 测试 MoE Block
    print("--- 测试 MoE Block ---")
    config.use_moe = True
    moe_block = StellarByteBlock(layer_id=1, config=config)
    moe_output, _ = moe_block(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        use_cache=False
    )
    print(f"MoE Block 输出形状: {moe_output.shape}")
    
    print("StellarByteBlock 模块测试通过!\n")
