"""
StellarByte 主模型定义：Embedding + N x Block + final RMSNorm。

模型支持 KV cache、RoPE（预计算 cos/sin）以及 MoE 的辅助损失汇总。
"""

import math
import torch
from torch import nn
from transformers import TextStreamer
from typing import Optional, List, Tuple
from config import StellarByteConfig
from RoPE import precompute_freqs_cis
from RMSNorm import RMSNorm
from StellarByteBlock import StellarByteBlock


class StellarByteModel(nn.Module):
    """主模型类，包含 embedding 层与若干 Transformer blocks。"""

    def __init__(self, config: StellarByteConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers

        #  Token ID 参数应用
        self.pad_token_id = config.pad_token_id
        self.bos_token_id = config.bos_token_id  
        self.eos_token_id = config.eos_token_id

        # token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # dropout
        self.dropout = nn.Dropout(config.dropout)
        # Transformer 层列表
        self.layers = nn.ModuleList([StellarByteBlock(l, config) for l in range(self.num_hidden_layers)])
        # 最后层 RMSNorm
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

        # 参数初始化
        self.apply(self._init_weights)

        # 权重绑定
        if config.tie_word_embeddings:
            self.tie_weights()
    
    def _init_weights(self, module):
        """使用 initializer_range 初始化模型权重"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # 线性层和嵌入层使用正态分布初始化
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                # 偏置项初始化为零
                module.bias.data.zero_()
        elif isinstance(module, RMSNorm):
            # RMSNorm 权重初始化为 1
            module.weight.data.fill_(1.0)
        
        # ==================== 残差投影的特殊缩放初始化 ====================
        # 对 FFN 输出投影 (down_proj) 和注意力输出投影 (o_proj) 进行特殊初始化
        param_name = None
        # 获取参数的完整名称
        for name, param in self.named_parameters():
            if param is module.weight if hasattr(module, 'weight') else None:
                param_name = name
                break
            
        if param_name is not None:
            # 检查是否是残差投影层
            is_residual_projection = (
                param_name.endswith('down_proj.weight') or  # FFN 输出投影
                param_name.endswith('o_proj.weight')        # 注意力输出投影
            )

            if is_residual_projection:
                # 使用缩放初始化：std = 0.02 / sqrt(2 * n_layers)
                scale = 0.02 / math.sqrt(2 * self.config.num_hidden_layers)
                module.weight.data.normal_(mean=0.0, std=scale)
                if self.config.verbose_init:
                    print(f"应用缩放初始化到: {param_name}, std={scale:.6f}")
    
    def tie_weights(self):
        """绑定输入词嵌入和输出层权重"""
        if hasattr(self, 'embed_tokens') and hasattr(self, 'lm_head'):
            self.lm_head.weight = self.embed_tokens.weight

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

        #  注意力掩码处理（使用 pad_token_id） 
        if attention_mask is None and self.pad_token_id is not None:
            # 自动生成注意力掩码
            attention_mask = (input_ids != self.pad_token_id).long()

        # 处理 past_key_values 的兼容性
        if hasattr(past_key_values, "layers"):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        
        # 计算 start_pos（基于 past key 的长度）
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # embedding + dropout
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 构造位置嵌入（cos, sin）切片
        current_freqs_cos = self.freqs_cos[start_pos:start_pos + seq_length]
        current_freqs_sin = self.freqs_sin[start_pos:start_pos + seq_length]
        position_embeddings = (current_freqs_cos, current_freqs_sin)

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

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        attention_mask: Optional[torch.Tensor] = None,
        stop_tokens: Optional[List[int]] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        do_sample: bool = True,
        streamer: Optional[TextStreamer] = None,
        **kwargs
    ):
        """
        序列生成函数，支持多种采样策略和停止条件。

        Args:
            input_ids: 输入token序列，形状为 (batch_size, seq_len)
            max_new_tokens: 最大生成token数量
            temperature: 温度参数，控制随机性 (0.0-1.0)
            top_k: top-k采样参数，保留概率最高的k个token
            top_p: nucleus采样参数，保留累积概率达到p的token
            repetition_penalty: 重复惩罚参数，>1.0降低重复token概率
            attention_mask: 注意力掩码，形状同input_ids
            stop_tokens: 停止token列表，遇到任一token停止生成
            pad_token_id: padding token ID
            eos_token_id: 结束token ID
            do_sample: 是否使用采样，False则使用贪心解码
            streamer: 流式输出处理器
            **kwargs: 其他参数

        Returns:
            generated_sequences: 生成的序列
        """
        # 参数校验和初始化
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # 使用模型配置中的token ID
        if pad_token_id is None:
            pad_token_id = self.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.eos_token_id

        # 设置停止token
        stop_tokens = stop_tokens or []
        if eos_token_id is not None and eos_token_id not in stop_tokens:
            stop_tokens.append(eos_token_id)

        # 初始化注意力掩码
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # 存储生成的序列
        generated_sequences = input_ids.clone()
        active_sequences = torch.ones(batch_size, dtype=torch.bool, device=device)

        # 初始化past_key_values
        past_key_values = None

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 自回归生成循环
        for step in range(max_new_tokens):
            # 前向传播
            hidden_states, past_key_values, _ = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True
            )

            # 获取最后一个token的隐藏状态
            next_token_logits = hidden_states[:, -1, :]

            # 重复惩罚
            if repetition_penalty != 1.0:
                self._apply_repetition_penalty(next_token_logits, input_ids, repetition_penalty)

            # 温度调节
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # 采样策略
            if do_sample:
                if top_p < 1.0:
                    # nucleus采样 (top-p)
                    next_token_probs = self._apply_top_p_filtering(next_token_logits, top_p)
                elif top_k is not None and top_k > 0:
                    # top-k采样
                    next_token_probs = self._apply_top_k_filtering(next_token_logits, top_k)
                else:
                    next_token_probs = torch.softmax(next_token_logits, dim=-1)

                # 采样下一个token
                next_tokens = torch.multinomial(next_token_probs, num_samples=1)
            else:
                # 贪心解码
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # 更新停止序列
            stopped_sequences = torch.zeros(batch_size, dtype=torch.bool, device=device)
            for stop_token in stop_tokens:
                stopped_sequences = stopped_sequences | (next_tokens.squeeze(-1) == stop_token)

            # 对于已停止的序列，使用pad_token_id
            next_tokens = torch.where(
                active_sequences.unsqueeze(-1),
                next_tokens,
                torch.tensor(pad_token_id, device=device).expand_as(next_tokens)
            )

            # 更新active序列状态
            active_sequences = active_sequences & (~stopped_sequences)

            if streamer is not None:
                streamer.put(next_tokens.cpu())

            # 如果所有序列都已停止，提前结束
            if not active_sequences.any():
                break
            
            # 准备下一轮输入
            input_ids = next_tokens
            attention_mask = torch.cat([
                attention_mask, 
                active_sequences.unsqueeze(-1).long()
            ], dim=1)

            # 更新生成的序列
            generated_sequences = torch.cat([generated_sequences, next_tokens], dim=1)

        if streamer is not None:
            streamer.end()

        return generated_sequences

    def _apply_repetition_penalty(self, logits, input_ids, penalty):
        """应用重复惩罚"""
        for batch_idx in range(logits.shape[0]):
            for token_id in set(input_ids[batch_idx].tolist()):
                if logits[batch_idx, token_id] < 0:
                    logits[batch_idx, token_id] *= penalty
                else:
                    logits[batch_idx, token_id] /= penalty

    def _apply_top_k_filtering(self, logits, top_k):
        """应用top-k过滤"""
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float('Inf')
        return torch.softmax(logits, dim=-1)

    def _apply_top_p_filtering(self, logits, top_p):
        """应用nucleus (top-p) 过滤"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # 移除累积概率超过top_p的token
        sorted_indices_to_remove = cumulative_probs > top_p
        # 保留第一个超过阈值的token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        for batch_idx in range(logits.shape[0]):
            indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
            logits[batch_idx, indices_to_remove] = -float('Inf')

        return torch.softmax(logits, dim=-1)

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
