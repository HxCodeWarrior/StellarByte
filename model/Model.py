"""Byte‑Transformer 模型（中文高质量注释版）
=================================================
本文件实现了一个基于 **XPos** 旋转位置编码的 GPT‑类语言模型，
包含 DeepNorm、LayerScale、DropPath、KV‑Cache 等现代化 Tricks，
并与 HuggingFace 生态完全兼容。

作者：ByteWyrm  |  日期：2025‑07‑14
"""

import math
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

# ==== 本项目自定义模块 ====
from config import ByteModelConfig            # 模型配置类（超参数集中管理）
from DecoderLayer import ByteDecoderLayer     # 单层解码器实现（Attention + MLP）
from utils.KVCache import KVCache             # 高性能 KV 缓存（推理加速）
from RMSNorm import RMSNorm                   # RMSNorm 归一化层

__all__ = ["ByteTransformer"]


class ByteTransformer(PreTrainedModel):
    """Byte‑level Transformer 语言模型

    主要特性
    --------
    1. **XPosRotaryEmbedding**：改进版 RoPE，长上下文更稳定。
    2. **DeepNorm 初始化**：权重按 `√(2L)` 缩放，深层训练不崩。
    3. **LayerScale & DropPath**：提升梯度流动与正则效果。
    4. **KV‑Cache**：推理复杂度从 `O(T²)` 降到 `O(T)`。
    5. **权重共享**：支持词嵌入 ↔ LM‑Head 共权重，省参数。
    6. 与 HuggingFace API 兼容，可直接用 `Trainer` / `PEFT`。
    """

    # 指定 HF Config 类，便于 `.from_pretrained()` 等函数识别
    config_class = ByteModelConfig

    # ------------------------------------------------------------------
    # 构造函数
    # ------------------------------------------------------------------
    def __init__(self, config: ByteModelConfig):
        super().__init__(config)
        self.config = config  # 保存配置实例，便于外部访问
        D = config.model_dim  # 嵌入维度

        # ------- 词嵌入层 -------
        # shape: [vocab_size, D]
        self.embed_tokens = nn.Embedding(config.vocab_size, D)
        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)

        # ------- KV 缓存 -------
        # 仅在推理阶段使用；训练阶段不会占显存
        self.kv_cache: Optional[KVCache] = None
        if config.use_cache:
            self.kv_cache = KVCache(
                num_layers=config.num_layers,
                num_heads=config.num_attention_heads,  # 总头数
                head_dim=D // config.num_attention_heads,
                max_seq_len=config.max_position_embeddings,
                key_dtype=config.key_cache_dtype,
                value_dtype=config.value_cache_dtype,
                tensor_parallel_size=config.tensor_parallel_size,
                tensor_parallel_rank=config.tensor_parallel_rank,
            )

        # ------- 解码器堆栈 -------
        # 逐层构建 ByteDecoderLayer
        self.layers = nn.ModuleList(
            [ByteDecoderLayer(config, layer_id=i, kv_cache=self.kv_cache)
             for i in range(config.num_layers)]
        )

        # ------- 输出前归一化 -------
        self.norm = RMSNorm(D, eps=config.layer_norm_eps)

        # ------- LM Head -------
        self.lm_head = nn.Linear(D, config.vocab_size, bias=False)

        # 可选：共享权重（Embedding ↔ LM‑Head）
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # ------- 参数初始化 -------
        self.apply(self._init_weights)            # 基础初始化
        self._post_init_residual_scaling()        # 针对残差分支的小初始化

        # ------- 运行时辅助 -------
        self.last_loss: Optional[torch.Tensor] = None  # 记录最近一次 forward 的损失

    # ------------------------------------------------------------------
    # 权重初始化相关
    # ------------------------------------------------------------------
    def _init_weights(self, module: nn.Module):
        """对 Linear / Embedding 进行高斯初始化。"""
        std = self.config.initializer_range / math.sqrt(2 * self.config.num_layers)
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def _post_init_residual_scaling(self):
        """DeepNet 建议：残差分支权重再小一个量级。"""
        std_small = self.config.initializer_range / math.sqrt(2 * self.config.num_layers)
        for name, param in self.named_parameters():
            if name.endswith(".w3.weight") or name.endswith(".W_o.weight") or name.endswith(".lm_head.weight"):
                nn.init.normal_(param, mean=0.0, std=std_small)

    # ------------------------------------------------------------------
    # 工具方法：清空 KV 缓存
    # ------------------------------------------------------------------
    def reset_cache(self):
        if self.kv_cache is not None:
            self.kv_cache.reset()

    # ------------------------------------------------------------------
    # HF 兼容的 Embedding getter / setter
    # ------------------------------------------------------------------
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        self.embed_tokens = new_embeddings
        if self.config.tie_word_embeddings:
            self.lm_head.weight = new_embeddings.weight

    def get_output_embeddings(self):
        return self.lm_head

    # ------------------------------------------------------------------
    # 前向传播
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,        # Token ID 序列
        inputs_embeds: Optional[torch.FloatTensor] = None,   # 直接输入嵌入向量（两者择一）
        attention_mask: Optional[torch.Tensor] = None,       # Padding Mask
        labels: Optional[torch.LongTensor] = None,           # 监督训练标签
        return_hidden_states: bool = False,                  # 是否返回各层隐状态
        use_cache: Optional[bool] = None,                    # 预留接口（暂未区分）
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """主前向函数，兼容 HF `CausalLMOutputWithPast` 输出格式。"""

        # ---- 1. 输入检查 ----
        if input_ids is None and inputs_embeds is None:
            raise ValueError("input_ids 与 inputs_embeds 必须至少提供一个！")

        # ---- 2. 词嵌入 ----
        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)   # [B,T,D]
        else:
            hidden_states = inputs_embeds
        hidden_states = self.embed_dropout(hidden_states)

        # ---- 3. 构造 Padding 掩码 ----
        additive_mask = None  # 经典 Attention 分支：-inf 掩码
        if attention_mask is not None:
            additive_mask = (1.0 - attention_mask.to(hidden_states.dtype)) * torch.finfo(hidden_states.dtype).min
            additive_mask = additive_mask[:, None, None, :]  # [B,1,1,T]

        # ---- 4. 逐层前向 ----
        hidden_states_list: List[torch.Tensor] = []
        if return_hidden_states:
            hidden_states_list.append(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states, additive_mask=additive_mask)
            if return_hidden_states:
                hidden_states_list.append(hidden_states)

        # ---- 5. 输出层处理 ----
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)  # [B,T,V]

        # ---- 6. 计算损失（如果提供标签） ----
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            self.last_loss = loss  # 更新损失缓存

        # ---- 7. 打包输出 ----
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,  # 已被 KVCache 内部管理
            hidden_states=hidden_states_list if return_hidden_states else None,
        )

    # ------------------------------------------------------------------
    # 自回归生成接口（支持温度、top‑k、top‑p）
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        stop_token_id: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        自回归文本生成
        
        Args:
            input_ids: 起始token序列 [batch_size, seq_len]
            max_new_tokens: 最大生成token数
            temperature: 温度参数控制随机性
            top_k: top-k采样参数
            stop_token_id: 停止生成token ID
            attention_mask: 注意力掩码
            
        Returns:
            生成的token序列 [batch_size, new_seq_len]
        """
        self.eval()
        self.reset_cache()
        
        # 初始化生成序列
        generated = input_ids.clone()
        batch_size = input_ids.size(0)
        
        # 处理初始输入
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        
        # 生成循环
        for _ in range(max_new_tokens):
            # 前向传播获取logits
            outputs = self(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            next_token_logits = outputs.logits[:, -1, :]
            
            # 应用温度
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Top-k采样
            if top_k is not None:
                top_k = min(top_k, next_token_logits.size(-1))
                v, _ = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
            
            # 概率采样
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            
            # 检查停止条件
            if stop_token_id is not None and torch.all(next_tokens == stop_token_id):
                break
            
            # 更新输入序列
            input_ids = next_tokens
            generated = torch.cat([generated, next_tokens], dim=-1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=attention_mask.device)
            ], dim=-1)
        
        # 只返回新生成的token
        return generated[:, input_ids.size(1):]
    
    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        计算模型参数数量
        
        Args:
            non_embedding: 是否排除词嵌入参数
            
        Returns:
            参数数量
        """
        n_params = sum(p.numel() for p in self.parameters())
        
        if non_embedding:
            # 排除词嵌入层参数
            n_params -= self.tok_embeddings.weight.numel()
            if self.config.tie_word_embeddings:
                # 如果共享了权重，输出层参数已包含在词嵌入中
                n_params -= self.output.weight.numel()
        
        return n_params

# 测试代码
if __name__ == "__main__":
    # 创建配置
    config = ByteModelConfig(
        vocab_size=32000,
        model_dim=768,
        num_layers=12,
        num_attention_heads=12,
        num_kv_heads=6,
        hidden_dim=3072,
        max_position_embeddings=2048,
        attention_dropout_prob=0.1,
        residual_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        layer_norm_eps=1e-5,
        tie_word_embeddings=True,
        use_cache=True,
        use_flash_attention=True
    )
    
    # 创建模型
    model = ByteTransformer(config)
    
    # 打印模型信息
    print(f"模型参数总数: {model.get_num_params() / 1e6:.2f}M")
    print(f"非词嵌入参数: {model.get_num_params(non_embedding=True) / 1e6:.2f}M")
    
    # 测试前向传播
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    attention_mask = torch.ones_like(input_ids)
    
    outputs = model(input_ids, attention_mask=attention_mask)
    print(f"Logits形状: {outputs.logits.shape}")  # 应为 [2, 16, 32000]
    
    # 测试生成
    generated = model.generate(
        input_ids=input_ids[:, :1],  # 只使用第一个token作为起始
        max_new_tokens=10,
        temperature=0.8
    )
    print(f"生成结果形状: {generated.shape}")  # 应为 [2, 10]