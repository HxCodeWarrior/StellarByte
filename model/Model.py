"""Byte‑Transformer 模型
=================================================
本文件实现了一个基于 **XPos** 旋转位置编码的 GPT‑类语言模型，
包含 DeepNorm、LayerScale、DropPath、KV‑Cache 等现代化 Tricks，
并与 HuggingFace 生态完全兼容。

作者：ByteWyrm  |  日期：2025‑07‑14
"""

import math
from typing import Optional, List, Tuple

import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

# ==== 本项目自定义模块 ====
try:
    from .config        import ByteModelConfig      # 模型配置类（超参数集中管理）
    from .DecoderLayer  import ByteDecoderLayer     # 单层解码器实现（Attention + MLP）
    from .utils.KVCache import KVCache              # 高性能 KV 缓存（推理加速）
    from .RMSNorm       import RMSNorm              # RMSNorm 归一化层
except:
    from config         import ByteModelConfig       
    from DecoderLayer   import ByteDecoderLayer
    from utils.KVCache  import KVCache        
    from RMSNorm        import RMSNorm              

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
                num_layers           = config.num_layers,
                num_heads            = config.num_attention_heads,  # 总头数
                head_dim             = D // config.num_attention_heads,
                max_seq_len          = config.max_seq_len,
                key_dtype            = config.key_cache_dtype,
                value_dtype          = config.value_cache_dtype,
                tensor_parallel_size = config.tensor_parallel_size,
                tensor_parallel_rank = config.tensor_parallel_rank
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

        # 记录一次权重 dtype，用于推理时安全 cast
        self._infer_dtype: torch.dtype = next(self.parameters()).dtype
        # 缓存首帧掩码 (batch_shape, additive_mask)
        self._cached_mask: Tuple[Tuple[int, int], torch.Tensor] = ((0, 0), torch.empty(0))

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
        """DeepNorm + µParam Init"""
        μ = 1.0 / math.sqrt(2 * self.config.num_layers)
        for name, p in self.named_parameters():
            if any(name.endswith(s) for s in (".w3.weight", ".W_o.weight", ".lm_head.weight")):
                nn.init.normal_(p, mean=0.0, std=μ * self.config.initializer_range)

    # ------------------------------------------------------------------
    # 工具方法：清空 KV 缓存
    # ------------------------------------------------------------------
    def reset_cache(self):
        if self.kv_cache is not None:
            self.kv_cache.reset()
    
    def gradient_checkpointing_enable(self):
        """为所有 DecoderLayer 打开 Torch Utils Checkpoint，加速训练"""
        for layer in self.layers:
            layer.gradient_checkpointing_enable = True
        return self

    def compile(self, mode: str = "default"):
        """torch.compile wrapper"""
        if not hasattr(torch, "compile"):
            warnings.warn("当前 PyTorch 版本不支持 torch.compile，已跳过。")
            return self
        return torch.compile(self, mode=mode, fullgraph=False)


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
        use_cache = self.config.use_cache if use_cache is None else use_cache

        # ---- 1. 输入检查 ----
        if input_ids is None and inputs_embeds is None:
            raise ValueError("input_ids 与 inputs_embeds 必须至少提供一个！")

        # ---- 2. 词嵌入 ----
        hidden_states = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds   # [B,T,D]
        hidden_states = self.embed_dropout(hidden_states)

        # ---- 3. 构造 Padding mask ----
        base_mask = None
        if attention_mask is not None:
            base_mask = attention_mask.to(hidden_states.dtype)

        # ---- 4. 底层前向 (KVCache-aware) ----
        B = hidden_states.size(0)
        hidden_states_list: List[torch.Tensor] = []
        if return_hidden_states:
            hidden_states_list.append(hidden_states)

        for i, layer in enumerate(self.layers):
            # 4.1 缓存长度
            past_len = self.kv_cache.layer_length(i) if self.kv_cache is not None else 0

            # 4.2 构造 layer-specific mask
            additive_mask = None
            if base_mask is not None:
                if past_len > 0:
                    ones = torch.ones(B, past_len, dtype=base_mask.dtype, device=base_mask.device)
                    mask = torch.cat([ones, base_mask], dim=1).to(hidden_states.dtype).to(hidden_states.device)
                else:
                    mask = base_mask
                additive_mask = (1.0 - mask) * -1e4
                additive_mask = additive_mask[:, None, None, :]

            if getattr(layer, "gradient_checkpointing_enable", False) and self.training:
                def custom_forward(*inputs):
                    return layer(*inputs)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    custom_forward, 
                    hidden_states, 
                    additive_mask,
                    use_reentrant=False # 采用调度器方式，稳定性更高，在实际训练中更可控。
                )
            else:
                hidden_states = layer(hidden_states, additive_mask)

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
        top_p: Optional[float] = None,
        stop_token_id: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        自回归文本生成
        
        Args:
            input_ids: 起始token序列 [batch_size, seq_len]
            max_new_tokens: 最大生成token数
            temperature: 温度参数控制随机性
            top_k: top-k采样参数
            top_p: top-p采样参数
            stop_token_id: 停止生成token ID
            attention_mask: 注意力掩码
            use_caceh: 是否使用KVCache
            
        Returns:
            生成的token序列 [batch_size, new_seq_len]
        """
        self.eval()
        self.reset_cache()
        use_kv = self.kv_cache is not None and self.config.use_cache
        
        # 初始化生成序列
        prefix_len = input_ids.size(1)
        generated = input_ids.clone()
        batch_size = input_ids.size(0)
        
        # 处理初始输入
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        # -------------------------------------------------------------
        # 1. 预热阶段：一次性把前缀送进去，填满 KV‑Cache
        # -------------------------------------------------------------
        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_kv,
        )
        next_token_logits = outputs.logits[:, -1, :]   # 取前缀最后一个 token 的 logits

        generated = input_ids.clone()                  # 记录完整序列

        # -------------------------------------------------------------
        # 2. 进入增量采样‑生成循环
        # -------------------------------------------------------------
        for _ in range(max_new_tokens):  
            # — 2‑1 采样 —
            # 应用温度
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Top-k采样
            if top_k is not None and top_k > 0:
                top_k = min(top_k, next_token_logits.size(-1))
                v, _ = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
            
            if top_p is not None and 0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                # 保证至少有一个 token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits = torch.where(indices_to_remove, torch.full_like(next_token_logits, -float("inf")), next_token_logits)
            
            # 概率采样
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            
            # — 2‑2 停止条件 —
            if stop_token_id is not None and torch.all(next_tokens == stop_token_id):
                break
            
            # — 2‑3 写入输出 & KV‑Cache 已自动扩充 —
            generated = torch.cat([generated, next_tokens], dim=-1)

            # — 2‑4 前向增量：**只喂新 token，不传 attention_mask** —
            outputs = self(
                input_ids=next_tokens,           # [B, 1]
                attention_mask=None,             # 让 forward 自行推断 past_len
                use_cache=use_kv,
            )
            next_token_logits = outputs.logits[:, -1, :]

        # 只返回新生成的token
        return generated[:, prefix_len:]
    
    # ------------------------------------------------------------------
    # 单步推理接口 (增量解码原子 API)
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def inference_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        reset_cache: bool = False,
        return_logits: bool = True,
        kv_clip: Optional[int] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        单步推理 (Single‑step Inference)

        参数
        ----
        input_ids   : LongTensor, 形状 [B,T] / [B] / [B,1]
                      * 首帧 (reset_cache=True 或 KVCache 为空) 允许任意 T≤max_seq_len
                      * 后续步骤通常传入 1 个新 token
        attention_mask : (可选) 仅首帧需要；后续自动推断
        reset_cache : 若 True 则先清空 KV‑Cache，从头推理
        return_logits : 是否返回最新 logits（常用于采样）
        kv_clip : 限制 KV‑Cache 的最大保留长度；None 表示不截断

        返回
        ----
        hidden_last : None (本实现不返回隐状态，可自行扩展)
        logits_last : FloatTensor | None, 形状 [B,V]
        """

        # ===== 0. 前置处理 =====
        if reset_cache:
            self.reset_cache()                       # 开启新一轮推理

        # 保证 input_ids 形状为 [B,T]
        if input_ids.ndim == 1:
            input_ids = input_ids[:, None]           # [B] -> [B,1]
        elif input_ids.ndim != 2:
            raise ValueError("input_ids 应为 [B] / [B,1] / [B,T]")

        B, T = input_ids.shape
        is_first_frame = (self.kv_cache is None) or (self.kv_cache.global_length() == 0)

        # ===== 1. dtype 强一致 =====
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids 必须是 torch.long 类型")
        # 统计参数 dtype（首次在 __init__ 已记录）
        param_dtype = self._infer_dtype
        if self.embed_tokens.weight.dtype != param_dtype:
            # 若外部调用了 model.to(...) 改变 dtype，则更新记录
            param_dtype = self.embed_tokens.weight.dtype
            self._infer_dtype = param_dtype

        # ===== 2. 掩码构造 (仅首帧) =====
        additive_mask = None
        if is_first_frame:
            # 若用户没给 attention_mask，默认全 1（无 PAD）
            if attention_mask is None:
                attention_mask = torch.ones(B, T, dtype=torch.long, device=input_ids.device)

            # 查看是否已缓存同形状的掩码模板
            if self._cached_mask[0] != (B, T):
                # 计算 additive mask 并缓存
                m = attention_mask.to(param_dtype)        # [B,T]
                causal_inf = torch.finfo(param_dtype).min
                additive_mask_tmpl = (1.0 - m)[:, None, None, :] * causal_inf  # [B,1,1,T]
                self._cached_mask = ((B, T), additive_mask_tmpl)

            additive_mask = self._cached_mask[1]
        # 单 token 步骤跳过掩码构造 (Attention 内部会处理 causal)

        # ===== 3. 调用 forward =====
        output: CausalLMOutputWithPast = self(
            input_ids=input_ids,
            attention_mask=attention_mask if is_first_frame else None,
            use_cache=True,                # 始终使用 KV‑Cache
            return_hidden_states=False,    # 推理不需要逐层 hidden
        )

        logits_last = output.logits[:, -1, :] if return_logits else None

        # ===== 4. KV‑Cache 长度控制 =====
        if kv_clip is not None and self.kv_cache is not None:
            # 若 KVCache 对象实现 clip() & total_len
            if hasattr(self.kv_cache, "global_length") and self.kv_cache.global_length() > kv_clip:
                if hasattr(self.kv_cache, "clip"):
                    self.kv_cache.clip(kv_clip)
                else:
                    warnings.warn("KVCache 未实现 clip()，已跳过 kv_clip")

        # 返回 (hidden_last, logits_last)
        return None, logits_last

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
            n_params -= self.embed_tokens.weight.numel()
            if self.config.tie_word_embeddings:
                # 如果共享了权重，输出层参数已包含在词嵌入中
                n_params -= self.lm_head.weight.numel()
        
        return n_params
    
    def get_optimizer_param_groups(self, weight_decay: float = 0.01):
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() == 1 or n.endswith("bias"):
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]


# 测试代码
if __name__ == "__main__":
    # 创建配置
    cfg = ByteModelConfig(
        vocab_size=32000,
        model_dim=768,
        num_layers=12,
        num_attention_heads=12,
        num_kv_heads=6,
        hidden_dim=3072,
        max_seq_len=2048,
        attention_dropout_prob=0.1,
        residual_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        layer_norm_eps=1e-5,
        tie_word_embeddings=True,
        use_cache=True,
        use_flash_attention=True,
    )

    # 创建模型
    model = ByteTransformer(cfg)
    print(f"模型参数总数: {model.get_num_params()/1e6:.2f}M")
    print(f"非词嵌入参数: {model.get_num_params(non_embedding=True)/1e6:.2f}M")

    # ----- 测试 forward -----
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    attention_mask = torch.ones_like(input_ids)

    outputs = model(input_ids, attention_mask=attention_mask)
    print(f"Logits 形状: {outputs.logits.shape}")  # [2,16,32000]

    # ----- 测试 step() -----
    # 首帧
    _, logits = model.inference_step(input_ids, attention_mask=attention_mask, reset_cache=True)
    print(f"step() 首帧 logits: {logits.shape}")  # [2,32000]
    # 单 token
    next_token = torch.tensor([[1],[2]])
    _, logits = model.inference_step(next_token)
    print(f"step() 单 token logits: {logits.shape}")  # [2,32000]

    # ----- 测试 generate -----
    gen = model.generate(input_ids[:, :1], max_new_tokens=10, temperature=0.8)
    print(f"生成序列形状: {gen.shape}")  # [2,10]