"""
StellarByte 注意力模块（支持因果注意力、RoPE、KV-cache 与 FlashAttention）。

包含详细中文行注释，接口与原代码保持兼容。
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn.functional as F
from torch import nn

from RoPE import precompute_freqs_cis, apply_rotary_pos_emb, repeat_kv
from config import StellarByteConfig


class MultiHeadAttention(nn.Module):
    """自注意力模块，支持 flash attention、因果 mask、RoPE 以及 KV cache。

    设计目标：在训练和推理阶段都尽可能高效且易于维护。
    """

    def __init__(self, config: StellarByteConfig):
        """初始化 Attention 模块。

        参数：
        - config: StellarByteConfig 实例，包含注意力相关超参
        """
        super().__init__()
        # num_key_value_heads 默认为 num_attention_heads 的值或指定值
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        # 确保 attention heads 能被 kv heads 整除
        assert config.num_attention_heads % self.num_key_value_heads == 0

        # 本地头数量 = query 的头数；n_local_kv_heads = kv 的头数
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = config.num_key_value_heads
        # 重复倍数（用于将 kv 头扩展到 query 头数）
        self.n_rep = self.n_local_heads // self.n_local_kv_heads

        # 每个头的维度
        self.head_dim = config.hidden_size // config.num_attention_heads

        # 使用一次性投影来计算 Q,K,V（避免多个 GEMM 调用）
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # 输出投影：将多头连接后的结果映射回隐藏维
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        # dropout 层
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        # 若 PyTorch 提供了 scaled_dot_product_attention，则可尝试使用 flash attention
        self.flash = hasattr(F, "scaled_dot_product_attention") and config.flash_attn

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """前向计算注意力并可选返回 updated KV cache。

        Args:
            x: 输入张量，形状为 [batch, seq_len, hidden]
            position_embeddings: (cos, sin) 用于 RoPE，形状为 [seq_len, head_dim*2]
            past_key_value: 可选的历史 kv，用于自回归推理
            use_cache: 是否返回新的 kv 缓存
            attention_mask: 可选的注意力 mask，shape 常为 [batch, seq_len]

        Returns:
            output: 注意力输出，形状 [batch, seq_len, hidden]
            past_kv: 若 use_cache=True 返回 (k, v) 的 tuple，否则为 None
        """
        bsz, seq_len, _ = x.shape

        # 投影 Q/K/V
        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)

        # 重新 reshape 成多头格式：Q -> [batch, seq_len, n_heads, head_dim]
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # position_embeddings 由 StellarByteModel 提供为 (cos, sin)
        cos, sin = position_embeddings
        # 仅取序列实际长度的 cos/sin 切片并应用 RoPE
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # 当存在 past kv 时，将其拼接在时间维度（dim=1）上实现缓存
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)

        # 获取当前键值序列的总长度
        kv_seq_len = xk.shape[1]
        past_kv = (xk, xv) if use_cache else None

        # 将各张量转成 [batch, heads, seq_len, head_dim] 以便进行点积
        xq = xq.transpose(1, 2)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        # 若支持 FlashAttention 并且没有 attention mask，则可使用加速内核
        if self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            # F.scaled_dot_product_attention 已接收 is_causal=True
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # 经典实现：scores = Q @ K^T / sqrt(head_dim)
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # 添加上三角因果 mask（上半三角为 -inf），保证自回归因果性
            causal_mask = torch.triu(
                torch.full((seq_len, kv_seq_len), float("-inf"), device=scores.device), 
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)
            scores = scores + causal_mask

            # 若提供 attention_mask，则把其扩展并加入到 scores（将 pad 位置设为 -inf）
            if attention_mask is not None:
                # 扩展 attention_mask 以匹配 KV 序列长度
                if attention_mask.shape[1] < kv_seq_len:
                    # 在左侧填充 1（表示有效token）
                    pad_len = kv_seq_len - attention_mask.shape[1]
                    extended_mask = torch.cat([
                        torch.ones(bsz, pad_len, device=attention_mask.device),
                        attention_mask
                    ], dim=1)
                else:
                    extended_mask = attention_mask[:, :kv_seq_len]

                extended_attention_mask = extended_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            # softmax 得到注意力分布并乘以 V
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        # 恢复形状并做输出投影
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

# 测试 Attention 模块  
if __name__ == "__main__":
    print("=== 测试 Attention 模块 ===")
    
    # 创建模拟配置
    class TestConfig:
        hidden_size = 256
        num_attention_heads = 8
        num_key_value_heads = 4
        dropout = 0.1
        flash_attn = False
    
    config = TestConfig()
    
    # 初始化注意力模块
    attention = MultiHeadAttention(config)
    
    # 创建输入数据
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # 预计算位置编码
    cos, sin = precompute_freqs_cis(config.hidden_size // config.num_attention_heads, end=seq_len)
    
    # 前向传播
    output, past_kv = attention(x, (cos, sin))
    print(f"输入形状: {x.shape}, 输出形状: {output.shape}")
    
    # 测试 KV cache
    output_with_cache, new_past_kv = attention(x, (cos, sin), past_key_value=past_kv, use_cache=True)
    print(f"使用 KV cache 后输出形状: {output_with_cache.shape}")
    print(f"新的 KV cache 形状: k={new_past_kv[0].shape}, v={new_past_kv[1].shape}")
    
    print("Attention 模块测试通过!\n")
