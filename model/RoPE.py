from typing import Optional, Tuple
import math
import torch

def precompute_freqs_cis(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """预计算 RoPE 的 cos 和 sin 矩阵，用于后续快速应用。

    Args:
        dim: 每个头的维度（head_dim）或旋转维度的两倍。
        end: 预计算的最大序列长度（通常为 max_position_embeddings）。
        rope_base: RoPE 基数（论文常用 1e6）。
        rope_scaling: 可选的 RoPE 外推缩放字典（用于 YaRN）

    Returns:
        freqs_cos, freqs_sin：两个形状为 [end, dim] 的张量，分别为 cos 和 sin。
    """
    # 计算频率向量：1 / (rope_base ** (i / dim))，i 按偶数索引
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2).float() / dim))

    # 如果启用外推缩放，则对频率进行调整（YaRN 风格）
    if rope_scaling is not None:
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 4)
        beta_fast = rope_scaling.get("beta_fast", 4.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        # 若请求的序列长度超过训练时原始最大长度
        # 对频率进行按位缩放以获得更好的外推能力
        # 计算 corr_dim：小于等于 dim//2 的第一个索引，使得 2π/freqs[i] > orig_max
        if end / orig_max > 1.0:
            corr_dim = next((i for i in range(dim // 2) if 2 * math.pi / freqs[i] > orig_max), dim // 2)
            # 通过线性插值计算 beta
            power = torch.arange(0, dim // 2).float() / max(dim // 2 - 1, 1)
            beta = beta_slow + (beta_fast - beta_slow) * power
            # YaRN 标准公式：λ = (β·α - β + 1)/(β·α)
            scale = torch.where(
                torch.arange(dim // 2) < corr_dim,
                (beta * factor - beta + 1) / (beta * factor),
                1.0 / factor,
            )
            freqs = freqs * scale

    # t 为 0..end-1 的位置索引向量
    t = torch.arange(end)
    # 将位置索引与频率外积以得到角度矩阵
    freqs = torch.outer(t, freqs).float()
    # 使用 concat 将 cos 和 sin 扩展到完整维度（dim 的两倍）
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """对 q/k 应用旋转位置编码（RoPE）。

    Args:
        q, k: 张量形状通常为 [batch, seq_len, heads, head_dim]
        cos, sin: 位置相关的 cos/sin，形状为 [seq_len, head_dim*2]
        position_ids: 保留接口（当前实现使用连续位置）
        unsqueeze_dim: 在哪个维度插入位置维度（默认为 1，适配 [batch, seq, ...]）

    返回:
        q_embed, k_embed：应用 RoPE 后的 q, k
    """

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """将最后一维一分为二并做交换与取负（RoPE 的标准操作）。"""
        # 假设最后一维为 head_dim，先取后一半取负并拼回
        return torch.cat((-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1)

    # 按位应用 RoPE：q * cos + rotate_half(q) * sin
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """将 key/value 的头从 num_kv_heads 重复以匹配 query 的头数量。

    例如：若 num_attention_heads=8, num_key_value_heads=2，则 n_rep=4，
    表示对 kv 头在原来基础上扩展 4 倍以便点乘。
    """
    # x 的形状通常为 [batch, seq_len, num_key_value_heads, head_dim]
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # 先在中间插入一维再 expand，然后 reshape 回去以得到新的头数
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )

# 测试 RoPE 模块
if __name__ == "__main__":
    print("=== 测试 RoPE 模块 ===")
    
    # 模拟配置参数
    dim = 64  # 头维度
    seq_len = 10  # 序列长度
    batch_size = 2
    num_heads = 4
    
    # 预计算 RoPE
    cos, sin = precompute_freqs_cis(dim, end=seq_len)
    print(f"cos shape: {cos.shape}, sin shape: {sin.shape}")
    
    # 创建模拟的 Q, K 张量
    q = torch.randn(batch_size, seq_len, num_heads, dim)
    k = torch.randn(batch_size, seq_len, num_heads, dim)
    
    # 应用 RoPE
    q_rotated, k_rotated = apply_rotary_pos_emb(q, k, cos, sin)
    print(f"旋转前 Q shape: {q.shape}, 旋转后 Q shape: {q_rotated.shape}")
    
    # 测试 repeat_kv
    kv_heads = 2
    n_rep = num_heads // kv_heads
    x_kv = torch.randn(batch_size, seq_len, kv_heads, dim)
    x_repeated = repeat_kv(x_kv, n_rep)
    print(f"KV 重复前 shape: {x_kv.shape}, 重复后 shape: {x_repeated.shape}")
    
    print("RoPE 模块测试通过!\n")