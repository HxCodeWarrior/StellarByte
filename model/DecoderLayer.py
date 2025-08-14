import math
import torch
import torch.nn as nn
from typing import Optional

try:
    from .config           import ByteModelConfig
    from .utils.KVCache    import ByteKVCache
    from .utils.DropPath   import DropPath
    from .RMSNorm          import ByteRMSNorm
    from .Attention        import ByteMultiHeadSelfAttention
    from .MLP              import ByteMLP
except:
    from config           import ByteModelConfig
    from utils.KVCache    import ByteKVCache
    from utils.DropPath   import DropPath
    from RMSNorm          import ByteRMSNorm
    from Attention        import ByteMultiHeadSelfAttention
    from MLP              import ByteMLP


class ByteDecoderLayer(nn.Module):
    """
    Decoder Layer
    -------
    1. **DeepNorm**：残差缩放  1/√(2L)  (预设：开启)
    2. **LayerScale**：对残差分支引入可学习 γ (默认 1e‑4)
    3. **DropPath**：训练期随机丢弃残差 (默认随层深度线性增长)
    4. **并行模式**（可选）：Attention 与 MLP 并行计算，减少两次读写
    5. 参数双层初始化：DeepNet‑style + Small Init
    """
    def __init__(
        self,
        args: ByteModelConfig,
        layer_id: int = 0,      # 必传：当前层编号
    ):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        D = args.model_dim

        # ---------------- Norm ----------------
        self.norm_attn = ByteRMSNorm(D, eps=args.layer_norm_eps)
        self.norm_ffn  = ByteRMSNorm(D, eps=args.layer_norm_eps)

        # ---------------- Blocks --------------
        self.self_attn = ByteMultiHeadSelfAttention(
            args,
            layer_id=layer_id,
            num_layers=args.num_layers
        )
        self.mlp = ByteMLP(
            dim=D,
            hidden_dim=args.hidden_dim,
            multiple_of=args.dim_multiplier,
            dropout=args.hidden_dropout_prob
        )

        # ---------------- Residual Tricks -----
        L = args.num_layers
        self.resid_scale = (
            1.0 / math.sqrt(2 * L)
            if getattr(args, "use_deepnorm", True) else 1.0
        )
        # LayerScale γ   — 两条残差各自独立
        ls_init = getattr(args, "layerscale_init", 1e-4)
        self.ls_attn = nn.Parameter(torch.ones(D) * ls_init)
        self.ls_mlp  = nn.Parameter(torch.ones(D) * ls_init)

        # DropPath 概率：随深度线性增长
        dp_max = getattr(args, "drop_path_prob", 0.0)
        drop_p = dp_max * layer_id / max(1, L - 1)
        self.drop_path = DropPath(drop_p)

        # 是否启用并行分支
        self.parallel_residual = getattr(args, "parallel_residual", False)

    # --------------------------------------------------------------------- #
    def forward(
        self, 
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[ByteKVCache] = None
    ) -> torch.Tensor:
        """
        x: [B, T, D]
        attention_mask: 与 MultiHeadSelfAttention 相同
        """
        if self.parallel_residual:
            # -------- Parallel Residual (PaLM / GPT‑NeoX 风格) ----------
            attn_norm_x    = self.norm_attn(x)
            ffn_norm_x     = self.norm_ffn(x)
            attn_out, meta = self.self_attn(attn_norm_x, attention_mask, kv_cache)
            ffn_out        = self.mlp(ffn_norm_x)
            out = x + self.drop_path(
                self.ls_attn * attn_out * self.resid_scale
                + self.ls_mlp  * ffn_out  * self.resid_scale
            )
            return out
        else:
            # ------------------- 顺序 Residual -------------------------
            # 1) Self‑Attention
            attn_out, meta = self.self_attn(self.norm_attn(x), attention_mask, kv_cache)
            x = x + self.drop_path(self.ls_attn * attn_out * self.resid_scale)

            # 2) Feed‑Forward
            ffn_out = self.mlp(self.norm_ffn(x))
            x = x + self.drop_path(self.ls_mlp * ffn_out * self.resid_scale)
            return x

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    cfg = ByteModelConfig(
        model_dim=128,
        num_layers=24,
        num_attention_heads=8,
        layerscale_init=1e-4,
        drop_path_prob=0.2,
        parallel_residual=True,
        use_flash_attention=False      # CPU quick test
    )
    layer = ByteDecoderLayer(cfg, layer_id=12)
    x = torch.randn(2, 16, cfg.model_dim)
    y = layer(x)          # 训练阶段不传 mask
    print(f"Input Shape: {x.shape}\nOutput Shape: {y.shape}")   # torch.Size([2, 16, 512])
