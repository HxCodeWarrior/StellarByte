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
    from .MoE              import ByteMoE
except:
    from config           import ByteModelConfig
    from utils.KVCache    import ByteKVCache
    from utils.DropPath   import DropPath
    from RMSNorm          import ByteRMSNorm
    from Attention        import ByteMultiHeadSelfAttention
    from MLP              import ByteMLP
    from MoE              import ByteMoE


class ByteDecoderLayer(nn.Module):
    """Transformer 解码器层，包含自注意力机制和前馈网络。
    
    实现特性：
    - DeepNorm：残差连接缩放 (1/√(2L))
    - LayerScale：残差分支的可学习缩放因子
    - DropPath：训练时的随机残差路径丢弃
    - 并行残差模式（可选）：同时计算注意力和MLP输出
    
    参数初始化：
    - 采用 DeepNet 风格的双层初始化策略
    
    参数:
        args (ByteModelConfig): 模型配置对象
        layer_id (int, optional): 当前层在模型中的索引位置. 默认为 0.
    """
    def __init__(
        self,
        args: ByteModelConfig,
        layer_id: int = 0,    # 必传：当前层编号
    ):
        super().__init__()
        self.args     = args
        self.layer_id = layer_id
        D             = args.model_dim

        # ---------------- Norm ----------------
        # 注意力前的RMSNorm
        self.norm_attn = ByteRMSNorm(D, eps=args.layer_norm_eps)
        # 前馈网络前的RMSNorm
        self.norm_ffn  = ByteRMSNorm(D, eps=args.layer_norm_eps)

        # ---------------- Blocks --------------
        # 多头自注意力机制
        self.self_attn = ByteMultiHeadSelfAttention(
            args,
            layer_id   = layer_id,
            num_layers = args.num_layers
        )

        # 前馈神经网络(二选一)
        # 混合专家系统（Mixture of Experts）
        self.moe = ByteMoE(
            dim = D,
            k   = args.moe_k,
            hidden_dim      = args.hidden_dim,
            num_experts     = args.moe_num_experts,
            capacity_factor = args.moe_capacity_factor,
            dropout         = args.hidden_dropout_prob,
            world_size      = args.world_size,
            rank            = args.rank,
            activation      = "gelu"
        )
        # 多层感知机（MLP）
        self.mlp = ByteMLP(
            dim=D,
            hidden_dim  = args.hidden_dim,
            multiple_of = args.dim_multiplier,
            dropout     = args.hidden_dropout_prob
        )

        # ---------------- Residual Tricks -----
        L = args.num_layers # 模型总层数
        
        # DeepNorm 缩放因子
        self.resid_scale = (
            1.0 / math.sqrt(2 * L)
            if getattr(args, "use_deepnorm", True) else 1.0
        )

        # LayerScale γ   — 两条残差各自独立
        ls_init = getattr(args, "layerscale_init", 1e-4)
        # 注意力输出的可学习缩放因子 (维度为模型隐藏层大小)
        self.ls_attn = nn.Parameter(torch.ones(D) * ls_init)
        # FFN输出的可学习缩放因子
        self.ls_ffn  = nn.Parameter(torch.ones(D) * ls_init)

        # DropPath 概率：随深度线性增长
        dp_max         = getattr(args, "drop_path_prob", 0.0) # 最大丢弃概率
        # 当前层的丢弃概率：随深度线性增加 (第0层=0, 最后一层=dp_max)
        drop_p         = dp_max * layer_id / max(1, L - 1)
        self.drop_path = DropPath(drop_p)

        # 是否启用并行分支
        self.parallel_residual = getattr(args, "parallel_residual", False)

    # --------------------------------------------------------------------- #
    def forward(
        self, 
        x: torch.Tensor,
        kv_cache: Optional[ByteKVCache] = None
    ) -> torch.Tensor:
        """前向传播逻辑。
        
        参数:
            x (torch.Tensor): 输入张量 [batch_size, seq_len, hidden_dim]
            kv_cache (ByteKVCache, optional): 用于存储KV值的缓存对象（推理加速）
            
        返回:
            torch.Tensor: 输出张量 [batch_size, seq_len, hidden_dim]
        """
        if self.parallel_residual:
            # ============= 并行残差模式 (PaLM/GPT-NeoX) =============
            # 1. 并行计算两个归一化结果
            attn_norm_x    = self.norm_attn(x)  # 注意力输入归一化
            ffn_norm_x     = self.norm_ffn(x)   # MLP输入归一化
            
            # 2. 并行计算注意力和FFN输出
            attn_out, meta = self.self_attn(attn_norm_x, kv_cache)
            if self.args.use_moe:
                ffn_out, aux_loss = self.moe(ffn_norm_x)  # 接收专家输出+辅助损失
            else:
                ffn_out  = self.mlp(ffn_norm_x)  # 标准MLP输出
                aux_loss = None

            # 3. 融合并缩放残差连接
            residual = self.drop_path(
                self.ls_attn * attn_out * self.resid_scale
                + self.ls_ffn  * ffn_out  * self.resid_scale
            )
            out = x + self.drop_path(residual)  # 应用DropPath

            return out, aux_loss
        else:
            # ============= 顺序残差模式 (原始Transformer) =============
            # 1: 自注意力 + 残差连接
            attn_input = self.norm_attn(x)  # 输入归一化
            attn_out, meta = self.self_attn(attn_input, kv_cache)
            
            # 残差连接（应用LayerScale和DeepNorm缩放）
            attn_residual = self.ls_attn * attn_out * self.resid_scale
            x = x + self.drop_path(attn_residual)  # 添加DropPath

            # 2: 前馈网络 + 残差连接
            ffn_input = self.norm_ffn(x)  # 输入归一化
            if self.args.use_moe:
                ffn_out, aux_loss = self.moe(ffn_input)  # 接收专家输出+辅助损失
            else:
                ffn_out  = self.mlp(ffn_input)
                aux_loss = None
            
            # 残差连接（应用LayerScale和DeepNorm缩放）
            ffn_residual = self.ls_ffn * ffn_out * self.resid_scale
            x = x + self.drop_path(ffn_residual)

            return x, aux_loss


if __name__ == "__main__":
    cfg = ByteModelConfig(
        model_dim=128,
        num_layers=24,
        num_attention_heads=8,
        layerscale_init=1e-4,
        drop_path_prob=0.2,
        parallel_residual=True,
        use_flash_attention=False,      # CPU quick test
        use_moe=True,
        moe_num_experts=1,        
        moe_k= 1,
        moe_capacity_factor=1.25,
        moe_loss_coefficient=0.01,
        moe_world_size=1,         
        moe_rank=0,               
    )
    layer = ByteDecoderLayer(cfg, layer_id=12)
    x = torch.randn(2, 16, cfg.model_dim)
    y, aux_loss = layer(x)          # 训练阶段不传 mask
    print(f"Input Shape: {x.shape}\nOutput Shape: {y.shape}\nAux Loss: {aux_loss}")   # torch.Size([2, 16, 128])
