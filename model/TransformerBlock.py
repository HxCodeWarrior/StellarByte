# 导入必要的PyTorch模块和类型注解支持
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict  # 用于类型注解

# 尝试从本地模块导入相关组件
try:
    from .config import StellarByteModelArgs  # 模型配置类
    from .RMSNorm import StellarByteRMSNorm  # RMS归一化层
    from .PositionEmbedding import StellarByteRoPE  # 位置编码
    from .Attention import StellarByteAttention  # 注意力机制
    from .FeedForward import StellarByteFeedForward  # 普通前馈网络
    from .MoE import StellarByteMOEFeedForward  # 混合专家前馈网络
except:
    # 如果相对导入失败，尝试绝对导入（用于独立运行或测试）
    from config import StellarByteModelArgs
    from RMSNorm import StellarByteRMSNorm
    from PositionEmbedding import StellarByteRoPE
    from Attention import StellarByteAttention
    from FeedForward import StellarByteFeedForward
    from MoE import StellarByteMOEFeedForward


class StellarByteBlock(nn.Module):
    """StellarByte模型的Transformer块实现。
    
    该类实现了Transformer架构中的一个完整块，包含自注意力机制和前馈网络。
    支持普通前馈网络和混合专家(MoE)前馈网络的配置。
    
    Attributes:
        num_heads (int): 注意力头的数量
        dim (int): 模型维度
        head_dim (int): 每个注意力头的维度
        attention (StellarByteAttention): 自注意力机制实例
        enabled_moe (bool): 是否启用混合专家模式
        feed_forward (Union[StellarByteFeedForward, StellarByteMOEFeedForward]): 前馈网络
        layer_id (int): 当前层的ID标识
        attention_norm (StellarByteRMSNorm): 注意力前的归一化层
        ffn_norm (StellarByteRMSNorm): 前馈前的归一化层
        moe_aux_loss (Optional[torch.Tensor]): MoE辅助损失
        moe_expert_usage (Optional[Dict[int, int]]): MoE专家使用统计
    """
    
    def __init__(self, layer_id: int, args: StellarByteModelArgs):
        """初始化StellarByteBlock模块。
        
        Args:
            layer_id: 当前层的ID，用于标识不同层
            args: 包含模型配置参数的StellarByteModelArgs对象
        """
        super().__init__()
        # 保存注意力头数量
        self.num_heads = args.num_heads
        # 保存模型维度
        self.dim = args.dim
        # 计算每个注意力头的维度
        self.head_dim = args.dim // args.num_heads
        # 初始化自注意力机制
        self.attention = StellarByteAttention(args)
        # 记录是否启用混合专家模式
        self.enabled_moe = args.enabled_moe

        # 初始化前馈网络
        # 根据配置选择使用MoE前馈网络或普通MLP
        if args.enabled_moe:
            self.feed_forward = StellarByteMOEFeedForward(args)
        else:
            self.feed_forward = StellarByteFeedForward(
                dim=args.dim,  # 输入维度
                hidden_dim=4 * args.dim,  # 隐藏层维度（通常为4倍输入维度）
                multiple_of=args.multiple_of,  # 确保维度是该值的倍数
                ffn_dim_multiplier=args.ffn_dim_multiplier,  # 前馈网络维度乘数
            )
        
        # 保存层ID
        self.layer_id = layer_id
        # 初始化注意力前的RMS归一化层
        self.attention_norm = StellarByteRMSNorm(args.dim, eps=args.norm_eps)
        # 初始化前馈前的RMS归一化层
        self.ffn_norm = StellarByteRMSNorm(args.dim, eps=args.norm_eps)

        # 初始化MoE相关的统计信息存储变量
        self.moe_aux_loss = None
        self.moe_expert_usage = None

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """前向传播函数。
        
        实现Transformer块的计算流程：归一化 -> 自注意力 -> 残差连接 -> 
        归一化 -> 前馈网络 -> 残差连接。
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, dim)
            freqs_cis: 预先计算的频率张量，用于旋转位置编码
            freqs_sin: 预先计算的频率张量，用于旋转位置编码
            past_key_value: 可选的缓存键值对，形状为 (batch_size, seq_len, dim)
            
        Returns:
            输出张量，形状与输入相同 (batch_size, seq_len, dim)
            更新后的键值缓存元组（如果使用缓存）
        """
        # 保存残差连接
        residual = x

        # 注意力层计算 (带残差连接)
        # 1. 对输入进行归一化
        attn_norm_output = self.attention_norm(x)
        # 2. 应用自注意力机制
        attn_output, present_key_value = self.attention(
            attn_norm_output, freqs_cos, freqs_sin, past_key_value
        )
        # 3. 与原始输入相加 (残差连接)
        hidden_states = residual + attn_output
        
        # 前馈层计算 (可能是MoE或普通MLP)
        # 1. 对注意力输出进行归一化
        ffn_input = self.ffn_norm(hidden_states)
        # 2. 应用前馈网络
        ffn_output = self.feed_forward(ffn_input)

        # 如果是MoE模式，保存辅助损失和专家使用情况
        if self.enabled_moe:
            # 获取MoE辅助损失（如果存在）
            self.moe_aux_loss = self.feed_forward.aux_loss
            # 获取专家使用统计
            self.moe_expert_usage = self.feed_forward.get_expert_usage()
        
        # 应用残差连接并返回结果
        out = hidden_states + ffn_output

        return out, present_key_value


if __name__ == "__main__":
    print("="*50)
    print("StellarByteBlock 形状验证测试")
    print("="*50)
    
    # 1. 创建模型配置
    class StellarByteModelArgs:
        vocab_size = 32000
        dim = 512
        num_heads = 8
        num_kv_heads = 4
        max_batch_size = 2
        max_seq_len = 1024
        rope_theta = 10000.0
        enabled_flash_attn = False
        enabled_kv_cache = True
        attention_dropout = 0.1
        resid_dropout = 0.1
        model_parallel_size = 1
        ffn_dropout = 0.1
        rms_norm_eps = 1e-6
        ffn_dim_multiplier = 1
        norm_eps = 1e-6
        multiple_of = 256
        enabled_moe = False
        gating_dim = 512
        num_experts = 4
        num_experts_per_tok = 2
        num_routed_experts = 4
        num_shared_experts = 1
        scoring_func = 'softmax'
        aux_loss_alpha = 0.1
        seq_aux = True
        norm_topk_prob = True

    args = StellarByteModelArgs()
    
    # 2. 创建Transformer块实例
    block = StellarByteBlock(layer_id=0, args=args)
    
    # 3. 创建模拟输入
    batch_size = 2
    seq_len = 64
    x = torch.randn(batch_size, seq_len, args.dim)
    past_key_value = None  # 缓存位置
    
    # 创建正确的频率张量
    head_dim = args.dim // args.num_heads
    rope = StellarByteRoPE(dim=head_dim, max_seq_len=args.max_seq_len, theta=args.rope_theta)
    freqs_cos, freqs_sin = rope.precompute_freqs_cis(head_dim, args.max_seq_len, args.rope_theta)
    
    # 调整频率张量形状以匹配注意力头的数量
    freqs_cos = freqs_cos[:seq_len]  # 截取到当前序列长度
    freqs_sin = freqs_sin[:seq_len]  # 截取到当前序列长度
    
    print(f"输入形状: {x.shape}")
    print(f"freqs_cos形状: {freqs_cos.shape}")
    print(f"freqs_sin形状: {freqs_sin.shape}")
    
    # 4. 前向传播
    output, present_key_value = block(x, freqs_cos, freqs_sin, past_key_value=past_key_value)
    print(f"输出形状: {output.shape}")
    print(f"present_key_value形状: {present_key_value[0].shape if present_key_value else 'None'}")
    
    # 5. 形状验证
    assert output.shape == x.shape, \
        f"形状验证失败! 预期: {x.shape}, 实际: {output.shape}"
    
    # 6. MoE功能验证（如果启用）
    if args.enabled_moe:
        print("\nMoE功能验证:")
        aux_loss = block.moe_aux_loss
        expert_usage = block.moe_expert_usage
        
        print(f"辅助损失: {aux_loss is not None}")
        print(f"专家使用统计: {expert_usage}")
        
        assert aux_loss is not None, "MoE辅助损失未正确设置"
        assert expert_usage is not None and len(expert_usage) > 0, "专家使用统计未正确设置"
    else:
        print("\n普通前馈网络模式")
    
    # 7. 测试启用MoE的情况
    print("\n" + "="*50)
    print("测试启用MoE的情况")
    print("="*50)
    
    args.enabled_moe = True
    moe_block = StellarByteBlock(layer_id=1, args=args)
    
    moe_output, moe_present_key_value = moe_block(x, freqs_cos, freqs_sin, past_key_value=past_key_value)
    print(f"MoE输出形状: {moe_output.shape}")
    
    # 验证MoE输出形状
    assert moe_output.shape == x.shape, \
        f"MoE形状验证失败! 预期: {x.shape}, 实际: {moe_output.shape}"
    
    # 验证MoE相关属性
    print(f"MoE辅助损失: {moe_block.moe_aux_loss is not None}")
    print(f"MoE专家使用统计: {moe_block.moe_expert_usage}")
    
    print("\n✅ 所有测试通过：输入输出形状一致")
    print("="*50)
