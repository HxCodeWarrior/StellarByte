# 导入必要的PyTorch模块和类型注解支持
import torch
import torch.nn as nn
from typing import Optional, Dict  # 用于类型注解

# 尝试从本地模块导入相关组件
try:
    from .config import StellarByteModelArgs  # 模型配置类
    from .RMSNorm import StellarByteRMSNorm  # RMS归一化层
    from .Attention import StellarByteAttention  # 注意力机制
    from .FeedForward import StellarByteMLP  # 普通前馈网络
    from .MoE import StellarByteMOEFeedForward  # 混合专家前馈网络
except:
    # 如果相对导入失败，尝试绝对导入（用于独立运行或测试）
    from config import StellarByteModelArgs
    from RMSNorm import StellarByteRMSNorm
    from Attention import StellarByteAttention
    from FeedForward import StellarByteMLP
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
        feed_forward (Union[StellarByteMLP, StellarByteMOEFeedForward]): 前馈网络
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
        
        # 根据配置选择使用MoE前馈网络或普通MLP
        if args.enabled_moe:
            self.feed_forward = StellarByteMOEFeedForward(args)
        else:
            self.feed_forward = StellarByteMLP(
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
        start_pos: int,
        freqs_cis: torch.Tensor
    ) -> torch.Tensor:
        """前向传播函数。
        
        实现Transformer块的计算流程：归一化 -> 自注意力 -> 残差连接 -> 
        归一化 -> 前馈网络 -> 残差连接。
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, dim)
            start_pos: 当前序列在缓存中的起始位置，用于增量解码
            freqs_cis: 预先计算的频率张量，用于旋转位置编码
            
        Returns:
            输出张量，形状与输入相同 (batch_size, seq_len, dim)
        """
        # 注意力层计算 (带残差连接)
        # 1. 对输入进行归一化
        # 2. 应用自注意力机制
        # 3. 与原始输入相加 (残差连接)
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis)
        
        # 前馈层计算 (可能是MoE或普通MLP)
        # 1. 对注意力输出进行归一化
        ffn_input = self.ffn_norm(h)
        # 2. 应用前馈网络
        ffn_output = self.feed_forward(ffn_input)

        # 如果是MoE模式，保存辅助损失和专家使用情况
        if self.enabled_moe:
            # 获取MoE辅助损失（如果存在）
            self.moe_aux_loss = getattr(self.feed_forward, 'aux_loss', None)
            # 获取专家使用统计
            self.moe_expert_usage = self.feed_forward.get_expert_usage()
        
        # 应用残差连接并返回结果
        out = h + ffn_output
        return out
    
    def get_moe_aux_loss(self) -> Optional[torch.Tensor]:
        """获取MoE辅助损失。
        
        MoE模型使用辅助损失来平衡专家使用率，防止某些专家被过度使用而其他专家被忽略。
        
        Returns:
            MoE辅助损失张量，如果未启用MoE或没有辅助损失则返回None
        """
        return self.moe_aux_loss

    def get_moe_expert_usage(self) -> Optional[Dict[int, int]]:
        """获取MoE专家使用统计。
        
        返回每个专家在当前前向传播中被使用的次数统计。
        
        Returns:
            专家ID到使用次数的映射字典，如果未启用MoE则返回None
        """
        return self.moe_expert_usage