import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    from .config import StellarByteModelArgs
    from .FeedForward import StellarByteFeedForward
except ImportError:
    from config import StellarByteModelArgs
    from FeedForward import StellarByteFeedForward

class StellarByteMoEGate(nn.Module):
    def __init__(self, args: StellarByteModelArgs):
        super().__init__()
        self.args = args
        self.top_k = args.num_experts_per_tok
        self.num_routed_experts = args.num_routed_experts
        self.scoring_func = args.scoring_func
        self.alpha = args.aux_loss_alpha
        self.seq_aux = args.seq_aux
        self.norm_topk_prob = args.norm_topk_prob
        self.gating_dim = args.gating_dim
        
        # 参数初始化
        self.weight = nn.Parameter(torch.empty((self.num_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 使用更合适的初始化方法
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, h = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, h)
        
        # 计算门控分数
        logits = F.linear(hidden_states_flat, self.weight, None)
        
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'Unsupported scoring function for MoE gating: {self.scoring_func}')

        # 选择top-k专家
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # 可选: 标准化top-k权重
        if self.top_k > 1 and self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

        # 计算辅助损失
        aux_loss = self._calculate_aux_loss(scores, topk_idx, bsz, seq_len, hidden_states.device)
        
        return topk_idx, topk_weight, aux_loss

    def _calculate_aux_loss(self, scores: torch.Tensor, topk_idx: torch.Tensor, 
                           bsz: int, seq_len: int, device: torch.device) -> torch.Tensor:
        """计算负载均衡辅助损失"""
        if not self.training or self.alpha <= 0.0:
            return torch.tensor(0.0, device=device)
        
        aux_topk = self.top_k
        scores_for_aux = scores
        
        if self.seq_aux:
            # 序列级别的辅助损失
            scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
            ce = torch.zeros(bsz, self.num_routed_experts, device=device)
            
            # 使用bincount替代scatter_add以提高效率
            for i in range(bsz):
                flat_indices = topk_idx[i*seq_len:(i+1)*seq_len].view(-1)
                ce[i] = torch.bincount(flat_indices, minlength=self.num_routed_experts).float()
            
            ce = ce / (seq_len * aux_topk / self.num_routed_experts)
            aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
        else:
            # Token级别的辅助损失
            mask_ce = F.one_hot(topk_idx.view(-1), num_classes=self.num_routed_experts)
            ce = mask_ce.float().mean(0)
            Pi = scores_for_aux.mean(0)
            fi = ce * self.num_routed_experts
            aux_loss = (Pi * fi).sum() * self.alpha
            
        return aux_loss


class StellarByteMOEFeedForward(nn.Module):
    def __init__(self, args: StellarByteModelArgs):
        super().__init__()
        self.args = args
        self.num_routed_experts = args.num_routed_experts
        self.num_shared_experts = args.num_shared_experts
        
        # 路由专家
        self.experts = nn.ModuleList([
            StellarByteFeedForward(
                dim=args.dim,
                hidden_dim=args.dim * 4,
                multiple_of=args.multiple_of,
                ffn_dim_multiplier=args.ffn_dim_multiplier,
                dropout=args.ffn_dropout,
                eps=args.rms_norm_eps
            ) for _ in range(args.num_routed_experts)
        ])
        
        self.gate = StellarByteMoEGate(args)
        
        # 共享专家
        if args.num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                StellarByteFeedForward(
                dim=args.dim,
                hidden_dim=args.dim * 4,
                multiple_of=args.multiple_of,
                ffn_dim_multiplier=args.ffn_dim_multiplier,
                dropout=args.ffn_dropout,
                eps=args.rms_norm_eps
            ) for _ in range(args.num_shared_experts)
            ])
        
        # 缓存专家计算信息用于分析
        self.expert_usage = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x_flat = x.view(-1, x.shape[-1])
        
        # 根据训练/推理模式选择不同的计算路径
        if self.training:
            y = self._moe_train(x_flat, topk_idx, topk_weight)
        else:
            y = self._moe_infer(x_flat, topk_idx, topk_weight)
        
        y = y.view(*orig_shape)
        
        # 添加共享专家计算
        if self.num_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        
        # 保存辅助损失和专家使用情况
        self.aux_loss = aux_loss
        return y

    def _moe_train(self, x: torch.Tensor, topk_idx: torch.Tensor, 
                  topk_weight: torch.Tensor) -> torch.Tensor:
        """训练模式下的MoE计算"""
        # 扩展输入以匹配top_k专家
        x_repeated = x.repeat_interleave(self.args.num_experts_per_tok, dim=0)
        
        # 预分配输出张量
        y = torch.zeros(x_repeated.shape[0], x.shape[-1], 
                       dtype=torch.float16 if x.dtype == torch.float16 else torch.float32,
                       device=x.device)
        
        # 收集每个专家的处理索引
        expert_indices = []
        for i in range(self.num_routed_experts):
            mask = (topk_idx.view(-1) == i)
            expert_indices.append((i, mask))
        
        # 并行处理所有专家（如果设备支持）
        for i, mask in expert_indices:
            if mask.any():
                expert_input = x_repeated[mask]
                expert_output = self.experts[i](expert_input).to(y.dtype)
                y[mask] = expert_output
        
        # 应用门控权重并聚合结果
        y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
        return y

    @torch.no_grad()
    def _moe_infer(self, x: torch.Tensor, topk_idx: torch.Tensor, 
                  topk_weight: torch.Tensor) -> torch.Tensor:
        """推理模式下的MoE计算"""
        flat_expert_indices = topk_idx.view(-1)
        flat_expert_weights = topk_weight.view(-1, 1)
        
        # 预分配输出张量
        expert_cache = torch.zeros_like(x)
        
        # 对专家索引进行排序以提高缓存效率
        sorted_indices = flat_expert_indices.argsort()
        sorted_expert_indices = flat_expert_indices[sorted_indices]
        sorted_token_indices = sorted_indices // self.args.num_experts_per_tok
        
        # 计算每个专家处理的token范围
        unique_experts, expert_counts = torch.unique_consecutive(
            sorted_expert_indices, return_counts=True)
        
        # 记录专家使用情况
        self.expert_usage = {
            int(expert): count.item() 
            for expert, count in zip(unique_experts, expert_counts)
        }
        
        # 为每个专家处理其分配的token
        start_idx = 0
        for expert_id, count in zip(unique_experts, expert_counts):
            if count == 0:
                continue
                
            end_idx = start_idx + count
            token_indices = sorted_token_indices[start_idx:end_idx]
            expert = self.experts[expert_id]
            
            # 获取对应token的输入和权重
            expert_input = x[token_indices]
            expert_weights = flat_expert_weights[sorted_indices[start_idx:end_idx]]
            
            # 计算专家输出并应用权重
            expert_output = expert(expert_input).to(expert_cache.dtype)
            weighted_output = expert_output * expert_weights
            
            # 累加到输出缓存
            expert_cache.index_add_(0, token_indices, weighted_output)
            start_idx = end_idx
        
        return expert_cache

    def get_expert_usage(self) -> dict:
        """返回专家使用统计信息"""
        return self.expert_usage if self.expert_usage else {}


if __name__ == "__main__":
    import torch
    from config import StellarByteModelArgs  # 确保从正确位置导入配置类
    
    print("="*50)
    print("StellarByteMOEFeedForward 形状验证测试")
    print("="*50)
    
    # 1. 创建模型配置
    class StellarByteModelArgs:
        dim=512  # 模型维度
        multiple_of=256  # 维度倍数约束
        ffn_dropout=0.1
        rms_norm_eps=1e-6
        ffn_dim_multiplier=1
        enabled_moe=True
        gating_dim=512
        num_experts=4
        num_experts_per_tok=2
        num_routed_experts=4
        num_shared_experts=1
        scoring_func='softmax'
        aux_loss_alpha=0.1
        seq_aux=True
        norm_topk_prob=True
    args = StellarByteModelArgs()

    # 2. 创建MoE模块实例
    moe = StellarByteMOEFeedForward(args)
    
    # 3. 创建模拟输入
    batch_size = 2
    seq_len = 64
    x = torch.randn(batch_size, seq_len, args.dim)
    
    print(f"输入形状: {x.shape}")
    
    # 4. 前向传播
    output = moe(x)
    
    print(f"输出形状: {output.shape}")
    
    # 5. 形状验证
    assert output.shape == x.shape, \
        f"形状验证失败! 预期: {x.shape}, 实际: {output.shape}"
    
    # 6. 专家选择验证
    print("\n专家选择验证:")
    print(f"辅助损失: {moe.aux_loss.item():.4f} (应为非零值)")
    print(f"专家使用统计: {moe.get_expert_usage()}")
    
    # 7. 验证专家选择逻辑
    total_tokens = batch_size * seq_len
    expected_min_usage = total_tokens * args.num_experts_per_tok // args.num_experts
    for expert_id, usage in moe.get_expert_usage().items():
        assert usage > 0, f"专家 {expert_id} 未被使用!"
        assert usage >= expected_min_usage, f"专家 {expert_id} 使用率过低!"
    
    print("\n✅ 测试通过：输入输出形状一致，专家选择逻辑正常")
    print("="*50)
