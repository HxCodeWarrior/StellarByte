import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

try:
    from .MoERouter import ByteContextAwareRouter
    from .RMSNorm import ByteRMSNorm
    from .MLP import ByteMLP
except ImportError:
    from MoERouter import ByteContextAwareRouter
    from RMSNorm import ByteRMSNorm
    from MLP import ByteMLP


class ByteMoELayer(nn.Module):
    """
    基础的工业级 MoE Layer，参考 DeepSpeed-MoE 风格实现。
    支持多专家动态路由、负载均衡、上下文增强，暂不考虑分布式通信，预留接口。

    Args:
        hidden_size (int): 输入维度
        ffn_hidden_size (int): 专家FFN隐层维度
        num_experts (int): 专家数量
        k (int): 每个token路由到的专家数
        dropout (float): dropout概率
        aux_loss_coef (float): 负载均衡loss系数
        capacity_factor (float): 专家容量因子
    """
    def __init__(self,
                 hidden_size: int,
                 ffn_hidden_size: int,
                 num_experts: int,
                 k: int = 1,
                 dropout: float = 0.1,
                 aux_loss_coef: float = 0.01,
                 capacity_factor: float = 1.25,
                 multiple_of: int = 256):
        super().__init__()

        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.num_experts = num_experts
        self.k = k
        self.capacity_factor = capacity_factor
        self.aux_loss_coef = aux_loss_coef
        self.multiple_of = multiple_of

        # 输入归一化层，应用RMSNorm优化输入特征
        self.input_norm = ByteRMSNorm(hidden_size)

        # 初始化路由器（上下文感知）
        self.router = ByteContextAwareRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            k=k,
            aux_loss_coef=aux_loss_coef,
            capacity_factor=capacity_factor,
            context_dim=64,
        )

        # 初始化专家组（共享参数结构）
        self.experts = nn.ModuleList([
            ByteMLP(
                dim=hidden_size, 
                hidden_dim=ffn_hidden_size, 
                multiple_of=multiple_of,
                dropout=dropout,
            )
            for _ in range(num_experts)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x: torch.Tensor,
                positions: Optional[torch.Tensor] = None,
                prev_hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入张量 [B, S, H]
            positions: 位置索引 [B, S]
            prev_hidden: 上一隐藏状态 [B, H]

        Returns:
            output: MoE输出 [B, S, H]
            aux_loss: 负载均衡辅助损失
        """
        B, S, H = x.size()
        N = B * S
        # 输入先做RMSNorm归一化
        x_normed = self.input_norm(x)
        x_flat = x.view(N, H)

        # === 路由阶段 ===
        dispatch_info, aux_loss, next_context = self.router(x_normed, positions, prev_hidden)
        expert_indices = dispatch_info["expert_indices"]  # [N, k]
        expert_weights = dispatch_info["expert_weights"]  # [N, k]
        buffer_positions = dispatch_info["buffer_positions"]  # [N, k]
        assigned_mask = dispatch_info["assigned_mask"]  # [N, k]
        capacity = dispatch_info["capacity"]

        # === 构建专家输入缓存 ===
        expert_inputs = [
            torch.zeros(capacity, H, device=x.device)
            for _ in range(self.num_experts)
        ]
        input_mask = torch.zeros(self.num_experts, capacity, dtype=torch.bool, device=x.device)
        token_outputs = torch.zeros(N, H, device=x.device)

        for i in range(self.k):
            tok_mask = assigned_mask[:, i]
            tok_idx = tok_mask.nonzero(as_tuple=False).squeeze(-1)
            if tok_idx.numel() == 0:
                continue
            expert_ids = expert_indices[tok_idx, i]  # [T]
            slot_ids = buffer_positions[tok_idx, i]  # [T]
            weights = expert_weights[tok_idx, i].unsqueeze(1)  # [T, 1]
            input_vals = x_flat[tok_idx] * weights  # [T, H]

            for e in range(self.num_experts):
                e_mask = (expert_ids == e)
                if e_mask.any():
                    e_slot = slot_ids[e_mask]
                    e_input = input_vals[e_mask]
                    expert_inputs[e][e_slot] = e_input
                    input_mask[e][e_slot] = True

        # === 专家前向计算 ===
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            valid_idx = input_mask[i].nonzero(as_tuple=False).squeeze(-1)
            if valid_idx.numel() == 0:
                expert_outputs.append(torch.zeros_like(expert_inputs[i]))
            else:
                out = expert(expert_inputs[i][valid_idx])
                full_out = torch.zeros_like(expert_inputs[i])
                full_out[valid_idx] = out
                expert_outputs.append(full_out)

        # === 汇聚专家输出 ===
        for i in range(self.k):
            tok_mask = assigned_mask[:, i]
            tok_idx = tok_mask.nonzero(as_tuple=False).squeeze(-1)
            if tok_idx.numel() == 0:
                continue
            expert_ids = expert_indices[tok_idx, i]
            slot_ids = buffer_positions[tok_idx, i]
            for j, idx in enumerate(tok_idx):
                expert_out = expert_outputs[expert_ids[j]][slot_ids[j]]
                token_outputs[idx] += expert_out

        out = self.dropout(token_outputs.view(B, S, H))
        return out, aux_loss
