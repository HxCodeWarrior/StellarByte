"""
DeepSpeed-MoE 风格分布式 Mixture-of-Experts 层
特性：
  - top-k routing 完全向量化
  - 使用 all_to_all 进行分布式 token 交换
  - 支持容量限制和负载均衡 loss
  - 单卡和多卡兼容
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class ExpertFFN(nn.Module):
    """单个专家 FFN 层"""
    def __init__(self, d_model: int, d_ff: int, activation: str = "gelu") -> None:
        super().__init__()
        # 输入到隐藏层线性变换
        self.fc1 = nn.Linear(d_model, d_ff)
        # 隐藏层到输出线性变换
        self.fc2 = nn.Linear(d_ff, d_model)
        # 激活函数选择
        if activation not in ("gelu", "relu", "swish"):
            raise ValueError("不支持的激活函数")
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 前向计算
        x = self.fc1(x)
        # 激活函数
        if self.activation == "gelu":
            x = F.gelu(x)
        elif self.activation == "swish":
            x = x * torch.sigmoid(x)
        else:
            x = F.relu(x)
        # 输出线性变换
        return self.fc2(x)


class MoELayerDistributedOptimized(nn.Module):
    """分布式 MoE 层"""
    def __init__(self, d_model: int, d_ff: int, n_experts: int = 32, k: int = 1,
                 capacity_factor: float = 1.25, dropout: float = 0.0, activation: str = "gelu",
                 world_size: int = 1, rank: int = 0) -> None:
        super().__init__()
        # 只支持 top-1 或 top-2 路由
        assert k in (1, 2), "只支持 top-1 或 top-2"
        self.d_model = d_model              # token 表示维度
        self.d_ff = d_ff                    # 专家隐藏层维度
        self.n_experts = n_experts          # 专家总数
        self.k = k                          # top-k 路由数量
        self.capacity_factor = capacity_factor  # 容量因子
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()  # dropout 层
        self.world_size = world_size        # 分布式总 GPU 数
        self.rank = rank                    # 当前 GPU rank

        # 每个 GPU 的本地专家数量
        self.n_experts_per_rank = n_experts // world_size
        assert n_experts % world_size == 0, "n_experts 必须能整除 world_size"

        # gating 网络，用于计算每个 token 对各个专家的权重
        self.w_gate = nn.Linear(d_model, n_experts, bias=False)
        # 本地专家列表
        self.experts = nn.ModuleList([ExpertFFN(d_model, d_ff, activation)
                                      for _ in range(self.n_experts_per_rank)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, D = x.shape
        # 将输入 reshape 为二维 [S, D]，方便后续处理
        tokens = x.view(-1, D)
        S = tokens.size(0)  # 总 token 数

        # ---------- 1) top-k gating ----------
        logits = self.w_gate(tokens)           # [S, n_experts] 计算每个 token 对每个专家的分数
        probs = F.softmax(logits, dim=-1)     # 概率化
        topk_vals, topk_idx = torch.topk(probs, self.k, dim=-1)  # 每个 token 选择 top-k 专家

        # ---------- 2) 负载均衡 loss ----------
        # importance 表示每个专家被选中的概率和
        importance = probs.sum(dim=0)
        # topk_onehot 用于统计每个专家被选中 token 的数量
        topk_onehot = torch.zeros(S, self.n_experts, device=x.device)
        topk_onehot.scatter_(1, topk_idx, 1.0)
        load = topk_onehot.sum(dim=0)
        # 计算负载均衡 loss
        load_balance_loss = (torch.mean(importance * load) * (self.n_experts ** 2)) / (S * S) if self.training else None

        # ---------- 3) 计算容量 ----------
        capacity = int(math.ceil((S / self.n_experts) * self.capacity_factor))  # 每个专家容量

        # ---------- 4) flatten top-k routing ----------
        expert_idx_flat = topk_idx.flatten()       # 展平为一维 [S*k]
        gate_vals_flat = topk_vals.flatten()      # 展平对应权重 [S*k]
        token_idx_flat = torch.arange(S, device=x.device).unsqueeze(1).repeat(1, self.k).flatten()  # token 索引

        # ---------- 5) 分布式路由 ----------
        # 计算每个 token 属于哪个 GPU 的专家
        expert_rank = expert_idx_flat // self.n_experts_per_rank
        local_mask = expert_rank == self.rank   # 当前 GPU 的 token 掩码
        local_expert_idx = expert_idx_flat[local_mask] - self.rank * self.n_experts_per_rank  # 本地专家索引
        local_token_idx = token_idx_flat[local_mask]   # 对应 token 索引
        local_gate_vals = gate_vals_flat[local_mask]   # 对应 gate 权重

        # ---------- 6) 向量化容量控制 ----------
        # 按本地专家排序，便于后续位置计算
        sorted_idx = torch.argsort(local_expert_idx)
        local_expert_idx = local_expert_idx[sorted_idx]
        local_token_idx = local_token_idx[sorted_idx]
        local_gate_vals = local_gate_vals[sorted_idx]

        # one-hot 编码每个 token 属于的本地专家
        one_hot = F.one_hot(local_expert_idx, num_classes=self.n_experts_per_rank).float()  # [N_local, n_experts_local]
        # 累加计算每个 token 在专家 buffer 中的位置
        positions = one_hot.cumsum(dim=0).argmax(dim=1) - 1  # [N_local]
        # 容量上限
        positions = positions.clamp(max=capacity-1)

        # ---------- 7) 构建专家输入 buffer ----------
        expert_input = torch.zeros(self.n_experts_per_rank, capacity, D, device=x.device, dtype=tokens.dtype)
        # 将 token 放入对应专家的 buffer
        expert_input.index_put_((local_expert_idx, positions), tokens[local_token_idx], accumulate=False)

        # ---------- 8) 专家前向 ----------
        expert_output = torch.zeros_like(expert_input)
        for i, expert in enumerate(self.experts):
            expert_output[i] = expert(expert_input[i])  # 每个专家独立计算输出

        # ---------- 9) all-to-all 分布式通信 ----------
        # 目标：将每个 GPU 上的本地专家输出，按专家所在 rank 分发到对应 GPU
        # 输入：expert_output -> [n_experts_per_rank, capacity, D]
        # 输出：expert_output_full -> [n_experts, capacity, D]
        
        # 获取本地专家数量和特征维度
        n_local_experts, capacity, D = expert_output.shape
        
        # 1. 将本地专家数据按世界大小切分，每份对应一个目标 GPU
        #    chunk 后每个 chunk 是 [n_local_experts_per_dest, capacity, D]
        n_chunk = self.world_size
        assert n_local_experts % n_chunk == 0, "本地专家数必须能被 world_size 整除"
        split_experts = expert_output.chunk(n_chunk, dim=0)  # list of tensors
        
        # 2. flatten 为二维 [n_experts_per_chunk * capacity, D]，便于 all_to_all_single 传输
        send_tensors = [t.contiguous().view(-1, D) for t in split_experts]
        send_tensor_flat = torch.cat(send_tensors, dim=0)
        
        # 3. 计算发送/接收量，每个 GPU 接收的 token 数量
        tokens_per_gpu = send_tensor_flat.shape[0] // self.world_size
        recv_tensor_flat = torch.zeros(tokens_per_gpu * self.world_size, D, device=expert_output.device, dtype=expert_output.dtype)
        
        # 4. 调用 all_to_all_single 进行通信
        #    send_tensor_flat -> 所有 GPU 按顺序发送
        #    recv_tensor_flat -> 接收来自所有 GPU 的数据
        dist.all_to_all_single(recv_tensor_flat, send_tensor_flat)
        
        # 5. reshape 回三维 [n_experts, capacity, D]
        #    total_experts = n_experts_per_rank * world_size
        total_experts = self.n_experts_per_rank * self.world_size
        expert_output_full = recv_tensor_flat.view(total_experts, capacity, D)

        # ---------- 10) 合并回原 token ----------
        combined = torch.zeros(S, D, device=x.device, dtype=tokens.dtype)
        weight_sums = torch.zeros(S, device=x.device, dtype=tokens.dtype)
        # 将专家输出按 gate 权重累加到对应 token
        combined.index_add_(0, local_token_idx, expert_output_full[local_expert_idx, positions] * local_gate_vals)
        weight_sums.index_add_(0, local_token_idx, local_gate_vals)
        # 对非零权重的 token 做归一化
        nonzero = weight_sums > 0
        combined[nonzero] = combined[nonzero] / weight_sums[nonzero].unsqueeze(1)

        # reshape 回原始 [B, T, D] 形状
        y = combined.view(B, T, D)
        y = self.dropout(y)

        return y, load_balance_loss
