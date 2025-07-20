import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple

try:
    from .RMSNorm import ByteRMSNorm
    from .MLP import ByteMLP
except ImportError:
    from RMSNorm import ByteRMSNorm
    from MLP import ByteMLP
    

class ByteContextAwareRouter(nn.Module):
    """
    上下文感知的MoE路由系统

    Args:
        hidden_size: 输入特征维度
        num_experts: 专家数量
        k: 每个token选择的专家数
        capacity_factor: 基础容量因子
        max_capacity: 专家最大容量
        min_capacity: 专家最小容量
        aux_loss_coef: 负载均衡loss权重
        context_dim: 上下文特征维度
    """
    def __init__(self,
                 hidden_size: int,
                 num_experts: int,
                 k: int = 2,
                 capacity_factor: float = 1.25,
                 max_capacity: int = 2048,
                 min_capacity: int = 4,
                 aux_loss_coef: float = 0.01,
                 context_dim: int = 64,
                 multiple_of: int = 256,
                 dropout: float = 0.1,):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.k = k
        self.capacity_factor = capacity_factor
        self.max_capacity = max_capacity
        self.min_capacity = min_capacity
        self.aux_loss_coef = aux_loss_coef
        self.context_dim = context_dim
        
        # 上下文特征提取器
        self.context_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, context_dim)
        )
        
        # 门控网络 - 上下文增强
        self.gate_norm = ByteRMSNorm(hidden_size + context_dim)  # 输入归一化
        self.gate_mlp  = ByteMLP(
            dim=hidden_size + context_dim,
            hidden_dim=None,              # 默认自动计算
            multiple_of=multiple_of,              # 与主网络对齐
            dropout=dropout,
            bias=False
        )
        self.gate_proj    = nn.Linear(hidden_size + context_dim, num_experts, bias=False)  # 输出专家 logits
        self.gate_dropout = nn.Dropout(dropout)
        
        # 位置编码
        self.position_emb = nn.Embedding(4096, context_dim // 2)
        
        # 专家状态跟踪
        self.register_buffer("expert_load", torch.zeros(num_experts))
        self.register_buffer("expert_utilization", torch.ones(num_experts) * 0.5)
        self.register_buffer("expert_priority", torch.ones(num_experts))
        self.register_buffer("total_tokens", torch.tensor(0))
        self.register_buffer("expert_assignment_count", torch.zeros(num_experts, dtype=torch.long))
        self.register_buffer('expert_cold_priority', torch.ones(num_experts))

        
        # 溢出缓冲区
        self.overflow_buffer = None
        self.overflow_count = torch.zeros(num_experts, dtype=torch.long)
        
        # 动态温度控制
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # 动态fallback权重
        self.fallback_weight = nn.Parameter(torch.zeros(num_experts))

    def forward(self, 
                x: torch.Tensor, 
                positions: Optional[torch.Tensor] = None,
                prev_hidden: Optional[torch.Tensor] = None) -> Tuple[Dict, torch.Tensor]:
        """
        向量化的路由前向传播
        
        Args:
            x: 输入token [batch_size, seq_len, hidden_size]
            positions: 位置索引 [batch_size, seq_len]
            prev_hidden: 前一隐藏状态 [batch_size, hidden_size]
            
        Returns:
            dispatch_info: 分发信息字典
            aux_loss: 负载均衡损失
        """
        B, S, H = x.shape
        N = B * S
        x_flat = x.view(N, H)
        
        # ===== 1. 上下文特征提取 =====
        # 位置编码
        if positions is None:
            positions = torch.arange(S, device=x.device).unsqueeze(0).expand(B, S)
        pos_emb = self.position_emb(positions).view(N, -1)
        
        # 语义上下文
        context_feat = self.context_net(x_flat)
        
        # 历史状态上下文
        if prev_hidden is not None:
            hist_context = prev_hidden.unsqueeze(1).expand(B, S, -1).contiguous().view(N, -1)
            context_feat = torch.cat([context_feat, hist_context], dim=-1)
        
        # 组合上下文特征
        context_feat = torch.cat([context_feat, pos_emb], dim=-1)
        
        # ===== 2. 上下文感知门控 =====
        gate_input = torch.cat([x_flat, context_feat], dim=-1)
        gate_input = self.gate_norm(gate_input)
        gate_hidden = self.gate_mlp(gate_input)
        gate_hidden = gate_hidden + gate_input
        gate_logits = self.gate_proj(gate_hidden)
        gate_logits = self.gate_dropout(gate_logits)
        
        # 引入专家优先级
        gate_logits = gate_logits + self.expert_priority.log().unsqueeze(0)

        # 负载感知温度调整
        load_imbalance = self.expert_load.std() / (self.expert_load.mean() + 1e-6)
        temp = self.temperature * (1.0 + load_imbalance.detach())
        gate_scores = F.softmax(gate_logits / temp.clamp(min=0.3), dim=-1)
        
        # ===== 3. Top-K专家选择 =====
        topk_scores, topk_indices = torch.topk(gate_scores, self.k, dim=-1)
        
        # ===== 4. 动态容量计算 =====
        capacity = self._compute_dynamic_capacity(N)
        
        # ===== 5. 向量化分配 =====
        dispatch_info = self._vectorized_dispatch(
            topk_indices, topk_scores, capacity, N
        )
        
        # ===== 6. 溢出处理 =====
        if dispatch_info['overflow_mask'].any():
            self._handle_overflow_vectorized(
                x_flat, gate_scores, topk_indices, dispatch_info, capacity
            )
        
        # ===== 7. 负载均衡Loss =====
        aux_loss = self._compute_balance_loss(
            gate_scores, dispatch_info, N
        )
        
        # 更新专家状态
        self._update_expert_state(dispatch_info, N)
        
        # 返回上下文用于下一层
        next_context = context_feat.detach().view(B, S, -1).mean(dim=1)
        
        return dispatch_info, aux_loss, next_context

    def _compute_dynamic_capacity(self, num_tokens: int) -> int:
        """负载感知的动态容量计算"""
        base_cap = math.ceil(num_tokens * self.capacity_factor / self.num_experts)
        
        # 基于专家利用率调整
        util_factor = 1.0 / (self.expert_utilization.clamp(min=0.1, max=0.9) + 0.1)
        capacity = min(
            max(int(base_cap * util_factor.mean().item()), self.min_capacity),
            self.max_capacity
        )
        return capacity

    def _vectorized_dispatch(self, 
                             expert_indices: torch.Tensor, 
                             expert_weights: torch.Tensor,
                             capacity: int,
                             num_tokens: int) -> Dict:
        """
        使用 torch_scatter 进行高效的向量化专家分配。

        Args:
            expert_indices: [N, k] 每个 token 的 top-k 专家索引
            expert_weights: [N, k] 每个 token 的 top-k 权重
            capacity: 每个专家最大容量
            num_tokens: token 总数

        Returns:
            dispatch_info: 包含分配信息的字典
        """
        device = expert_indices.device
        N, K   = expert_indices.shape  # token数量, top-k数量
        E      = self.num_experts      # 专家数量

        # === Step 1: 展平 ===
        flat_token_idx = torch.arange(N, device=device).unsqueeze(1).expand(-1, K).reshape(-1)  # [N*K]
        flat_expert_idx = expert_indices.reshape(-1)                                            # [N*K]
        flat_weight = expert_weights.reshape(-1)                                                # [N*K]

        # (2) 引入冷启动优先级: 对长期未活跃专家增加路由优先级
        # 冷启动优先级因子：expert_priority ∈ [1.0, 2.0]
        expert_priority = self.expert_priority.clamp(min=1.0, max=2.0)
        flat_priority   = expert_priority[flat_expert_idx]                                       # [N*K]
        flat_weight     = flat_weight * flat_priority                                            # 加权优先调度

        # (3) 每个专家只选择 top-Capacity 个 token（专家主导式分配）
        # 为每个 expert 构造 mask
        topk_mask  = torch.zeros_like(flat_token_idx, dtype=torch.bool)
        slot_index = torch.full_like(flat_token_idx, fill_value=-1, dtype=torch.long)

        # 初始化每个专家接收 token 的计数器
        expert_counter = torch.zeros(E, dtype=torch.long, device=device)

        # 按 expert 维度进行排序，从专家角度选择 token
        sorted_weight, sorted_idx = torch.sort(flat_weight, descending=True)

        for idx in sorted_idx:
            e = flat_expert_idx[idx].item()
            if expert_counter[e] < capacity:
                expert_counter[e] += 1
                slot_index[idx] = expert_counter[e] - 1
                topk_mask[idx] = True

        # (4) 反向 reshape 成原始 token-k 格式
        assigned_mask    = topk_mask.view(N, K)
        buffer_positions = slot_index.clamp(min=0).view(N, K)  # -1为非法位置

        # (5) 溢出 token 检测：所有 k 个专家都未接受此 token
        overflow_mask = ~assigned_mask.any(dim=-1)
        
        # 构建分配信息
        dispatch_info = {
            "expert_indices": expert_indices,           # [N, k]
            "expert_weights": expert_weights,           # [N, k]
            "assigned_mask": assigned_mask,             # [N, k]
            "buffer_positions": buffer_positions,       # [N, k]
            "overflow_mask": overflow_mask,             # [N]
            "capacity": capacity,
            "expert_count": expert_counter,             # [E]
        }
        return dispatch_info

    def _handle_overflow_vectorized(self,
                                    x_flat: torch.Tensor,
                                    gate_scores: torch.Tensor,
                                    topk_indices: torch.Tensor,
                                    dispatch_info: Dict,
                                    capacity: int):
        """
        向量化的溢出处理函数（Overflow Handling with Vectorization）

        该函数用于处理专家容量限制下的溢出 token（即被选中专家容量已满的 token）。
        它尝试通过重排、备选专家分配或降级策略（如冷启动专家、随机专家等）来最大程度保留 token 分发，
        保证模型性能与专家负载均衡。

        一般在 MoE 路由中，token 会根据 gate 分数被分配给 top-k 专家。如果所分配的专家已满，就称为“溢出”。
        本函数在向量化方式下处理这些溢出 token，避免频繁 for-loop，从而提升处理效率与并发性。

        Args:
            x_flat (torch.Tensor): 展平后的输入 token 表示，形状为 [N, D]，其中 N 是 token 数，D 是特征维度。
            gate_scores (torch.Tensor): token 在 top-k 专家上的得分，形状为 [N, k]。
            topk_indices (torch.Tensor): 每个 token 被选中的 top-k 专家索引，形状为 [N, k]。
            dispatch_info (Dict): 分发相关信息的字典，包括以下关键字段：
              - "dispatch_mask": [N, E, C] 的三维布尔张量，表示每个 token 是否被分配到专家某一容量位置。
              - "overflow_mask": [N] 的布尔张量，标记当前是否为溢出 token。
              - "expert_counter": [E] 的整数张量，表示每个专家当前已分配的 token 数。
              - "route_prob": [N, k] 的 float 张量，表示 token 分配到各个 top-k 专家的概率。
            capacity (int): 每个专家最多可接收的 token 数（容量限制）。

        Returns:
            None.（就地更新 dispatch_info 字典中的 dispatch_mask、overflow_mask 和 expert_counter）

        Functions:
            - 分析哪些 token 当前仍处于溢出状态（dispatch_info["overflow_mask"]）。
            - 为这些 token 重新尝试分配专家（通常不是它们的 top-1，而是 top-2 或 top-3）。
            - 检查新专家是否仍有空位，如果有则更新 dispatch_mask 与 expert_counter。
            - 若所有尝试失败，则该 token 最终保持溢出状态，等待后续 fallback 策略处理。
        """
        device = x_flat.device
        overflow_mask = dispatch_info["overflow_mask"]
        if not overflow_mask.any():
            return

        # =============== [Step 1] 获取溢出 token 的备选候选专家 ===============
        overflow_idx = overflow_mask.nonzero(as_tuple=False).squeeze(1)  # [M]
        M = overflow_idx.size(0)
        all_scores = gate_scores[overflow_idx]        # [M, E]
        original_topk = topk_indices[overflow_idx]    # [M, k]

        # 构造备选 mask：非topk部分
        topk_mask = torch.zeros_like(all_scores, dtype=torch.bool)
        topk_mask.scatter_(1, original_topk, True)
        candidate_mask = ~topk_mask

        # 保留备选专家得分
        candidate_scores = all_scores.masked_fill(~candidate_mask, float("-inf"))

        # 获取备选专家中分数靠前的 backup_k 个
        backup_k = min(4, self.num_experts - self.k)
        backup_scores, backup_experts = torch.topk(candidate_scores, backup_k, dim=-1)  # [M, backup_k]

        # =============== [Step 2] 尝试为溢出 token 分配 backup 专家 ===============
        flat_token_idx = overflow_idx.unsqueeze(1).expand(-1, backup_k).reshape(-1)  # [M * backup_k]
        flat_expert_idx = backup_experts.reshape(-1)                                 # [M * backup_k]
        flat_scores = backup_scores.reshape(-1)

        expert_counter = dispatch_info["expert_count"].clone()                       # [E]
        expert_avail_mask = (expert_counter[flat_expert_idx] < capacity)            # [M * backup_k]

        assign_idx = expert_avail_mask.nonzero(as_tuple=False).squeeze(1)
        if assign_idx.numel() == 0:
            return self._sticky_fallback(overflow_idx, dispatch_info, capacity)

        assigned_tokens = flat_token_idx[assign_idx]           # [M']
        assigned_experts = flat_expert_idx[assign_idx]
        assigned_scores = flat_scores[assign_idx]

        # =============== [Step 3] 向 dispatch_info 写入分配结果 ===============
        assigned_mask = dispatch_info["assigned_mask"]         # [N, k]
        expert_indices = dispatch_info["expert_indices"]
        expert_weights = dispatch_info["expert_weights"]
        buffer_positions = dispatch_info["buffer_positions"]
        expert_used_slot = dispatch_info["expert_count"]

        # 1. 计算每个 token 首个未分配槽位位置（非稀疏矩阵 -> 行优先稀疏向量）
        free_slots = (assigned_mask[assigned_tokens] == False).float()  # [M', k]
        insert_pos = free_slots.argmax(dim=-1)                          # [M']

        # 2. 用 index_put_ 写入（防止稀疏冲突）
        expert_indices.index_put_((assigned_tokens, insert_pos), assigned_experts, accumulate=False)
        expert_weights.index_put_((assigned_tokens, insert_pos), assigned_scores, accumulate=False)
        assigned_mask.index_put_((assigned_tokens, insert_pos), torch.ones_like(insert_pos, dtype=torch.bool), accumulate=False)

        # 3. 写入 slot index
        updated_pos = expert_used_slot[assigned_experts]                # [M']
        buffer_positions.index_put_((assigned_tokens, insert_pos), updated_pos, accumulate=False)

        # 4. 更新专家计数器
        expert_used_slot.index_add_(0, assigned_experts, torch.ones_like(assigned_experts, dtype=torch.long))

        # 5. 移除已成功分配的 token 的 overflow 标记
        dispatch_info["overflow_mask"][assigned_tokens] = False
        dispatch_info["expert_count"] = expert_used_slot

        # =============== [Step 4] 粘性 fallback: 粘贴历史专家 ===============
        still_overflow = dispatch_info["overflow_mask"].nonzero(as_tuple=False).squeeze(1)
        if still_overflow.numel() > 0:
            self._sticky_fallback(still_overflow, dispatch_info, capacity)

            # =============== [Step 5] fallback回退策略 ========================
            still_overflow = dispatch_info["overflow_mask"].nonzero(as_tuple=False).squeeze(1)
            if still_overflow.numel() > 0:
                self._handle_overflow_fallback(still_overflow, dispatch_info, capacity)        

    def _handle_overflow_fallback(self, 
                                 overflow_indices: torch.Tensor, 
                                 dispatch_info: Dict,
                                 capacity: int):
        """溢出回退策略"""
        device = overflow_indices.device
        utilization = self.expert_utilization                      # [E]
        expert_count = dispatch_info["expert_count"]              # [E]
        N, K = dispatch_info["expert_indices"].shape              # token数, top-k数
        E = self.num_experts

        # === 1. 动态计算“冷启动优先级” ===
        # self.expert_cold_priority 是形状为 [E] 的张量，范围 [1.0, 2.0]
        cold_priority = getattr(self, 'expert_cold_priority', torch.ones(E, device=device))

        # 利用 fallback_weight 作为可学习参数，初始为0，需在 init 定义
        fallback_weight = getattr(self, 'fallback_weight', torch.zeros(E, device=device))

        # 计算 fallback score: 综合考虑利用率、冷启动优先级和学习权重
        # 先计算基础分数，负载越低，分数越高
        base_score = (1.0 - utilization.clamp(0, 1)) * cold_priority * torch.exp(fallback_weight)

        # 只考虑未满容量的专家
        available_mask = expert_count < capacity
        available_experts = torch.where(available_mask)[0]        # [E_avail]

        if available_experts.numel() == 0:
            # 退化：所有专家满载，选择所有专家
            available_experts = torch.arange(E, device=device)

        available_scores = base_score[available_experts]           # [E_avail]

        # === 2. 为每个溢出token分配专家 ===
        # 直接选出score最高的专家，广播所有overflow token分配相同专家会造成冲突，故针对每token选其最优专家
        # 为避免for循环，采取如下策略：对所有overflow token随机扰动分数（break ties），然后选最大
        M = overflow_indices.size(0)
        noise = -torch.log(-torch.log(torch.rand_like(scores))) * self.temperature  # 微扰分数，打破tie

        scores = available_scores.unsqueeze(0).expand(M, -1) + noise  # [M, E_avail]
        max_scores, max_indices = torch.max(scores, dim=1)            # [M]

        selected_experts = available_experts[max_indices]             # [M]

        # === 3. 检查每个专家容量是否允许分配 ===
        expert_slots = expert_count[selected_experts]                 # [M]
        can_accept_mask = expert_slots < capacity
        valid_idx = torch.where(can_accept_mask)[0]

        if valid_idx.numel() == 0:
            # 全部失败，不处理（可日志记录）
            return

        final_tokens = overflow_indices[valid_idx]                     # [M']
        final_experts = selected_experts[valid_idx]
        final_slots = expert_slots[valid_idx]

        # === 4. 写入 dispatch_info，向量化插入 ===
        assigned_mask = dispatch_info["assigned_mask"]
        free_pos = (~assigned_mask[final_tokens]).float().argmax(dim=1)  # [M']

        dispatch_info["expert_indices"][final_tokens, free_pos] = final_experts
        dispatch_info["expert_weights"][final_tokens, free_pos] = 1.0
        dispatch_info["assigned_mask"][final_tokens, free_pos] = True
        dispatch_info["buffer_positions"][final_tokens, free_pos] = final_slots

        # === 5. 更新专家状态 ===
        expert_count.index_add_(0, final_experts, torch.ones_like(final_experts))
        # 更新利用率：按分配比例更新，归一化到capacity范围
        self.expert_utilization.index_add_(0, final_experts, torch.ones_like(final_experts, dtype=torch.float32) / capacity)

        # === 6. 清除分配成功的 overflow 标记 ===
        dispatch_info["overflow_mask"][final_tokens] = False

        # === 7. fallback分配记录监控 ===
        self.overflow_count.index_add_(0, final_experts, torch.ones_like(final_experts, dtype=torch.long))

    def _sticky_fallback(self, overflow_idx: torch.Tensor, dispatch_info: Dict, capacity: int):
        """
        粘性路由策略：将仍无法分配的 token 路由至其上轮分配成功的专家（只要未满）
        """
        if overflow_idx.numel() == 0:
            return

        device = overflow_idx.device
        expert_count = dispatch_info["expert_count"]
        k = dispatch_info["expert_indices"].shape[1]

        last_expert = dispatch_info["expert_indices"][overflow_idx, 0]  # 使用历史第一个成功专家
        can_assign = expert_count[last_expert] < capacity
        valid_idx = torch.where(can_assign)[0]

        if valid_idx.numel() == 0:
            return  # 最终无法分配，交给 padding expert 或 dropout

        token_ids = overflow_idx[valid_idx]
        expert_ids = last_expert[valid_idx]
        slot_pos = expert_count[expert_ids]

        # 更新分配信息
        for i in range(valid_idx.numel()):
            tok = token_ids[i]
            exp = expert_ids[i]
            pos = slot_pos[i]

            insert_pos = (dispatch_info["assigned_mask"][tok] == False).nonzero(as_tuple=False)
            if insert_pos.numel() == 0:
                continue
            slot = insert_pos[0].item()

            dispatch_info["expert_indices"][tok, slot] = exp
            dispatch_info["expert_weights"][tok, slot] = 1.0  # 粘性路由 score 可设为1.0
            dispatch_info["assigned_mask"][tok, slot] = True
            dispatch_info["buffer_positions"][tok, slot] = pos
            dispatch_info["overflow_mask"][tok] = False

            dispatch_info["expert_count"][exp] += 1

    def _compute_balance_loss(self, 
                             gate_scores: torch.Tensor, 
                             dispatch_info: Dict, 
                             num_tokens: int) -> torch.Tensor:
        """细粒度负载均衡损失计算"""
        device = gate_scores.device
        E = self.num_experts

        # 计算专家重要性：所有token分配给专家的概率之和 [E]
        expert_importance = gate_scores.sum(dim=0)  # [E]

        # 计算专家实际负载：统计被分配的token数目
        assigned_mask = dispatch_info["assigned_mask"]  # [N, k]
        expert_indices = dispatch_info["expert_indices"]  # [N, k]

        # 获取所有被分配的专家索引（展平）
        flat_expert_idx = expert_indices[assigned_mask]  # 一维，包含所有被分配的专家索引

        # 统计专家负载（token数）
        expert_load = torch_scatter.scatter_add(
            torch.ones_like(flat_expert_idx, dtype=torch.float32, device=device),
            flat_expert_idx,
            dim_size=E
        )

        # 防止除零
        expert_load_sum = expert_load.sum() + 1e-6
        importance_sum = expert_importance.sum() + 1e-6

        # 归一化分布
        P = expert_importance / importance_sum  # 专家重要性分布
        L = expert_load / expert_load_sum       # 专家实际负载分布

        # 计算token分配概率的熵，衡量不确定性
        token_entropy = (-gate_scores * torch.log(gate_scores + 1e-9)).sum(dim=-1)  # [N]

        # token重要性权重，令分配不确定性高的token权重更大
        token_importance = 1.0 + token_entropy
        weighted_importance = (gate_scores * token_importance.unsqueeze(-1)).sum(dim=0)  # [E]

        # 多目标损失：KL散度 + MSE + 负载方差
        kl_loss = F.kl_div(P.log(), L, reduction='sum')
        mse_loss = F.mse_loss(L, torch.ones_like(L) / E)
        load_var = expert_load.float().var(unbiased=False) / (expert_load.float().mean() ** 2 + 1e-6)

        loss = self.aux_loss_coef * (kl_loss + 0.2 * mse_loss + 0.1 * load_var)
        return loss

    def _update_expert_state(self, dispatch_info: Dict, num_tokens: int):
        """更新专家状态（EMA）"""
        # 提取分配信息
        assigned_mask = dispatch_info["assigned_mask"]
        flat_expert_idx = dispatch_info["expert_indices"][assigned_mask].view(-1)
        
        # 计算当前负载
        current_load = torch_scatter.scatter_add(
            torch.ones_like(flat_expert_idx), flat_expert_idx, dim_size=self.num_experts
        )
        
        # 计算分配成功率
        assignment_success = torch_scatter.scatter_add(
            assigned_mask.float().view(-1), 
            dispatch_info["expert_indices"].view(-1),
            dim_size=self.num_experts
        ) / (current_load + 1e-6)
        
        # 计算利用率
        capacity = dispatch_info["capacity"]
        utilization = current_load.float() / capacity
        
        # EMA更新状态
        decay = 0.9
        self.expert_load = decay * self.expert_load + (1 - decay) * current_load
        self.expert_utilization = decay * self.expert_utilization + (1 - decay) * utilization
        
        # 更新专家优先级
        priority_update = 0.5 * assignment_success + 0.3 * (1 - utilization) + 0.2 * (1 - current_load.std()/current_load.mean())
        self.expert_priority = decay * self.expert_priority + (1 - decay) * priority_update
        self.total_tokens += num_tokens

    def generate_dispatch_plan(self, dispatch_info: Dict) -> Dict:
        """生成高效的分发计划（向量化实现）"""
        assigned_mask = dispatch_info["assigned_mask"]
        expert_idx = dispatch_info["expert_indices"][assigned_mask]
        token_idx = torch.nonzero(assigned_mask, as_tuple=True)[0]
        positions = dispatch_info["buffer_positions"][assigned_mask]
        weights = dispatch_info["expert_weights"][assigned_mask]
        
        expert_buffers = {
            "expert_idx": expert_idx,
            "token_idx": token_idx,
            "positions": positions,
            "weights": weights,
            "capacity": dispatch_info["capacity"],
            "num_tokens": token_idx.size(0)
        }

        return expert_buffers