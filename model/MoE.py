import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ByteMoE(nn.Module):
    """
    高效MoE层
    
    Args:
        hidden_size: 输入和输出隐藏维度
        expert_ffn_dim: 专家FFN隐藏维度
        num_experts: 专家数
        k: top-k专家数，默认为2
        capacity_factor: 专家容量因子
        loss_coef: 负载均衡loss权重
        max_capacity: 专家最大容量限制
    """

    def __init__(
        self,
        hidden_size,
        expert_ffn_dim,
        num_experts,
        k=2,
        capacity_factor=1.25,
        loss_coef=1e-2,
        max_capacity=512,
        moe_dropout=0.0,
        max_backup=4,
    ):
        super().__init__()
        self.hidden_size     = hidden_size      
        self.expert_ffn_dim  = expert_ffn_dim
        self.num_experts     = num_experts      # 专家数量
        self.k               = k                # 每个 token 分配给 top_k 个专家，通常为 1 或 2
        self.capacity_factor = capacity_factor  # 每个专家最大 token 容量的扩张系数
        self.loss_coef       = loss_coef
        self.max_capacity    = max_capacity
        self.moe_dropout     = moe_dropout
        self.max_backup      = max_backup         # 最多尝试备份专家数

        # 专家网络，使用ModuleList
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, expert_ffn_dim),
                nn.GELU(),
                nn.Linear(expert_ffn_dim, hidden_size)
            ) for _ in range(num_experts)
        ])

        # 门控网络，输出num_experts维度logits
        self.gate = nn.Linear(hidden_size, num_experts)

    def forward(self, x):
        """
        输入:
            x: (B, S, H)
        输出:
            y: (B, S, H)
            aux_loss: 负载均衡辅助loss
        """
        B, S, H = x.shape
        N       = B * S    # 总token数
        E       = self.num_experts
        K       = self.k
        x_flat  = x.reshape(N, H)  # (N, H), N=B*S

        # === Step 1: MoE Dropout (增强泛化能力) ===
        # 训练时的门控dropout，用于提升泛化能力
        if self.training and self.moe_dropout > 0.0:
            drop_mask = torch.rand(N, self.num_experts, device=x.device) > self.moe_dropout
        else:
            drop_mask = torch.ones(N, self.num_experts, device=x.device, dtype=torch.bool)

        # === Step 2: 计算门控分数 === 
        gate_logits             = self.gate(x_flat)               # (N, E)
        gate_logits[~drop_mask] = float('-inf')                   # 掉落的专家置为-inf
        gate_scores             = F.softmax(gate_logits, dim=-1)  # (N, E)

        # === Step 3: 选top-k专家 ===
        # 2.0 计算备份专家数
        backup_k = min(K * 4, E)
        # 2.1 计算每个token分配给每个专家的得分
        topk_scores_all, topk_indices_all = torch.topk(gate_scores, backup_k, dim=-1)  # (N, k)
        # 2.2 Top-k得分归一化
        topk_scores_normalized    = topk_scores_all / (topk_scores_all.sum(dim=-1, keepdim=True) + 1e-9)  # (N, k)

        # === Step 4:  ===

        # === Step 5: 动态容量计算 ===
        capacity = min(
            int(self.capacity_factor * math.ceil(N / E)),
            self.max_capacity
        )  # 每个专家最大token数

        # === Step 4: 构造稀疏分配矩阵 ===
        # 4.1 展平 top-k 维度，得到每个 token-expert 对
        topk_indices_flat = topk_indices_all.reshape(-1)           # (N*K,)
        topk_scores_flat = topk_scores_normalized.reshape(-1)  # (N*K,)

        # 4.2 构造 (E, N*K) 的one-hot专家分配矩阵，用于后续计算序号slot
        expert_assign_matrix = torch.zeros(
            E, topk_indices_flat.size(0),
            device=x.device,
            dtype=torch.bool
        )
        expert_assign_matrix[topk_indices_flat, torch.arange(topk_indices_flat.size(0), device=x.device)] = True

        # 4.3 计算每个token-expert对在对应专家的slot索引（从0开始）
        expert_cumsum = torch.cumsum(expert_assign_matrix.to(torch.long), dim=1) - 1  # (E, N*K)

        # 4.4 取出每个token-expert对对应的slot编号
        token_positions = expert_cumsum[topk_indices_flat, torch.arange(topk_indices_flat.size(0), device=x.device)]  # (N*K,)

        # 4.5 根据容量限制过滤超出容量的token-expert对
        mask_in_capacity = token_positions < capacity  # (N*K,)

        # 4.6 筛选有效token-expert对索引
        valid_idx = torch.nonzero(mask_in_capacity, as_tuple=False).squeeze(-1)  # (M,)

        # 4.7 计算有效token-expert对的原token索引，专家索引，slot索引，权重
        token_idx = torch.div(valid_idx, K, rounding_mode='floor')  # (M,)
        expert_slot_idx = token_positions[valid_idx]               # (M,)
        expert_idx = topk_indices_flat[valid_idx]                  # (M,)
        expert_score = topk_scores_flat[valid_idx]                 # (M,)

        # 4.8 按权重缩放对应token特征
        selected_x = x_flat[token_idx] * expert_score.unsqueeze(-1)  # (M, H)

        # 4.9 构造专家输入buffer，大小为 (E, capacity, H)
        expert_input_buf = torch.zeros(E, capacity, H, device=x.device)

        # 4.10 利用index_put_将token特征写入专家buffer对应位置
        expert_input_buf.index_put_(
            (expert_idx, expert_slot_idx),
            selected_x,
            accumulate=True  # 理论上不会重复写同一个位置，安全起见开启累加
        )

        # === Step 5: 并行执行专家前向处理 ===
        expert_output_buf = torch.zeros_like(expert_input_buf)

        # 统计每个专家的实际负载，防止溢出
        expert_load = torch.bincount(expert_idx, minlength=E).clamp(max=capacity)  # (E,)

        # 依次调用每个专家前向，只计算有输入的部分
        for eid in range(E):
            load = expert_load[eid].item()
            if load > 0:
                expert_output_buf[eid, :load] = self.experts[eid](expert_input_buf[eid, :load])

        # === Step 6: 重组专家输出回原token顺序 ===
        y_flat = torch.zeros_like(x_flat)  # (N, H)

        # 取出有效token-expert对对应的专家输出
        selected_output = expert_output_buf[expert_idx, expert_slot_idx]  # (M, H)

        # 由于每个token可能对应多个专家，需加权累加专家输出
        y_flat.index_add_(0, token_idx, selected_output)

        # 恢复为 (B, S, H)
        y = y_flat.view(B, S, H)

        # === Step 7: 负载均衡loss计算 ===
        aux_loss = self._balanced_loss(gate_scores, topk_indices_all)

        return y, aux_loss

    def _balanced_loss(self, gate_probs, topk_indices):
        """
        负载均衡loss计算方法：结合top-k使用频率和期望负载,鼓励专家均匀利用

        Args:
            gate_probs: [N, E] 门控概率
            topk_indices: [N, k] top-k专家索引

        Returns:
            加权的KL散度作为负载均衡loss
        """
        N, E = gate_probs.size()
        k = topk_indices.size(1)

        # 构造专家出现的掩码矩阵 [N, E]
        expert_mask = torch.zeros_like(gate_probs)
        token_idx = torch.arange(N, device=gate_probs.device).unsqueeze(1).expand(-1, k)
        expert_mask[token_idx.reshape(-1), topk_indices.reshape(-1)] = 1

        # 计算实际专家负载概率分布
        actual_load = expert_mask.sum(0).float() / expert_mask.sum()

        # 理想均匀负载分布
        expected_load = torch.full_like(actual_load, 1.0 / E)

        # 计算KL散度衡量分布差异
        kl_div = F.kl_div(actual_load.log(), expected_load, reduction="sum")

        return self.loss_coef * kl_div

    @staticmethod
    def register_gradient_clipping(model, max_norm):
        """
        注册全局反向钩子，实现自动梯度裁剪
        """
        def _clip_grad(module, grad_input, grad_output):
            torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm)
        model.register_full_backward_hook(_clip_grad)

    # --- 以下为分布式辅助函数 ---
    def distributed_expert_sync(self, expert_outputs):
        """
        跨设备同步专家输出
        实际生产建议用通信库DeepSpeed MoE或FSDP封装。
        """
        pass

    def integrate_xformers(self, x):
        """
        xFormers MoE加速示范接口占位
        实际需要依赖xformers库，替换核心逻辑
        """
        # TODO: 使用xformers.sparsemoe.MoE等
        pass

    def integrate_deepspeed(self, x):
        """
        DeepSpeed MoE加速示范接口占位
        需要依赖DeepSpeed MoE组件
        """
        # TODO: 调用deepspeed.ops.moe.MoE层
        pass
