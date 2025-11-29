"""
StellarByte 的前馈网络与 MoE 实现模块。

包含 FeedForward、MoEGate、MOEFeedForward。
所有函数均附带中文逐行注释以便理解。
"""

from typing import List
import math
import torch
import torch.nn.functional as F
from torch import nn
import torch.nn.init as init
from transformers.activations import ACT2FN
from config import StellarByteConfig


class FeedForward(nn.Module):
    """标准的 FFN 模块，使用 gate + activation 的结构（例如 SiLU/GELU）。

    采用 gate_proj, up_proj, down_proj 的实现顺序，以支持更高效的矩阵合并。
    """

    def __init__(self, config: StellarByteConfig):
        """初始化 FFN。

        若 config.intermediate_size 未指定，则按 8/3 规则估算并向上对齐到 64 的倍数（性能考量）。
        """
        super().__init__()
        # 若未指定中间层大小则按启发规则计算并对齐到 64 的倍数
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)

        # gate_proj: 用于 gate 分支的线性层（W1）
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        # down_proj: 将中间层投影回隐藏维（W3）
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        # up_proj: 用于激活分支的线性层（W2）
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        # 获取激活函数（如 silu/gelu）
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向计算：采用 gate * activation 的结构，然后投影回 hidden_size 并 dropout。

        计算顺序说明（便于合并计算）：
        1. gate = W1(x)
        2. up = W2(x)
        3. out = W3(gate * act(up))
        """
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module):
    """MoE 的 gating 网络：根据 hidden state 计算专家分配概率并选 top-k。

    该实现同时支持训练时的 auxiliary loss（load balancing），
    可选择在序列级别或扁平化上计算辅助损失。
    """

    def __init__(self, config: StellarByteConfig):
        super().__init__()
        self.config = config
        # top_k：每个 token 路由到多少个专家
        self.top_k = config.num_experts_per_tok
        # 总专家数量
        self.n_routed_experts = config.n_routed_experts
        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux
        self.norm_topk_prob = config.norm_topk_prob
        # gating 的输入维度通常为 hidden_size
        self.gating_dim = config.hidden_size
        # gating 权重矩阵（num_experts x gating_dim）
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """使用 kaiming 初始化 gating 权重。"""
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor):
        """执行 gating：返回 topk 的索引、概率与辅助损失。

        Args:
            hidden_states: 形状 [batch, seq_len, hidden]

        Returns:
            topk_idx: 扁平化后的 topk 专家索引，形状 [batch*seq_len, top_k]
            topk_weight: 对应的 topk 权重
            aux_loss: 辅助负载平衡损失（训练时）或 0（推理）
        """
        bsz, seq_len, h = hidden_states.shape
        # 扁平化到 [batch*seq_len, hidden]
        hidden_states_flat = hidden_states.view(-1, h)
        # logits = hidden_states_flat @ weight.T
        logits = F.linear(hidden_states_flat, self.weight, None)
        # 计算分数（目前仅支持 softmax）
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f"insupportable scoring function for MoE gating: {self.scoring_func}")

        # 取 top_k 概率与索引
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # 若 top_k > 1 可选对 topk 权重进行标准化
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        aux_loss = 0.0
        # 训练时计算 auxiliary load balancing loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # topk_idx_for_aux_loss: 变形为 [batch, seq_len*topk]
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                # 序列级别辅助 loss
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device))
                ce.div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                # 扁平化级别的辅助 loss
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha

        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    """基于专家的 MoE 前馈层实现。

    说明：
    - training 模式下使用分派并行（按 token 分配到专家）
    - 推理模式下使用排序+批量专家推理以提高效率
    """

    def __init__(self, config: StellarByteConfig):
        super().__init__()
        self.config = config
        # 构建专家模块列表
        self.experts = nn.ModuleList([FeedForward(config) for _ in range(config.n_routed_experts)])
        # gating 模块
        self.gate = MoEGate(config)
        # 若配置存在共享专家，则也构建共享专家
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([FeedForward(config) for _ in range(config.n_shared_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MoE 前向：根据 gating 将 tokens 分配给专家并组合输出。

        训练时（self.training=True）会使用 repeat_interleave 的简单实现。
        推理时会调用 moe_infer 做高效批量专家推理。
        """
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape

        # gating 输出 topk 专家索引和权重
        topk_idx, topk_weight, aux_loss = self.gate(x)
        # 将输入扁平化为 [batch*seq_len, hidden]
        x_flat = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)

        if self.training:
            # 训练：对每个 token 重复 top_k 次，分批传入各专家
            x_rep = x_flat.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            # 创建输出缓存（使用 fp16 以节省内存/计算）
            y = torch.empty_like(x_rep, dtype=torch.float16)
            # 逐个专家调用（单卡场景可接受；多卡需优化路由）
            for i, expert in enumerate(self.experts):
                mask = flat_topk_idx == i
                if mask.any():
                    y[mask] = expert(x_rep[mask]).to(y.dtype)
            # 将 y 重塑回 topk 维度并按权重求和
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            # 推理：使用高效排序批量专家推理
            y = self.moe_infer(x_flat, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)

        # 将共享专家的输出累加（若存在）
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)

        # 将 auxiliary loss 暂存到模块属性，供上层汇总
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x: torch.Tensor, flat_expert_indices: torch.Tensor, flat_expert_weights: torch.Tensor) -> torch.Tensor:
        """推理时的 MoE 批量专家调用实现。

        实现要点：按专家分组 token，批量调用对应专家，然后将结果 scatter 回原始位置。
        """
        # 创建缓存用于累积专家输出，形状与 x 相同
        expert_cache = torch.zeros_like(x)
        # argsort 得到按专家分组后的索引顺序
        idxs = flat_expert_indices.argsort()
        # bincount 得到各专家的 token 数量并做累加以得到分割点
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # token_idxs：原位置索引（每个位置代表一个 token）
        token_idxs = idxs // self.config.num_experts_per_tok

        # 按专家批量调用
        prev = 0
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = prev
            prev = end_idx
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # 按权重加权
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # scatter_add 将结果写回原始位置
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache

# 测试 FFN 和 MoE 模块
if __name__ == "__main__":
    print("=== 测试 FFN 和 MoE 模块 ===")
    
    # 创建模拟配置
    class TestConfig:
        hidden_size = 256
        intermediate_size = 512
        hidden_act = "silu"
        dropout = 0.1
        # MoE 相关配置
        n_routed_experts = 4
        num_experts_per_tok = 2
        n_shared_experts = 1
        scoring_func = "softmax"
        aux_loss_alpha = 0.01
        seq_aux = False
        norm_topk_prob = True
    
    config = TestConfig()
    
    # 测试标准 FFN
    print("--- 测试标准 FFN ---")
    ffn = FeedForward(config)
    x = torch.randn(2, 10, config.hidden_size)
    output = ffn(x)
    print(f"FFN 输入形状: {x.shape}, 输出形状: {output.shape}")
    
    # 测试 MoE Gate
    print("--- 测试 MoE Gate ---")
    gate = MoEGate(config)
    topk_idx, topk_weight, aux_loss = gate(x)
    print(f"TopK 索引形状: {topk_idx.shape}, TopK 权重形状: {topk_weight.shape}")
    print(f"辅助损失: {aux_loss}")
    
    # 测试 MoE FFN
    print("--- 测试 MoE FFN ---")
    moe_ffn = MOEFeedForward(config)
    
    # 训练模式
    moe_ffn.train()
    output_train = moe_ffn(x)
    print(f"MoE 训练模式输出形状: {output_train.shape}, 辅助损失: {moe_ffn.aux_loss}")
    
    # 推理模式
    moe_ffn.eval()
    with torch.no_grad():
        output_eval = moe_ffn(x)
    print(f"MoE 推理模式输出形状: {output_eval.shape}")
    
    print("FFN/MoE 模块测试通过!")
    