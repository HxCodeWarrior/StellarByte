import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class ExpertFFN(nn.Module):
    """专家前馈网络模块 (单个专家)"""
    def __init__(
        self,
        dim: int,
        d_ff: int,
        activation: str = "gelu",
        dropout: float = 0.0,
        use_layernorm: bool = False,
        residual: bool = True
    ):
        """
        初始化专家前馈网络
        Args:
            dim: 输入/输出维度
            d_ff: 前馈层中间维度
            activation: 激活函数类型 ("gelu", "relu", "swish", "silu")
            dropout: Dropout 概率
            use_layernorm: 是否使用 LayerNorm
            residual: 是否添加残差连接
        """
        super().__init__()
        self.fc1 = nn.Linear(dim, d_ff)
        self.fc2 = nn.Linear(d_ff, dim)
        self.dropout = nn.Dropout(dropout)
        self.use_layernorm = use_layernorm
        self.residual = residual
        if use_layernorm:
            self.layernorm = nn.LayerNorm(dim)
        
        # 激活函数映射
        self.act = self._get_activation(activation)
        
        # 权重初始化
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def _get_activation(self, activation: str):
        """激活函数选择器"""
        act_map = {
            "gelu": F.gelu,
            "relu": F.relu,
            "swish": lambda x: x * torch.sigmoid(x),
            "silu": F.silu  # Swish 等价函数
        }
        if activation not in act_map:
            raise ValueError(f"不支持的激活函数: {activation}")
        return act_map[activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：
        输入 -> fc1 -> 激活 -> Dropout -> fc2 -> Dropout -> (残差 + LayerNorm)
        """
        residual = x if self.residual else None
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        if residual is not None:
            x = x + residual
        if self.use_layernorm:
            x = self.layernorm(x)
        return x

class ByteMoE(nn.Module):
    """
    工业级 MoE 层 (专家并行版)
    特性:
        - 双 all_to_all 通信 (分发输入 / 组合输出)
        - top-1 / top-2 路由 (含容量裁剪)
        - 负载均衡损失 (辅助损失)
        - 自动适配单卡 / 多卡环境
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        world_size: int = 1,
        rank: int = 0,
        k: int = 1,
        capacity_factor: float = 1.25,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        """
        初始化 MoE 层
        Args:
            dim: 输入/输出维度
            hidden_dim: 前馈层隐藏维度
            num_experts: 专家总数
            world_size: 分布式进程数 (并行组大小)
            rank: 当前进程在并行组中的 rank
            k: 每个 token 路由到的专家数 (top-1 或 top-2)
            capacity_factor: 容量因子 (限制每个专家最多接收的 token 数)
            dropout: dropout 概率
            activation: 激活函数类型
        """
        super().__init__()
        assert k in (1, 2), "仅支持 top-1/top-2 路由"
        assert k <= num_experts, "k 必须小于等于专家数"

        # 保存配置参数
        self.dim = dim
        self.d_ff = hidden_dim
        self.n_experts = num_experts
        self.world_size = world_size
        self.rank = rank
        self.k = k
        self.capacity_factor = capacity_factor
        # dropout 层 (若为 0 则不使用)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 路由器：输入 token -> 专家权重 (fp32 保证数值稳定)
        self.w_gate = nn.Linear(dim, num_experts, bias=False)

        # 验证专家总数能被进程数整除 (保证分布式划分均匀)
        assert num_experts % max(world_size, 1) == 0, "专家数必须能被并行组大小整除"
        self.n_local = num_experts // max(world_size, 1)  # 当前进程持有的专家数

        # 初始化本地专家列表 (每个 rank 上存放部分专家)
        self.experts = nn.ModuleList([
            ExpertFFN(dim, hidden_dim, activation) for _ in range(self.n_local)
        ])

        # 初始化损失
        self.aux_loss = None

    @torch.no_grad()
    def _global_token_count(self, s_local: int) -> int:
        """
        计算全局 token 数 (跨卡聚合)
        Args:
            s_local: 当前进程的 token 数
        """
        if self.world_size == 1:
            return s_local
        # 将本地 token 数打包成张量
        t = torch.tensor([s_local], device=self.w_gate.weight.device, dtype=torch.long)
        # all_reduce 求和 (得到全局总数)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return int(t.item())

    def _compute_aux_loss(self, probs: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
        """
        计算负载均衡辅助损失
        Args:
            probs: [S, n_experts] 每个 token 的路由概率
            topk_idx: [S, k] 每个 token 选中的专家索引
        Returns:
            aux_loss: 标量辅助损失
        """
        S = probs.size(0)
        n_experts = probs.size(1)
        # 每个专家被选中的平均概率 (重要性)
        importance = probs.mean(dim=0)  # [n_experts]
        # 统计每个专家被 top-k 选中的次数（避免构造 S x n_experts 稠密矩阵）
        idx_flat = topk_idx.reshape(-1).long()  # [S*k]
        counts = torch.bincount(idx_flat, minlength=n_experts).to(probs.dtype)  # [n_experts]
        # 每个专家实际分配的 token 比例 (负载)
        load = counts / float(S)  # [n_experts],与 one_hot.mean(dim=0) 等价

        # 损失 = n_experts * sum(重要性 * 负载)
        aux_loss = n_experts * torch.sum(importance * load)
        return aux_loss

    def _compute_routing(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        计算路由 (token -> 专家概率分布)
        Args:
            tokens: [S, D] 输入 tokens
        Returns:
            router_probs: [S, n_experts] softmax 概率
            topk_vals: [S, k] 选择的专家概率
            topk_idx: [S, k] 选择的专家索引
        """
        # 线性映射到专家 logits
        router_logits = self.w_gate(tokens.to(torch.float32))  # [S, n_experts]
        # 转换为概率分布
        router_probs = F.softmax(router_logits, dim=-1).to(tokens.dtype)
        # 取 top-k 专家
        topk_vals, topk_idx = torch.topk(router_probs, k=self.k, dim=-1)
        return router_probs, topk_vals, topk_idx.long()

    def _dispatch_tokens(
        self,
        tokens: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_vals: torch.Tensor,
        s_all: Optional[torch.Tensor],
        offsets: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        将 tokens 分发到目标专家（支持跨卡通信），并传递完整元数据。

        元数据 `meta_int` 包含三列: 
            [目标专家本地索引, 源token本地索引, 源设备rank]
        
        参数:
            tokens: 输入token特征 [S_local, D]
            topk_idx: 每个token选中的top-k专家索引 (全局索引) [S_local, k]
            topk_vals: 每个token对应的门控值 [S_local, k]
            s_all: 各 rank 的 token 数 (长度 world_size)
            offsets: 前缀和 offsets (长度 world_size)，offsets[r] 为 rank r 的全局起始 index

        返回:
            recv_tokens:  接收到的token特征 [N_recv, D]
            recv_gate:    对应的门控值 [N_recv]
            recv_expert_local:  本地专家索引 [N_recv] (0..n_local-1)
            recv_origin_local_idx: 源设备上的本地token索引 [N_recv]
            recv_src_rank: 源设备rank [N_recv]
            send_splits_back: 各源设备发送给本rank的token数量 [world_size]
        """
        device = tokens.device
        S_local = tokens.size(0)  # 当前设备上的token数量

        # 创建本地token索引的平坦视图 [S_local * k]
        token_idx_flat_local = torch.arange(S_local, device=device).unsqueeze(-1).expand(-1, self.k).reshape(-1)
        # 展平专家索引 [S_local * k]
        expert_idx_flat = topk_idx.reshape(-1)  # 全局专家索引 (0..n_experts-1)
        # 展平门控值 [S_local * k]
        gate_flat = topk_vals.reshape(-1)

        # 计算目标rank和该rank上的本地专家索引
        if self.world_size > 1:  # 多卡情况
            # 目标rank = 全局专家索引 // 每卡专家数
            dest_rank = (expert_idx_flat // self.n_local).long()
            # 本地专家索引 = 全局专家索引 % 每卡专家数
            expert_local = (expert_idx_flat % self.n_local).long()
        else:  # 单卡情况
            dest_rank = torch.zeros_like(expert_idx_flat, dtype=torch.long)
            expert_local = expert_idx_flat.long()

        # 计算发送到每个目标rank的token数量
        if self.world_size > 1:
            # 统计发送到各rank的数量 [world_size]
            send_counts = torch.bincount(dest_rank, minlength=self.world_size).to(torch.long)
            # 全局通信：收集所有rank的send_counts
            all_send_counts = [torch.zeros_like(send_counts) for _ in range(self.world_size)]
            dist.all_gather(all_send_counts, send_counts)  # 集合通信
            # 计算本rank将从各src rank接收的数量
            recv_counts = torch.stack(all_send_counts, dim=0)[:, self.rank].to(torch.long)
        else:  # 单卡情况
            send_counts = torch.tensor([gate_flat.size(0)], device=device, dtype=torch.long)
            recv_counts = torch.tensor([gate_flat.size(0)], device=device, dtype=torch.long)

        # 根据目标rank排序 (便于后续连续内存操作)
        order = torch.argsort(dest_rank)
        token_idx_o_local = token_idx_flat_local[order]  # 排序后的本地token索引
        expert_local_o = expert_local[order]             # 排序后的本地专家索引
        gate_o = gate_flat[order]                        # 排序后的门控值

        # 准备发送的token特征 [N_send, D]
        send_tokens = tokens[token_idx_o_local]
        # 构造负载：特征 + 门控值 (最后一列) [N_send, D+1]
        send_payload = torch.cat([send_tokens, gate_o.unsqueeze(-1)], dim=-1)

        # 使用 offsets[self.rank] 生成 global token id
        # offsets 必须来自 forward 的 all_gather 结果
        # 如果没有提供 offsets，则无法计算 token 的全局唯一索引
        if offsets is None:
            raise ValueError("offsets must be provided to _dispatch_tokens (use forward to compute s_all and offsets)")
        # 如果没有提供 s_all，也无法知道每个 rank 的 token 数量，从而无法计算全局索引
        if s_all is None:
            raise ValueError("s_all must be provided to _dispatch_tokens")
        s_all = s_all.to(device)
        # offsets_all: 每个 rank 的全局起始 token index
        offsets_all = torch.cat([torch.tensor([0], device=device), s_all.cumsum(0)[:-1]])

        # token_idx_o_local 是当前 rank 排序后的本地 token 索引 [0..S_local-1]
        # global_idx_o = offsets_all[self.rank] + token_idx_o_local
        # 计算每个 token 的全局唯一索引 (全局 token id)
        # 这样可以保证不同 rank 的 token 在全局序列中不会冲突
        global_idx_o = offsets_all[self.rank] + token_idx_o_local

        meta_int = torch.stack([
            expert_local_o.to(torch.long),   # 目标专家本地索引
            global_idx_o.to(torch.long),     # 源token本地索引
        ], dim=-1).to(torch.int64)  # [N_send, 3]

        # 将分割数量转为Python列表 (用于all_to_all通信)
        in_splits = send_counts.tolist()    # 发送到各rank的数量
        out_splits = recv_counts.tolist()   # 从各rank接收的数量
        recv_total = sum(out_splits)        # 总接收量

        # 多卡通信处理
        if self.world_size > 1:
            # 准备接收缓冲区
            recv_payload = torch.empty(recv_total, tokens.size(-1) + 1, device=device, dtype=tokens.dtype)
            recv_meta = torch.empty(recv_total, 2, device=device, dtype=torch.int64)

            # 第一步：传输特征+门控值
            dist.all_to_all_single(
                recv_payload, 
                send_payload,
                output_split_sizes=out_splits,
                input_split_sizes=in_splits
            )

            # 第二步：传输元数据
            dist.all_to_all_single(
                recv_meta, 
                meta_int,
                output_split_sizes=out_splits,
                input_split_sizes=in_splits
            )

            # 拆分接收的数据
            recv_tokens = recv_payload[:, :-1]   # 特征部分 [N_recv, D]
            recv_gate = recv_payload[:, -1]       # 门控值 [N_recv]
            recv_expert_local = recv_meta[:, 0].long()       # 专家本地索引
            recv_origin_local_idx = recv_meta[:, 1].long()   # 源token本地索引

            # 记录各源rank发送到本rank的数量 (用于后续聚合)
            send_splits_back = recv_counts  # [world_size]

        # 单卡处理 (无通信)
        else:
            recv_tokens = send_tokens
            recv_gate = gate_o
            recv_expert_local = expert_local_o
            recv_origin_local_idx = global_idx_o
            send_splits_back = torch.tensor([recv_tokens.size(0)], device=device, dtype=torch.long)

        return recv_tokens, recv_gate, recv_expert_local, recv_origin_local_idx, send_splits_back

    def _local_expert_computation(
        self,
        tokens: torch.Tensor,
        expert_idx: torch.Tensor,
        gate_vals: torch.Tensor,
        origin_idx: torch.Tensor,
        capacity: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        在本地对分配到本设备的tokens进行专家计算，并进行容量裁剪。

        本函数处理以下核心任务：
        1. 将tokens按分配的专家分组
        2. 对每个专家的token进行容量裁剪（保留前capacity个）
        3. 调用专家网络处理保留的tokens
        4. 返回处理结果及关联元数据

        参数:
            tokens: 输入token特征 [N, D] (N: token数量, D: 特征维度)
            expert_idx: 每个token分配的本地专家索引 [N] (取值范围: 0..n_local-1)
            gate_vals: 每个token对应的门控值 [N] (路由权重)
            origin_idx: token在源设备上的本地索引 [N]
            src_rank: token来源的设备rank [N]
            capacity: 每个专家最多处理的token数 (容量上限)

        返回:
            expert_output_kept: 专家处理后的特征 [N_keep, D] (N_keep: 保留的token数)
            gate_kept: 保留token的门控值 [N_keep]
            origin_kept: 保留token在源设备上的本地索引 [N_keep]

            特殊返回:
                (None, None, None) - 当输入为空或无token被保留时
        """
        # 获取当前设备信息
        device = tokens.device

        # 边界情况处理：无输入token时直接返回None
        if tokens.numel() == 0:
            return None, None, None, None

        # Step 1: 按专家索引排序，使同一专家的token连续存储
        # 排序后，同一专家的token在内存中连续，便于后续分段处理
        perm = torch.argsort(expert_idx)  # 获取排序索引
        tokens_sorted = tokens[perm]       # 排序后的token特征
        expert_idx_sorted = expert_idx[perm]  # 排序后的专家索引
        gate_sorted = gate_vals[perm]      # 排序后的门控值
        origin_sorted = origin_idx[perm]   # 排序后的源索引

        # Step 2: 统计每个专家分配的token数量
        # counts[i] = 分配给专家i的token数量
        counts = torch.bincount(expert_idx_sorted, minlength=self.n_local)

        # Step 3: 计算每个专家token段的起始偏移量
        # 示例: counts = [3, 2, 4] → offsets = [0, 3, 5]
        offsets = torch.cat([
            torch.tensor([0], device=device, dtype=torch.long),  # 起始偏移为0
            counts.cumsum(0)[:-1]  # 累积和并去掉最后一个元素
        ])

        # Step 4: 容量裁剪 - 确定每个专家保留的token索引
        keep_indices = []  # 存储所有保留token的全局索引
        for i in range(self.n_local):  # 遍历每个本地专家
            # 计算当前专家token段的起始和结束位置
            start = int(offsets[i].item())
            end = start + int(counts[i].item())
            n_expert_tokens = end - start  # 当前专家的token总数

            # 确定保留数量 (不超过容量上限)
            n_keep = min(n_expert_tokens, capacity)

            # 如果有token需要保留，记录其索引
            if n_keep > 0:
                # 保留当前专家token段的前n_keep个token
                # 注意: 这里是顺序保留，实际应用中可能需要按门控值排序
                expert_keep = torch.arange(start, start + n_keep, device=device)
                keep_indices.append(expert_keep)

        # 边界情况处理：无token被保留时返回None
        if not keep_indices:
            return None, None, None, None

        # Step 5: 合并所有保留token的索引
        keep_indices = torch.cat(keep_indices)

        # Step 6: 从排序后的数组中提取保留的token及相关信息
        tokens_kept = tokens_sorted[keep_indices]      # 保留的token特征
        expert_idx_kept = expert_idx_sorted[keep_indices]  # 保留token的专家索引
        gate_kept = gate_sorted[keep_indices]          # 保留token的门控值
        origin_kept_global = origin_sorted[keep_indices]      # 保留token的源索引

        # Step 7: 初始化专家输出张量 (全零，与输入特征维度相同)
        expert_output = torch.zeros_like(tokens_kept)

        # Step 8: 重新统计裁剪后各专家的token数量
        kept_counts = torch.bincount(expert_idx_kept, minlength=self.n_local)

        # Step 9: 计算裁剪后各专家的偏移量
        kept_offsets = torch.cat([
            torch.tensor([0], device=device, dtype=torch.long),  # 起始偏移
            kept_counts.cumsum(0)[:-1]  # 累积和（去掉最后一个元素）
        ])

        # Step 10: 逐专家进行前向计算
        for i in range(self.n_local):  # 遍历每个本地专家
            # 计算当前专家在保留token中的起止位置
            start = int(kept_offsets[i].item())
            end = start + int(kept_counts[i].item())

            # 检查当前专家是否有需要处理的token
            if end > start:
                # 提取当前专家的token段
                expert_tokens = tokens_kept[start:end]

                # 调用专家网络进行前向计算
                expert_output[start:end] = self.experts[i](expert_tokens)

        # Step 11: 返回计算结果及关联元数据
        return expert_output, gate_kept, origin_kept_global


    def _combine_results(
        self,
        expert_out: torch.Tensor,   
        gate_vals: torch.Tensor,    
        origin_global_idx: torch.Tensor,
        send_splits_back: torch.Tensor,
        s_all: Optional[torch.Tensor],
        offsets: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        将本地专家计算得到的输出按源rank分组并回传给对应的源rank。

        本函数处理分布式MoE（Mixture of Experts）中专家计算结果回传阶段：
        1. 对计算结果按源rank排序分组
        2. 通过All-to-All通信将计算结果返回给原始请求的rank
        3. 重组接收到的计算结果并返回索引信息

        参数说明：
            expert_out: 专家层计算后的输出张量，形状为 [N_keep, D]
            gate_vals: 对应token的门控值，形状为 [N_keep]
            origin_global_idx: token在源rank上的本地索引，形状为 [N_keep]
            send_splits_back: [world_size] 各源rank发送到本rank的token数量（由 _dispatch_tokens 提供）
            s_all: 各rank的token计数（可选，仅用于日志记录或校验）
            offsets: 分组偏移量（可选，仅用于日志记录或校验）

        返回元组：
            combined_out: 重组后的输出张量，形状为 [N_returned, D]
            combined_gate: 重组后的门控值，形状为 [N_returned]
            origin_local_idx: 当前rank上原始token的本地索引，形状为 [N_returned]
                该索引用于后续的index_add操作，将专家输出写回正确位置

        注意事项：
            - 返回的origin_local_idx是相对于当前rank的本地索引
            - 在分布式环境中，本函数通过All-to-All通信实现数据交换
            - 单机环境直接返回输入数据（无通信）
        """
        device     = expert_out.device  # 获取设备信息
        world_size = self.world_size  # 获取分布式设备数量

        # === 单机处理 ===
        if self.world_size == 1:
            # 单机环境无需通信，直接返回原始数据
            return expert_out, gate_vals, origin_global_idx

        # === 数据排序：按源rank分组 ===
        # Step 1: 根据 s_all 构造每个源 rank 的 token区间
        # offsets[r]: rank r token 全局起始 idx
        # s_all[r]: rank r token 数量
        rank_intervals = [(offsets[r], offsets[r] + s_all[r]) for r in range(world_size)]

        # Step 2: 对每个 token 确定源 rank
        src_rank = torch.zeros_like(origin_global_idx, device=device)
        for r, (start, end) in enumerate(rank_intervals):
            mask = (origin_global_idx >= start) & (origin_global_idx < end)
            src_rank[mask] = r


        # 对src_rank进行排序，使相同rank的token连续存放
        order = torch.argsort(src_rank)
        expert_out_sorted = expert_out[order]      # 按源rank排序的专家输出
        gate_sorted = gate_vals[order]             # 按源rank排序的门控值
        origin_sorted = origin_global_idx[order]   # 按源rank排序的原始索引
        src_sorted = src_rank[order]               # 排序后的源rank列表

        # === 通信计数计算 ===
        # 计算发往每个rank的token数量（send_counts_back）
        send_counts_back = send_splits_back.tolist()  # 当前rank发送到各rank的块大小

        # 全局通信：收集所有rank的发送计数
        all_send_counts = [
            torch.zeros_like(send_counts_back) 
            for _ in range(self.world_size)
        ]
        dist.all_gather(all_send_counts, send_counts_back)

        # 提取当前rank将接收的token数量（recv_counts_back）
        recv_counts_back = torch.stack(all_send_counts, dim=0)[:, self.rank].tolist()

        # === 数据打包 ===
        # 合并专家输出和门控值（特征维度拼接）
        # 形状 [N_send_back, D+1]
        send_payload = torch.cat(
            [expert_out_sorted, gate_sorted.unsqueeze(-1)], 
            dim=-1
        ) # [N_send_back, D + 1]

        # 准备元数据（原始本地索引）
        send_meta = origin_sorted.to(torch.int64).unsqueeze(-1)  # [N_send_back, 1]

        # === All-to-All 通信 ===
        # 计算接收数据总长度
        recv_total_back = sum(recv_counts_back)

        # 初始化接收缓冲区
        recv_payload = torch.empty(
            recv_total_back, 
            self.dim + 1, 
            device=device, 
            dtype=expert_out.dtype
        )
        recv_meta = torch.empty(
            recv_total_back, 
            1, 
            device=device, 
            dtype=torch.int64
        )

        # 执行负载数据通信（专家输出+门控值）
        dist.all_to_all_single(
            output_tensor=recv_payload,
            input_tensor=send_payload,
            output_split_sizes=recv_counts_back,  # 各rank输出块大小
            input_split_sizes=send_counts_back    # 各rank输入块大小
        )

        # 执行元数据通信（原始索引）
        dist.all_to_all_single(
            output_tensor=recv_meta,
            input_tensor=send_meta,
            output_split_sizes=recv_counts_back,
            input_split_sizes=send_counts_back
        )

        # === 数据解包 ===
        combined_out = recv_payload[:, :-1]  # 拆分专家输出 [N_recv, D]
        combined_gate = recv_payload[:, -1]  # 拆分门控值 [N_recv]
        origin_global_recv = recv_meta[:, 0].long()  # 转换为长整型索引 [N_recv]

        # 将 global idx 转为当前 rank 的 local idx: local_idx = global_idx - offsets[self.rank]
        local_offset = offsets[self.rank].to(device).long()
        origin_local_idx = origin_global_recv - local_offset

        return combined_out, combined_gate, origin_local_idx


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MoE（Mixture of Experts）层的前向传播

        参数:
            x: 输入张量，形状为 [B, T, D]（每个计算设备上的本地批次）
                B - 批次大小
                T - 序列长度/时间步长
                D - 特征维度

        返回:
            output: 输出张量，形状与输入相同 [B, T, D]
            aux_loss: 辅助损失张量（保留梯度，可直接加到主损失）
        """
        # 获取输入张量的形状和设备信息
        B, T, D = x.shape
        device = x.device

        # 将输入展平为二维张量 [总令牌数, 特征维度]
        # [B, T, D] -> [B*T, D] = [S_local, D]
        tokens = x.view(-1, D)

        # 1. 路由计算：确定每个token分配给哪些专家
        # router_probs: 路由概率 [S_local, n_experts]
        # topk_vals: 每个token选择的前k个专家的权重 [S_local, k]
        # topk_idx: 每个token选择的前k个专家的索引 [S_local, k]
        router_probs, topk_vals, topk_idx = self._compute_routing(tokens)

        # 计算辅助损失（负载均衡损失）
        aux_loss = self._compute_aux_loss(router_probs, topk_idx)

        # 分布式环境下的辅助损失处理（用于日志记录）
        if self.world_size > 1:  # 多设备情况
            # 复制分离的辅助损失用于全局聚合
            aux_loss_log = aux_loss.detach().clone()
            # 全局求和
            dist.all_reduce(aux_loss_log, op=dist.ReduceOp.SUM)
            # 计算全局平均值
            aux_loss_log = aux_loss_log / float(self.world_size)
            # 存储日志值（标量）
            self.latest_aux_loss_for_log = float(aux_loss_log.item())
        else:  # 单设备情况
            self.latest_aux_loss_for_log = float(aux_loss.detach().item())

        # 2. 容量计算：确定每个专家处理的最大token数
        # 计算全局token总数（分布式环境下跨所有设备）
        S_global = self._global_token_count(tokens.size(0))
        # 计算每个专家的容量（考虑容量因子）
        capacity = max(1, math.ceil(
            S_global * self.k / self.n_experts * self.capacity_factor
        ))

        # 3. 分布式通信准备：收集各设备的token数量信息
        if self.world_size > 1:  # 多设备情况
            # 创建本地token数量张量
            s_local = torch.tensor([tokens.size(0)], device=device, dtype=torch.long)
            # 准备接收缓冲区
            s_all_list = [torch.zeros_like(s_local) for _ in range(self.world_size)]
            # 全局收集所有设备的token数量
            dist.all_gather(s_all_list, s_local)
            # 组合为全局token计数向量 [world_size]
            s_all = torch.stack(s_all_list).view(-1)
            # 计算各设备在全局缓冲区中的偏移量
            offsets = torch.cat([
                torch.tensor([0], device=device, dtype=torch.long),
                s_all.cumsum(0)[:-1]  # 累加并去掉最后一个元素
            ])
        else:  # 单设备情况
            s_all = torch.tensor([tokens.size(0)], device=device, dtype=torch.long)
            offsets = torch.tensor([0], device=device, dtype=torch.long)

        # 4. 令牌分发：根据路由结果分配token到专家
        # recv_tokens: 接收到的令牌数据 [总接收令牌数, D]
        # recv_gate: 对应的门控值 [总接收令牌数]
        # recv_expert: 分配的专家索引 [总接收令牌数]
        # recv_origin_global_idx: 原始本地索引 [总接收令牌数]
        # send_splits_back: 返回发送给各设备的令牌数量列表
        recv_tokens, recv_gate, recv_expert, recv_origin_global_idx, send_splits_back = \
            self._dispatch_tokens(
                tokens, topk_idx, topk_vals, 
                s_all, offsets
            )

        # 5. 本地专家计算：处理分配到的令牌
        # expert_out: 专家输出 [已处理令牌数, D]
        # gate_kept: 保留的门控值 [已处理令牌数]
        # origin_kept_global: 保留的原始索引 [已处理令牌数]
        expert_out, gate_kept, origin_kept_global = self._local_expert_computation(
            recv_tokens, recv_expert, recv_gate, 
            recv_origin_global_idx, capacity
        )

        # 处理所有token被裁剪的情况（超出专家容量）
        if expert_out is None:
            # 创建零输出（保持原始形状）
            output = torch.zeros_like(tokens).view(B, T, D)
            # 应用dropout后返回（此时aux_loss可能非零）
            return self.dropout(output), aux_loss

        # 6. 结果组合：收集并准备回传结果
        # combined_out: 组合后的专家输出 [待发送令牌数, D]
        # combined_gate: 组合后的门控值 [待发送令牌数]
        # origin_idx_local: 本地重组后的原始索引 [待发送令牌数]
        combined_out, combined_gate, origin_idx_local = self._combine_results(
            expert_out, gate_kept, origin_kept_global, 
            send_splits_back, s_all, offsets
        )

        # 7. 聚合输出：合并同一token来自多个专家的结果
        # 初始化输出缓冲区和权重累加器
        combined = torch.zeros(tokens.size(0), D, device=device, dtype=tokens.dtype)
        weight_sum = torch.zeros(tokens.size(0), device=device, dtype=tokens.dtype)

        # 加权聚合：对同一原始token的输出进行加权求和
        # combined_out * combined_gate.unsqueeze(-1): 加权输出
        combined.index_add_(0, origin_idx_local, combined_out * combined_gate.unsqueeze(-1))

        # 累加权重
        weight_sum.index_add_(0, origin_idx_local, combined_gate)

        # 归一化处理（避免除零）
        mask = weight_sum > 0  # 找到有权重的token位置
        combined[mask] /= weight_sum[mask].unsqueeze(-1)  # 归一化

        # 恢复原始形状 [B, T, D]
        output = combined.view(B, T, D)

        # 应用dropout后返回结果和辅助损失
        return self.dropout(output), aux_loss

if __name__ == "__main__":
    # 单卡测试配置
    dim = 128
    hidden_dim = 32
    n_experts = 4
    batch_size = 2
    seq_len = 16
    k = 2  # top-2路由
    capacity_factor = 1.25

    # 创建单卡MoE层
    moe_layer = ByteMoE(
        dim=dim,
        hidden_dim=hidden_dim,
        num_experts=n_experts,
        world_size=1,
        rank=0,
        k=k,
        capacity_factor=capacity_factor,
        dropout=0.1,
        activation="gelu"
    )

    # 创建测试输入
    x = torch.randn(batch_size, seq_len, dim)
    
    # 前向传播
    output, aux_loss = moe_layer(x)
    
    # 验证输出形状
    assert output.shape == (batch_size, seq_len, dim), "输出形状不正确"
    assert aux_loss.dim() == 0 and aux_loss.item() >= 0, "辅助损失应为非负标量"
    print(f"单卡测试通过!\nInput shape: {x.shape}, \nOutput shape: {output.shape}, \nAux Loss: {aux_loss.item():.4f}")
