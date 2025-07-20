import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional, Tuple, Dict

try:
    from .MoERouter import ByteContextAwareRouter
    from .RMSNorm import ByteRMSNorm
except ImportError:
    from MoERouter import ByteContextAwareRouter
    from RMSNorm import ByteRMSNorm

class ExpertBlock(nn.Module):
    """专家网络模块（带门控的GLU变体）"""
    def __init__(self, 
                 hidden_size: int, 
                 ffn_dim: int, 
                 activation: str = 'gelu',
                 device: torch.device = None,
                 dtype: torch.dtype = torch.float16):
        super().__init__()
        self.device = device or torch.cuda.current_device()
        self.dtype = dtype
        
        # 门控线性层
        self.gate_proj = nn.Linear(hidden_size, ffn_dim, bias=False, device=device, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, ffn_dim, bias=False, device=device, dtype=dtype)
        
        # 激活函数
        self.act_fn = self._get_activation_fn(activation)
        
        # 下投影层
        self.down_proj = nn.Linear(ffn_dim, hidden_size, bias=False, device=device, dtype=dtype)
        
        # 归一化层
        self.norm = ByteRMSNorm(hidden_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        x = gate * up  # GLU门控
        x = self.down_proj(x)
        return x + residual  # 残差连接

    def _get_activation_fn(self, activation: str):
        if activation == "gelu":
            return nn.GELU()
        elif activation == "silu":
            return nn.SiLU()
        elif activation == "relu":
            return nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

class MoELayer(nn.Module):
    """
    MoE层设计
    - 支持专家并行(Expert Parallelism)
    - 动态容量调整
    - 容错路由机制
    - 零浪费内存管理
    
    Args:
        hidden_size: 输入特征维度
        ffn_dim: 专家网络FFN维度
        num_experts: 专家总数
        num_local_experts: 当前设备上的专家数
        router_config: 路由配置参数
        activation: 激活函数
        device: 计算设备
        dtype: 数据类型
    """
    def __init__(self,
                 hidden_size: int,
                 ffn_dim: int,
                 num_experts: int,
                 num_local_experts: int,
                 router_config: dict,
                 activation: str = 'gelu',
                 device: torch.device = None,
                 dtype: torch.dtype = torch.float16):
        super().__init__()
        self.hidden_size       = hidden_size
        self.ffn_dim           = ffn_dim
        self.num_experts       = num_experts
        self.num_local_experts = num_local_experts
        self.device            = device or torch.cuda.current_device()
        self.dtype             = dtype
        self.rank              = dist.get_rank() if dist.is_initialized() else 0
        
        # ===== 1. 专家网络 =====
        self.experts = nn.ModuleList([
            ExpertBlock(hidden_size, ffn_dim, activation, device, dtype)
            for _ in range(num_local_experts)
        ])
        
        # ===== 2. 路由系统 =====
        router_config.setdefault('hidden_size', hidden_size)
        router_config.setdefault('num_experts', num_experts)
        self.router = ByteContextAwareRouter(**router_config).to(device)
        
        # ===== 3. 通信组管理 =====
        self.expert_groups = []
        self._init_parallel_groups()
        
        # ===== 4. 内存缓冲区 =====
        max_capacity = router_config.get('max_capacity', 2048)
        self.register_buffer('expert_inputs', torch.zeros(
            num_local_experts, max_capacity, hidden_size,
            device=device, dtype=dtype
        ))
        self.register_buffer('expert_outputs', torch.zeros_like(self.expert_inputs))
        self.register_buffer('expert_counts', torch.zeros(num_local_experts, dtype=torch.long))
        
        # ===== 5. 容错机制 =====
        self.fallback_expert = ExpertBlock(hidden_size, ffn_dim//4, activation, device, dtype)
        self.dropout_rate = router_config.get('dropout', 0.1)
        
        # ===== 6. 性能监控 =====
        self.register_buffer('total_routed', torch.tensor(0, dtype=torch.long))
        self.register_buffer('total_overflow', torch.tensor(0, dtype=torch.long))

    def _init_parallel_groups(self):
        """初始化专家并行通信组"""
        if dist.is_initialized():
            # 按专家ID划分通信组
            world_size = dist.get_world_size()
            expert_world_size = world_size // self.num_local_experts
            
            # 确保设备数能被本地专家数整除
            if world_size % self.num_local_experts != 0:
                raise ValueError(f"World size {world_size} must be divisible by num_local_experts {self.num_local_experts}")
                
            self.expert_groups = []
            for i in range(self.num_local_experts):
                group_ranks = list(range(i * expert_world_size, (i+1) * expert_world_size))
                self.expert_groups.append(dist.new_group(group_ranks))
        else:
            self.expert_groups = [None] * self.num_local_experts

    def forward(self, 
                hidden_states: torch.Tensor,
                positions: Optional[torch.Tensor] = None,
                prev_context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """
        MoE层前向传播
        
        Args:
            hidden_states: 输入张量 [B, S, H]
            positions: 位置编码 [B, S]
            prev_context: 上一层路由上下文 [B, C]
            
        Returns:
            output: 输出张量 [B, S, H]
            metrics: 路由统计指标
        """
        B, S, H = hidden_states.shape
        self.expert_counts.fill_(0)  # 重置专家计数器
        
        # ===== 1. 路由决策 =====
        dispatch_info, aux_loss, next_context = self.router(
            hidden_states, positions, prev_context
        )
        
        # ===== 2. 生成分发计划 =====
        dispatch_plan = self.router.generate_dispatch_plan(dispatch_info)
        
        # ===== 3. 收集输入数据 =====
        self._gather_inputs(hidden_states, dispatch_plan)
        
        # ===== 4. 并行专家计算 =====
        self._compute_experts()
        
        # ===== 5. 分发计算结果 =====
        output = self._scatter_outputs(hidden_states, dispatch_plan, B, S, H)
        
        # ===== 6. 处理溢出token =====
        if dispatch_info['overflow_mask'].any():
            overflow_output = self._handle_overflow(
                hidden_states, dispatch_info['overflow_mask']
            )
            # 确保overflow_mask形状正确
            overflow_mask = dispatch_info['overflow_mask'].view(B, S)
            output[overflow_mask] = overflow_output
        
        # ===== 7. 收集性能指标 =====
        metrics = self._collect_metrics(dispatch_info, aux_loss, B*S)
        
        return output, metrics

    def _gather_inputs(self, 
                      hidden_states: torch.Tensor, 
                      plan: Dict):
        """收集需要发送给专家的输入token"""
        token_idx = plan['token_idx']         # 全局token索引
        expert_idx = plan['expert_idx']       # 目标专家ID
        positions = plan['positions']         # 在专家缓冲区的位置
        
        # 仅处理本地专家
        local_mask = (expert_idx // self.num_local_experts) == self.rank
        local_experts = expert_idx[local_mask] % self.num_local_experts
        local_tokens = token_idx[local_mask]
        local_positions = positions[local_mask]
        
        # 展平hidden_states以便索引
        flat_hidden = hidden_states.view(-1, self.hidden_size)
        
        # 填充专家输入缓冲区
        for e, p, t in zip(local_experts, local_positions, local_tokens):
            self.expert_inputs[e, p] = flat_hidden[t]
            self.expert_counts[e] += 1

        # 跨设备通信
        if dist.is_initialized() and dist.get_world_size() > 1:
            self._alltoall_communication(plan, flat_hidden)

    def _alltoall_communication(self, plan: Dict, flat_hidden: torch.Tensor):
        """处理跨设备的专家通信"""
        world_size = dist.get_world_size()
        send_buffers = [torch.empty(0, device=self.device) for _ in range(world_size)]
        recv_buffers = [torch.empty(0, device=self.device) for _ in range(world_size)]
        
        # 组织发送数据
        for expert_id, token_idx, pos in zip(plan['expert_idx'], plan['token_idx'], plan['positions']):
            target_rank = expert_id // self.num_local_experts
            if target_rank == self.rank:
                continue
                
            token_data = torch.cat([
                token_idx.unsqueeze(0).float(),
                pos.unsqueeze(0).float(),
                flat_hidden[token_idx].unsqueeze(0)
            ], dim=-1)
            
            if send_buffers[target_rank].numel() == 0:
                send_buffers[target_rank] = token_data
            else:
                send_buffers[target_rank] = torch.cat([send_buffers[target_rank], token_data])
        
        # 执行All-to-All通信
        dist.all_to_all(recv_buffers, send_buffers)
        
        # 处理接收数据
        for data in recv_buffers:
            if data.numel() > 0:
                tokens = data[:, :2].long()
                expert_id = tokens[:, 0]
                positions = tokens[:, 1]
                inputs = data[:, 2:]
                
                for e, p, x in zip(expert_id, positions, inputs):
                    local_e = e % self.num_local_experts
                    self.expert_inputs[local_e, p] = x
                    self.expert_counts[local_e] += 1

    def _compute_experts(self):
        """并行执行本地专家计算"""
        for e in range(self.num_local_experts):
            count = self.expert_counts[e].item()
            if count > 0:
                inputs = self.expert_inputs[e, :count]
                self.expert_outputs[e, :count] = self.experts[e](inputs)

    def _scatter_outputs(self, 
                        base_states: torch.Tensor,
                        plan: Dict,
                        B: int, S: int, H: int) -> torch.Tensor:
        """将专家输出分发回原始位置"""
        output = base_states.clone()
        token_idx = plan['token_idx']
        expert_idx = plan['expert_idx']
        positions = plan['positions']
        weights = plan['weights']
        
        # 展平输出以便索引
        flat_output = output.view(-1, H)
        
        # 处理本地专家输出
        local_mask = (expert_idx // self.num_local_experts) == self.rank
        local_experts = expert_idx[local_mask] % self.num_local_experts
        local_tokens = token_idx[local_mask]
        local_positions = positions[local_mask]
        local_weights = weights[local_mask]
        
        for e, t, p, w in zip(local_experts, local_tokens, local_positions, local_weights):
            expert_out = self.expert_outputs[e, p]
            flat_output[t] += w * expert_out
        
        # 处理跨设备输出
        if dist.is_initialized() and dist.get_world_size() > 1:
            flat_output = self._scatter_remote_outputs(flat_output, plan)
        
        return flat_output.view(B, S, H)

    def _scatter_remote_outputs(self, flat_output: torch.Tensor, plan: Dict) -> torch.Tensor:
        """处理跨设备的输出分发"""
        world_size = dist.get_world_size()
        H = self.hidden_size
        
        # 组织发送数据
        send_requests = []
        for e, t, p, w in zip(plan['expert_idx'], plan['token_idx'], plan['positions'], plan['weights']):
            source_rank = e // self.num_local_experts
            if source_rank == self.rank:
                continue
                
            local_e = e % self.num_local_experts
            expert_out = self.expert_outputs[local_e, p]
            send_data = torch.cat([
                t.unsqueeze(0).float(),
                w.unsqueeze(0),
                expert_out.unsqueeze(0)
            ], dim=-1)
            
            # 异步发送
            req = dist.isend(send_data, dst=source_rank)
            send_requests.append(req)
        
        # 接收数据
        recv_count = 0
        while recv_count < len(send_requests):
            status = dist.Status()
            probe = dist.iprobe(status=status)
            if probe:
                src_rank = status.source_rank
                size = status.message_size()
                recv_data = torch.empty(size // 4, dtype=torch.float32, device=self.device)  # 假设float32
                dist.recv(recv_data, src=src_rank)
                
                # 处理接收数据
                recv_data = recv_data.view(-1, H + 2)  # token_idx + weight + output
                for data in recv_data:
                    t = int(data[0].item())
                    w = data[1].item()
                    out = data[2:]
                    flat_output[t] += w * out
                
                recv_count += 1
        
        # 等待所有发送完成
        for req in send_requests:
            req.wait()
        
        return flat_output

    def _handle_overflow(self,
                        hidden_states: torch.Tensor,
                        overflow_mask: torch.Tensor) -> torch.Tensor:
        """处理溢出token的降级计算"""
        self.total_overflow += overflow_mask.sum().item()
        
        # 提取溢出token
        B, S, H = hidden_states.shape
        overflow_mask_2d = overflow_mask.view(B, S)
        overflow_states = hidden_states[overflow_mask_2d]
        
        # 方案1: 使用轻量级fallback专家
        fallback_output = self.fallback_expert(overflow_states)
        
        # 方案2: 随机丢弃部分溢出token
        if self.training and self.dropout_rate > 0:
            dropout_mask = torch.rand(fallback_output.shape[0], device=fallback_output.device) < self.dropout_rate
            fallback_output[dropout_mask] = 0
        
        return fallback_output

    def _collect_metrics(self, 
                        dispatch_info: Dict, 
                        aux_loss: torch.Tensor, 
                        num_tokens: int) -> Dict:
        """收集性能指标"""
        metrics = {
            'aux_loss': aux_loss.item(),
            'router_z_loss': 0.0,  # 可添加z-loss正则
            'expert_utilization': self.router.expert_utilization.mean().item(),
            'load_imbalance': self.router.expert_load.std().item() / (self.router.expert_load.mean().item() + 1e-6),
            'overflow_rate': self.total_overflow / (self.total_routed + 1e-6)
        }
        
        # 更新全局统计
        self.total_routed += num_tokens
        
        # 专家负载分布
        for e in range(self.num_experts):
            metrics[f'expert_{e}_load'] = self.router.expert_load[e].item()
            metrics[f'expert_{e}_util'] = self.router.expert_utilization[e].item()
        
        return metrics

    @torch.no_grad()
    def rebalance_experts(self):
        """动态专家负载均衡策略"""
        util = self.router.expert_utilization
        overloaded = util > 0.9
        underloaded = util < 0.3
        
        # 过载专家分裂
        for e in overloaded.nonzero(as_tuple=False):
            self._split_expert(e.item())
        
        # 低载专家合并
        if underloaded.sum() >= 2:
            candidates = underloaded.nonzero(as_tuple=False).squeeze(1)
            self._merge_experts(candidates[:2])
        
        # 优先级重置
        self.router.expert_priority.fill_(1.0)
        self.router.expert_cold_priority.fill_(1.0)

    def _split_expert(self, expert_id: int):
        """分裂过载专家"""
        if self.num_local_experts >= 8:  # 单设备专家数上限
            return
        
        # 复制专家参数
        new_expert = ExpertBlock(
            self.hidden_size, 
            self.ffn_dim, 
            self.activation, 
            self.device, 
            self.dtype
        )
        new_expert.load_state_dict(self.experts[expert_id].state_dict())
        
        # 添加扰动
        for p in new_expert.parameters():
            p.data += 0.01 * torch.randn_like(p)
        
        # 添加到专家列表
        self.experts.append(new_expert)
        self.num_local_experts += 1
        
        # 更新路由器
        self.router.num_experts += 1
        self.router.expert_priority = torch.cat([
            self.router.expert_priority, 
            torch.tensor([1.5], device=self.device)
        ])
        
        # 记录分裂操作
        print(f"Split expert {expert_id} into new expert {self.num_local_experts-1}")

    def _merge_experts(self, expert_ids: list):
        """合并低载专家"""
        if len(expert_ids) < 2:
            return
        
        # 平均合并参数
        state_dict = {}
        for key in self.experts[expert_ids[0]].state_dict():
            params = [self.experts[i].state_dict()[key] for i in expert_ids]
            state_dict[key] = torch.stack(params).mean(dim=0)
        
        # 替换第一个专家
        self.experts[expert_ids[0]].load_state_dict(state_dict)
        
        # 移除多余专家
        for i in sorted(expert_ids[1:], reverse=True):
            del self.experts[i]
        
        # 更新路由器
        self.num_local_experts -= len(expert_ids) - 1
        self.router.num_experts -= len(expert_ids) - 1
        mask = torch.ones(self.router.num_experts, dtype=torch.bool, device=self.device)
        mask[expert_ids[1:]] = False
        self.router.expert_priority = self.router.expert_priority[mask]
        
        # 记录合并操作
        print(f"Merged experts {expert_ids} into expert {expert_ids[0]}")
