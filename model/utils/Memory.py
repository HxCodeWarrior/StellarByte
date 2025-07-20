import torch
from typing import Optional, List

class ByteMemoryManager:
    """
    Transformer Memory Manager for caching past hidden states across multiple layers.

    适用场景：Transformer-XL 风格的长上下文记忆机制。

    功能：
    - 多层记忆缓存管理
    - 记忆更新（拼接+裁剪）
    - 梯度隔离 detach
    - 支持清空所有记忆
    - 支持动态设备转移
    - 支持批量更新所有层记忆

    """

    def __init__(
        self, 
        n_layers: int, 
        mem_lens: int, 
        device=None,
        batch_mismatch_policy: str = "strict"
    ):
        """
        构造函数

        Args:
            n_layers: int
            mem_lens: int
            device: str,optinal,设备类型
            batch_mismatch_policy: str
              - strict: 严格模式(默认),batch不一致时报错
              - select: 裁剪/复制模式,自动匹配batch维度
              - repeat: 广播模式,特别适合单样本推理场景
        Return:

        Functions:

        """
        self.n_layers = n_layers
        self.device = torch.device(device) if device is not None else None
        self.memory = [None for _ in range(n_layers)]
        self.fusion_weights = [0.7, 0.3]  # [旧记忆权重, 新输入权重]

        # 处理记忆长度配置
        if isinstance(mem_lens, int):
            self.mem_lens = [mem_lens] * n_layers
        else:
            assert len(mem_lens) == n_layers, "mem_lens长度需与层数一致"
            self.mem_lens = mem_lens

        # 策略验证
        assert batch_mismatch_policy in ["strict", "select", "repeat"], \
            "策略需为strict/select/repeat"
        self.batch_policy = batch_mismatch_policy

    def _match_batch_size(
        self,
        prev: torch.Tensor,
        new: torch.Tensor
    ) -> torch.Tensor:
        """智能匹配batch尺寸"""
        prev_bs, new_bs = prev.size(1), new.size(1)
        
        # 尺寸一致无需处理
        if prev_bs == new_bs:
            return prev
        
        # 根据策略处理尺寸不匹配
        if self.batch_policy == "strict":
            raise RuntimeError(
                f"Batch尺寸不匹配: 记忆{prev_bs} vs 输入{new_bs}。"
                "请调整策略或检查数据"
            )
        
        if self.batch_policy == "select":
            return prev[:, :new_bs] if prev_bs > new_bs else prev.repeat(1, new_bs, 1)[:, :new_bs]
        
        if self.batch_policy == "repeat":
            if prev_bs == 1:  # 单样本广播
                return prev.expand(-1, new_bs, -1)
            # 计算需要重复的次数
            n_repeat = (new_bs + prev_bs - 1) // prev_bs
            return prev.repeat(1, n_repeat, 1)[:, :new_bs]
    
    def _residual_fusion(
        self,
        prev_mem: torch.Tensor,
        new_input: torch.Tensor
    ) -> torch.Tensor:
        """残差加权融合新旧记忆"""
        # 拼接新输入到历史记忆
        combined = torch.cat([prev_mem, new_input], dim=0)
        
        # 当不超限时直接返回
        if combined.size(0) <= self.mem_lens[self.current_layer]:
            return combined
        
        # 计算需要保留的记忆长度
        retain_len = self.mem_lens[self.current_layer]
        prev_len = prev_mem.size(0)
        new_len = new_input.size(0)
        
        # 新旧内容融合策略
        if retain_len < new_len:  # 新输入占主导
            return new_input[-retain_len:]
        
        # 加权融合重叠部分
        overlap_len = prev_len + new_len - retain_len
        old_part = prev_mem[:prev_len - overlap_len]
        overlap_old = prev_mem[-overlap_len:]
        overlap_new = new_input[:overlap_len]
        
        # 计算加权融合
        fused_overlap = (
            self.fusion_weights[0] * overlap_old +
            self.fusion_weights[1] * overlap_new
        )
        
        return torch.cat([
            old_part,
            fused_overlap,
            new_input[overlap_len:]
        ], dim=0)

    def update(
        self, 
        layer_idx: int, 
        new_hidden: torch.Tensor,
        active_indices: Optional[torch.Tensor] = None
    ):
        assert 0 <= layer_idx < self.n_layers, f"layer_idx {layer_idx} 越界"

        self.current_layer = layer_idx  # 为融合方法提供上下文

        # 梯度隔离
        new_hidden = new_hidden.detach()

        # 设备转移
        if self.device is not None and new_hidden.device != self.device:
            new_hidden = new_hidden.to(self.device)

        prev_memory = self.memory[layer_idx]

        # 处理batch尺寸变化
        if prev_memory is not None:
            # 使用索引选择活跃序列
            if active_indices is not None:
                prev_memory = prev_memory.index_select(1, active_indices)
            # 自动尺寸适配
            elif prev_memory.size(1) != new_hidden.size(1):
                prev_memory = self._match_batch_size(prev_memory, new_hidden)
        
        # 初始记忆设置
        if prev_memory is None:
            self.memory[layer_idx] = new_hidden[-min(new_hidden.size(0), self.mem_lens[layer_idx]):]
            return
        
        # 残差融合更新记忆
        self.memory[layer_idx] = self._residual_fusion(prev_memory, new_hidden)

    def update_all(
        self, 
        new_hiddens: list[torch.Tensor],
        active_indices_list: Optional[List[torch.Tensor]] = None
    ):
        assert len(new_hiddens) == self.n_layers, "输入层数不匹配"
        for idx, h in enumerate(new_hiddens):
            indices = active_indices_list[idx] if active_indices_list else None
            self.update(idx, h, indices)

    def get(self, layer_idx: int):
        assert 0 <= layer_idx < self.n_layers, f"layer_idx {layer_idx} 越界"
        return self.memory[layer_idx]

    def clear(self):
        self.memory = [None] * self.n_layers

    def to(self, device):
        device = torch.device(device)
        self.device = device
        for i in range(self.n_layers):
            if self.memory[i] is not None:
                self.memory[i] = self.memory[i].to(device)

    def memory_size(self):
        """
        返回每层当前记忆长度（时间步数）
        """
        return [mem.size(0) if mem is not None else 0 for mem in self.memory]
    
    def configure_fusion_weights(self, weights: List[float]):
        """配置记忆融合权重[旧记忆权重, 新输入权重]"""
        assert len(weights) == 2, "需提供两个权重值"
        assert abs(sum(weights) - 1.0) < 1e-5, "权重和应为1.0"
        self.fusion_weights = weights

    def __repr__(self):
        status = []
        for i, mem in enumerate(self.memory):
            mem_len = self.mem_lens[i]
            curr_len = mem.size(0) if mem is not None else 0
            status.append(
                f"层{i}: {curr_len}/{mem_len}步 | "
                f"设备={mem.device if mem is not None else 'None'} | "
                f"形状={list(mem.shape) if mem is not None else '[]'}"
            )
        return (
            f"ByteMemoryManager(层数={self.n_layers}, 策略={self.batch_policy})\n"
            + "\n".join(status)
        )
