import torch

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

    def __init__(self, n_layers: int, mem_len: int, device=None):
        self.n_layers = n_layers
        self.mem_len = mem_len
        self.device = torch.device(device) if device is not None else None
        self.memory = [None for _ in range(n_layers)]

    def update(self, layer_idx: int, new_hidden: torch.Tensor):
        assert 0 <= layer_idx < self.n_layers, f"layer_idx {layer_idx} 越界"

        new_hidden = new_hidden.detach()
        if self.device is not None and new_hidden.device != self.device:
            new_hidden = new_hidden.to(self.device)

        prev_memory = self.memory[layer_idx]

        if prev_memory is not None:
            # Batch size must match
            if prev_memory.size(1) != new_hidden.size(1):
                raise RuntimeError(
                    f"Batch size mismatch in layer {layer_idx}: "
                    f"prev {prev_memory.size(1)} vs new {new_hidden.size(1)}"
                )

        if prev_memory is None:
            self.memory[layer_idx] = new_hidden
        else:
            combined = torch.cat([prev_memory, new_hidden], dim=0)
            if combined.size(0) > self.mem_len:
                combined = combined[-self.mem_len:]
            self.memory[layer_idx] = combined

    def update_all(self, new_hiddens: list[torch.Tensor]):
        assert len(new_hiddens) == self.n_layers
        for idx, new_hidden in enumerate(new_hiddens):
            self.update(idx, new_hidden)

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

    def __repr__(self):
        mem_status = []
        for i, mem in enumerate(self.memory):
            if mem is None:
                mem_status.append(f"Layer {i}: None")
            else:
                mem_status.append(
                    f"Layer {i}: shape={list(mem.shape)}, device={mem.device}"
                )
        return f"MemoryManager(n_layers={self.n_layers}, mem_len={self.mem_len}, device={self.device})\n" + "\n".join(mem_status)
