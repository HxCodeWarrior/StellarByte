import torch
from typing import List, Tuple, Dict, Optional


class KVCache:
    r"""
    高性能 KV 缓存（支持批量、混合精度、Sliding-Window、张量并行）

    Shape 约定
    -----------
    key / value  : [B, T, H_local, D]
          B      : batch_size
          T      : 当前缓存长度 ≤ max_seq_len
          H_local: 当前张量并行 rank 的头数 = H_total / tensor_parallel_size
          D      : head_dim
    """
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int,
        key_dtype: torch.dtype = torch.float16,  # 更低精度用于 Key
        value_dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cuda",
        tensor_parallel_size: int = 1,
        tensor_parallel_rank: int = 0,
    ):
        assert num_heads % tensor_parallel_size == 0, "num_heads 必须能整除并行数"
        self.L = num_layers
        self.H_total = num_heads
        self.H_local = num_heads // tensor_parallel_size
        self.D = head_dim
        self.max_T = max_seq_len

        self.key_dtype = key_dtype
        self.value_dtype = value_dtype
        self.device = device

        self.tp_size = tensor_parallel_size
        self.tp_rank = tensor_parallel_rank

        # runtime attributes（运行时确定）
        self.batch_size: Optional[int] = None
        self.length: int = 0
        self.cache: List[Dict[str, torch.Tensor]] = []

    @torch.inference_mode()
    def append(self, layer_id: int, key: torch.Tensor, value: torch.Tensor) -> None:
        """
        追加新的 KV 对到指定层缓存，自动 sliding-window。
        要求 key / value: [B, T_new, H_local, D]
        """
        assert key.shape == value.shape, "K/V 形状不一致"
        B, T_new, H, D = key.shape
        assert H == self.H_local and D == self.D, f"KV head_dim 不一致：{H} vs {self.H_local}"

        if self.batch_size is None:
            self._allocate_buffers(B)
        assert B == self.batch_size, "batch_size 不一致"

        # sliding-window
        overflow = max(0, self.length + T_new - self.max_T)
        if overflow > 0:
            for buf in self.cache:
                buf["key"] = torch.roll(buf["key"], shifts=-overflow, dims=1)
                buf["value"] = torch.roll(buf["value"], shifts=-overflow, dims=1)

        start = max(0, self.length - overflow)
        end = start + T_new
        self.cache[layer_id]["key"][:, start:end].copy_(key)
        self.cache[layer_id]["value"][:, start:end].copy_(value)
        self.length = min(self.length + T_new, self.max_T)

    @torch.inference_mode()
    def get(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回有效的 KV 缓存（不拷贝），shape=[B, T_valid, H_local, D]
        """
        key = self.cache[layer_id]["key"][:, : self.length]
        value = self.cache[layer_id]["value"][:, : self.length]
        return key, value

    @torch.inference_mode()
    def reset(self) -> None:
        """清空缓存内容（保留显存）"""
        if self.batch_size is None:
            return
        for buf in self.cache:
            buf["key"].zero_()
            buf["value"].zero_()
        self.length = 0

    @torch.inference_mode()
    def to(self, device: str | torch.device) -> "KVCache":
        """将缓存迁移到其他设备"""
        for buf in self.cache:
            buf["key"] = buf["key"].to(device)
            buf["value"] = buf["value"].to(device)
        self.device = device
        return self

    def _allocate_buffers(self, batch_size: int) -> None:
        """分配每层缓存：shape=[B, T, H_local, D]"""
        self.batch_size = batch_size
        self.cache = [
            {
                "key": torch.empty(batch_size, self.max_T, self.H_local, self.D, dtype=self.key_dtype, device=self.device),
                "value": torch.empty(batch_size, self.max_T, self.H_local, self.D, dtype=self.value_dtype, device=self.device),
            }
            for _ in range(self.L)
        ]
        self.length = 0
