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
        self._seq_lens: List[int] = [0] * self.L
        self.cache: List[Dict[str, torch.Tensor]] = []

    def layer_length(self, layer_id: int) -> int:
        return self._seq_lens[layer_id]

    def global_length(self) -> int:
        return max(self._seq_lens)

    @torch.inference_mode()
    def reset(self) -> None:
        """清空缓存内容（保留显存）"""
        if self.batch_size is None:
            return
        for buf in self.cache:
            buf["key"].zero_()
            buf["value"].zero_()
        self._seq_lens = [0] * self.L
    
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
        self._seq_lens = [0] * self.L
    
    @torch.inference_mode()
    def get(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回有效的 KV 缓存（不拷贝），shape=[B, T_valid, H_local, D]
        """
        T_valid = self._seq_lens[layer_id]
        buf = self.cache[layer_id]
        key = buf["key"][:, :T_valid]
        value = buf["value"][:, :T_valid]
        return key, value

    @torch.inference_mode()
    def append(self, layer_id: int, key: torch.Tensor, value: torch.Tensor) -> None:
        """
        追加新的 KV 对到指定层缓存，超出窗口长度时自动“滑动窗口截断”。

        参数
        ----
        key / value: [B, T_new, H_local, D]
        约束:
            1. T_new 可以任意（≥1），若 > max_seq_len 仅保留最近 max_seq_len。
            2. 不会创建新张量；只做必要的 in‑place copy。
        """
        # —— 1. 基本合法性检查 ——
        assert key.shape == value.shape, "K/V 形状不一致"
        B, T_new, H, D = key.shape
        assert H == self.H_local and D == self.D, \
            f"KV head_dim 不一致: {H} vs {self.H_local}"
        if self.batch_size is None:
            self._allocate_buffers(B)
        assert B == self.batch_size, "batch_size 不一致"

        # —— 2. 若一次输入超过窗口，就直接截断 ——
        if T_new >= self.max_T:  # 只保留最近 max_T
            key  = key[:, -self.max_T:]
            value = value[:, -self.max_T:]
            T_new = self.max_T
            overflow = self._seq_lens[layer_id]  # 全部被替换
        else:
            current_len = self._seq_lens[layer_id]
            overflow = max(0, current_len + T_new - self.max_T)

        buf = self.cache[layer_id]

        # —— 3. 若存在 overflow，则左移 (current_len - overflow) 个 token ——
        if overflow > 0:
            keep_len = current_len - overflow  # 将要保留的旧 token 数
            if keep_len > 0:  # 仅当有内容需要保留时才复制
                src = slice(overflow, overflow + keep_len)   # 旧 token
                dst = slice(0, keep_len)                    # 左移后位置
                buf["key"][:, dst].copy_(buf["key"][:, src])
                buf["value"][:, dst].copy_(buf["value"][:, src])

        # —— 4. 写入新 token ——
        start = min(self._seq_lens[layer_id], self.max_T) - overflow
        end   = start + T_new
        buf["key"][:, start:end].copy_(key)
        buf["value"][:, start:end].copy_(value)

        # —— 5. 更新长度 ——
        self._seq_lens[layer_id] = min(self._seq_lens[layer_id] + T_new, self.max_T)
