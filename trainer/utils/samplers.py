###############################################################################
# 文件: utils/samplers.py
###############################################################################
"""
采样器工具
"""

from torch.utils.data import Sampler
from typing import Iterable, List


class SkipBatchSampler(Sampler[List[int]]):
    """跳过前若干 batch 的采样器，用于训练断点恢复或预热阶段。

    行为：在内部委托一个基础 sampler（通常为 RangeSampler / SequentialSampler / RandomSampler），
    将采样索引按 batch_size 切分为 batch，然后跳过前 skip_batches 个 batch，再 yield 后续 batch。
    """

    def __init__(self, base_sampler: Iterable[int], batch_size: int, skip_batches: int = 0):
        self.base_sampler = list(base_sampler)
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.base_sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total = (len(self.base_sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total - self.skip_batches)