import torch
import torch.nn as nn
class DropPath(nn.Module):
    """
    Implementation from timm;
    """
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.p
        # [B, 1, 1, 1]，支持任意维度
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rnd = torch.rand(shape, dtype=x.dtype, device=x.device)
        mask = (rnd < keep_prob).float()
        return x / keep_prob * mask