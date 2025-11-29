###############################################################################
# 文件: utils/seed.py
###############################################################################
"""\
随机种子设置工具
- 保证 numpy / random / torch 的可复现性
- 适配单卡与多卡环境
"""

import random
import numpy as np
import torch


def setup_seed(seed: int = 42, cudnn_deterministic: bool = True):
    """设置训练所需的随机种子。

    Args:
        seed (int): 种子值。
        cudnn_deterministic (bool): 是否启用 cudnn 的 deterministic 模式（可能牺牲性能）。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False