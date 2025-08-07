import swanlab
import os
import platform
import argparse
import time
import warnings
import math
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
from contextlib import nullcontext
from transformers import AutoTokenizer
from tqdm import tqdm
from model.Model import ByteModel
from model.config import ByteModelConfig
from datasets import PretrainDataset


def set_environment():
    pass

def cosine_annealing_lr(
    current_step: int, 
    total_steps: int, 
    max_lr: float = 1e-3,
    min_lr: float = 1e-6,
    warmup_steps: int = 0,
    hold_steps: int = 0,
    restart_step: int = None,
    decay_ratio: float = None
) -> float:
    """
    余弦退火学习率调度器，支持热身阶段和周期性重启

    阶段划分:
      1. 线性预热 (0 → max_lr)
      2. 稳定平台 (保持 max_lr)
      3. 余弦退火 (max_lr → min_lr)
      4. 最终衰减 (min_lr → final_lr)
    
    Args:
        current_step (int): 当前训练步数（从0开始计数）
        total_steps (int): 总训练步数
        max_lr (float): 最大学习率（余弦退火起点）
        min_lr (float): 最小学习率（余弦退火终点）,max_lr 的 30 %
        warmup_steps (int): 热身阶段步数（学习率线性增长）
        hold_steps (int): 最大学习率保持步数（从第warmup_steps步开始）
        restart_step (int): 周期性重启步长（None表示不重启）
        decay_ratio (float): 衰减阶段比例 (0.0-1.0, 覆盖total_steps)
    
    Return:
        float: 当前步对应的学习率
    
    Error:
        ValueError: 当输入参数不合法时抛出
    """
    # 参数校验
    if total_steps <= 0:
        raise ValueError("otal_step必须为正整数")
    if current_step < 0:
        raise ValueError("current_step不能为负数")
    if warmup_steps < 0:
        raise ValueError("warmup_steps不能为负数")
    if min_lr < 0 or max_lr < min_lr:
        raise ValueError(f"无效的学习率范围: min_lr={min_lr}, max_lr={max_lr}")
    
    # 使用decay_ratio动态计算退火阶段
    decay_end_step = total_steps
    if decay_ratio:
        decay_start_step = total_steps - int(total_steps * decay_ratio)
    else:
        decay_start_step = warmup_steps + hold_steps

    # 处理周期性重启逻辑
    effective_step = current_step
    effective_total = total_steps
    if restart_step is not None and restart_step > 0:
        # 当前在第几个周期（从0开始）
        cycle = effective_step // restart_step
        # 当前周期内的步数
        effective_step = current_step % restart_step
        # 动态调整周期内总步数 (防止最后一周期溢出)
        effective_total = min(restart_step, total_steps - cycle * restart_step)
        # 重启时重置衰减起始点
        decay_start_step = min(decay_start_step, effective_total)
    
    # 1. 热身阶段处理
    if warmup_steps > 0 and effective_step < warmup_steps:
        # 线性热身：从0到max_lr
        return max_lr * (effective_step + 1) / warmup_steps
    
    # 2. 稳定平台阶段 返回最大学习率
    if effective_step < decay_start_step:
        return max_lr
    
    # 3. 余弦退火阶段核心计算
    # 计算当前进度比例（热身阶段之后）
    progress = (effective_step - warmup_steps) / (decay_end_step - warmup_steps)
    # 应用余弦退火公式：η = η_min + 0.5*(η_max - η_min)*(1 + cos(π*progress))
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    current_lr = min_lr + (max_lr - min_lr) * cosine_decay
    
    return current_lr

def init_model():
    pass

def eval():
    pass

def train_epoch(epoch):
    pass

def train():
    pass
