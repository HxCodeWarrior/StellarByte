###############################################################################
# 文件: utils/checkpoint.py
###############################################################################
"""\
检查点管理器（保存/加载模型、优化器、调度器等）
- 支持 MoE 后缀命名约定
- 在保存时先写 tmp 文件再原子替换，降低损坏风险
- 在加载时自动处理 world_size 变更引起的 step 调整
"""

import os
import torch
from typing import Optional

from .distributed import get_world_size

class CheckpointManager:
    """检查点保存与恢复的封装类。"""

    def __init__(self, save_dir: str = './checkpoints', logger=None):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.logger = logger

    def _ckp_paths(self, prefix: str, hidden_size: int, use_moe: bool):
        moe_suffix = '_moe' if use_moe else ''
        ckp = os.path.join(self.save_dir, f"{prefix}_{hidden_size}{moe_suffix}.pth")
        resume = os.path.join(self.save_dir, f"{prefix}_{hidden_size}{moe_suffix}_resume.pth")
        return ckp, resume

    def save(self, prefix: str, lm_config, model: torch.nn.Module, optimizer=None, epoch: int = 0, step: int = 0, extras: dict = None):
        """保存模型权重与优化器/状态。

        Args:
            prefix: 权重前缀，例如 'pretrain' 或 'sft'
            lm_config: 模型配置对象（需包含 hidden_size、use_moe）
            model: 要保存的模型（可能被 DDP 包装）
            optimizer: 优化器实例（可选）
            epoch: 当前 epoch
            step: 训练步数
            extras: 其他要保存的键值对
        """
        ckp_path, resume_path = self._ckp_paths(prefix, lm_config.hidden_size, lm_config.use_moe)
        # 获取 state_dict（若为 DDP，则取 module）
        from torch.nn.parallel import DistributedDataParallel
        state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
        # 半精度存储节省空间（若需要）
        tmp_ckp = ckp_path + '.tmp'
        torch.save({k: v.half() if isinstance(v, torch.Tensor) and v.dtype == torch.float32 else v for k, v in state_dict.items()}, tmp_ckp)
        os.replace(tmp_ckp, ckp_path)

        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict() if optimizer is not None else None,
            'epoch': epoch,
            'step': step,
            'world_size': get_world_size() if torch.distributed.is_initialized() else 1,
        }
        if extras:
            resume_data.update(extras)
        tmp_resume = resume_path + '.tmp'
        torch.save(resume_data, tmp_resume)
        os.replace(tmp_resume, resume_path)
        if self.logger and hasattr(self.logger, 'info'):
            self.logger.info(f"Saved checkpoint to {resume_path}")
    
    def save_best_model(self, prefix: str, lm_config, model: torch.nn.Module):
        """保存当前最佳模型权重（仅模型参数，不包含优化器状态）。

        Args:
            prefix: 权重前缀，例如 'pretrain' 或 'sft'
            lm_config: 模型配置对象（需包含 hidden_size、use_moe）
            model: 要保存的模型（可能被 DDP 包装）
        """
        moe_suffix = '_moe' if lm_config.use_moe else ''
        best_path = os.path.join(self.save_dir, f"{prefix}_{lm_config.hidden_size}{moe_suffix}_best.pth")
        
        # 获取 state_dict（若为 DDP，则取 module）
        from torch.nn.parallel import DistributedDataParallel
        state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
        
        # 半精度存储节省空间
        tmp_best = best_path + '.tmp'
        torch.save({k: v.half() if isinstance(v, torch.Tensor) and v.dtype == torch.float32 else v for k, v in state_dict.items()}, tmp_best)
        os.replace(tmp_best, best_path)
        
        if self.logger and hasattr(self.logger, 'info'):
            self.logger.info(f"Saved best model to {best_path}")

    def load(self, prefix: str, lm_config, device: str = 'cpu') -> Optional[dict]:
        """加载检查点（resume 版本）并返回内容。

        Returns:
            dict 或 None: 如果存在 resume 文件则返回加载的字典，否则返回 None。
        """
        _, resume_path = self._ckp_paths(prefix, lm_config.hidden_size, lm_config.use_moe)
        if not os.path.exists(resume_path):
            return None
        data = torch.load(resume_path, map_location=device)
        # 处理 world_size 变化导致的 step 调整
        saved_ws = data.get('world_size', 1)
        cur_ws = get_world_size() if torch.distributed.is_initialized() else 1
        if saved_ws != cur_ws and 'step' in data:
            data['step'] = data['step'] * saved_ws // cur_ws
            if self.logger:
                self.logger.info(f"GPU 数量变化({saved_ws} -> {cur_ws})，已自动调整 step 为 {data['step']}")
        return data