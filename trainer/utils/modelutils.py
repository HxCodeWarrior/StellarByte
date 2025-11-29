###############################################################################
# Optimizer
###############################################################################
"""优化器工厂：统一构造 AdamW、Lion 等优化器的参数分组和权重衰减处理。"""

import os
import torch
import torch.optim as optim
from typing import Iterable
from trainer.utils.logger import Logger


def get_parameter_groups(model: torch.nn.Module, weight_decay: float, no_decay_names: Iterable[str] = ("bias", "LayerNorm.weight")):
    """构造参数分组，常见于 transformer 模型。

    返回值可直接传入优化器.
    """
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay_names):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0}
    ]


def build_optimizer(model: torch.nn.Module, optim_cfg: dict):
    """根据配置构造优化器。

    optim_cfg 示例：{'type':'AdamW', 'lr':1e-4, 'weight_decay':0.01}
    """
    pgroups = get_parameter_groups(model, optim_cfg.get('weight_decay', 0.0))
    opt_type = optim_cfg.get('type', 'AdamW').lower()
    lr = optim_cfg.get('lr', 1e-4)
    if opt_type == 'adamw':
        return optim.AdamW(pgroups, lr=lr)
    elif opt_type == 'sgd':
        return optim.SGD(pgroups, lr=lr)
    else:
        # 默认回退到 AdamW
        return optim.AdamW(pgroups, lr=lr)

###############################################################################
# Scheduler
###############################################################################
"""\
学习率调度器工厂
- 提供 cosine、linear warmup 等常见调度器
"""

from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup


def build_scheduler(optimizer, scheduler_cfg: dict, num_training_steps: int, num_warmup_steps: int):
    """构造调度器。

    scheduler_cfg 示例：{'type':'cosine'}
    """
    t = scheduler_cfg.get('type', 'cosine').lower()
    if t == 'cosine':
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    else:
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

###############################################################################
# Init Model
###############################################################################
"""\
模型与分词器初始化工具
- 封装 tokenizer 加载、模型从预训练/断点加载、DDP 包装的逻辑
"""

from transformers import AutoTokenizer
from torch.nn.parallel import DistributedDataParallel


def init_model_and_tokenizer(lm_config, tokenizer_path: str, model_class, from_weight: str = 'pretrain', checkpoint_dir: str = './checkpoints', device: str = 'cuda', strict: bool = False, logger: Logger = None):
    """初始化模型和 tokenizer。

    Args:
        lm_config: 模型配置对象
        tokenizer_path: tokenizer 保存路径
        model_class: 模型类（如 StellarByteForCausalLM）
        from_weight: 权重前缀或 'none' 表示不加载
        checkpoint_dir: 检查点目录
        device: 设备
        strict: load_state_dict 的 strict
    Returns:
        model, tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = model_class(lm_config)
    if from_weight != 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = os.path.join(checkpoint_dir, f"{from_weight}_{lm_config.hidden_size}{moe_suffix}_resume.pth")
        if os.path.exists(weight_path):
            data = torch.load(weight_path, map_location=device)
            state = data.get('model', data)
            model.load_state_dict(state, strict=strict)
            if logger:
                logger.info(f"Loaded weights from {weight_path}")
    model.to(device)
    # DDP 包装由外部调用决定
    return model, tokenizer

###############################################################################
# Init Swanlab
###############################################################################
"""
Swanlab 初始化工具
"""
import os
import swanlab
from typing import Dict, Optional

def init_swanlab(project: str, config: Dict, 
                 workspace: Optional[str] = None, 
                 api_key: Optional[str] = None) -> Optional[swanlab.Run]:
    """初始化 Swanlab 可视化工具
    
    Args:
        project: 项目名称
        config: 配置字典
        workspace: 工作空间（可选）
        api_key: API密钥，如果为None则从环境变量读取
    
    Returns:
        Swanlab Run对象，如果初始化失败则返回None
    """
    try:
        # 设置API密钥（优先使用参数，其次环境变量）
        swanlab.login(api_key=api_key, save=True)
        
        # 初始化SwanLab
        swanlab_run = swanlab.init(
            project=project,
            workspace=workspace,
            config=config,
        )
        
        print(f"SwanLab初始化成功! 项目: {project}, 运行ID: {swanlab_run.id}")
        return swanlab_run
        
    except Exception as e:
        print(f"SwanLab初始化失败: {e}")
        print("请确保已安装swanlab包并配置了正确的API密钥")
        return None

###############################################################################
# Count Model Parameters
###############################################################################
"""
模型参数计算工具函数
提供详细的模型参数统计和分析功能
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict

def count_parameters(model: nn.Module, 
                    verbose: bool = False, 
                    max_name_length: int = 50) -> Dict[str, Union[int, Dict]]:
    """计算模型参数数量
    
    Args:
        model: PyTorch模型
        verbose: 是否输出详细统计信息
        max_name_length: 模块名称最大显示长度
    
    Returns:
        包含参数统计信息的字典
    """
    total_params = 0
    trainable_params = 0
    layer_stats = {}
    
    for name, module in model.named_modules():
        # 跳过空模块
        if len(list(module.children())) > 0:
            continue
            
        # 计算该层的参数
        layer_params = sum(p.numel() for p in module.parameters())
        layer_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        if layer_params > 0:
            # 截断过长的模块名称
            display_name = name if len(name) <= max_name_length else f"...{name[-(max_name_length-3):]}"
            layer_stats[display_name] = {
                'total': layer_params,
                'trainable': layer_trainable,
                'percentage': (layer_params / sum(p.numel() for p in model.parameters())) * 100
            }
            
            total_params += layer_params
            trainable_params += layer_trainable
    
    # 按参数数量排序
    layer_stats = dict(sorted(layer_stats.items(), key=lambda x: x[1]['total'], reverse=True))
    
    # 输出详细统计信息
    if verbose:
        print("\n" + "="*80)
        print("模型参数详细统计")
        print("="*80)
        print(f"{'模块名称':<{max_name_length}} {'总参数':>12} {'可训练参数':>12} {'占比(%)':>10}")
        print("-"*80)
        
        for name, stats in layer_stats.items():
            print(f"{name:<{max_name_length}} {stats['total']:>12,} {stats['trainable']:>12,} {stats['percentage']:>9.2f}")
        
        print("-"*80)
        print(f"{'总计':<{max_name_length}} {total_params:>12,} {trainable_params:>12,} {'100.00':>10}")
        print("="*80)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'layer_statistics': layer_stats,
        'parameter_ratio': trainable_params / total_params if total_params > 0 else 0
    }

def count_moe_parameters(model: nn.Module, 
                        moe_layer_names: Optional[List[str]] = None,
                        verbose: bool = False) -> Dict[str, Union[int, Dict]]:
    """专门计算MoE模型参数数量
    
    Args:
        model: PyTorch模型（包含MoE层）
        moe_layer_names: MoE层名称列表，如果为None则自动检测
        verbose: 是否输出详细统计信息
    
    Returns:
        包含MoE参数统计信息的字典
    """
    if moe_layer_names is None:
        # 自动检测MoE层（名称包含'moe'或'expert'的层）
        moe_layer_names = []
        for name, module in model.named_modules():
            if 'moe' in name.lower() or 'expert' in name.lower():
                moe_layer_names.append(name)
    
    moe_params = 0
    moe_trainable = 0
    moe_layer_stats = {}
    
    for name, module in model.named_modules():
        if name in moe_layer_names:
            layer_params = sum(p.numel() for p in module.parameters())
            layer_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            moe_params += layer_params
            moe_trainable += layer_trainable
            moe_layer_stats[name] = {
                'total': layer_params,
                'trainable': layer_trainable
            }
    
    # 计算非MoE参数
    total_stats = count_parameters(model, verbose=False)
    non_moe_params = total_stats['total_parameters'] - moe_params
    non_moe_trainable = total_stats['trainable_parameters'] - moe_trainable
    
    if verbose:
        print("\n" + "="*80)
        print("MoE模型参数统计")
        print("="*80)
        print(f"MoE层数量: {len(moe_layer_names)}")
        print(f"MoE总参数: {moe_params:,} ({moe_params/total_stats['total_parameters']*100:.2f}%)")
        print(f"MoE可训练参数: {moe_trainable:,}")
        print(f"非MoE参数: {non_moe_params:,} ({non_moe_params/total_stats['total_parameters']*100:.2f}%)")
        print(f"非MoE可训练参数: {non_moe_trainable:,}")
        
        if moe_layer_stats:
            print("\nMoE层详细统计:")
            for name, stats in moe_layer_stats.items():
                print(f"  {name}: {stats['total']:,} 参数 ({stats['trainable']:,} 可训练)")
        print("="*80)
    
    return {
        'total_parameters': total_stats['total_parameters'],
        'trainable_parameters': total_stats['trainable_parameters'],
        'moe_parameters': moe_params,
        'moe_trainable_parameters': moe_trainable,
        'non_moe_parameters': non_moe_params,
        'non_moe_trainable_parameters': non_moe_trainable,
        'moe_ratio': moe_params / total_stats['total_parameters'] if total_stats['total_parameters'] > 0 else 0,
        'moe_layer_statistics': moe_layer_stats
    }
