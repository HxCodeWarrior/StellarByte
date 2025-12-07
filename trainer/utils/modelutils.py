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
# Validation Functions
###############################################################################
"""\
通用验证和评估工具函数
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable, Any
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import time


def validate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: Optional[Callable] = None,
    metrics: Optional[Dict[str, Callable]] = None,
    num_batches: Optional[int] = None,
    desc: str = "Validating",
    disable_progress: bool = False,
    use_amp: bool = False,
    distributed: bool = False
) -> Dict[str, float]:
    """通用模型验证函数
    
    Args:
        model: 要验证的模型
        dataloader: 验证数据加载器
        device: 设备
        criterion: 损失函数，如果为None则只计算指标
        metrics: 额外指标函数字典 {name: function}
        num_batches: 最大批次数，如果为None则使用整个验证集
        desc: 进度条描述
        disable_progress: 是否禁用进度条
        use_amp: 是否使用自动混合精度
        distributed: 是否在分布式训练中
    
    Returns:
        包含所有指标结果的字典
    """
    model.eval()
    
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    if metrics is None:
        metrics = {}
    
    total_loss = 0.0
    batch_count = 0
    results = defaultdict(float)
    
    # 初始化所有指标
    for metric_name in metrics.keys():
        results[metric_name] = 0.0
    
    with torch.no_grad():
        # 创建进度条
        pbar = tqdm(
            dataloader, 
            desc=desc, 
            disable=disable_progress or not disable_progress and distributed and torch.distributed.get_rank() != 0
        )
        
        for batch_idx, batch in enumerate(pbar):
            if num_batches is not None and batch_idx >= num_batches:
                break
            
            # 处理不同类型的batch格式
            if isinstance(batch, dict):
                inputs = batch.get('input_ids')
                labels = batch.get('labels')
                
                # 对于语言模型，通常input_ids和labels相同但错位
                if labels is None and 'input_ids' in batch:
                    labels = batch['input_ids'].clone()
                    if inputs is not None:
                        labels = labels[:, 1:].contiguous()
                        inputs = inputs[:, :-1].contiguous()
            elif isinstance(batch, (list, tuple)):
                inputs, labels = batch[0], batch[1]
            else:
                raise ValueError(f"Unsupported batch type: {type(batch)}")
            
            # 移动到设备
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # 前向传播
            if use_amp:
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(inputs) if labels is None else model(inputs, labels=labels)
            else:
                outputs = model(inputs) if labels is None else model(inputs, labels=labels)
            
            # 计算损失
            if criterion is not None and labels is not None:
                # 首先尝试从模型输出中获取损失
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    loss = outputs.loss
                else:
                    # 如果没有现成的损失，自己计算
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    if logits is not None:
                        # 正确展平logits和labels
                        loss = criterion(
                            logits.reshape(-1, logits.size(-1)),  # [batch_size * seq_len, vocab_size]
                            labels.reshape(-1)                     # [batch_size * seq_len]
                        )
                    else:
                        loss = torch.tensor(0.0, device=device)
                
                total_loss += loss.item()
            
            # 计算额外指标
            for metric_name, metric_fn in metrics.items():
                try:
                    metric_value = metric_fn(outputs, labels)
                    results[metric_name] += metric_value
                except Exception as e:
                    print(f"计算指标 {metric_name} 时出错: {e}")
                    continue
            
            batch_count += 1
            
            # 更新进度条
            if criterion is not None and labels is not None and batch_count > 0:
                current_loss = total_loss / batch_count
                pbar.set_postfix({'loss': f'{current_loss:.4f}'})
    
    # 计算平均值
    if batch_count > 0:
        if criterion is not None and labels is not None:
            results['loss'] = total_loss / batch_count
        
        for metric_name in metrics.keys():
            results[metric_name] /= batch_count
    
    # 如果是分布式训练，同步所有进程的结果
    if distributed and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        
        # 创建所有结果的张量
        keys = list(results.keys())
        local_results = torch.tensor([results[k] for k in keys], device=device, dtype=torch.float32)
        global_results = torch.zeros_like(local_results)
        
        # 同步所有进程
        torch.distributed.all_reduce(local_results, op=torch.distributed.ReduceOp.SUM)
        global_results = local_results / world_size
        
        # 更新结果字典
        for i, key in enumerate(keys):
            results[key] = global_results[i].item()
    
    model.train()
    return dict(results)


def calculate_perplexity(loss: float) -> float:
    """根据损失计算困惑度
    
    Args:
        loss: 交叉熵损失值
    
    Returns:
        困惑度
    """
    return np.exp(loss)


def calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """计算分类准确率
    
    Args:
        logits: 模型输出logits [batch_size, seq_len, vocab_size]
        labels: 真实标签 [batch_size, seq_len]
    
    Returns:
        准确率 (0-1)
    """
    predictions = logits.argmax(dim=-1)
    correct = (predictions == labels).float()
    
    # 忽略padding token (id为-100)
    mask = labels != -100
    if mask.any():
        correct = correct[mask]
        return correct.mean().item()
    return correct.mean().item()


def calculate_token_accuracy(logits: torch.Tensor, labels: torch.Tensor, ignore_tokens: List[int] = None) -> Dict[str, float]:
    """计算token级别的准确率（忽略特定token）
    
    Args:
        logits: 模型输出logits
        labels: 真实标签
        ignore_tokens: 要忽略的token ID列表
    
    Returns:
        包含准确率和详细统计的字典
    """
    if ignore_tokens is None:
        ignore_tokens = [-100]  # 默认忽略padding token
    
    predictions = logits.argmax(dim=-1)
    mask = torch.ones_like(labels, dtype=torch.bool)
    
    for token_id in ignore_tokens:
        mask = mask & (labels != token_id)
    
    if mask.any():
        correct = (predictions[mask] == labels[mask]).float()
        accuracy = correct.mean().item()
        total_tokens = mask.sum().item()
        correct_tokens = correct.sum().item()
    else:
        accuracy = 0.0
        total_tokens = 0
        correct_tokens = 0
    
    return {
        'accuracy': accuracy,
        'total_tokens': total_tokens,
        'correct_tokens': correct_tokens
    }


def validate_with_gradient_accumulation(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    gradient_accumulation_steps: int = 1,
    **kwargs
) -> Dict[str, float]:
    """使用梯度累积进行验证（适用于内存不足的情况）
    
    Args:
        model: 要验证的模型
        dataloader: 验证数据加载器
        device: 设备
        gradient_accumulation_steps: 梯度累积步数
        **kwargs: 传递给validate_model的其他参数
    
    Returns:
        验证结果
    """
    original_batch_size = dataloader.batch_size
    effective_batch_size = original_batch_size * gradient_accumulation_steps
    
    # 这里简化处理，实际使用时可能需要更复杂的逻辑
    return validate_model(model, dataloader, device, **kwargs)


def get_validation_summary(results: Dict[str, float], epoch: int = None, step: int = None) -> str:
    """生成验证结果摘要字符串
    
    Args:
        results: 验证结果字典
        epoch: 当前epoch（可选）
        step: 当前训练步数（可选）
    
    Returns:
        格式化的摘要字符串
    """
    summary_parts = []
    
    if epoch is not None:
        summary_parts.append(f"Epoch {epoch}")
    if step is not None:
        summary_parts.append(f"Step {step}")
    
    summary_parts.append("Validation Results:")
    
    for metric_name, value in results.items():
        if isinstance(value, float):
            summary_parts.append(f"  {metric_name}: {value:.6f}")
        else:
            summary_parts.append(f"  {metric_name}: {value}")
    
    return "\n".join(summary_parts)


def create_validation_metrics_for_lm(
    tokenizer = None,
    ignore_token_ids: List[int] = None
) -> Dict[str, Callable]:
    """为语言模型创建标准验证指标集合
    
    Args:
        tokenizer: 分词器（用于解码示例）
        ignore_token_ids: 要忽略的token ID列表
    
    Returns:
        指标函数字典
    """
    if ignore_token_ids is None:
        ignore_token_ids = [-100]
    
    def perplexity_metric(outputs, labels):
        """计算困惑度指标"""
        if isinstance(outputs, dict) and 'loss' in outputs:
            loss = outputs['loss'].item()
        else:
            # 如果没有损失，使用交叉熵计算
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            criterion = nn.CrossEntropyLoss(ignore_index=-100)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1)).item()
        
        return calculate_perplexity(loss)
    
    def accuracy_metric(outputs, labels):
        """计算准确率指标"""
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        return calculate_accuracy(logits, labels)
    
    def token_accuracy_metric(outputs, labels):
        """计算token准确率（忽略特定token）"""
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        token_acc = calculate_token_accuracy(logits, labels, ignore_token_ids)
        return token_acc['accuracy']
    
    metrics = {
        'perplexity': perplexity_metric,
        'accuracy': accuracy_metric,
        'token_accuracy': token_accuracy_metric,
    }
    
    return metrics
    
###############################################################################
# Init Swanlab
###############################################################################
"""
Swanlab 初始化工具
"""
import os
import swanlab
from typing import Dict, Optional

def init_swanlab(project: str,
                 workspace: Optional[str] = None, 
                 experiment_name: Optional[str] = None,
                 api_key: Optional[str] = None,
                 config: Dict = None, ) -> Optional[swanlab.Run]:
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
            experiment_name=experiment_name,
            config=config,
        )
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
