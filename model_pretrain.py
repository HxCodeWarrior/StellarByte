import os
import logging
import yaml
import time
import warnings
import math
import random
import numpy as np

import swanlab

import torch
from torch import optim
from torch.utils.data import DataLoader
from contextlib import nullcontext
from transformers import AutoTokenizer

from tqdm import tqdm
from types import SimpleNamespace

from model.Model import ByteModel
from model.config import ByteModelConfig
from datasets import PretrainDataset

# 创建日志记录器
logger = logging.getLogger(__name__)

def parse_args(config_path: str):
    """
    读取并解析两级结构的 YAML 配置文件为嵌套对象。

    Args:
        config_path (str): YAML 配置文件路径。

    Returns:
        config (SimpleNamespace): 以属性形式访问配置项的嵌套命名空间。
    
    Usage:
        >>> config=parse_args(config_path="./configs/pretrain.yaml")
        >>> config.training.train_epochs
        3
        >>> model = AutoModel(config=config)
    """
    def dict_to_namespace(d):
        """
        将字典递归转换为 SimpleNamespace。
        """
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        else:
            return d

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    return dict_to_namespace(config_dict)

def set_environment(config, seed: int = 42, use_cuda: bool = True):
    """
    设置训练所需的基本环境，并初始化 SwanLab 实验追踪。

    Args:
        config (SimpleNamespace): 配置对象，包含训练参数。
        seed (int): 随机种子，默认 42。
        use_cuda (bool): 是否启用 GPU 训练，默认 True。
    """

    # 禁用并行 tokenizer 警告和 Python 警告
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁止 tokenizer 多线程警告
    os.environ["PYTHONWARNINGS"] = "ignore"         # 忽略所有 Python 警告

    # 设置日志格式（如果未统一设定）
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 设置随机种子，保证可复现性
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果有多卡也统一

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 设备选择
    if use_cuda and torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device = torch.device("cuda")
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        logger.info(f"Using CUDA device(s): {device_count} GPU(s) detected.")
        for i in range(device_count):
            logger.info(f"  - [{i}] {torch.cuda.get_device_name(i)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS backend.")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA / MPS not available. Using CPU.")

    # 设置浮点精度表现
    torch.set_printoptions(precision=4, sci_mode=False)

    # 屏蔽 PyTorch 警告
    warnings.filterwarnings("ignore", category=UserWarning)

    # 初始化 SwanLab 实验追踪
    try:
        swanlab.init(
            project=config.experiment.name or "default_project",
            name=config.experiment.run_name or f"run_{int(time.time())}",
            config=config,  # 自动记录所有配置参数
            mode="online",  # 改为 "offline" 可离线记录
        )
        logger.info("SwanLab initialized.")
    except Exception as e:
        logger.warning(f"Failed to initialize SwanLab: {e}")

    return device

def cosine_annealing_lr(
    current_step: int, 
    total_steps: int, 
    warmup_ratio: float = 0.02,
    plateau_ratio: float = 0.01,
    max_lr: float = 1e-3,
    min_lr: float = 1e-6,
) -> float:
    """
    标准 LLM 预训练学习率调度器: Warmup + Plateau  + Cosine Decay
    
    Args:
        Args:
        current_step (int): 当前步数（从0开始）
        total_steps  (int): 总训练步数
        warmup_ratio  (float): warmup阶段占总步数的比例（0~1）
        plateau_ratio (float): 平台期步数比例（0~1之间）
        max_lr (float): 最大学习率
        min_lr (float): 最小学习率
    
    Return:
        float: 当前步对应的学习率
    """
    # 参数校验
    if total_steps <= 0:
        raise ValueError("total_steps必须为正整数")
    if not (0.0 <= warmup_ratio <= 1.0):
        raise ValueError("warmup_ratio必须在[0, 1]之间")
    if not (0 <= plateau_ratio <= 1):
        raise ValueError("plateau_ratio必须在[0, 1]之间")
    if warmup_ratio + plateau_ratio >= 1.0:
        raise ValueError("warmup_ratio + plateau_ratio 必须小于 1")
    if current_step < 0:
        raise ValueError("current_step不能为负数")
    if min_lr < 0 or max_lr < min_lr:
        raise ValueError(f"无效的学习率范围: min_lr={min_lr}, max_lr={max_lr}")
    
    # 步数换算
    warmup_steps = int(total_steps * warmup_ratio)
    plateau_steps = int(total_steps * plateau_ratio)
    decay_start_step = warmup_steps + plateau_steps
    decay_end_step = total_steps

    # 1. 热身阶段处理
    if warmup_steps > 0 and current_step < warmup_steps:
        # 线性热身：从0到max_lr
        return max_lr * (current_step + 1) / warmup_steps
    
    # 2. Plateau阶段：保持最大学习率
    if current_step < decay_start_step:
        return max_lr

    # 3. 余弦退火阶段核心计算
    # 计算当前进度比例（热身阶段之后）
    decay_progress = (current_step - decay_start_step) / (decay_end_step - decay_start_step)
    # 应用余弦退火公式：η = η_min + 0.5*(η_max - η_min)*(1 + cos(π*progress))
    cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_progress))
    current_lr = min_lr + (max_lr - min_lr) * cosine_decay
    
    return current_lr

def init_model(config):
    """
    初始化分词器和模型，并移动到适当设备。

    Args:
        config: 由 parse_args() 返回的配置对象，包含 model 和 training 部分。

    Returns:
        model: 初始化后的 ByteModel 实例
        tokenizer: 初始化后的 AutoTokenizer 实例
    """
    # ===== 1. 模型配置 =====
    model_config = ByteModelConfig(config)

    # ===== 2. 初始化模型 =====
    model = ByteModel(model_config)

    # ===== 3. 初始化Tokenizer =====
    # config.model.tokenizer_path 应指向已有的 tokenizer 路径或预训练模型名
    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_path, use_fast=True)

    # ===== 4. 设备处理 =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, tokenizer

def evaluate(model, eval_dataset, batch_size=8, device="cuda"):
    f"""
    工业级的LLM预训练模型评估函数。

    Args:
        model (torch.nn.Module): 需要评估的模型。
        eval_dataset (torch.utils.data.Dataset): 用于评估的数据集。
        batch_size (int): 每次评估的批次大小。
        device (str): 执行评估的设备（默认使用cuda）。
    Returns:
        type: dict
        {
            "loss": final_loss,
            "perplexity": final_perplexity,
            "accuracy": final_accuracy
        }
    """
    
    model.eval()  # 将模型设置为评估模式（会禁用dropout等）
    model.to(device)  # 将模型移至指定设备（如GPU）

    # 构建评估用的DataLoader（不打乱顺序）
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # 初始化累计变量
    total_loss = 0.0             # 总损失
    total_tokens = 0             # 总样本数量
    correct_predictions = 0      # 正确预测数量
    total_predictions = 0        # 总预测数量

    step = 0  # 当前评估步数

    # 禁用梯度计算，加快评估速度、减少显存占用
    with torch.no_grad():
        # 使用tqdm显示进度条
        pbar = tqdm(eval_dataloader, desc="Evaluating", unit="batch")

        for batch in pbar:
            step += 1

            # 将batch中的输入转移到指定设备
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # 如果没有显式提供labels，默认将input_ids作为labels（如自回归模型）
            labels = batch.get("labels", input_ids).to(device)

            # 前向传播，获取损失和输出logits
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            # 计算预测值（取最大概率位置作为预测token）
            predictions = torch.argmax(logits, dim=-1)

            # 创建mask用于忽略padding或-100标记
            mask = labels != -100

            # 计算正确预测的token数量
            correct = (predictions == labels) & mask
            correct_predictions += correct.sum().item()
            total_predictions += mask.sum().item()

            # 累加损失和样本数（按样本个数加权）
            total_loss += loss.item() * input_ids.size(0)
            total_tokens += input_ids.size(0)

            # 动态计算当前平均loss、困惑度和准确率
            avg_loss = total_loss / total_tokens
            perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

            # 更新进度条中的状态信息
            pbar.set_postfix({
                "Loss": f"{avg_loss:.4f}",
                "PPL": f"{perplexity:.2f}",
                "Acc": f"{accuracy:.2%}"
            })

    # 评估结束后，计算最终平均损失、困惑度、准确率
    final_loss = total_loss / total_tokens
    final_perplexity = math.exp(final_loss) if final_loss < 100 else float("inf")
    final_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

    # 打印最终评估结果
    logger.info(f"[Eval] Loss: {final_loss:.4f}, PPL: {final_perplexity:.2f}, Accuracy: {final_accuracy:.2%}")

    # 返回评估指标结果（可用于日志、保存等）
    return {
        "loss": final_loss,
        "perplexity": final_perplexity,
        "accuracy": final_accuracy
    }

def train_epoch(
    epoch: int,
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    config: SimpleNamespace,
    global_step: int,
    lr_scheduler: callable,
    scaler: torch.cuda.amp.GradScaler,
    eval_dataset: PretrainDataset = None
) -> int:
    """
    工业级LLM单轮训练函数，支持混合精度、梯度累积等高级特性
    
    Args:
        epoch (int): 当前训练轮次
        model (nn.Module): 待训练模型
        train_dataloader (DataLoader): 训练数据加载器
        optimizer (Optimizer): 优化器实例
        device (torch.device): 训练设备
        config (SimpleNamespace): 训练配置
        global_step (int): 全局训练步数
        lr_scheduler (callable): 学习率调度函数
        scaler (GradScaler): 混合精度梯度缩放器
        eval_dataset (PretrainDataset): 验证数据集
        
    Returns:
        int: 更新后的全局训练步数
    """
    # ===== 1. 训练准备 =====
    model.train()  # 设置模型为训练模式（启用dropout等）
    train_config       = config.training # 获取训练相关配置
    iter_per_epoch     = len(train_dataloader)  # 每epoch的迭代次数
    accumulation_steps = train_config.gradient_accumulation_steps  # 梯度累积步数
    max_grad_norm      = train_config.max_grad_norm                # 梯度裁剪阈值
    total_loss         = 0.0             # 累计整个epoch的总损失
    accumulated_loss   = 0.0             # 当前梯度累积周期内的损失
    tokens_processed   = 0               # 当前梯度累积周期内处理的token总数
    start_time         = time.time()     # 当前梯度累积周期的开始时间
    
    # 从配置获取日志间隔（添加默认值）
    log_interval = getattr(train_config, 'log_interval', 10)
    save_interval = getattr(train_config, 'save_interval', 1000)
    eval_interval = getattr(train_config, 'eval_steps', 2000)

    # 创建进度条（显示epoch信息，以batch为单位）
    pbar = tqdm(
        train_dataloader, 
        desc=f"Epoch {epoch+1}/{train_config.train_epochs}", 
        unit="batch"
    )
    
    # 混合精度上下文管理器
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if scaler else nullcontext()
    
    # ===== 2. 批次训练循环 =====
    for step, batch in enumerate(pbar):
        # 2.1 数据准备
        input_ids   = batch["input_ids"].to(device, non_blocking=True) # 输入token IDs，异步传输到设备
        labels      = batch["labels"].to(device, non_blocking=True)    # 标签token IDs，异步传输
        loss_mask   = batch["loss_mask"].to(device, non_blocking=True) # 损失掩码，异步传输
        batch_size, seq_len = input_ids.shape
        
        # 2.2 动态学习率更新（每一步都更新）
        current_lr = lr_scheduler(global_step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # 2.3 混合精度前向传播
        with amp_ctx: # 进入混合精度上下文（自动管理FP16计算）
            outputs = model(
                input_ids    = input_ids, 
                labels       = labels,
                padding_mask = loss_mask
            ) # 模型前向传播
            loss = outputs.loss / accumulation_steps  # 梯度累积损失缩放
        
        # 2.4 反向传播
        scaler.scale(loss).backward()
        
        # 2.4 损失统计
        current_loss     = loss.item() * accumulation_steps  # 还原实际损失值（去除梯度累积的缩放）
        accumulated_loss += current_loss                     # 累加到当前累积周期的损失
        total_loss       += current_loss                     # 累加到epoch总损失
        tokens_processed += batch_size * seq_len             # 累加处理的token数量
        
        # 2.5 梯度累积更新
        if (step + 1) % accumulation_steps == 0:
            #  梯度裁剪（防止梯度爆炸）
            if max_grad_norm > 0:  # 检查是否启用梯度裁剪
                scaler.unscale_(optimizer)  # 取消缩放梯度（恢复原始梯度值）
                # 裁剪梯度（L2范数不超过max_grad_norm）
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # 参数更新
            scaler.step(optimizer) # 使用缩放后的梯度更新参数
            scaler.update()        # 缩放器更新梯度缩放因子
            optimizer.zero_grad()  # 清空梯度（准备下个累积周期）
            
            # 2.6 指标计算与记录
            global_step    += 1  # 全局步数增加
            step_time      = time.time() - start_time  # 计算当前累积周期的耗时
            tokens_per_sec = tokens_processed / step_time  # 计算吞吐量（token/秒）
            avg_loss       = accumulated_loss / accumulation_steps  # 计算平均损失
            
            # 2.7 重置累积状态（准备下一个累积周期）
            accumulated_loss = 0.0          # 重置累积损失
            tokens_processed = 0            # 重置token计数器
            start_time       = time.time()  # 重置计时器
            
            # 2.8 日志记录（按全局步骤）
            if global_step % log_interval == 0:
                # 计算剩余时间
                elapsed_time = time.time() - step_time
                batches_per_sec = (step + 1) / elapsed_time
                remaining_batches = iter_per_epoch - step - 1
                remaining_time = remaining_batches / batches_per_sec
                
                # 获取显存使用情况
                mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)  # GB
                mem_reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
                
                logger.info(
                    f"Epoch:[{epoch+1}/{train_config.train_epochs}] "
                    f"Global Step:[{global_step}] "
                    f"Loss:{avg_loss:.3f} "
                    f"LR:{current_lr:.3e} "
                    f"Speed:{tokens_per_sec:,.0f} tok/s "
                    f"Mem:{mem_alloc:.1f}/{mem_reserved:.1f} GB"
                    f"Remaining:[{int(remaining_time)} secs]"
                )
                
                # SwanLab记录
                swanlab.log({
                    "train/loss"        : avg_loss,
                    "train/lr"          : current_lr,
                    "train/speed"       : tokens_per_sec,
                    "train/step_time"   : step_time,
                    "train/mem_alloc"   : mem_alloc,
                    "train/mem_reserved": mem_reserved
                }, step=global_step)

            # 更新进度条
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",       # 格式化显示损失
                "lr"  : f"{current_lr:.2e}",     # 科学计数法显示学习率
                "t/s" : f"{tokens_per_sec:.0f}"  # 整数显示token/秒
            })
            
            
            # 2.8 模型保存
            if global_step % save_interval == 0:
                model.eval()
                save_path = f"{train_config.output_dir}/model_step_{global_step}.pth"
                
                # 自定义模型保存
                
                # 恢复训练
                model.train()
            
            # 2.9 模型验证
            if eval_dataset and global_step % eval_interval == 0:
                logger.info(f"开始验证 @ step {global_step}")
                eval_start = time.time()
                eval_metrics = evaluate(
                    model, 
                    eval_dataset, 
                    batch_size=train_config.eval_batch_size,
                    device=device
                )
                
                # 记录验证指标
                swanlab.log({
                    "eval/loss": eval_metrics["loss"],
                    "eval/perplexity": eval_metrics["perplexity"],
                    "eval/accuracy": eval_metrics["accuracy"],
                    "eval/duration": time.time() - eval_start
                }, step=global_step)
                
                # 恢复训练模式
                model.train()
                logger.info(f"验证完成 | 耗时: {time.time()-eval_start:.1f}s")
    
    # ===== 3. 轮次结束处理 =====
    # 计算整个epoch的平均损失
    epoch_loss = total_loss / len(train_dataloader)
    # 记录epoch完成日志
    logger.info(f"Epoch {epoch+1} completed | Avg Loss: {epoch_loss:.4f}")
    # 记录epoch损失到SwanLab
    swanlab.log({"train/epoch_loss": epoch_loss}, step=global_step)
    
    # 保存最终检查点
    if (epoch + 1) % train_config.save_epochs == 0:
        pass
    
    return global_step

def train():
    pass
