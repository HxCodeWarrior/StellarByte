import os
import multiprocessing
import argparse
import logging
import yaml
import time
import warnings
import math
import random
import numpy as np

from tqdm import tqdm
from types import SimpleNamespace

import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score

import swanlab

import torch
from torch import optim
from torch.utils.data import DataLoader
from contextlib import nullcontext
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from utils.logger import setup_logger
from utils.checkpoint import CheckpointManager 
from model.Model import ByteModel
from model.config import ByteModelConfig
from datasets import PretrainDataset

# nltk需要下载必要资源
# 使用环境变量标记是否已下载，避免重复下载
if not os.environ.get('NLTK_DATA_DOWNLOADED'):
    nltk.data.path.append('./sources/')
    nltk.download('punkt', download_dir='./sources')
    nltk.download('punkt_tab', download_dir='./sources')
    nltk.download('wordnet', download_dir='./sources')
    nltk.download('omw-1.4', download_dir='./sources')
    # 设置环境变量标记已下载
    os.environ['NLTK_DATA_DOWNLOADED'] = 'True'

# 设置全局logger
logger = logging.getLogger("StellarByte")

def parse_args(config_path: str):
    """
    读取并解析 YAML 配置文件，将所有嵌套字典的键值直接拉平到一级 SimpleNamespace对象，所有参数都在同一级。

    Args:
        config_path (str): YAML 配置文件路径。

    Returns:
        config (SimpleNamespace): 以属性形式访问配置项的嵌套命名空间。
    
    Usage:
        >>> config=parse_args(config_path="./configs/pretrain.yaml")
        >>> config.train_epochs
        3
        >>> model = AutoModel(config=config)
    """
    def convert_value(v):
        # 如果已经是数字或布尔值，直接返回
        if isinstance(v, (int, float, bool)):
            return v
        
        # 如果是字符串，尝试转换
        if isinstance(v, str):
            low_v = v.strip().lower()
            # 布尔值识别
            if low_v in ("true", "yes", "on"):
                return True
            if low_v in ("false", "no", "off"):
                return False
            # 数值识别（包含科学计数法）
            try:
                num = float(v)
                # 如果是整数形式，转 int
                if num.is_integer():
                    return int(num)
                return num
            except ValueError:
                return v  # 转换失败则保留原字符串
        return v

    def flatten_dict_no_prefix(d):
        flat = {}
        for k, v in d.items():
            if isinstance(v, dict):
                flat.update(flatten_dict_no_prefix(v))
            else:
                flat[k] = convert_value(v)
        return flat

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # 递归处理，将嵌套字典展平成一维字典
    flat_config = flatten_dict_no_prefix(config_dict)
    return SimpleNamespace(**flat_config)

def parse_device(config):
    """
    直接从 config 对象中读取 device 字符串并解析，
    支持：
      - "cpu"
      - "cuda:0"
      - "cuda:[0,1,2]"
      
    返回:
      device (torch.device): 主设备
      device_ids (List[int]): 多设备ID列表
    """
    device_str = getattr(config, "device", "cpu").lower().strip()
    
    import re
    if device_str == "cpu":
        return torch.device("cpu"), []
    
    multi_gpu_match = re.match(r"cuda:\[(.*)\]", device_str)
    if multi_gpu_match:
        device_ids_str = multi_gpu_match.group(1)
        device_ids = [int(x.strip()) for x in device_ids_str.split(",") if x.strip().isdigit()]
        if not device_ids:
            raise ValueError(f"Invalid cuda device list in '{device_str}'")
        main_device = torch.device(f"cuda:{device_ids[0]}")
        return main_device, device_ids
    
    single_gpu_match = re.match(r"cuda:(\d+)", device_str)
    if single_gpu_match:
        device_id = int(single_gpu_match.group(1))
        return torch.device(f"cuda:{device_id}"), [device_id]
    
    # 兜底使用cpu
    return torch.device("cpu"), []

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

    # 创建一个日志器（logger）
    logger_config = config
    setup_logger(
        name=logger_config.logger_name,               # 自定义logger名字，可为None使用root logger
        log_dir=logger_config.log_dir,                # 日志文件保存目录
        log_file=logger_config.log_file,              # 日志文件名
        level=logger_config.log_level,                # logger总体日志等级
        console_level=logger_config.console_level,    # 控制台日志等级
        file_level=logger_config.file_level,          # 文件日志等级
        use_color=logger_config.use_color,            # 控制台启用彩色日志
        when=logger_config.rotation,                  # 文件轮转周期，午夜
        backup_count=logger_config.backup_count,      # 保留最近7个日志文件
        is_rank_0=logger_config.is_rank_0             # 是否为rank 0（多进程场景控制文件写入）
    )

    # 设置随机种子，保证可复现性
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果有多卡也统一

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 解析设备
    device, device_ids = parse_device(config)

    # 设备选择
    if use_cuda and torch.cuda.is_available():
        torch.cuda.set_device(device)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        logger.info(f"Using CUDA device(s): {device_ids}")
        for i in device_ids:
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
    if config.use_swanlab and config.api_key:
        try:
            swanlab.login(api_key=config.api_key)

            # 将SimpleNamespace转换为字典
            config_dict = vars(config) if hasattr(config, '__dict__') else {}

            swanlab.init(
                project=config.project_name or "default_project",
                name=config.run_name or f"run_{int(time.time())}",
                config=config_dict,  # 自动记录所有配置参数
                logdir=logger_config.log_dir or None,
            )
            logger.info("SwanLab initialized.")
        except Exception as e:
            logger.warning(f"Failed to initialize SwanLab: {e}")
    else:
        if not config.use_swanlab:
            logger.info("SwanLab tracking is disabled in configuration.")
        elif not config.api_key:
            logger.warning("SwanLab API key is missing. Tracking disabled.")

    return device

def cosine_annealing_lr(
    current_step: int, 
    total_steps: int, 
    warmup_ratio: float = 0.02,
    plateau_ratio: float = 0.01,
    max_lr: float = 1e-3,
    min_lr_ratio: float = 0.10,
) -> float:
    """
    LLM 预训练学习率调度器: Warmup + Plateau  + Cosine Decay
    
    Args:
        Args:
        current_step  (int)   : 当前步数（从0开始）
        total_steps   (int)   : 总训练步数
        warmup_ratio  (float) : warmup阶段占总步数的比例（0~1）,推荐 0.01 ~ 0.03
        plateau_ratio (float) : 平台期步数比例（0~1之间）, 推荐 0.00 ~ 0.02
        max_lr        (float) : 最大学习率
        min_lr_ratio  (float) : 最小学习率比例（相对于max_lr）
    
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
    if min_lr_ratio <= 0 or min_lr_ratio > 1.0:
        raise ValueError("min_lr_ratio必须在(0, 1.0]之间")
    
    # 动态计算最小学习率
    min_lr = max_lr * min_lr_ratio

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

def init_model(config, device):
    """
    初始化分词器和模型，并移动到适当设备。

    Args:
        config: 由 parse_args() 返回的配置对象，包含 model 和 training 部分。

    Returns:
        model: 初始化后的 ByteModel 实例
        tokenizer: 初始化后的 AutoTokenizer 实例
    """
    # ===== 1. 模型配置 =====
    # 从config中提取需要的参数，而不是直接传递整个config对象
    model_config = ByteModelConfig(
        vocab_size=config.vocab_size,
        model_dim=config.model_dim,
        num_layers=config.num_layers,
        num_attention_heads=config.num_heads,
        num_kv_heads=config.num_kv_heads,
        hidden_dim=config.hidden_dim,
        dim_multiplier=config.dim_multiplier,
        max_seq_len=config.max_seq_len,
        drop_path_prob=config.drop_path_prob,
        hidden_dropout_prob=config.hidden_dropout_prob,
        attention_dropout_prob=config.attention_dropout_prob,
        residual_dropout_prob=config.residual_dropout_prob,
        layer_norm_eps=float(config.layer_norm_eps),
        base_theta=config.base_theta,
        ntk_alpha=config.ntk_alpha,
        use_flash_attention=config.use_flash_attention,
        use_cache=config.use_cache,
        key_cache_dtype=torch.float16 if config.key_cache_dtype == 'float16' else torch.float32,
        value_cache_dtype=torch.float16 if config.value_cache_dtype == 'float16' else torch.float32,
        attention_window_size=config.attention_window_size,
        parallel_residual=config.parallel_residual,
        tensor_parallel_size=config.tensor_parallel_size,
        tensor_parallel_group=None,
        layerscale_init=float(config.layerscale_init),
        initializer_range=config.initializer_range
    )

    # ===== 2. 初始化模型 =====
    model = ByteModel(model_config)

    # ===== 3. 初始化Tokenizer =====
    # config.model.tokenizer_path 应指向已有的 tokenizer 路径或预训练模型名
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, use_fast=True)

    # ===== 4. 设备处理 =====
    model.to(device)

    return model, tokenizer

def evaluate(model, eval_dataset, tokenizer, batch_size=8, device=torch.device("cpu")):
    """
    工业级的LLM预训练模型评估函数。

    Args:
        model (torch.nn.Module): 需要评估的模型。
        eval_dataset (torch.utils.data.Dataset): 用于评估的数据集。
        tokenizer (AutoTokenizer): 分词器。
        batch_size (int): 每次评估的批次大小。
        device (torch.device): 执行评估的设备（默认使用cpu）。
    Returns:
        type: dict
        {
            "loss": final_loss,
            "perplexity": final_perplexity,
            "accuracy": final_accuracy,
            "bleu": bleu_score,
            "rouge-1": avg_rouge1,
            "rouge-l": avg_rougeL,
            "meteor": avg_meteor,
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

    # 准备存储生成文本与参考文本，供BLEU/ROUGE/METEOR计算
    all_references = []  # list of list of references (tokenized)
    all_hypotheses = []  # list of hypotheses (tokenized)

    # Rouge scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge1_scores = []
    rougeL_scores = []
    meteor_scores = []

    step = 0  # 当前评估步数

    # 禁用梯度计算，加快评估速度、减少显存占用
    with torch.no_grad():
        # 使用tqdm显示进度条
        pbar = tqdm(eval_dataloader, desc="Evaluating", unit="batch")

        for batch in pbar:
            step += 1

            # 将batch中的输入转移到指定设备
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            # 前向传播，获取损失和输出logits
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            # 计算预测值（取最大概率位置作为预测token）
            predictions = torch.argmax(logits, dim=-1)

            # 创建mask用于忽略padding或-100标记
            mask = labels != -100
            
            # 计算预测的token数量
            num_tokens   = mask.sum().item()
            total_loss   += loss.item() * num_tokens  # 按token加权
            total_tokens += num_tokens

            # 计算准确率
            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions == labels) & mask
            correct_predictions += correct.sum().item()
            total_predictions += num_tokens

            # 动态计算当前平均loss、困惑度和准确率
            avg_loss = total_loss / total_tokens
            perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

            # --- 文本生成部分 ---
            # 解码参考文本和生成文本（忽略-100）
            for ref_ids, pred_ids in zip(labels.cpu().tolist(), predictions.cpu().tolist()):
                # 过滤掉-100标签，转为token字符串
                ref_tokens = [token for token in ref_ids if token != -100]
                pred_tokens = pred_ids[:len(ref_tokens)]  # 预测长度与参考对齐
                # 使用tokenizer解码为字符串（需要从外层传入tokenizer或定义为全局）
                ref_text = tokenizer.decode(ref_tokens, skip_special_tokens=True).strip()
                pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()

                # 分词（使用 nltk 分词）
                ref_tokens_split = word_tokenize(ref_text)
                pred_tokens_split = word_tokenize(pred_text)

                all_references.append([ref_tokens_split])  # BLEU要求参考是list of list
                all_hypotheses.append(pred_tokens_split)

                # 计算ROUGE分数
                rouge_score = scorer.score(ref_text, pred_text)
                rouge1_scores.append(rouge_score['rouge1'].fmeasure)
                rougeL_scores.append(rouge_score['rougeL'].fmeasure)

                # 计算METEOR分数
                meteor_scores.append(meteor_score([ref_tokens_split], pred_tokens_split))

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

    # 计算BLEU
    # smoothing_method=4可避免0分问题
    bleu_score = corpus_bleu(
        all_references,
        all_hypotheses,
        smoothing_function=SmoothingFunction().method4
    )

    # 计算ROUGE均值
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0

    # 计算METEOR均值
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0

    # 打印最终评估结果
    logger.info(
        f"[Eval] Loss: {final_loss:.4f}, "
        f"PPL: {final_perplexity:.2f}, "
        f"Accuracy: {final_accuracy:.2%}, "
        f"BLEU: {bleu_score:.4f}, "
        f"ROUGE-1: {avg_rouge1:.4f}, "
        f"ROUGE-L: {avg_rougeL:.4f}, "
        f"METEOR: {avg_meteor:.4f}")

    # 返回评估指标结果（可用于日志、保存等）
    return {
        "loss": final_loss,
        "perplexity": final_perplexity,
        "accuracy": final_accuracy,
        "bleu": bleu_score,
        "rouge-1": avg_rouge1,
        "rouge-l": avg_rougeL,
        "meteor": avg_meteor,
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
    tokenizer: PreTrainedTokenizerBase,
    checkpoint_manager: CheckpointManager,
    eval_dataset: PretrainDataset = None,
) -> int:
    """
    LLM单轮训练函数，支持混合精度、梯度累积等高级特性
    
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
        float: 当前epoch的平均
    """
    # ===== 1. 训练准备 =====
    model.train()               # 设置模型为训练模式（启用dropout等）
    train_config       = config # 获取训练相关配置
    iter_per_epoch     = len(train_dataloader)  # 每epoch的迭代次数
    accumulation_steps = train_config.gradient_accumulation_steps  # 梯度累积步数
    max_grad_norm      = train_config.max_grad_norm                # 梯度裁剪阈值
    # === 初始化损失与token计数统计变量 ===
    total_token_loss_sum = 0.0  # 整个epoch累积的token级loss和（loss * token数）
    total_token_count = 0       # 整个epoch累积的有效token数

    acc_token_loss_sum = 0.0    # 当前梯度累积周期内token级loss和
    acc_token_count = 0         # 当前梯度累积周期内有效token数
    
    # 从配置获取日志间隔（添加默认值）
    log_interval = getattr(train_config, 'log_interval', 10)
    save_interval = getattr(train_config, 'save_interval', 1000)
    eval_interval = getattr(train_config, 'eval_steps', 2000)

    # 创建进度条（显示epoch信息，以batch为单位）
    pbar = tqdm(
        train_dataloader, 
        total=iter_per_epoch, 
        desc=f"Epoch {epoch+1}/{train_config.train_epochs}", 
        unit="batch"
    )
    
    # 混合精度上下文管理器
    amp_ctx = torch.autocast(device_type=device.type, dtype=torch.float16) if scaler else nullcontext()
    
    # 确保在开始之前梯度清零
    optimizer.zero_grad()

    start_time         = time.time()     # 当前梯度累积周期的开始时间
    epoch_start_time   = start_time      # 当前epoch的开始时间
    local_start_time   = start_time  # 用于 tokens/sec 计算（每次累积重置）

    # ===== 2. 批次训练循环 =====
    for step, batch in enumerate(pbar):
        # 2.1 数据准备
        input_ids      = batch["input_ids"].to(device, non_blocking=True)      # 输入token IDs，异步传输到设备
        labels         = batch["labels"].to(device, non_blocking=True)         # 标签token IDs，异步传输
        attention_mask = batch["attention_mask"].to(device, non_blocking=True) # 注意力掩码，异步传输
        loss_mask      = batch["loss_mask"].to(device, non_blocking=True)      # 损失掩码，异步传输
        batch_size, seq_len = input_ids.shape
        
        # 2.2 动态学习率更新（每一步都更新）
        current_lr = lr_scheduler(global_step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # 2.3 混合精度前向传播
        with amp_ctx: # 进入混合精度上下文（自动管理FP16计算）
            outputs = model(
                input_ids      = input_ids, 
                labels         = labels,
                attention_mask = attention_mask,
            ) # 模型前向传播
            loss      = outputs.loss / accumulation_steps  # 梯度累积损失缩放
            # loss_mask = loss_mask.view(-1)
            # loss      = torch.sum(loss * loss_mask) / loss_mask.sum()

        # 2.4 反向传播
        scaler.scale(loss).backward()
        
        # 2.4 损失统计
        # 计算 activa tokens 数量
        active_tokens = int(loss_mask.sum().item())

        # 恢复该batch实际loss（未除以accumulation_steps）
        batch_loss = loss.detach().item() * accumulation_steps  

        # 累计当前梯度累积周期的loss*token数和token数
        acc_token_loss_sum += batch_loss * active_tokens
        acc_token_count += active_tokens
        
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
            # 统计epoch总loss和token数
            total_token_loss_sum += acc_token_loss_sum
            total_token_count += acc_token_count

            # 计算当前累积周期平均token loss
            avg_loss = acc_token_loss_sum / acc_token_count if acc_token_count > 0 else 0.0

            # 计算吞吐量 (token/s)
            step_time = time.time() - local_start_time
            tokens_per_sec = acc_token_count / step_time if step_time > 0 else 0
            
            # 2.7 重置累积状态（准备下一个累积周期）
            acc_token_loss_sum = 0.0          # 重置累积token损失
            acc_token_count    = 0            # 重置累积token计数器
            start_time         = time.time()  # 重置计时器
            
            # 2.8 日志记录（按全局步骤）
            if global_step % log_interval == 0:
                # 计算剩余时间
                elapsed_time      = time.time() - epoch_start_time
                batches_per_sec   = (step + 1) / elapsed_time
                remaining_batches = max(0, iter_per_epoch - step - 1)
                remaining_time    = remaining_batches / batches_per_sec
                
                mem_alloc    = 0.0
                mem_reserved = 0.0
                # 获取显存信息（仅 cuda）
                if device.type == "cuda":
                    mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
                    mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                else:
                    mem_alloc = mem_reserved = 0.0
                
                logger.info(
                    f"Epoch:[{epoch+1}/{train_config.train_epochs}] "
                    f"Global Step:[{global_step}] "
                    f"Loss:{avg_loss:.3f} "
                    f"LR:{current_lr:.3e} "
                    f"Speed:{tokens_per_sec:,.0f} tok/s "
                    f"Mem:{mem_alloc:.1f}/{mem_reserved:.1f} GB"
                    f"Remaining:[{int(remaining_time)} secs]"
                )
                
                if config.use_swanlab:
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
                
                # 自定义模型保存
                checkpoint_path = checkpoint_manager.save_checkpoint(
                    epoch=epoch,
                    global_step=global_step,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    metrics={"train_loss": avg_loss}  # 记录当前训练损失
                )
                
                # 恢复训练
                model.train()

                # 记录保存信息
                if checkpoint_path:
                    logger.info(f"已保存检查点: {checkpoint_path}")

                # 可选：SwanLab记录
                if config.use_swanlab:
                    swanlab.log({"train/checkpoint_saved": 1}, step=global_step)

            # 2.9 模型验证与最佳模型保存
            if eval_dataset and global_step % eval_interval == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None # 清空显存缓存（避免OOM）

                logger.info(f"开始验证 @ step {global_step}")
                eval_start_time = time.time()
                eval_metrics = evaluate(
                    model, 
                    eval_dataset,
                    tokenizer, 
                    batch_size=train_config.eval_batch_size,
                    device=device
                )
                
                # 保存最佳模型（基于验证损失）
                best_path = checkpoint_manager.save_best(
                    metric_name="loss",
                    metric_value=eval_metrics["loss"],
                    epoch=epoch,
                    global_step=global_step,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    extra={"eval_metrics": eval_metrics}  # 保存完整评估指标
                )

                if best_path:
                    logger.info(f"保存新的最佳模型: {best_path}")
                
                # 记录验证指标
                if config.use_swanlab:
                    swanlab.log({
                        "eval/loss": eval_metrics["loss"],
                        "eval/perplexity": eval_metrics["perplexity"],
                        "eval/accuracy": eval_metrics["accuracy"],
                        "eval/bleu": eval_metrics["bleu"],
                        "eval/rouge-1": eval_metrics["rouge-1"],
                        "eval/rouge-l": eval_metrics["rouge-l"],
                        "eval/meteor": eval_metrics["meteor"],
                        "eval/duration": time.time() - eval_start_time
                    }, step=global_step)
                
                # 恢复训练模式
                model.train()
                logger.info(f"验证完成 | 耗时: {time.time()-eval_start_time:.1f}s")
    
    # 2.10 处理残余梯度(如果最后没有整除accumulation_steps)
    if acc_token_count > 0:
        # 梯度裁剪（防止梯度爆炸）
        if max_grad_norm > 0:  # 检查是否启用梯度裁剪
            scaler.unscale_(optimizer)  # 取消缩放梯度（恢复原始梯度值）
            # 裁剪梯度（L2范数不超过max_grad_norm）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # 取消缩放梯度（恢复原始梯度值）
        scaler.step(optimizer) # 使用缩放后的梯度更新参数
        scaler.update()        # 缩放器更新梯度缩放因子
        
        # 梯度清零
        optimizer.zero_grad()
        global_step += 1

        # 日志记录
        total_token_loss_sum += acc_token_loss_sum
        total_token_count += acc_token_count
        avg_loss = acc_token_loss_sum / acc_token_count if acc_token_count > 0 else 0.0
        logger.info(f"[Epoch End] Residual update done. Avg loss (residual): {avg_loss:.4f}")

    # ===== 3. 轮次结束处理 =====
    # 计算整个epoch的平均损失
    epoch_loss = total_token_loss_sum / total_token_count
    # 记录epoch完成日志
    logger.info(f"Epoch {epoch+1} completed | Avg Loss: {epoch_loss:.4f}")
    # 记录epoch损失到SwanLab
    if config.use_swanlab:
        swanlab.log({"train/epoch_loss": epoch_loss}, step=global_step)
    
    return global_step, epoch_loss

def train(config_path: str):
    """
    LLM 预训练主函数，整合训练流程所有组件
    
    Args:
        config_path (str): YAML 配置文件路径，默认为"./configs/pretrain.yaml"
    """
    # ==================== 1. 配置解析与环境设置 ====================
    config = parse_args(config_path)
    device = set_environment(config)

    # ==================== 2. 检查点管理器初始化 ====================
    # 确保检查点目录存在
    checkpoint_dir = getattr(config, "checkpoints_dir", config.output_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info(f"检查点保存目录: {checkpoint_dir}")

    checkpoint_manager = CheckpointManager(
        output_dir      = config.checkpoints_dir,
        monitor         = config.checkpoints_monitor,  # 用于性能指标监控的字段名
        mode            = config.checkpoints_mode,     # 监控模式："min" | "max"
        max_checkpoints = config.max_checkpoints,      # 最多保存的检查点数
        prefix          = config.model_type,           # 检查点命名前缀
        keep_last_n     = config.keep_last_n,          # 保存最近n个检查点
        is_master       = config.is_master,           # 主进程
    )
    # 注册信号处理器 (Ctrl+C/SIGTERM时保存紧急检查点)
    checkpoint_manager.register_signal_handlers()
    
    # ==================== 3. 模型与数据初始化 ====================
    model, tokenizer = init_model(config, device)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型总参数: {total_params:,} | 可训练参数: {trainable_params:,}")
    
    # 初始化数据集
    train_dataset = PretrainDataset(
        data_path=config.train_data,
        tokenizer=tokenizer,
        max_length=config.max_seq_len
    )
    
    eval_dataset = PretrainDataset(
        data_path=config.eval_data,
        tokenizer=tokenizer,
        max_length=config.max_seq_len
    ) if config.eval_data else None
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # ==================== 4. 训练基础设施初始化 ====================
    # 优化器配置
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2)
    )
    
    # 混合精度训练
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None
    
    # 学习率调度器
    total_steps  = len(train_loader) * config.train_epochs
    lr_scheduler = lambda step: cosine_annealing_lr(
        current_step  = step,
        total_steps   = total_steps,
        warmup_ratio  = config.warmup_ratio,
        plateau_ratio = config.plateau_ratio,
        max_lr        = config.learning_rate,
        min_lr_ratio  = config.min_lr_ratio
    )
    
    # ==================== 5. 恢复训练状态 ====================
    global_step = 0
    start_epoch = 0
    
    # 尝试加载最新检查点
    resume_training = getattr(config, "resume_training", False)
    if resume_training:
        latest_ckpt = checkpoint_manager.latest_checkpoint()
        if latest_ckpt:
            logger.info(f"恢复训练: {latest_ckpt}")
            ckpt = checkpoint_manager.load_checkpoint(
                str(latest_ckpt),
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                map_location=device
            )
            global_step = ckpt.get("global_step", 0)
            start_epoch = ckpt.get("epoch", 0) + 1
            # 恢复最佳指标
            if "metrics" in ckpt and "loss" in ckpt["metrics"]:
                checkpoint_manager.best_metric = ckpt["metrics"]["loss"]
                logger.info(f"恢复最佳验证损失: {checkpoint_manager.best_metric:.4f}")
        else:
            logger.info("没有找到可恢复的检查点，从头开始训练")
    
    # ==================== 6. 训练循环 ====================
    # 训练计时
    total_start_time = time.time()

    logger.info(f"开始训练，共 {config.train_epochs} 个轮次，总步数 {total_steps}")
    
    for epoch in range(start_epoch, config.train_epochs):
        epoch_start_time = time.time()
        
        global_step, epoch_loss = train_epoch(
            epoch            = epoch,
            model            = model,
            train_dataloader = train_loader,
            optimizer        = optimizer,
            device           = device,
            config           = config,
            global_step      = global_step,
            lr_scheduler     = lr_scheduler,
            scaler           = scaler,
            tokenizer        = tokenizer,
            checkpoint_manager = checkpoint_manager,
            eval_dataset     = eval_dataset
        )

        # 计算epoch耗时
        epoch_duration = time.time() - epoch_start_time

        # 轮次结束保存
        if (epoch + 1) %  getattr(config, "save_epochs", 1) == 0:
            checkpoint_path = checkpoint_manager.save_checkpoint(
                epoch=epoch,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                tag=f"epoch{epoch+1}",
                metrics={"epoch_loss": epoch_loss}
            )
            logger.info(f"轮次 {epoch+1} 结束 | 保存检查点: {checkpoint_path}")

        # 轮次结束评估
        if eval_dataset:
            logger.info(f"轮次 {epoch+1} 结束，开始完整验证集评估...")
            eval_metrics = evaluate(
                model        = model,
                eval_dataset = eval_dataset,
                tokenizer    = tokenizer,
                batch_size   = config.eval_batch_size,
                device       = device
            )

            # 保存最佳模型（基于验证损失）
            best_path = checkpoint_manager.save_best(
                metric_name="loss",
                metric_value=eval_metrics["loss"],
                epoch=epoch,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                extra={"full_eval_metrics": eval_metrics}
            )

            if best_path:
                logger.info(f"新的最佳模型 @ loss={eval_metrics['loss']:.4f}: {best_path}")

            # 记录验证指标
            if config.use_swanlab:
                swanlab.log({
                    "eval/epoch_loss": eval_metrics["loss"],
                    "eval/epoch_perplexity": eval_metrics["perplexity"],
                    "eval/epoch_accuracy": eval_metrics["accuracy"],
                    "eval/epoch_bleu": eval_metrics["bleu"],
                    "eval/epoch_rouge1": eval_metrics["rouge-1"],
                    "eval/epoch_rougeL": eval_metrics["rouge-l"],
                    "eval/epoch_meteor": eval_metrics["meteor"]
                }, step=global_step)
    
    # ==================== 7. 训练结束处理 ====================
    total_time = time.time() - total_start_time
    logger.info(f"训练完成! 总耗时: {total_time/3600:.2f} 小时")
    
    # 保存最终模型
    final_path = checkpoint_manager.save_checkpoint(
        epoch=config.train_epochs,
        global_step=global_step,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        tag="final",
        metrics={"total_time": total_time}
    )
    logger.info(f"最终模型保存到: {final_path}")
    
    # 记录训练完成
    if config.use_swanlab:
        swanlab.finish()

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="StellarByte-LLM PreTraining",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 添加配置文件参数
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/model_pretrain.yaml",
        help="模型预训练参数配置YAML文件路径"
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"配置文件不存在: {args.config}")
    
    # 启动训练
    print(f"使用配置文件: {args.config}")
    train(args.config)

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
