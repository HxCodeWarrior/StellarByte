# -----------------------------------------------------------------------------
# 导入依赖
# -----------------------------------------------------------------------------
import os
import math
import time
import json
import random
import argparse
from contextlib import nullcontext

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# 分布式训练核心
# from accelerate import Accelerator, DistributedDataParallelKwargs, DeepSpeedPlugin
# from accelerate.logging import get_logger as get_accel_logger

# 实验追踪
import swanlab

# 终端美化输出
from rich.console import Console
from rich.table import Table

# 项目内部模块
from datasets import PretrainDataset
from model.Model import ByteTransformer
from model.config import ByteModelConfig
from utils.checkpoint import CheckpointManager
from utils.progressbar import RichProgressBar
from utils.logger import get_logger

console = Console()

# -----------------------------------------------------------------------------
# 随机种子与学习率调度器
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """固定随机种子，保证实验可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr(it: int, all_iters: int, args) -> float:
    """
    余弦退火 + 线性预热 + 周期重启 学习率调度器。

    参数说明
    --------
    it : int
        当前全局 step。
    all_iters : int
        单个 epoch 内的 step 数乘以总 epoch 数，表示单周期迭代数。
    args : Namespace
        命令行参数集合。
    """
    # 计算关键节点
    warmup_iters = int(args.warmup_steps_ratio * all_iters)  # 预热步数
    decay_steps = int(args.lr_decay_steps_ratio * all_iters) # 每次衰减的步长

    # 便捷变量
    min_lr           = args.min_lr
    warmup_start_lr  = args.warmup_start_lr or args.learning_rate / 1_000
    num_restarts     = args.num_restarts
    lr_decay_rate    = args.lr_decay_rate

    cycle_length = all_iters                 # 一个周期长度
    total_iters  = all_iters * (num_restarts + 1)  # 所有周期的总步数

    # 如果超过最大训练步数，返回最小学习率
    if it >= total_iters:
        return min_lr

    # ------- 1. 线性 + 余弦预热 -------
    if it < warmup_iters:
        ratio  = it / max(1, warmup_iters)
        cosine = 0.5 * (1 - math.cos(math.pi * ratio))
        return warmup_start_lr + cosine * (args.learning_rate - warmup_start_lr)

    # ------- 2. 多周期余弦退火 -------
    cycle_step       = (it - warmup_iters) % cycle_length   # 当前周期内的步数
    cycle_idx        = (it - warmup_iters) // cycle_length  # 周期索引
    decay_steps_cnt  = cycle_step // decay_steps            # 已触发的衰减次数

    decayed_lr       = args.learning_rate * (lr_decay_rate ** decay_steps_cnt)
    decay_ratio      = (cycle_step % decay_steps) / max(1, decay_steps)
    cosine_coeff     = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    cycle_base_lr    = decayed_lr * (lr_decay_rate ** cycle_idx)
    current_lr       = min_lr + cosine_coeff * (max(cycle_base_lr, min_lr) - min_lr)
    return max(current_lr, min_lr)


# -----------------------------------------------------------------------------
# 训练过程辅助函数
# -----------------------------------------------------------------------------

def grad_global_norm(parameters) -> float:
    """计算所有可训练参数梯度的 L2 范数（用于监控 & 梯度裁剪）。"""
    total_norm = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        param_norm  = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    return math.sqrt(total_norm)


def format_size(num_bytes: int) -> str:
    """将字节数格式化为可读字符串，如 256.0 MiB。"""
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PiB"


# -----------------------------------------------------------------------------
# 验证循环
# -----------------------------------------------------------------------------

def evaluate(model, dataloader, args, logger, global_step):
    """
    整个验证集前向推理，不计算梯度。

    返回
    ----
    avg_loss : float
        验证集平均损失，用于早停或学习率调度。
    """
    model.eval()  # 切换到评估模式
    total_loss   = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(args.device)
            labels    = batch["labels"].to(args.device)
            outputs   = model(input_ids=input_ids, labels=labels)
            # 累加 *样本数* 方便最后求平均
            total_loss += outputs.loss.item() * input_ids.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    ppl      = math.exp(min(20, avg_loss))  # 防止溢出

    logger.info(f"[Eval] Step {global_step} | 验证损失 {avg_loss:.4f} | PPL {ppl:.2f}")

    if args.use_swanlab:
        swanlab.log({
            "val/loss": avg_loss,
            "val/perplexity": ppl,
        }, step=global_step)

    model.train()  # 评估完记得切回训练模式
    return avg_loss


# -----------------------------------------------------------------------------
# 模型与分词器初始化
# -----------------------------------------------------------------------------

def init_model(args, logger):
    """根据 CLI 参数构造 ByteTransformer 与分词器。"""

    # 1. 组装配置
    config = ByteModelConfig(
        vocab_size           = args.vocab_size,
        dim                  = args.model_dim,
        n_layers             = args.num_layers,
        n_heads              = args.num_attention_heads,
        n_kv_heads           = args.num_kv_heads,
        hidden_dim           = args.hidden_dim,
        multiple_of          = args.dim_multiplier,
        max_seq_len          = args.max_seq_len,
        drop_path_prob       = args.drop_path_prob,
        hidden_dropout       = args.hidden_dropout_prob,
        attention_dropout    = args.attention_dropout_prob,
        residual_dropout     = args.residual_dropout_prob,
        layer_norm_eps       = args.layer_norm_eps,
        initializer_range    = args.initializer_range,
        layerscale_init_value= args.layerscale_init,
        tie_word_embeddings  = args.tie_word_embeddings,
        xpos_rope_theta      = args.xpos_rope_theta,
        xpos_scale_base      = args.xpos_scale_base,
        use_flash_attention  = args.use_flash_attention,
        causal               = args.use_causal,
        use_cache            = args.use_cache,
        key_dtype            = args.key_cache_dtype,
        value_dtype          = args.value_cache_dtype,
        model_parallel_size  = args.model_parallel_size,
        tensor_parallel_size = args.tensor_parallel_size,
        tensor_parallel_rank = args.tensor_parallel_rank,
    )

    # 2. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # 3. 构建模型
    model = ByteTransformer(config)
    if torch.cuda.device_count() > 1 and not args.ddp:
        model = torch.nn.DataParallel(model)
    model = model.to(args.device)

    # 4. 打印参数规模
    param_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数总量: {param_cnt/1e6:.2f}M")

    return model, tokenizer, config


# -----------------------------------------------------------------------------
# 单个 epoch 训练
# -----------------------------------------------------------------------------
def train_epoch(model, dataloader, optimizer, scaler, ctx, args, epoch,
                total_iters, logger, global_step):
    """执行一个 epoch 的前向、反向与梯度更新。"""

    model.train()
    loss_sum = 0.0

    pb_total = len(dataloader)
    # 使用 RichProgressBar 可视化训练进度
    with RichProgressBar(total_steps=pb_total, total_batches=pb_total,
                         total_epochs=args.epochs, desc=f"Epoch {epoch+1}") as pbar:

        start_wall = time.perf_counter()

        # --------------------------------------------------
        # 遍历数据集
        # --------------------------------------------------
        for step, batch in enumerate(dataloader, 1):
            input_ids          = batch["input_ids"].to(args.device)
            labels             = batch["labels"].to(args.device)
            tokens_this_batch  = input_ids.numel()

            # —— 动态学习率 ——
            lr = get_lr(global_step, total_iters, args)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # —— 前向 + 反向 ——
            with ctx:  # 支持 AMP autocast
                outputs = model(input_ids=input_ids, labels=labels)
                loss    = outputs.loss / args.accumulation_steps

            scaler.scale(loss).backward()

            # —— 梯度累积 ——
            if (step % args.accumulation_steps) == 0:
                # 反缩放后裁剪 & 记录梯度范数
                scaler.unscale_(optimizer)
                total_grad_norm = grad_global_norm(model.parameters())
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                # 更新参数
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            else:
                total_grad_norm = 0.0  # 非更新步，梯度范数置 0 仅做日志占位

            # —— 指标统计 ——
            loss_item     = loss.item() * args.accumulation_steps
            loss_sum     += loss_item
            tokens_per_s  = tokens_this_batch / max(1e-6, time.perf_counter() - start_wall)
            gpu_mem       = torch.cuda.memory_allocated(args.device) if torch.cuda.is_available() else 0

            # —— 日志打印 & SwanLab ——
            if global_step % args.log_interval == 0:
                ppl = math.exp(min(20, loss_item))
                logger.info(
                    f"E{epoch+1} S{global_step} | loss {loss_item:.4f} | ppl {ppl:.1f} "
                    f"| lr {lr:.6g} | gnorm {total_grad_norm:.2f} | tok/s {tokens_per_s:.0f} "
                    f"| mem {format_size(gpu_mem)}")

                if args.use_swanlab:
                    swanlab.log({
                        "train/loss": loss_item,
                        "train/perplexity": ppl,
                        "train/lr": lr,
                        "train/grad_norm": total_grad_norm,
                        "train/tokens_per_sec": tokens_per_s,
                        "train/gpu_mem": gpu_mem / (1024**2),  # 转 MB
                    }, step=global_step)

            # —— 验证 & 保存 ——
            if args.val_data_path and (global_step % args.eval_interval == 0):
                evaluate(model, args.val_loader, args, logger, global_step)

            if global_step % args.save_interval == 0:
                args.ckpt_mgr.save_checkpoint(model, optimizer, None, epoch, step=global_step)

            # —— 更新进度条 ——
            pbar.update_loader(step)
            pbar.update_train(global_step, epoch+1, loss=loss_item, lr=lr)

            global_step += 1

    avg_loss = loss_sum / pb_total
    logger.info(f"[Epoch {epoch+1}] 平均损失 {avg_loss:.4f}")

    return global_step


# -----------------------------------------------------------------------------
# 主训练流程
# -----------------------------------------------------------------------------
def train(args, logger):
    """主训练流程，包含模型初始化、训练与验证、SwanLab 接入等。"""
    set_seed(args.seed)

    # 初始化 SwanLab 实验
    if args.use_swanlab:
        swanlab.login(api_key=args.swanlab_api_key)
        swanlab.init(
            project=args.swanlab_project,
            experiment_name=args.swanlab_experiment_name,
            config=vars(args)
        )

    # 初始化模型、分词器与配置
    model, tokenizer, config = init_model(args, logger)

    # 加载训练集
    train_dataset = PretrainDataset(args.train_data_path, tokenizer, max_length=config.max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # 加载验证集（如果提供）
    val_loader = None
    if args.val_data_path:
        val_dataset = PretrainDataset(args.val_data_path, tokenizer, max_length=config.max_seq_len)
        val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
        args.val_loader = val_loader

    # 构建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # 学习率调度与AMP配置
    total_iters = args.epochs * len(train_loader)
    use_amp = args.amp or args.dtype in ['float16', 'bfloat16']
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ctx = nullcontext() if args.device == 'cpu' else torch.cuda.amp.autocast(dtype=getattr(torch, args.dtype)) if use_amp else nullcontext()

    # 初始化检查点管理器
    ckpt_mgr = CheckpointManager(args.checkpoints_dir)
    args.ckpt_mgr = ckpt_mgr

    start_epoch = 0
    global_step = 0
    if ckpt_mgr.has_checkpoint():
        logger.info("🔁 检测到历史检查点，正在恢复中…")
        checkpoint = ckpt_mgr.load_checkpoint(model, optimizer, None)
        start_epoch = checkpoint.get("epoch", 0) + 1
        global_step = checkpoint.get("step", 0)

    logger.info("🚀 开始训练…")
    for epoch in range(start_epoch, args.epochs):
        global_step = train_epoch(model, train_loader, optimizer, scaler, ctx,
                                  args, epoch, total_iters, logger, global_step)
        if val_loader:
            evaluate(model, val_loader, args, logger, global_step)
        ckpt_mgr.save_checkpoint(model, optimizer, None, epoch)

    logger.info("✅ 训练完成。")



# 程序入口
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # =============================
    #       实验基础配置
    # =============================
    parser.add_argument("--seed", type=int, default=42, help="全局随机种子，确保实验可复现性")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",help="训练设备 (cuda/cpu)")
    parser.add_argument("--use_swanlab", action="store_true", help="是否启用SwanLab实验追踪")
    parser.add_argument("--swanlab_project", type=str, default="Happy-LLM", help="SwanLab项目名称")
    parser.add_argument("--swanlab_experiment_name", type=str, default="Pretrain-215M", help="SwanLab实验名称")
    parser.add_argument("--swanlab_api_key", type=str, default="",  help="SwanLab API认证密钥")
    parser.add_argument("--logs_dir", type=str, default="./logs", help="日志输出目录")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="模型检查点输出目录")

    # =============================
    #       数据集配置
    # =============================
    parser.add_argument("--tokenizer_path", type=str, default="./tokenizer", help="分词器配置文件路径")
    parser.add_argument("--train_data_path", type=str, default="./datasets/test/train.jsonl", help="训练数据集文件路径")
    parser.add_argument("--val_data_path", type=str, default="./datasets/test/val.jsonl", help="验证数据集文件路径（可选）")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载器工作进程数")

    # =============================
    #       模型架构配置
    # =============================
    parser.add_argument("--vocab_size", type=int, default=32000, help="词汇表大小")
    parser.add_argument("--model_dim", type=int, default=768, help="模型隐藏层维度")
    parser.add_argument("--num_layers", type=int, default=12, help="Transformer层数")
    parser.add_argument("--num_attention_heads", type=int, default=16, help="注意力头数")
    parser.add_argument("--num_kv_heads", type=int, default=8, help="Key/Value注意力头数（头分离）")
    parser.add_argument("--hidden_dim", type=int, default=None, help="FFN隐藏层维度（默认4*model_dim）")
    parser.add_argument("--dim_multiplier", type=int, default=4, help="隐藏层维度对齐基数")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="最大序列长度")
    
    # Dropout 参数
    parser.add_argument("--drop_path_prob", type=float, default=0.0, help="残差路径Dropout概率")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1, help="隐藏层Dropout概率")
    parser.add_argument("--attention_dropout_prob", type=float, default=0.1, help="注意力Dropout概率")
    parser.add_argument("--residual_dropout_prob", type=float, default=0.1, help="残差连接Dropout概率")
    
    # 归一化参数
    parser.add_argument("--layer_norm_eps", type=float, default=1e-5, help="层归一化epsilon")
    parser.add_argument("--initializer_range", type=float, default=0.02, help="参数初始范围")
    parser.add_argument("--layerscale_init", type=float, default=1e-5, help="层缩放初始化值")
    
    # 嵌入参数
    parser.add_argument("--tie_word_embeddings", action="store_true", help="绑定输入输出词嵌入")
    
    # 位置编码参数
    parser.add_argument("--xpos_rope_theta", type=float, default=10000.0, help="XPos位置编码theta")
    parser.add_argument("--xpos_scale_base", type=float, default=512.0, help="XPos缩放因子")
    
    # 注意力机制
    parser.add_argument("--use_flash_attention", action="store_true", help="启用FlashAttention")
    parser.add_argument("--use_causal", action="store_true", help="使用因果注意力掩码")
    
    # 推理优化
    parser.add_argument("--use_cache", action="store_true", help="启用KV缓存加速推理")
    parser.add_argument("--key_cache_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"], help="Key缓存数据类型")
    parser.add_argument("--value_cache_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"], help="Value缓存数据类型")
    
    # 并行训练
    parser.add_argument("--model_parallel_size", type=int, default=1, help="模型并行大小")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="张量并行大小")
    parser.add_argument("--tensor_parallel_rank", type=int, default=0, help="张量并行rank")

    # =============================
    #       训练超参数配置
    # =============================
    parser.add_argument("--batch_size", type=int, default=32, help="每批次样本数量")
    parser.add_argument("--epochs", type=int, default=5, help="训练总轮次")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数（模拟更大batch size）")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值（防止梯度爆炸）")

    # =============================
    #       优化器参数配置
    # =============================
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="基础学习率")
    parser.add_argument("--min_lr", type=float, default=5e-6, help="学习率最小值（余弦退火下限）")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW优化器权重衰减系数")

    # =============================
    #       学习率调度器配置
    # =============================
    parser.add_argument("--warmup_steps_ratio", type=float, default=0.05, help="预热阶段占总训练步数的比例")
    parser.add_argument("--warmup_start_lr", type=float, default=5e-7, help="预热起始学习率（默认: learning_rate/1000）")
    parser.add_argument("--lr_decay_rate", type=float, default=0.8,  help="学习率衰减率（多周期余弦退火）")
    parser.add_argument("--lr_decay_steps_ratio", type=int, default=0.3, help="学习率衰减间隔比例（步进式衰减）")
    parser.add_argument("--num_restarts", type=int, default=0,  help="余弦退火重启次数（0表示单周期）")

    # =============================
    #       混合精度训练配置
    # =============================
    parser.add_argument("--dtype", type=str, default="bfloat16", 
                       choices=["float32", "float16", "bfloat16"],
                       help="训练数据类型（float32/full precision, float16/half, bfloat16/brain float）")
    parser.add_argument("--amp", action="store_true", help="启用自动混合精度训练（与dtype互斥）")

    # =============================
    #       日志与保存配置
    # =============================
    parser.add_argument("--log_interval", type=int, default=100, help="训练日志打印间隔（步数）")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型检查点保存间隔（步数）")
    parser.add_argument("--eval_interval", type=int, default=2000, help="验证集评估间隔（步数）")
    parser.add_argument("--max_checkpoints", type=int, default=5, help="最大保留的检查点数量")

    # =============================
    #       验证配置
    # =============================
    parser.add_argument("--eval_batch_size", type=int, default=64, help="验证批次大小")
    parser.add_argument("--eval_max_steps", type=int, default=100, help="最大验证步数（全验证集过大时使用）")
    args = parser.parse_args()
    
    # -------- 日志系统 --------
    logger = get_logger(args.logs_dir, args.swanlab_experiment_name)
    logger.info("配置参数:\n" + str(vars(args)))

    train(args, logger)  # 启动训练
