import os
import math
import torch
import random
import argparse
import numpy as np
from contextlib import nullcontext
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model.Model import ByteTransformer
from datasets import PretrainDataset
from utils.logger import Logger
from utils.checkpoint import CheckpointManager
from config import ModelConfig


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr(it, all_iters, args):
    warmup_iters = int(args.warmup_steps_ratio * all_iters)
    decay_steps = int(args.lr_decay_steps_ratio * all_iters)

    min_lr = args.min_lr
    warmup_start_lr = args.warmup_start_lr
    num_restarts = args.num_restarts
    lr_decay_rate = args.lr_decay_rate

    cycle_length = all_iters
    total_iters = all_iters * (num_restarts + 1)

    if it >= total_iters:
        return min_lr

    if it < warmup_iters:
        ratio = it / max(1, warmup_iters)
        cosine = 0.5 * (1 - math.cos(math.pi * ratio))
        return warmup_start_lr + cosine * (args.learning_rate - warmup_start_lr)

    cycle_step = (it - warmup_iters) % cycle_length
    cycle_idx = (it - warmup_iters) // cycle_length
    decay_steps_count = cycle_step // decay_steps

    decayed_lr = args.learning_rate * (lr_decay_rate ** decay_steps_count)
    decay_ratio = (cycle_step % decay_steps) / max(1, decay_steps)
    cosine_coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    cycle_base_lr = decayed_lr * (lr_decay_rate ** cycle_idx)
    current_lr = min_lr + cosine_coeff * (max(cycle_base_lr, min_lr) - min_lr)
    return max(current_lr, min_lr)


def evaluate(model, dataloader, args, logger):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(args.device)
            labels = batch['labels'].to(args.device)
            outputs = model(input_ids=input_ids, labels=labels)
            total_loss += outputs.loss.item()
    avg_loss = total_loss / len(dataloader)
    logger.info(f"[Eval] Validation Loss: {avg_loss:.4f}")
    return avg_loss


def init_model(args):
    lm_config = ModelConfig(
        vocab_size=args.vocab_size,
        dim=args.model_dim,
        n_layers=args.num_layers,
        n_heads=args.num_attention_heads,
        n_kv_heads=args.num_kv_heads,
        hidden_dim=args.hidden_dim,
        multiple_of=args.dim_multiplier,
        max_seq_len=args.max_seq_len,
        drop_path_prob=args.drop_path_prob,
        hidden_dropout=args.hidden_dropout_prob,
        attention_dropout=args.attention_dropout_prob,
        residual_dropout=args.residual_dropout_prob,
        layer_norm_eps=args.layer_norm_eps,
        initializer_range=args.initializer_range,
        layerscale_init_value=args.layerscale_init,
        tie_word_embeddings=args.tie_word_embeddings,
        xpos_rope_theta=args.xpos_rope_theta,
        xpos_scale_base=args.xpos_scale_base,
        use_flash_attention=args.use_flash_attention,
        causal=args.use_causal,
        use_cache=args.use_cache,
        key_dtype=args.key_cache_dtype,
        value_dtype=args.value_cache_dtype,
        model_parallel_size=args.model_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        tensor_parallel_rank=args.tensor_parallel_rank,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = ByteTransformer(lm_config)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(args.device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Logger(f"模型参数总量：{param_count / 1e6:.2f}M")
    return model, tokenizer, lm_config


def train_epoch(model, dataloader, optimizer, scaler, ctx, args, epoch, total_iters, logger, global_step):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(args.device)
        labels = batch['labels'].to(args.device)

        lr = get_lr(global_step, total_iters, args)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        global_step += 1

        if global_step % args.log_interval == 0:
            logger.info(f"[Epoch {epoch} | Step {global_step}] Loss: {loss.item() * args.accumulation_steps:.4f}, LR: {lr:.8f}")

        if args.val_data_path and global_step % args.eval_interval == 0:
            evaluate(model, dataloader, args, logger)

        if global_step % args.save_interval == 0:
            args.ckpt_mgr.save_checkpoint(model, optimizer, None, epoch, step=global_step)

    avg_loss = total_loss / len(dataloader)
    logger.info(f"[Epoch {epoch}] 平均损失: {avg_loss:.4f}")
    return global_step


def train(args):
    logger = Logger(os.path.join(args.out_dir, "train.log"))
    set_seed(args.seed)

    model, tokenizer, config = init_model(args)
    dataset = PretrainDataset(args.train_data_path, tokenizer, max_length=config.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_loader = None
    if args.val_data_path:
        val_dataset = PretrainDataset(args.val_data_path, tokenizer, max_length=config.max_seq_len)
        val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_iters = args.epochs * len(dataloader)

    use_amp = args.amp or args.dtype in ['float16', 'bfloat16']
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ctx = nullcontext() if args.device == 'cpu' else torch.cuda.amp.autocast(dtype=getattr(torch, args.dtype)) if use_amp else nullcontext()

    ckpt_mgr = CheckpointManager(args.out_dir)
    start_epoch = 0
    global_step = 0
    if ckpt_mgr.has_checkpoint():
        logger.info("恢复模型权重中...")
        checkpoint = ckpt_mgr.load_checkpoint(model, optimizer, None)
        start_epoch = checkpoint.get("epoch", 0) + 1
        global_step = checkpoint.get("step", 0)

    logger.info("开始训练...")
    for epoch in range(start_epoch, args.epochs):
        global_step = train_epoch(model, dataloader, optimizer, scaler, ctx, args, epoch, total_iters, logger, global_step)
        if val_loader:
            evaluate(model, val_loader, args, logger)
        ckpt_mgr.save_checkpoint(model, optimizer, None, epoch)
    logger.info("训练结束。")


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
    parser.add_argument("--out_dir", type=str, default="./checkpoints", help="模型检查点和日志输出目录")

    # =============================
    #       数据集配置
    # =============================
    parser.add_argument("--tokenizer_path", type=str, default="./tokenizer", help="分词器配置文件路径")
    parser.add_argument("--train_data_path", type=str, default="./test/train.jsonl", help="训练数据集文件路径")
    parser.add_argument("--val_data_path", type=str, default="./test/val.jsonl", help="验证数据集文件路径（可选）")
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

    train(args)  # 启动训练
