import os
import sys
import math
import time
import random
import logging
from datetime import datetime
from contextlib import nullcontext
from types import SimpleNamespace

import torch
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP

# 分布式训练核心
from torch.utils.data.distributed import DistributedSampler

# 实验追踪
import swanlab

# 终端美化输出
from rich.console import Console

# 项目内部模块
from datasets import PretrainDataset
from model.Model import ByteTransformer
from model.config import ByteModelConfig
from utils.checkpoint import CheckpointManager, GracefulKiller
from utils.progressbar import ProgressBarManager
from utils.logger import register_global_exception_handler, _build_logger
from utils.config_params import load_config

console = Console()

# ========= 全局性能 / 显存优化） =========
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # 减碎片 :contentReference[oaicite:0]{index=0}
torch.backends.cuda.matmul.allow_tf32 = True        # Ampere+ TF32 ⾃动降精度
torch.backends.cudnn.allow_tf32 = True              # Turing+ TF32 ⾃动降精度
torch.backends.cudnn.benchmark = True               # cuDNN 算法自动搜索

# -----------------------------------------------------------------------------
# 随机种子与学习率调度器
# -----------------------------------------------------------------------------
def set_seed(seed: int, args=None) -> int:
    """
    固定随机种子；DDP 时将 rank0 的 seed 广播给所有进程，
    返回最终 seed 以便调用者复用。
    """
    if args and args.enable_ddp:
        # rank0 决定随机种子并广播
        seed_tensor = torch.tensor([seed], dtype=torch.long, device=f"cuda:{args.local_rank}")
        torch.distributed.broadcast(seed_tensor, src=0)
        seed = int(seed_tensor.item())

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


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

def compute_label_smoothing(logits, labels, loss_mask, label_smoothing=0.1):
    """
    使用 Label Smoothing 计算交叉熵损失，同时考虑 loss_mask。
    logits: (B, T, V)
    labels: (B, T)
    loss_mask: (B, T)
    """
    vocab_size = logits.size(-1)
    log_probs = F.log_softmax(logits, dim=-1)  # (B, T, V)

    # 构造平滑标签
    with torch.no_grad():
        true_dist = torch.zeros_like(log_probs)  # (B, T, V)
        true_dist.fill_(label_smoothing / (vocab_size - 1))
        ignore_mask = (labels == -100)
        labels = labels.clone()
        labels[ignore_mask] = 0
        true_dist.scatter_(2, labels.unsqueeze(2), 1.0 - label_smoothing)
        true_dist[ignore_mask] = 0  # 忽略 pad 部分

    # 交叉熵损失
    loss = -(true_dist * log_probs).sum(dim=-1)  # (B, T)
    loss = loss * loss_mask  # 仅对有效 token 求损失
    return loss.sum() / (loss_mask.sum() + 1e-8)

def format_size(num_bytes: int) -> str:
    """将字节数格式化为可读字符串，如 256.0 MiB。"""
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PiB"


# -----------------------------------------------------------------------------
# 学习率调度器
# -----------------------------------------------------------------------------
def get_lr(global_step: int, total_iters: int, args) -> float:
    """
    余弦退火 + 线性预热 + 周期重启 学习率调度器。

    参数说明
    --------
    global_step : int,当前全局 step。
    total_iters : int,总训练步数（所有 epoch * (重启周期+1)），表示单周期迭代数。
    args : Namespace,命令行参数集合。
    """
    # 计算关键节点
    warmup_iters = int(args.warmup_steps_ratio * total_iters)  # 预热步数
    decay_steps = int(args.lr_decay_steps_ratio * total_iters) # 每次衰减的步长

    # 便捷变量
    min_lr           = args.min_lr
    warmup_start_lr  = args.warmup_start_lr or args.learning_rate / 1_000
    num_restarts     = args.num_restarts
    lr_decay_rate    = args.lr_decay_rate

    cycle_length = total_iters // (num_restarts + 1) # 一个周期长度

    # 如果超过最大训练步数，返回最小学习率
    if global_step >= total_iters:
        return min_lr

    # ------- 1. 线性 + 余弦预热 -------
    if global_step < warmup_iters:
        ratio  = global_step / max(1, warmup_iters)
        cosine = 0.5 * (1 - math.cos(math.pi * ratio))
        return warmup_start_lr + cosine * (args.learning_rate - warmup_start_lr)

    # ------- 2. 多周期余弦退火 -------
    cycle_step       = (global_step - warmup_iters) % cycle_length   # 当前周期内的步数
    cycle_idx        = (global_step - warmup_iters) // cycle_length  # 周期索引
    decay_steps_cnt  = cycle_step // decay_steps            # 已触发的衰减次数

    decayed_lr       = args.learning_rate * (lr_decay_rate ** decay_steps_cnt)
    decay_ratio      = (cycle_step % decay_steps) / max(1, decay_steps)
    cosine_coeff     = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    cycle_base_lr    = decayed_lr * (lr_decay_rate ** cycle_idx)
    current_lr       = min_lr + cosine_coeff * (max(cycle_base_lr, min_lr) - min_lr)
    return max(current_lr, min_lr)


# -----------------------------------------------------------------------------
# 验证循环
# -----------------------------------------------------------------------------
def evaluate(model, dataloader, args, logger, epoch=None, progressor=None):
    """
    整个验证集前向推理，不计算梯度。

    返回
    ----
    avg_loss : float
        验证集平均损失，用于早停或学习率调度。
    """
    model.eval()  # 切换到评估模式
    total_loss   = torch.tensor(0.0, device=args.device)
    total_tokens = torch.tensor(0,   device=args.device)
    num_batches = len(dataloader)

    # -------- 初始化进度条 --------
    if progressor and is_main_process(args) and epoch is not None:
        progressor.set_epoch(epoch)
        progressor.start_phase(total_steps=num_batches, phase='val')

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(args.device)
            labels    = batch["labels"].to(args.device)
            outputs   = model(input_ids=input_ids, labels=labels)
            # 累加 *样本数* 方便最后求平均
            total_loss   += outputs.loss.detach() * input_ids.size(0)
            total_tokens += input_ids.size(0)

            # 更新进度条
            if progressor and is_main_process(args):
                avg_loss_sofar = (total_loss / total_tokens).item()
                progressor.update_phase(loss=avg_loss_sofar)

     # -------- 分布式汇总 --------
    if args.ddp:
        torch.distributed.all_reduce(total_loss,  op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_tokens, op=torch.distributed.ReduceOp.SUM)

    avg_loss = (total_loss / total_tokens).item()
    ppl      = math.exp(min(20, avg_loss))

    if is_main_process(args):
        logger.info(f"[Eval] Step {epoch} | loss {avg_loss:.4f} | ppl {ppl:.2f}")
        if args.use_swanlab:
            swanlab.log({"val/loss": avg_loss, "val/ppl": ppl}, step=epoch)
    
    if progressor and is_main_process(args):
        progressor.end_phase()

    model.train()  # 评估完记得切回训练模式
    return avg_loss


# -----------------------------------------------------------------------------
# 模型与分词器初始化
# -----------------------------------------------------------------------------
def init_model(args, logger):
    """根据 CLI 参数构造 ByteTransformer 与分词器。"""

    # 1. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # 2. 检查词表大小是否变化（比如添加了pad token）
    if len(tokenizer) > args.vocab_size:
        logger.info(f"检测到 tokenizer 词表大小变为 {len(tokenizer)}，更新模型配置 vocab_size")
        args.vocab_size = len(tokenizer)

    # 3. 组装配置
    config = ByteModelConfig(
        vocab_size           = args.vocab_size,
        dim                  = args.dim,
        n_layers             = args.n_layers,
        n_heads              = args.n_heads,
        n_kv_heads           = args.n_kv_heads,
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
        xpos_rope_theta      = args.rope_theta,
        xpos_scale_base      = args.scale_base,
        use_flash_attention  = args.use_flash_attention,
        causal               = args.use_causal,
        use_cache            = args.use_cache,
        key_dtype            = args.key_dtype,
        value_dtype          = args.value_dtype,
        model_parallel_size  = args.model_parallel_size,
        tensor_parallel_size = args.tensor_parallel_size,
        tensor_parallel_rank = args.tensor_parallel_rank,
    )

    # 4. 构建模型
    model = ByteTransformer(config).to(args.device)

    # 5. 梯度检查点
    if getattr(args, "grad_checkpoint", False):
        model.gradient_checkpointing_enable()
        logger.info("✅ 已启用 Gradient Checkpointing")
 
    # 6. torch.compile（吞吐+显存双赢）
    if getattr(args, "use_torch_compile", False) and hasattr(torch, "compile"):
        mode = getattr(args, "compile_mode", "max-autotune")
        model = torch.compile(model, mode=mode, fullgraph=False)
        logger.info(f"🚀 torch.compile(mode='{mode}') 已启用")  # :contentReference[oaicite:1]{index=1}
 
    # 7. 并行包装
    if args.enable_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False
        )
    elif torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # 8. 打印参数规模
    param_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数总量: {param_cnt/1e6:.2f}M")

    return model, tokenizer, config


# -----------------------------------------------------------------------------
# 分布式训练
# -----------------------------------------------------------------------------
def init_distributed(args):
    """DDP 初始化（torchrun 环境下自动读取 env 变量）"""
    # ---- 检查 CUDA 设备 ----
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("DDP 已启用，但当前未检测到可用 CUDA，请安装 GPU 版 PyTorch 或设置 enable_ddp=False")

    # ---- torchrun 注入的环境变量 ----
    args.rank = int(os.environ["RANK"])            # 全局 rank
    args.world_size = int(os.environ["WORLD_SIZE"])# 全局进程数
    args.local_rank = int(os.environ["LOCAL_RANK"])# 本节点局部 rank

    # ---- 设备、进程组 ----
    backend="nccl" if torch.cuda.is_available() and sys.platform != "win32" else "gloo"
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend=backend,
        init_method="env://"
    )
    torch.distributed.barrier(device_ids=[args.local_rank])

def cleanup_distributed():
    """训练结束后销毁进程组，释放资源。"""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

def is_main_process(args) -> bool:
    """判断当前进程是否主进程（rank0）"""
    return (not args.enable_ddp) or args.rank == 0

def sync_metrics(total_loss, total_correct, total_tokens, args):
    if args.enable_ddp and dist.is_initialized():
        t_loss = torch.tensor(total_loss, device=args.device)
        t_correct = torch.tensor(total_correct, device=args.device)
        t_tokens = torch.tensor(total_tokens, device=args.device)

        dist.all_reduce(t_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_tokens, op=dist.ReduceOp.SUM)

        total_loss = t_loss.item()
        total_correct = t_correct.item()
        total_tokens = t_tokens.item()

    return total_loss, total_correct, total_tokens

# -----------------------------------------------------------------------------
# 单个 epoch 训练
# -----------------------------------------------------------------------------
def train_epoch(model, dataloader, tokenizer, optimizer, scaler, ctx, args, epoch, 
                total_iters, global_state, best_val_loss, logger, ckpt_mgr, killer):
    """执行一个 epoch 的前向、反向与梯度更新。"""

    model.train()

    global_step = global_state.global_step
    total_loss    = 0.0 # 累计总损失
    total_correct = 0   # 正确预测的 token 数
    total_tokens  = 0   # 总预测 token 数
    num_batches   = len(dataloader)
    
    # DDP sampler 洗牌
    if args.enable_ddp and isinstance(dataloader.sampler, DistributedSampler):
        dataloader.sampler.set_epoch(epoch)

    # 清理梯度
    optimizer.zero_grad(set_to_none=True)

    # 初始化吞吐率时间基准（为每步计算吞吐）
    start_wall = time.perf_counter()

    # --------------------------------------------------
    # 遍历数据集
    # --------------------------------------------------
    for step, batch in enumerate(dataloader):
        # 每 N 步再清一次，避免强同步导致吞吐下降
        if (step % args.empty_cache_interval) == 0:
            torch.cuda.empty_cache()
        input_ids          = batch["input_ids"].to(args.device)
        labels             = batch["labels"].to(args.device)
        loss_mask          = batch["loss_mask"].to(args.device)
        attention_mask     = (input_ids != tokenizer.pad_token_id).long()
        tokens_this_batch  = input_ids.numel()

        # 计算全局步数  
        global_state.global_step += 1

        # —— 动态学习率 ——
        lr = get_lr(global_state.global_step, total_iters, args)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # —— 前向 + 反向 ——
        with ctx:  # 支持 AMP autocast
            if torch.cuda.is_available():
                torch.compiler.cudagraph_mark_step_begin()  # 新图开始标记
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss    = outputs.loss / args.accumulation_steps
            logits  = outputs.logits.detach()  # [B, T, Vocab]
            predictions = torch.argmax(logits, dim=-1)  # [B, T],预测 token id
        # 自动混合精度
        scaler.scale(loss).backward()

        # —— 梯度累积 ——
        # 梯度累积到指定步数，进行梯度裁剪与参数更新
        if (global_state.global_step % args.accumulation_steps) == 0:
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
        total_loss    += loss_item * loss_mask.sum().item()

        # 准确率
        valid_mask    = (labels != -100)
        correct       = ((predictions == labels) & valid_mask).sum().item()
        total_correct += correct
        total_tokens  += valid_mask.sum().item()

        acc           = total_correct / (total_tokens + 1e-8)
        ppl           = math.exp(min(20, loss_item))
        
        # 计算每秒处理 token 数（通过时间差估算）
        tokens_per_s  = tokens_this_batch / max(1e-6, time.perf_counter() - start_wall)
        gpu_mem       = torch.cuda.memory_allocated(args.device) if torch.cuda.is_available() else 0

        if is_main_process(args):
            # —— 轻量 checkpoint(按步) ——
            # 每 N 步保存轻量 checkpoint
            if ckpt_mgr.should_save(global_state.global_step) and (step) % args.save_interval == 0:
                ckpt_mgr.save(model, 
                              optimizer, 
                              scaler,
                              epoch=epoch, 
                              step=global_state.global_step,
                              full=False)
            # —— 更新 Killer ——
            killer.update(epoch, global_step, best_val_loss)

            # —— 日志打印 & SwanLab ——
            if global_state.global_step % args.log_interval == 0:
                logger.info(
                    f"E{epoch+1} S{global_state.global_step} | Loss {loss_item:.4f} | PPL {ppl:.1f} | ACC {acc:.4f} "
                    f"| LR {lr:.6g} | GradNorm {total_grad_norm:.2f} | Tokens/s {tokens_per_s:.0f} "
                    f"| Mem {format_size(gpu_mem)}")
                if args.use_swanlab:
                    swanlab.log({
                        "loss": loss_item,
                        "perplexity": ppl,
                        "token_accuracy": acc,
                        "lr": lr,
                        "grad_norm": total_grad_norm,
                        "tokens_per_sec": tokens_per_s,
                        "gpu_mem": gpu_mem / (1024**2),  # 转 MB
                    })


    # —— 同步指标 ——
    if args.enable_ddp:
        total_loss, total_correct, total_tokens = sync_metrics(total_loss, total_correct, total_tokens, args)

    avg_loss = total_loss / (total_tokens + 1e-8)
    accuracy = total_correct / (total_tokens + 1e-8)
    if is_main_process(args):
        logger.info(f"[Epoch {epoch+1}] 平均损失 {avg_loss:.4f}, 准确率 {accuracy:.4f}")


# -----------------------------------------------------------------------------
# 主训练流程
# -----------------------------------------------------------------------------
def train(args, logger):
    """主训练流程，包含模型初始化、训练与验证、SwanLab 接入等。"""
    set_seed(args.seed, args)

    # --------- 初始化组件 ---------
    # 初始化 SwanLab 实验
    if is_main_process(args) and args.use_swanlab:
        swanlab.login(api_key=args.swanlab_api_key)
        swanlab.init(
            project=args.swanlab_project,
            experiment_name=args.swanlab_experiment_name,
            config=vars(args)
        )

     # 初始化检查点管理器
    ckpt_mgr = CheckpointManager(
        args.checkpoints_dir,
        keep_latest=args.keep_latest,
        keep_epoch=args.keep_epoch,
        keep_best=args.keep_best,
        save_every_n_steps=args.save_interval
    )

    # 初始化模型、分词器与配置
    model, tokenizer, config = init_model(args, logger)

    # 加载训练集
    # 1. 加载训练集
    train_dataset = PretrainDataset(
        args.train_data_path, 
        tokenizer, 
        max_length=config.max_seq_len,
        fields=args.dataset_loader.fields,
        template=args.dataset_loader.template if args.dataset_loader.template else None,
        add_bos=args.dataset_loader.add_bos
    )
    # 2. 根据分布式训练设置采样器
    if args.enable_ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        shuffle_flag  = False
    else:
        train_sampler = None
        shuffle_flag  = True
    # 3. 构建 Train_DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=shuffle_flag, 
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 加载验证集（如果提供）
    val_loader = None
    if args.val_data_path:
        # 1. 加载验证集
        val_dataset = PretrainDataset(
            args.val_data_path, 
            tokenizer, 
            max_length=config.max_seq_len,
            fields=args.dataset_loader.fields,
            template=args.dataset_loader.template if args.dataset_loader.template else None,
            add_bos=args.dataset_loader.add_bos
        )
        # 2. 根据分布式训练设置采样器
        if args.enable_ddp:
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
        else:
            val_sampler = None
        # 3. 构建 Val_DataLoader
        val_loader  = DataLoader(
            val_dataset, 
            batch_size=args.eval_batch_size, 
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )
        args.val_loader = val_loader

    # 构建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )

    # 学习率调度与AMP配置
    steps_per_epoch = len(train_loader)  
    total_iters = steps_per_epoch * args.epochs * (args.num_restarts + 1)
    use_amp     = args.amp or args.dtype in ['float16', 'bfloat16']
    scaler      = torch.amp.GradScaler(enabled=use_amp)
    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    ctx         = (nullcontext() if device_type == 'cpu'
                   else torch.amp.autocast(device_type=device_type, dtype=getattr(torch, args.dtype))
                   if use_amp else nullcontext())

    # 初始化 GracefulKiller，绑定 checkpoint 管理器
    killer = GracefulKiller(model, optimizer, scaler, ckpt_mgr, logger, sync=True)

    start_epoch   = 0
    global_state   = SimpleNamespace(global_step=0)
    val_loss      = None
    best_val_loss = float("inf")

    # ---------- 恢复 ----------
    try:
        logger.info("🔁 检测到历史检查点，正在恢复中…")
        ckpt = ckpt_mgr.load_latest(model, optimizer, scaler)
        start_epoch  = ckpt['epoch'] + 1
        global_state.global_step  = ckpt['global_step']
        best_val_loss = ckpt.get('val_loss', float('inf'))
        logger.info(f"🪄 已恢复到 epoch {start_epoch}, step {global_state.global_step}")
    except FileNotFoundError:
        logger.info("🆕 未检测到 checkpoint，开始全新训练")
        start_epoch, best_val_loss = 0, float('inf')

    # ---------- 训练 ----------
    try:
        for epoch in range(start_epoch, args.epochs):
            total_iters = len(train_loader)
            train_epoch(
                model, train_loader, tokenizer, optimizer, scaler, ctx, args, epoch,
                total_iters, global_state, best_val_loss, logger,  ckpt_mgr, killer
            )

            # -------------- 验证 (只在主进程) --------------------
            if val_loader and is_main_process(args) and (epoch + 1) % args.eval_interval == 0:
                val_loss = evaluate(model, val_loader, args, logger, epoch, progress=None)

                # 保存最优模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    ckpt_mgr.save(model, 
                                  optimizer, 
                                  scaler,
                                  epoch=epoch, 
                                  step=global_state.global_step,
                                  val_loss=best_val_loss, 
                                  full=True)
                    logger.info(f"🎉 验证集损失下降至 {best_val_loss:.4f}，保存最优模型权重。")

            # --------- 每 Epoch 末保存完整检查点 ------------------
            if is_main_process(args) and (epoch + 1) % args.save_interval == 0:
                ckpt_mgr.save(model, 
                              optimizer, 
                              scaler,
                              epoch=epoch,
                              step=global_state.global_step,
                              val_loss=val_loss, 
                              full=True)
                killer.update(epoch, global_state.global_step, best_val_loss)  # 保存完整检查点
            killer.update(epoch, global_state.global_step, best_val_loss)   # 同步最新 best

        if is_main_process(args):
            logger.info("✅ 训练完成")
    except Exception as e:
        if is_main_process(args):
            logger.error(f"训练异常: {e}")
            logger.info("💀 异常退出，正在保存检查点…")
        raise e
    finally:
        cleanup_distributed()


# 程序入口
if __name__ == "__main__":
    # ==== 加载配置 ====
    args = load_config("./configs/pretrain_config.yaml")

    # ==== 日志 ====
    log_file_path = os.path.join(args.logs_dir, f"pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = _build_logger(
        logger_name="ByteLogger",
        log_file=log_file_path,
        log_level=logging.DEBUG,
        console_level=logging.INFO,
        enable_color=True,
    )
    register_global_exception_handler(logger)

    # ==== 自动降级为 CPU（若无 CUDA）====                         
    if args.device.startswith("cuda") and not torch.cuda.is_available():  
        logger.warning("⚠️  当前环境未检测到 CUDA，已自动回退到 CPU 训练模式。")    
        args.device = "cpu"
        args.enable_ddp = False

    # ==== 初始化分布式 ====
    if args.enable_ddp:
        init_distributed(args)
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0

    # ==== 设置设备字符串 ====
    if args.device.startswith("cuda"):
        rank_device = f"cuda:{args.local_rank}"
        args.device = rank_device if args.enable_ddp else "cuda"
    else:
        logger.warning("⚠️  检测不到 CUDA，已自动切换到 CPU。若想用 GPU，请安装带 CUDA 的 PyTorch。")
        args.device = "cpu"

    # ==== 进程信息 ====
    if is_main_process(args):
        logger.info(f"主进程 rank={args.rank}, local_rank={args.local_rank}")
    else:
        logger.info(f"子进程 rank={args.rank}, local_rank={args.local_rank}")

    # ==== 训练 ====
    train(args, logger)
