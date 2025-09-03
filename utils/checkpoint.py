import os
import time
import signal
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List, Tuple

import torch
from types import SimpleNamespace
import logging

logger = logging.getLogger(__name__)


def _is_master(is_master: Optional[bool]) -> bool:
    """
    确保在分布式训练时只由主进程保存（caller可传入is_master）
    如果传入 None，则认为是主进程（适合非分布式）。
    """
    return True if is_master is None else bool(is_master)


def _model_state_dict(model: torch.nn.Module) -> Dict[str, Any]:
    """处理 DataParallel/DistributedDataParallel wrapper 的state_dict获取"""
    if hasattr(model, "module"):
        return model.module.state_dict()
    return model.state_dict()


class CheckpointManager:
    """
    Checkpoint 管理器（原子保存、最佳模型、轮替保留、断点恢复、异常/信号保护等）
    主要功能：
      - save_checkpoint(step, epoch, model, optimizer, scaler=None, metrics=None)
      - save_best_if_improved(metric_name, metric_value, ...)
      - load_checkpoint(path, model=None, optimizer=None, scaler=None)
      - register_signal_handlers() : 在接收到 SIGTERM/SIGINT 时自动保存
    """

    def __init__(
        self,
        output_dir: str,
        monitor: str = "loss",
        mode: str = "min",
        max_checkpoints: int = 5,
        prefix: str = "ckpt",
        keep_last_n: int = 1,
        is_master: Optional[bool] = None,
        save_tokenizer_fn: Optional[Callable[[str], None]] = None,
        save_config_fn: Optional[Callable[[str], None]] = None,
        best_model_weights_only: bool = True,
    ):
        """
        Args:
            output_dir              : 输出目录
            monitor                 : 用于决定"最优"的度量名称（如 "loss" 或 "perplexity"）
            mode                    : "min" 或 "max"；若 "min" 则 metric 越小越好
            max_checkpoints         : 保存最近 N 个检查点（包含最优和最后的）
            prefix                  : 保存文件名前缀
            keep_last_n             : 强制保留的最近 N 个 checkpoint（会与 max_checkpoints 一起工作）
            is_master               : 若在分布式训练中，应传入是否为主进程（rank0）。None 表示非分布式（主进程）
            save_tokenizer_fn       : 可选函数，负责把 tokenizer 保存到 output_dir （签名：fn(output_dir)）
            save_config_fn          : 可选函数，负责把配置保存到 output_dir （签名：fn(output_dir)）
            best_model_weights_only : 是否只保存最佳模型权重（不包含优化器状态等）
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.mode = mode
        self.max_checkpoints = int(max_checkpoints)
        self.prefix = prefix
        self.keep_last_n = int(keep_last_n)
        self.is_master = _is_master(is_master)
        self.save_tokenizer_fn = save_tokenizer_fn
        self.save_config_fn = save_config_fn
        self.best_model_weights_only = best_model_weights_only

        # internal state
        self.best_metric: Optional[float] = None
        self.best_path: Optional[Path] = None
        self.ckpt_history: List[Tuple[float, Path]] = []  # (timestamp, path)
        self._register_signal = False

    # ----------------------------- Utilities -----------------------------
    def _atomic_save(self, save_fn: Callable[[str], None], target_path: Path):
        """
        原子化保存：先写入临时文件/目录，再原子重命名到目标路径
        save_fn: 接收临时路径字符串并写入内容
        target_path: 最终目标
        """
        # 检查是否同一文件系统
        target_dev = os.stat(str(target_path.parent)).st_dev
        tmp_dir = tempfile.mkdtemp(prefix=str(target_path.name) + ".tmp.", dir=str(self.output_dir))
        tmp_dev = os.stat(tmp_dir).st_dev
        if target_dev != tmp_dev:
            logger.warning(
                f"Temp dir {tmp_dir} and target dir {target_path.parent} are on different file systems. "
                "Atomic move may not be guaranteed."
            )
        try:
            tmp_target = Path(tmp_dir) / target_path.name
            save_fn(str(tmp_target))
            # 如果是文件，确保目标父目录存在
            tmp_target_parent = target_path.parent
            tmp_target_parent.mkdir(parents=True, exist_ok=True)
            # 原子移动
            shutil.move(str(tmp_target), str(target_path))
        finally:
            # 删除临时目录（若仍在）
            if os.path.exists(tmp_dir):
                try:
                    shutil.rmtree(tmp_dir)
                except Exception:
                    pass

    def _cleanup_old_checkpoints(self):
        """
        循环删除旧的 checkpoint，保留最新 self.max_checkpoints 个（并保留 keep_last_n 的最近）
        ckpt_history 存的是 (timestamp, path)
        """
        # 按时间排序（最旧 -> 最新）
        self.ckpt_history.sort(key=lambda t: t[0])
        # 我们希望保留最新的 max_checkpoints，但也要确保最近的 keep_last_n 一定保留
        if len(self.ckpt_history) <= self.max_checkpoints:
            return

        # 先标记哪些需要保留（最新的 max_checkpoints）
        to_keep = set(p for _, p in self.ckpt_history[-self.max_checkpoints :])
        # 强制保留最近 keep_last_n
        to_keep.update(p for _, p in self.ckpt_history[-self.keep_last_n :])

        # 最佳模型路径确保保留
        if self.best_path:
            to_keep.add(self.best_path)

        # 删除不在 to_keep 的文件
        new_history = []
        for ts, path in self.ckpt_history:
            if path in to_keep:
                new_history.append((ts, path))
            else:
                try:
                    if path.exists():
                        path.unlink()
                        logger.info(f"Removed old checkpoint: {path}")
                except Exception as e:
                    logger.warning(f"Failed to delete checkpoint {path}: {e}")
        self.ckpt_history = new_history

    def _is_improved(self, metric_value: float) -> bool:
        if self.best_metric is None:
            return True
        if self.mode == "min":
            return metric_value < self.best_metric
        else:
            return metric_value > self.best_metric

    # ----------------------------- 保存/加载 -----------------------------
    def save_checkpoint(
        self,
        *,
        epoch: int,
        global_step: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        metrics: Optional[Dict[str, float]] = None,
        extra: Optional[Dict[str, Any]] = None,
        tag: Optional[str] = None,
        atomic: bool = True,
    ) -> Optional[Path]:
        """
        保存单个检查点。返回保存路径。
        - 名称格式: {prefix}_epoch{epoch}_step{global_step}_{timestamp}.pth
        - 会保存 model state_dict / optimizer / scaler / epoch / step / metrics / extra
        """
        if not self.is_master:
            # 非主进程不保存
            return None

        ts = int(time.time())
        filename = f"{self.prefix}_epoch{epoch}_step{global_step}_{ts}.pth"
        if tag:
            filename = f"{self.prefix}_{tag}_epoch{epoch}_step{global_step}_{ts}.pth"
        target_path = self.output_dir / filename

        def _save_to(path_str: str):
            # path_str 是 tmp 里面的最终文件名
            save_obj = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": _model_state_dict(model),
                "metrics": metrics or {},
                "extra": extra or {},
                "timestamp": ts,
            }
            if optimizer is not None:
                try:
                    save_obj["optimizer_state_dict"] = optimizer.state_dict()
                except Exception as e:
                    logger.warning(f"Failed to fetch optimizer state dict: {e}")
            if scaler is not None:
                try:
                    save_obj["scaler_state_dict"] = scaler.state_dict()
                except Exception:
                    # 某些版本的 GradScaler 可能无法序列化，忽略
                    logger.warning("Failed to save scaler state_dict; skipping.")

            torch.save(save_obj, path_str)

        if atomic:
            self._atomic_save(_save_to, target_path)
        else:
            _save_to(str(target_path))

        # 记录并执行清理
        self.ckpt_history.append((ts, target_path))
        self._cleanup_old_checkpoints()

        logger.info(f"Saved checkpoint: {target_path}")

        # 可选的 tokenizer / config 保存（如果用户提供了保存函数）
        if self.save_tokenizer_fn:
            try:
                self.save_tokenizer_fn(str(self.output_dir))
            except Exception as e:
                logger.warning(f"Failed to save tokenizer via provided function: {e}")
        if self.save_config_fn:
            try:
                self.save_config_fn(str(self.output_dir))
            except Exception as e:
                logger.warning(f"Failed to save config via provided function: {e}")

        return target_path

    def save_best(
        self,
        metric_name: str,
        metric_value: float,
        *,
        epoch: int,
        global_step: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        extra: Optional[Dict[str, Any]] = None,
        tag: Optional[str] = None,
    ) -> Optional[Path]:
        """
        如果 metric 更好，则保存为 best model（覆盖之前的 best）。
        返回 best_path (Path) if saved, else None.
        """
        if metric_name != self.monitor:
            logger.debug(f"Requested saving best for metric {metric_name}, but manager monitors {self.monitor}. Skipping.")
            return None

        if not _is_master(self.is_master):
            return None

        if self._is_improved(metric_value):
            logger.info(f"Metric improved ({self.best_metric} -> {metric_value}). Saving best model.")
            best_fn = f"{self.prefix}_best_{metric_name}.pth"
            best_path = self.output_dir / best_fn

            def _save_best(tmp_path: str):
                if self.best_model_weights_only:
                    # 只保存模型权重
                    torch.save(_model_state_dict(model), tmp_path)
                else:
                    # 保存完整检查点
                    save_obj = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "model_state_dict": _model_state_dict(model),
                        "metrics": {metric_name: metric_value},
                        "extra": extra or {},
                        "timestamp": int(time.time()),
                    }
                    if optimizer is not None:
                        try:
                            save_obj["optimizer_state_dict"] = optimizer.state_dict()
                        except Exception as e:
                            logger.warning(f"Failed to fetch optimizer state dict: {e}")
                    if scaler is not None:
                        try:
                            save_obj["scaler_state_dict"] = scaler.state_dict()
                        except Exception:
                            logger.warning("Failed to save scaler state dict; skipping.")
                    torch.save(save_obj, tmp_path)

            # 原子写入覆盖 best_path（先写 tmp 再移动）
            self._atomic_save(_save_best, best_path)
            self.best_metric = float(metric_value)
            self.best_path = best_path
            self.ckpt_history.append((int(time.time()), best_path))
            logger.info(f"Saved new best model to: {best_path}")
            return best_path
        return None

    def load_checkpoint(
        self,
        path: str,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        map_location: Optional[str] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        加载检查点并（可选）恢复到 model/optimizer/scaler。
        返回 checkpoint 对象的 dict。
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")

        map_loc = map_location or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(str(p), map_location=map_loc)

        # 处理只包含模型权重的检查点
        if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
            # 可能是只保存了模型权重的文件
            if model is not None:
                try:
                    model.load_state_dict(ckpt, strict=strict)
                    logger.info(f"Loaded model weights from {p}")
                except Exception as e:
                    logger.warning(f"Failed to load model weights: {e}")
            return {"model_state_dict": ckpt} if isinstance(ckpt, dict) else ckpt
            
        # 加载模型权重（如果提供model）
        if model is not None and "model_state_dict" in ckpt:
            try:
                model_state = ckpt["model_state_dict"]
                model.load_state_dict(model_state, strict=strict)
                logger.info(f"Loaded model_state_dict from {p}")
            except Exception as e:
                logger.warning(f"Failed to load model_state_dict strictly: {e}")
                # 若 strict 失败，尝试非 strict
                if strict:
                    logger.info("Retry loading with strict=False")
                    model.load_state_dict(ckpt["model_state_dict"], strict=False)

        # 恢复 optimizer / scaler
        if optimizer is not None and "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                logger.info("Loaded optimizer state.")
            except Exception as e:
                logger.warning(f"Failed to load optimizer state: {e}")

        if scaler is not None and "scaler_state_dict" in ckpt:
            try:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
                logger.info("Loaded GradScaler state.")
            except Exception as e:
                logger.warning(f"Failed to load scaler state: {e}")

        return ckpt

    # ----------------------------- 保护措施：异常 & 信号 -----------------------------
    def _signal_handler(self, signum, frame):
        """
        简单信号处理器：在收到 SIGINT/SIGTERM 时在 output_dir 处写入 emergency checkpoint文件（标记）
        注意：因为 signal handler 可能在任何线程/状态被调用，我们在这里只写一个标记文件，
        而真正的完整保存需要训练循环在捕获 KeyboardInterrupt / Exception 时主动调用 save_checkpoint.
        """
        try:
            pid = os.getpid()
            flag = self.output_dir / f"{self.prefix}_EMERGENCY_SIGNAL_{signum}_pid{pid}.flag"
            flag.write_text(str(time.time()))
            logger.warning(f"Received signal {signum} in pid {pid}. Wrote emergency flag to {flag}. Please call save_checkpoint in training loop to persist a full checkpoint.")
        except Exception as e:
            logger.error(f"Failed to write emergency flag: {e}")

    def register_signal_handlers(self, override: bool = False):
        """
        注册 SIGINT 与 SIGTERM 的 handler（仅在主进程注册）
        若要同时在子进程注册，则需要自行为子进程做同样的事情（此处默认只在主进程）。
        """
        if not _is_master(self.is_master):
            return
        if self._register_signal and not override:
            return
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self._register_signal = True
        logger.info("Registered SIGINT and SIGTERM handlers for checkpoint manager.")

    # ----------------------------- Helper getters -----------------------------
    def latest_checkpoint(self) -> Optional[Path]:
        if not self.ckpt_history:
            return None
        # 按时间倒序返回最新
        return sorted(self.ckpt_history, key=lambda t: t[0])[-1][1]

    def list_checkpoints(self) -> List[Path]:
        return [p for _, p in sorted(self.ckpt_history, key=lambda t: t[0])]
