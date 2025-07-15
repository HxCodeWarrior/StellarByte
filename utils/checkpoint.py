import os
import sys
import time
import signal
import threading
import torch
from collections import deque
from typing import Literal, Optional
try:
    from .logger import get_logger
except:
    from logger import get_logger

class CheckpointManager:
    """
    统一管理 three tiers of checkpoints:
    - latest_xxx.pt     : 轻量，保存频率高
    - epoch{n}_step{s}.pt: 完整，每个 epoch 1 份
    - best_{i}.pt        : Top‑K 最佳
    """
    def __init__(self,
                 save_dir: str,
                 keep_latest: int = 3,
                 keep_epoch: int = 5,
                 keep_best : int = 3,
                 save_every_n_steps: int = 1000):
        self.dir = save_dir
        os.makedirs(self.dir, exist_ok=True)

        self.keep_latest = keep_latest
        self.keep_epoch  = keep_epoch
        self.keep_best   = keep_best
        self.save_every_n_steps = save_every_n_steps

        self._latest_queue = deque()   # 轻量
        self._epoch_queue  = deque()   # 完整
        self._best_ckpts   = []        # (val_loss, path)

        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None


    # ---------- 公共接口 ----------
    def should_save(self, global_step: int) -> bool:
        return global_step % self.save_every_n_steps == 0

    def save(self,
             model,
             optimizer,
             scaler,
             epoch: int,
             step: int,
             val_loss: Optional[float] = None,
             full: bool = False):
        """
        Args:
            full: False➡轻量；True➡完整
        """
        state = self._collect_state(model, optimizer, scaler,
                                    epoch, step, val_loss, full)
        if full:
            filename = f"epoch{epoch}_step{step}.pt"
        else:
            filename = f"latest_step{step}.pt"

        path = os.path.join(self.dir, filename)
        self._async_save(state, path)

        # 队列管理 & top‑K
        self._enqueue(path, full)
        if val_loss is not None:
            self._update_best(path, val_loss)

    def save_sync(self, model, optimizer, scaler,
              epoch: int, step: int, val_loss=None):
        """同步写盘（主线程），专供 GracefulKiller 使用。"""
        state = self._collect_state(model, optimizer, scaler,
                                    epoch, step, val_loss, full=True)
        filename = f"epoch{epoch}_step{step}.pt"
        path = os.path.join(self.dir, filename)
        tmp = f"{path}.tmp"
        torch.save(state, tmp, _use_new_zipfile_serialization=True, pickle_protocol=4)
        os.replace(tmp, path)
        print(f"[Checkpoint] sync‑saved to {path}")

    def load_latest(self, model, optimizer=None, scaler=None):
        latest = self._find_most_recent("latest_")
        return self._load(latest, model, optimizer, scaler)

    def load_best(self, model, optimizer=None, scaler=None, idx: int = 0):
        if not self._best_ckpts:
            raise FileNotFoundError("No best checkpoint found.")
        path = sorted(self._best_ckpts)[idx][1]
        return self._load(path, model, optimizer, scaler)

    # ---------- 内部工具 ----------
    def _collect_state(self, model, optimizer, scaler,
                       epoch, step, val_loss, full):
        msd = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        state = {"model_state": msd,
                 "epoch": epoch,
                 "step": step,
                 "timestamp": time.time()}
        if full:
            state.update({
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict() if scaler else None,
                "val_loss": val_loss
            })
        return state

    def _async_save(self, payload, path):
        def worker():
            tmp = f"{path}.tmp"
            torch.save(payload, tmp, _use_new_zipfile_serialization=True, pickle_protocol=4)
            os.replace(tmp, path)          # 原子操作
        with self._lock:
            if self._thread and self._thread.is_alive():
                self._thread.join()
            self._thread = threading.Thread(target=worker, daemon=True)
            self._thread.start()

    # ---------- 队列 / best ----------
    def _enqueue(self, path, full: bool):
        queue = self._epoch_queue if full else self._latest_queue
        queue.append(path)
        keep = self.keep_epoch if full else self.keep_latest
        while len(queue) > keep:
            old = queue.popleft()
            if os.path.exists(old):
                os.remove(old)

    def _update_best(self, path, loss):
        self._best_ckpts.append((loss, path))
        self._best_ckpts.sort()            # 升序
        while len(self._best_ckpts) > self.keep_best:
            _, worst_path = self._best_ckpts.pop()
            if os.path.exists(worst_path):
                os.remove(worst_path)

    def _find_most_recent(self, prefix: str):
        files = [f for f in os.listdir(self.dir) if f.startswith(prefix)]
        if not files:
            raise FileNotFoundError(f"No {prefix} checkpoint in {self.dir}")
        return os.path.join(self.dir, sorted(files)[-1])

    def _load(self, path, model, optimizer=None, scaler=None):
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        if optimizer and "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if scaler and "scaler_state" in ckpt and ckpt["scaler_state"]:
            scaler.load_state_dict(ckpt["scaler_state"])
        return ckpt


class GracefulKiller:
    """
    捕获 SIGINT & SIGTERM，安全写入 **完整** 检查点：
      - model / optimizer / scaler / epoch / step / val_loss
      - 默认同步写盘；若磁盘极慢，可改 sync=False → 仍会 join
    """

    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scaler,                       # torch.cuda.amp.GradScaler | None
                 ckpt_mgr,                     # CheckpointManager
                 logger,
                 sync: bool = True,
                 best_val_loss: float = float("inf")):

        self.model          = model
        self.optimizer      = optimizer
        self.scaler         = scaler
        self.ckpt_mgr       = ckpt_mgr
        self.logger         = logger
        self.sync           = sync
        self.best_val_loss  = best_val_loss

        self._epoch = 0
        self._step  = 0
        self._lock  = threading.Lock()

        signal.signal(signal.SIGINT,  self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    # -------- 训练循环内实时刷新 ----------
    def update(self, epoch: int, step: int, best_val_loss: float | None = None):
        self._epoch, self._step = epoch, step
        if best_val_loss is not None:
            self.best_val_loss = best_val_loss

    # -------- 信号回调 ----------
    def _handler(self, signum, frame):
        sig = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        self.logger.warning(f"💀 收到 {sig}，写入完整检查点 …")

        with self._lock:
            # sync=True → 直接 torch.save；False → 走异步，但随后 join
            if self.sync:
                self.ckpt_mgr.save_sync(self.model, self.optimizer, self.scaler,
                                        epoch=self._epoch, step=self._step,
                                        val_loss=self.best_val_loss)
            else:
                self.ckpt_mgr.save(self.model, self.optimizer, self.scaler,
                                   epoch=self._epoch, step=self._step,
                                   val_loss=self.best_val_loss, full=True)
                # 等异步线程完成
                if self.ckpt_mgr._thread and self.ckpt_mgr._thread.is_alive():
                    self.ckpt_mgr._thread.join()

        self.logger.info("✅ 检查点保存完毕，安全退出。")
        sys.exit(0)

