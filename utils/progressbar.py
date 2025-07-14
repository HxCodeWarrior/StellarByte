"""
进度条工具 - 用于可视化模型训练、验证和数据加载进度
作者：ByteWyrm
日期：2025-07-14
"""
import time
import sys
from datetime import timedelta

class ProgressBar:
    """训练进度条可视化工具"""
    
    def __init__(self, total_epochs=None, total_steps=None, desc="Training", bar_len=30):
        """
        初始化进度条
        
        参数:
            total_epochs (int): 总训练轮数
            total_steps (int): 总训练步数
            desc (str): 进度条描述
        """
        self.total_epochs = total_epochs
        self.total_steps = total_steps
        self.desc = desc
        self.bar_len = bar_len

        self.start_time = time.time()
        self.last_update_time = 0
        self.current_step = 0
        self.current_epoch = 0
        self.loss = None
        self.lr = None

        self._print_header()  # 首次立即打印
    
    def update(self, step=None, epoch=None, loss=None, lr=None, force=False):
        now = time.time()
        # 控制刷新的最小时间间隔（默认 0.1s）
        if not force and now - self.last_update_time < 0.1:
            return

        self.last_update_time = now
        self.current_step = step if step is not None else self.current_step
        self.current_epoch = epoch if epoch is not None else self.current_epoch
        self.loss = loss if loss is not None else self.loss
        self.lr = lr if lr is not None else self.lr

        self._print_progress()
    
    def _print_header(self):
        sys.stdout.write(f"{self.desc} started...\n")
        sys.stdout.flush()

    def _print_progress(self):
        elapsed = time.time() - self.start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))

        # 计算总体进度
        if self.total_steps and self.total_epochs:
            total = self.total_steps * self.total_epochs
            current = self.current_step + (self.current_epoch - 1) * self.total_steps
        elif self.total_steps:
            total = self.total_steps
            current = self.current_step
        elif self.total_epochs:
            total = self.total_epochs
            current = self.current_epoch
        else:
            total = 1
            current = 0

        percent = min(current / total, 1.0)
        filled_len = int(self.bar_len * percent)
        bar = "█" * filled_len + "-" * (self.bar_len - filled_len)

        # 剩余时间估计
        if percent > 0:
            remaining = elapsed / percent - elapsed
            remaining_str = str(timedelta(seconds=int(remaining)))
        else:
            remaining_str = "??:??:??"

        # 构造打印信息
        output = f"\r{self.desc}: [{bar}] {percent*100:5.1f}%"
        if self.total_epochs:
            output += f" | Epoch {self.current_epoch}/{self.total_epochs}"
        if self.total_steps:
            output += f" | Step {self.current_step}/{self.total_steps}"
        if self.loss is not None:
            output += f" | Loss: {self.loss:.4f}"
        if self.lr is not None:
            output += f" | LR: {self.lr:.2e}"
        output += f" | Time: {elapsed_str} < {remaining_str}"

        sys.stdout.write(output)
        sys.stdout.flush()
    
    def close(self):
        """完成进度条"""
        elapsed = time.time() - self.start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        print(f"\n{self.desc} completed in {elapsed_str}")


class DataLoaderProgress:
    """数据加载器进度条"""
    
    def __init__(self, dataloader, desc="Loading", bar_len=30):
        """
        初始化数据加载进度条
        
        参数:
            dataloader (DataLoader): PyTorch 数据加载器
            desc (str): 进度条描述
        """
        self.dataloader = dataloader
        self.total = len(dataloader)
        self.desc = desc
        self.bar_len = bar_len
        self.start_time = time.time()
        self.last_update_time = 0
    
    def __iter__(self):
        """迭代器接口"""
        for i, batch in enumerate(self.dataloader):
            now = time.time()
            if now - self.last_update_time >= 0.1 or i == 0:
                self._print(i + 1)
                self.last_update_time = now
            yield batch
        self._final()
        
    def _print(self, current):
        elapsed = time.time() - self.start_time
        percent = current / self.total
        filled_len = int(self.bar_len * percent)
        bar = '█' * filled_len + '-' * (self.bar_len - filled_len)

        if percent > 0:
            remaining = elapsed / percent - elapsed
            remaining_str = str(timedelta(seconds=int(remaining)))
        else:
            remaining_str = "??:??:??"

        output = f"\r{self.desc}: [{bar}] {percent*100:5.1f}% ({current}/{self.total})"
        output += f" | Time: {str(timedelta(seconds=int(elapsed)))} < {remaining_str}"
        sys.stdout.write(output)
        sys.stdout.flush()

    def _final(self):
        total_time = str(timedelta(seconds=int(time.time() - self.start_time)))
        sys.stdout.write(f"\n{self.desc} completed in {total_time}\n")
        sys.stdout.flush()
