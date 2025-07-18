from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
)
from rich.table import Table
import time


class ProgressBarManager:
    def __init__(self, log_interval=10):
        self.console = Console()
        self.epoch = 0
        self.total_epochs = 0
        self.phase = 'train'
        self.progress = None  # 主进度对象由下面初始化
        self.train_task = None
        self.val_task = None
        self.dataloader_task = None
        self.val_start_time = None
        self.val_metrics = []
        self.log_interval = log_interval
        self._train_step = 0

    def start_training(self, total_epochs):
        self.total_epochs = total_epochs
        self.console.rule("[bold green]🚀 Start Training")
        # 主进度条对象，支持训练和验证共享但区分颜色
        self.progress = Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            auto_refresh=True,
            transient=False
        )
        self.progress.start()

    def end_training(self):
        self.console.rule("[bold green]🏁 Training Complete")
        if self.progress:
            self.progress.stop()

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.console.rule(f"[bold yellow]🌟 Epoch {epoch+1}/{self.total_epochs}")

    def start_dataloader(self, total_batches, desc="🔄 Loading Data"):
        if self.dataloader_task is not None:
            self.progress.remove_task(self.dataloader_task)
        self.dataloader_task = self.progress.add_task(desc, total=total_batches)

    def update_dataloader(self):
        if self.dataloader_task is not None:
            self.progress.update(self.dataloader_task, advance=1)

    def end_dataloader(self):
        if self.dataloader_task is not None:
            self.progress.remove_task(self.dataloader_task)
            self.dataloader_task = None

    def start_phase(self, total_steps, phase='train'):
        self.phase = phase
        self._train_step = 0  # 重置步数计数器
        desc = "🧠 Training" if phase == 'train' else "🧪 Validating"
        style = "bold cyan" if phase == 'train' else "bold green"

        # 清理旧任务
        if self.train_task is not None and phase == 'train':
            self.progress.remove_task(self.train_task)
            self.train_task = None
        if self.val_task is not None and phase != 'train':
            self.progress.remove_task(self.val_task)
            self.val_task = None

        task_id = self.progress.add_task(desc, total=total_steps, style=style)
        if phase == 'train':
            self.train_task = task_id
        else:
            self.val_task = task_id
            self.val_start_time = time.time()
            self.val_metrics.clear()

    def update_phase(self, loss=None, acc=None, **kwargs):
        """
        训练/验证阶段更新，验证时可传更多指标
        """
        task_id = self.train_task if self.phase == 'train' else self.val_task
        if task_id is not None:
            if self.phase == 'train':
                self._train_step += 1
                self.progress.update(task_id, advance=1)
                # 仅每 log_interval 步刷新一次 description，避免性能瓶颈
                if self._train_step % self.log_interval == 0 or self._train_step == 1:
                    msg = f"Train | Epoch {self.epoch + 1}/{self.total_epochs} | Step {self._train_step}"
                    if loss is not None:
                        msg += f" | Loss: {loss:.4f}"
                    self.progress.update(task_id, advance=1, description=msg)
            else:
                # 验证阶段显示更多指标
                msg = f"Val | Epoch {self.epoch + 1}/{self.total_epochs}"
                if loss is not None:
                    msg += f" | Loss: {loss:.4f}"
                    self.val_metrics.append(('Loss', loss))
                if acc is not None:
                    msg += f" | Acc: {acc:.2%}"
                    self.val_metrics.append(('Acc', acc))
                for k, v in kwargs.items():
                    msg += f" | {k.capitalize()}: {v}"
                    self.val_metrics.append((k.capitalize(), v))
                self.progress.update(task_id, advance=1, description=msg)

    def end_phase(self):
        task_id = self.train_task if self.phase == 'train' else self.val_task
        if task_id is not None:
            self.progress.remove_task(task_id)
            if self.phase == 'train':
                self.train_task = None
            else:
                self.val_task = None
                # 打印验证阶段统计汇总表
                self._print_val_summary()

    def _print_val_summary(self):
        if not self.val_metrics:
            return
        
        table = Table(title=f"Validation Summary Epoch {self.epoch + 1}")
        table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right", style="magenta")
        # 汇总相同指标的均值
        from collections import defaultdict
        metric_sum = defaultdict(float)
        metric_count = defaultdict(int)
        for k, v in self.val_metrics:
            metric_sum[k] += v
            metric_count[k] += 1
        for k in metric_sum:
            avg = metric_sum[k] / metric_count[k]
            if isinstance(avg, float):
                table.add_row(k, f"{avg:.4f}")
            else:
                table.add_row(k, str(avg))
        elapsed = time.time() - self.val_start_time
        table.caption = f"⏳ Validation took {elapsed:.2f} seconds"
        self.console.print(table)
