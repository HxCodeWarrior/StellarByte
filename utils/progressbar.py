"""
基于 Rich 的高颜值终端进度条工具
作者：ByteWyrm
日期：2025-07-14
"""
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
)
from contextlib import contextmanager

console = Console()


def create_shared_progress():
    return Progress(
        SpinnerColumn(style="bold cyan"),
        TextColumn("[bold blue]{task.fields[desc]}", justify="right"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        expand=True,
    )


class RichProgressBar:
    """用于训练 epoch + DataLoader 的多任务共享进度条"""

    def __init__(self, total_steps, total_batches, total_epochs, desc="Training"):
        self.progress = create_shared_progress()
        self.train_task = self.progress.add_task("", total=total_steps, desc=desc)
        self.load_task = self.progress.add_task("", total=total_batches, desc="Loading batch")
        self.total_epochs = total_epochs

    def __enter__(self):
        self.progress.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress.__exit__(exc_type, exc_val, exc_tb)

    def update_train(self, step, epoch, loss=None, lr=None):
        msg = f"Epoch {epoch}/{self.total_epochs} | Step {step}"
        if loss is not None:
            msg += f" | Loss: {loss:.4f}"
        if lr is not None:
            msg += f" | LR: {lr:.2e}"
        self.progress.update(self.train_task, completed=step, description=msg)

    def update_loader(self, step):
        self.progress.update(self.load_task, completed=step)
