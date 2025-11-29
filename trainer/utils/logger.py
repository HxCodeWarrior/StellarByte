###############################################################################
# 文件: utils/logger.py
###############################################################################
"""
日志与同步打印工具
- 提供主进程判定、统一打印接口、以及简单的文件日志写入
- 采用 Google style docstrings 和详尽中文注释
"""

import os
import sys
import time
import threading


def is_main_process():
    """判断当前进程是否为主进程（在分布式训练中为 rank 0）。

    Returns:
        bool: 如果未初始化分布式或当前进程 rank==0 则为 True。
    """
    try:
        import torch.distributed as dist
        return not dist.is_initialized() or dist.get_rank() == 0
    except Exception:
        return True


class Logger:
    """简单的进程安全日志器。

    特性：
    - 仅在主进程输出到 stdout，避免多卡重复打印
    - 可选写入文件（append 模式），并自动刷写
    - 提供时间戳和等级信息
    """

    def __init__(self, logfile: str = None, to_stdout: bool = True):
        """构造函数。

        Args:
            logfile (str): 可选的日志文件路径；为 None 则不写文件。
            to_stdout (bool): 主进程是否打印到标准输出。
        """
        self.logfile = logfile
        self.to_stdout = to_stdout
        self._lock = threading.Lock()
        if logfile:
            os.makedirs(os.path.dirname(logfile) or '.', exist_ok=True)

    def _format(self, level: str, msg: str) -> str:
        ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        return f"[{ts}] [{level}] {msg}"

    def info(self, msg: str):
        self._log('INFO', msg)

    def warn(self, msg: str):
        self._log('WARN', msg)

    def error(self, msg: str):
        self._log('ERROR', msg)

    def _log(self, level: str, msg: str):
        text = self._format(level, msg)
        with self._lock:
            if is_main_process() and self.to_stdout:
                print(text)
                sys.stdout.flush()
            if self.logfile:
                with open(self.logfile, 'a', encoding='utf-8') as f:
                    f.write(text + '\n')