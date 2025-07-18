import logging
import os
import sys
import traceback
from datetime import datetime
from typing import Optional
from logging.handlers import RotatingFileHandler

try:
    from colorlog import ColoredFormatter
    _COLORLOG_AVAILABLE = True
except ImportError:
    _COLORLOG_AVAILABLE = False

__all__ = ["get_logger", "_build_logger", "register_global_exception_handler"]

# === 关键常量定义 ===
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def register_global_exception_handler(logger, trace_file: str = "traceback.log"):
    """
    注册全局未捕获异常处理器，将异常写入日志和本地文件。

    Args:
        logger (logging.Logger): 日志记录器。
        trace_file (str): Traceback 文件路径。
    """

    def handle_exception(exc_type, exc_value, exc_traceback):
        # 忽略 Ctrl+C 中断
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        # 日志记录
        logger.error("未捕获异常", exc_info=(exc_type, exc_value, exc_traceback))

        # 保存 traceback 到文件
        try:
            with open(trace_file, "a", encoding="utf-8") as f:
                f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 未捕获异常：\n")
                traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
        except Exception as file_err:
            logger.warning(f"无法写入 trace 文件：{file_err}")

    sys.excepthook = handle_exception

def _build_logger(
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
    logger_name: Optional[str] = None,
    enable_color: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    构建并返回一个 logger。

    Args:
        log_file (str, optional): 要写入的日志文件路径。如果为 None，则仅输出到控制台。
        log_level (int): 日志级别。
        logger_name (str, optional): 日志器名称。
        enable_color (bool): 是否启用控制台彩色日志。
        max_bytes (int): 单个日志文件的最大字节数。
        backup_count (int): 最多保留的备份数量。

    Returns:
        logging.Logger: 已配置好的 logger 实例。
    """
    logger = logging.getLogger(logger_name or log_file or "ByteLogger")

    # 避免重复添加 handler（多次 import 时）
    if logger.handlers:
        return logger

    logger.setLevel(log_level)
    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)


    # === 控制台 handler ===
    console_handler = logging.StreamHandler(sys.stdout)
    if enable_color and _COLORLOG_AVAILABLE:
        color_fmt = ColoredFormatter(
            "%(log_color)s" + LOG_FORMAT,
            datefmt=DATE_FORMAT,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
        console_handler.setFormatter(color_fmt)
    else:
        console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # === 文件 handler（带回滚）===
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_logger(
    log_dir: str,
    exp_name: str = "experiment",
    log_level: int = logging.INFO,
    logger_name: Optional[str] = None,
    enable_color: bool = True,
) -> logging.Logger:
    """
    创建带时间戳的日志记录器。

    Args:
        log_dir (str): 日志输出目录。
        exp_name (str): 实验名称。
        log_level (int): 日志等级。
        logger_name (str): 可选指定 logger 名称。
        enable_color (bool): 是否启用控制台彩色输出。

    Returns:
        logging.Logger
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{exp_name}_{ts}.log")
    return _build_logger(
        log_file=log_file,
        log_level=log_level,
        logger_name=logger_name,
        enable_color=enable_color,
    )
