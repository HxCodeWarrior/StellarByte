import logging
import os
import sys
from datetime import datetime
from typing import Optional

__all__ = ["get_logger"]


def _build_logger(log_file: Optional[str] = None,
                  log_level: int = logging.INFO) -> logging.Logger:
    """
    构建并返回一个 logger。

    Args:
        log_file (str, optional): 要写入的日志文件路径。如果为 None，则仅输出到控制台。
        log_level (int): 日志级别。

    Returns:
        logging.Logger: 已配置好的 logger 实例。
    """
    logger = logging.getLogger(log_file or "ByteLogger")
    # 避免重复添加 handler（多次 import 时）
    if logger.handlers:
        return logger

    logger.setLevel(log_level)
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- 控制台 ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # --- 文件 ---
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    logger.propagate = False   # 禁止向父 logger 传递，防止重复打印
    return logger


def get_logger(log_dir: str,
               exp_name: str = "experiment",
               log_level: int = logging.INFO) -> logging.Logger:
    """
    根据日期‑时间与实验名自动生成日志文件，并返回 logger。

    Args:
        log_dir (str): 日志目录。
        exp_name (str): 实验名称，用于文件名前缀。
        log_level (int): 日志级别。

    Returns:
        logging.Logger
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{exp_name}_{ts}.log")
    return _build_logger(log_file, log_level)
