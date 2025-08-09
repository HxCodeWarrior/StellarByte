import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler
from typing import Optional, Union

try:
    import colorama
    colorama.init(autoreset=True)
    COLORAMA_IMPORTED = True
except ImportError:
    COLORAMA_IMPORTED = False

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[37m',     # 白色
        'INFO': '\033[36m',      # 青色
        'WARNING': '\033[33m',   # 黄色
        'ERROR': '\033[31m',     # 红色
        'CRITICAL': '\033[41m',  # 红底白字
    }
    RESET = '\033[0m'

    def __init__(self, fmt=None, datefmt=None, use_color=True):
        super().__init__(fmt, datefmt)
        self.use_color = use_color and COLORAMA_IMPORTED

    def format(self, record):
        msg = super().format(record)
        if self.use_color and record.levelname in self.COLORS:
            color = self.COLORS[record.levelname]
            return f"{color}{msg}{self.RESET}"
        else:
            return msg

def str_to_log_level(level: Union[str, int]) -> int:
    if isinstance(level, int):
        return level
    level = level.upper()
    mapping = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }
    return mapping.get(level, logging.DEBUG)

def setup_logger(
    name: Optional[str] = None,
    log_dir: str = "logs",
    log_file: str = "app.log",
    level: Union[str, int] = logging.DEBUG,
    console_level: Union[str, int] = logging.INFO,
    file_level: Union[str, int] = logging.DEBUG,
    use_color: bool = True,
    when: str = "midnight",
    backup_count: int = 7,
    encoding: str = "utf-8",
    is_rank_0: Optional[bool] = True
) -> logging.Logger:
    """
    创建并返回一个带彩色控制台输出和定时轮转文件日志的Logger。

    Args:
        name: Logger名称，None为root logger。
        log_dir: 日志文件保存目录。
        log_file: 日志文件名。
        level: logger总体日志级别。
        console_level: 控制台输出日志级别。
        file_level: 文件输出日志级别。
        use_color: 控制台是否启用颜色。
        when: 轮转周期，参考TimedRotatingFileHandler，如 'midnight', 'H', 'D' 等。
        backup_count: 保留旧日志文件个数。
        encoding: 日志文件编码。
        is_rank_0: 是否为rank 0进程，只有rank 0写文件日志。默认为True

    Returns:
        配置完成的Logger实例。
    """
    logger = logging.getLogger(name)
    logger.setLevel(str_to_log_level(level))

    if logger.hasHandlers():
        logger.handlers.clear()

    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    fmt = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # 控制台Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(str_to_log_level(console_level))
    console_formatter = ColoredFormatter(fmt, datefmt, use_color=use_color)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # 文件Handler - 定时轮转日志
    file_handler = TimedRotatingFileHandler(
        filename=log_path,
        when=when,
        backupCount=backup_count,
        encoding=encoding,
        utc=False,
    )

    if is_rank_0:
        file_handler.setLevel(str_to_log_level(file_level))
        file_formatter = logging.Formatter(fmt, datefmt)
        file_handler.setFormatter(file_formatter)

        logger.addHandler(file_handler)

        # 捕获异常堆栈信息示例（默认格式已支持stack info）
        # 使用 logger.error("msg", exc_info=True) 即可输出异常堆栈

    return logger