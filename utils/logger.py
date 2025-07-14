import logging
import os
import sys

class Logger:
    def __init__(self, log_file=None):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        # 控制台输出
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # 文件输出
        if log_file:
            # 确保使用绝对路径
            log_file = os.path.abspath(log_file)
            log_dir = os.path.dirname(log_file)
            os.makedirs(log_dir, exist_ok=True)

            fh = logging.FileHandler(log_file, encoding='utf-8')    # 指定UTF-8编码，避免输出的日志文本乱码
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
    
    def info(self, message):
        self.logger.info(message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def error(self, message):
        self.logger.error(message)

