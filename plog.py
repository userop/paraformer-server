"""
日志模块
 - 通用
 - 简洁
"""
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logger(
        name: str = "app",
        log_level: str = "INFO",
        log_dir: str = "logs",
        max_bytes: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5,
        console: bool = True,
        file: bool = True,
) -> logging.Logger:
    """
    配置并返回一个 logger 实例

    Args:
        name: logger 名称（建议用模块名）
        log_level: 日志级别 ("DEBUG", "INFO", ...)
        log_dir: 日志文件存储目录
        max_bytes: 单个日志文件最大字节数（轮转用）
        backup_count: 保留的旧日志文件数量
        console: 是否输出到控制台
        file: 是否写入文件
    """
    # 创建 logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # 避免重复添加 handler（重要！）
    if logger.handlers:
        return logger

    # 创建日志目录
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # 格式化
    formatter = logging.Formatter(
        fmt='%(asctime)s.%(msecs)03d [%(levelname)-8s] %(name)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # 控制台 Handler（带颜色）
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        # 简单彩色（Windows 可能不支持，Linux/macOS 正常）
        try:
            import colorlog
            color_formatter = colorlog.ColoredFormatter(
                fmt='%(log_color)s%(asctime)s.%(msecs)03d [%(levelname)-8s] %(name)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
            console_handler.setFormatter(color_formatter)
        except ImportError:
            console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 文件 Handler（轮转）
    if file:
        file_handler = RotatingFileHandler(
            filename=os.path.join(log_dir, f"{name}.log"),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# 全局默认 logger（可选）
_default_logger = setup_logger(name="global", console=True, file=True)


def get_logger(name: str = None) -> logging.Logger:
    """便捷函数：获取封装好的 logger"""
    if name is None:
        return _default_logger
    return setup_logger(name=name)
