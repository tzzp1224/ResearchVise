"""
Logger Configuration
统一日志配置
"""
import logging
import sys
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler
from rich.console import Console


# 全局 Console 实例
console = Console()

# 日志格式
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
LOG_FORMAT_SIMPLE = "%(message)s"

# 默认日志目录
LOG_DIR = Path(__file__).parent.parent / "logs"


def setup_logger(
    name: str = "academic_agent",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    use_rich: bool = True,
) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别
        log_file: 日志文件路径 (可选)
        use_rich: 是否使用 Rich 美化输出
        
    Returns:
        配置好的 Logger 实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加 handler
    if logger.handlers:
        return logger
    
    # Console Handler
    if use_rich:
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
        )
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT_SIMPLE))
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    
    console_handler.setLevel(level)
    logger.addHandler(console_handler)
    
    # File Handler (可选)
    if log_file:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_path = LOG_DIR / log_file
        
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "academic_agent") -> logging.Logger:
    """
    获取日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        Logger 实例
    """
    logger = logging.getLogger(name)
    
    # 如果没有配置过，进行默认配置
    if not logger.handlers:
        return setup_logger(name)
    
    return logger


# 预配置的模块日志器
def get_scraper_logger() -> logging.Logger:
    """获取抓取器日志器"""
    return get_logger("academic_agent.scrapers")


def get_storage_logger() -> logging.Logger:
    """获取存储日志器"""
    return get_logger("academic_agent.storage")


def get_processing_logger() -> logging.Logger:
    """获取处理日志器"""
    return get_logger("academic_agent.processing")


def get_rag_logger() -> logging.Logger:
    """获取 RAG 日志器"""
    return get_logger("academic_agent.rag")
