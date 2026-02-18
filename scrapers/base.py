"""
Base Scraper
所有抓取器的抽象基类
"""
from abc import ABC, abstractmethod
from typing import Callable, Generic, List, Optional, TypeVar
import asyncio
import logging

from config import get_settings
from models import SourceType


logger = logging.getLogger(__name__)

T = TypeVar("T")  # 泛型返回类型
R = TypeVar("R")


class BaseScraper(ABC, Generic[T]):
    """
    抓取器抽象基类
    所有具体抓取器都需要继承此类并实现抽象方法
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._session = None
    
    @property
    @abstractmethod
    def source_type(self) -> SourceType:
        """返回数据源类型"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """返回抓取器名称"""
        pass
    
    @abstractmethod
    async def search(self, query: str, max_results: Optional[int] = None) -> List[T]:
        """
        搜索接口
        
        Args:
            query: 搜索关键词
            max_results: 最大返回结果数
            
        Returns:
            搜索结果列表
        """
        pass
    
    @abstractmethod
    async def get_details(self, item_id: str) -> Optional[T]:
        """
        获取单个项目的详细信息
        
        Args:
            item_id: 项目ID
            
        Returns:
            项目详情
        """
        pass
    
    def is_configured(self) -> bool:
        """
        检查是否已正确配置
        子类可以覆盖此方法来检查必要的API密钥等
        """
        return True
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()
    
    async def close(self):
        """清理资源"""
        if self._session:
            await self._session.close()
            self._session = None

    async def _run_blocking(self, func: Callable[..., R], *args, **kwargs) -> R:
        """在线程池中执行阻塞函数，统一替代 run_in_executor 样板代码。"""
        return await asyncio.to_thread(func, *args, **kwargs)
    
    def _log_search(self, query: str, count: int):
        """记录搜索日志"""
        logger.info(f"[{self.name}] Search '{query}' returned {count} results")
    
    def _log_error(self, message: str, error: Exception):
        """记录错误日志"""
        logger.error(f"[{self.name}] {message}: {error}")


class RateLimitedScraper(BaseScraper[T]):
    """
    带速率限制的抓取器基类
    """
    
    def __init__(self, requests_per_second: float = 1.0):
        super().__init__()
        self._rate_limit = requests_per_second
        self._last_request_time = 0.0
        self._lock = asyncio.Lock()
    
    async def _wait_for_rate_limit(self):
        """等待满足速率限制"""
        async with self._lock:
            import time
            current_time = time.time()
            time_since_last = current_time - self._last_request_time
            min_interval = 1.0 / self._rate_limit
            
            if time_since_last < min_interval:
                await asyncio.sleep(min_interval - time_since_last)
            
            self._last_request_time = time.time()
