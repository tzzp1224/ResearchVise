"""
Cache
缓存模块
"""
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from pathlib import Path
from datetime import datetime, timedelta
import json
import hashlib
import pickle
import logging
from functools import wraps


logger = logging.getLogger(__name__)


class BaseCache(ABC):
    """
    缓存抽象基类
    """
    
    def __init__(self, ttl: Optional[int] = None):
        """
        初始化缓存
        
        Args:
            ttl: 缓存过期时间 (秒), None = 永不过期
        """
        self.ttl = ttl
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """设置缓存值"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> None:
        """删除缓存"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """清空缓存"""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        pass
    
    def get_or_set(
        self,
        key: str,
        factory,
        ttl: Optional[int] = None,
    ) -> Any:
        """
        获取缓存，不存在则通过 factory 创建
        
        Args:
            key: 缓存键
            factory: 值工厂函数
            ttl: 过期时间
            
        Returns:
            缓存值或新创建的值
        """
        value = self.get(key)
        if value is not None:
            return value
        
        value = factory()
        self.set(key, value, ttl)
        return value
    
    @staticmethod
    def make_key(*args, **kwargs) -> str:
        """
        根据参数生成缓存键
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            缓存键
        """
        key_parts = [str(arg) for arg in args]
        key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
        key_string = ":".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()


class MemoryCache(BaseCache):
    """
    内存缓存
    简单的字典缓存，适合开发和测试
    """
    
    def __init__(self, ttl: Optional[int] = None, max_size: int = 1000):
        """
        初始化内存缓存
        
        Args:
            ttl: 默认过期时间 (秒)
            max_size: 最大缓存条目数
        """
        super().__init__(ttl)
        self.max_size = max_size
        self._cache: Dict[str, Dict] = {}
    
    def _is_expired(self, entry: Dict) -> bool:
        """检查是否过期"""
        if entry.get("expires_at") is None:
            return False
        return datetime.now() > entry["expires_at"]
    
    def _cleanup(self):
        """清理过期条目"""
        expired_keys = [
            k for k, v in self._cache.items()
            if self._is_expired(v)
        ]
        for key in expired_keys:
            del self._cache[key]
        
        # 如果仍然超过限制，删除最旧的
        if len(self._cache) > self.max_size:
            sorted_keys = sorted(
                self._cache.keys(),
                key=lambda k: self._cache[k].get("created_at", datetime.min),
            )
            for key in sorted_keys[:len(self._cache) - self.max_size]:
                del self._cache[key]
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        entry = self._cache.get(key)
        if entry is None:
            return None
        
        if self._is_expired(entry):
            del self._cache[key]
            return None
        
        return entry["value"]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """设置缓存值"""
        self._cleanup()
        
        ttl = ttl or self.ttl
        expires_at = None
        if ttl:
            expires_at = datetime.now() + timedelta(seconds=ttl)
        
        self._cache[key] = {
            "value": value,
            "created_at": datetime.now(),
            "expires_at": expires_at,
        }
    
    def delete(self, key: str) -> None:
        """删除缓存"""
        self._cache.pop(key, None)
    
    def clear(self) -> None:
        """清空缓存"""
        self._cache.clear()
    
    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        return self.get(key) is not None
    
    def size(self) -> int:
        """返回缓存大小"""
        return len(self._cache)


class DiskCache(BaseCache):
    """
    磁盘缓存
    持久化缓存，适合长期存储
    """
    
    def __init__(
        self,
        cache_dir: str = "./cache",
        ttl: Optional[int] = None,
        use_json: bool = False,
    ):
        """
        初始化磁盘缓存
        
        Args:
            cache_dir: 缓存目录
            ttl: 默认过期时间 (秒)
            use_json: 是否使用 JSON 格式 (False = pickle)
        """
        super().__init__(ttl)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_json = use_json
        
        # 元数据文件
        self.meta_file = self.cache_dir / "_meta.json"
        self._load_meta()
    
    def _load_meta(self):
        """加载元数据"""
        if self.meta_file.exists():
            with open(self.meta_file, 'r') as f:
                self._meta = json.load(f)
        else:
            self._meta = {}
    
    def _save_meta(self):
        """保存元数据"""
        with open(self.meta_file, 'w') as f:
            json.dump(self._meta, f)
    
    def _get_path(self, key: str) -> Path:
        """获取缓存文件路径"""
        ext = ".json" if self.use_json else ".pkl"
        return self.cache_dir / f"{key}{ext}"
    
    def _is_expired(self, key: str) -> bool:
        """检查是否过期"""
        meta = self._meta.get(key, {})
        expires_at = meta.get("expires_at")
        if expires_at is None:
            return False
        return datetime.now().timestamp() > expires_at
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        path = self._get_path(key)
        
        if not path.exists():
            return None
        
        if self._is_expired(key):
            self.delete(key)
            return None
        
        try:
            if self.use_json:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                with open(path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load cache {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """设置缓存值"""
        path = self._get_path(key)
        
        ttl = ttl or self.ttl
        expires_at = None
        if ttl:
            expires_at = (datetime.now() + timedelta(seconds=ttl)).timestamp()
        
        # 保存数据
        try:
            if self.use_json:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(value, f, ensure_ascii=False, default=str)
            else:
                with open(path, 'wb') as f:
                    pickle.dump(value, f)
        except Exception as e:
            logger.error(f"Failed to save cache {key}: {e}")
            return
        
        # 更新元数据
        self._meta[key] = {
            "created_at": datetime.now().timestamp(),
            "expires_at": expires_at,
        }
        self._save_meta()
    
    def delete(self, key: str) -> None:
        """删除缓存"""
        path = self._get_path(key)
        if path.exists():
            path.unlink()
        self._meta.pop(key, None)
        self._save_meta()
    
    def clear(self) -> None:
        """清空缓存"""
        import shutil
        shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._meta = {}
        self._save_meta()
    
    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        return self.get(key) is not None
    
    def size(self) -> int:
        """返回缓存文件数量"""
        return len(list(self.cache_dir.glob("*"))) - 1  # 减去 meta 文件


# 缓存装饰器
def cached(cache: BaseCache, key_prefix: str = ""):
    """
    缓存装饰器
    
    Usage:
        @cached(my_cache, "my_func")
        def expensive_function(arg1, arg2):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            key = f"{key_prefix}:{cache.make_key(*args, **kwargs)}"
            
            # 尝试获取缓存
            result = cache.get(key)
            if result is not None:
                return result
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 存入缓存
            cache.set(key, result)
            
            return result
        
        return wrapper
    return decorator


# 工厂函数
_default_memory_cache = None
_default_disk_cache = None


def get_cache(
    provider: str = "memory",
    cache_dir: str = "./cache",
    ttl: Optional[int] = None,
    **kwargs,
) -> BaseCache:
    """
    获取缓存实例
    
    Args:
        provider: 提供商 (memory, disk)
        cache_dir: 磁盘缓存目录
        ttl: 过期时间
        
    Returns:
        缓存实例
    """
    global _default_memory_cache, _default_disk_cache
    
    if provider == "memory":
        if _default_memory_cache is None:
            _default_memory_cache = MemoryCache(ttl=ttl, **kwargs)
        return _default_memory_cache
    
    elif provider == "disk":
        if _default_disk_cache is None:
            _default_disk_cache = DiskCache(cache_dir=cache_dir, ttl=ttl, **kwargs)
        return _default_disk_cache
    
    else:
        raise ValueError(f"Unknown cache provider: {provider}")
