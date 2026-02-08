"""
Custom Exceptions
自定义异常类
"""


class AcademicAgentError(Exception):
    """学术研究助手基础异常类"""
    
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self):
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigurationError(AcademicAgentError):
    """配置错误"""
    pass


class ScraperError(AcademicAgentError):
    """抓取器错误"""
    
    def __init__(self, message: str, source: str = None, **kwargs):
        super().__init__(message, kwargs)
        self.source = source


class StorageError(AcademicAgentError):
    """存储错误"""
    pass


class VectorStoreError(StorageError):
    """向量存储错误"""
    pass


class CacheError(StorageError):
    """缓存错误"""
    pass


class ProcessingError(AcademicAgentError):
    """文档处理错误"""
    pass


class ChunkingError(ProcessingError):
    """文本分块错误"""
    pass


class EmbeddingError(AcademicAgentError):
    """向量化错误"""
    
    def __init__(self, message: str, model: str = None, **kwargs):
        super().__init__(message, kwargs)
        self.model = model


class RetrievalError(AcademicAgentError):
    """检索错误"""
    pass


class LLMError(AcademicAgentError):
    """LLM 调用错误"""
    
    def __init__(self, message: str, provider: str = None, **kwargs):
        super().__init__(message, kwargs)
        self.provider = provider
