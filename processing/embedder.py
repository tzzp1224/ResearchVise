"""
Embedder
文本向量化模块 - 支持多种 Embedding 提供商
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Union
import asyncio
import logging
import os
from functools import lru_cache

import numpy as np
import httpx

from config import get_settings
from utils.exceptions import EmbeddingError


logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """
    Embedder 抽象基类
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """返回向量维度"""
        pass
    
    @abstractmethod
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        同步嵌入文本
        
        Args:
            texts: 单个文本或文本列表
            
        Returns:
            向量数组 (n_texts, dimension)
        """
        pass
    
    async def aembed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        异步嵌入文本
        
        Args:
            texts: 单个文本或文本列表
            
        Returns:
            向量数组
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed, texts)
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        嵌入查询文本 (某些模型对查询有特殊处理)
        
        Args:
            query: 查询文本
            
        Returns:
            向量
        """
        return self.embed(query)[0]
    
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """
        嵌入文档列表
        
        Args:
            documents: 文档列表
            
        Returns:
            向量数组
        """
        return self.embed(documents)


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    基于 SentenceTransformers 的 Embedder
    支持多种开源模型，本地运行
    """
    
    # 推荐的模型
    RECOMMENDED_MODELS = {
        "small": "all-MiniLM-L6-v2",           # 384维，速度快
        "medium": "all-mpnet-base-v2",          # 768维，效果好
        "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",  # 384维，多语言
        "chinese": "shibing624/text2vec-base-chinese",  # 768维，中文优化
    }
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        """
        初始化 SentenceTransformer Embedder
        
        Args:
            model_name: 模型名称
            device: 运行设备 (cuda/cpu/mps)
            normalize: 是否归一化向量
        """
        super().__init__(model_name)
        self.device = device
        self.normalize = normalize
        self._dimension = None
    
    def _load_model(self):
        """延迟加载模型"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                
                logger.info(f"Loading SentenceTransformer model: {self.model_name}")
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                )
                self._dimension = self._model.get_sentence_embedding_dimension()
                logger.info(f"Model loaded. Dimension: {self._dimension}")
                
            except ImportError:
                raise EmbeddingError(
                    "Please install sentence-transformers: pip install sentence-transformers",
                    model=self.model_name,
                )
            except Exception as e:
                raise EmbeddingError(
                    f"Failed to load model: {e}",
                    model=self.model_name,
                )
        
        return self._model
    
    @property
    def dimension(self) -> int:
        """返回向量维度"""
        if self._dimension is None:
            self._load_model()
        return self._dimension
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """嵌入文本"""
        model = self._load_model()
        
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            embeddings = model.encode(
                texts,
                normalize_embeddings=self.normalize,
                show_progress_bar=len(texts) > 100,
            )
            return np.array(embeddings)
            
        except Exception as e:
            raise EmbeddingError(
                f"Embedding failed: {e}",
                model=self.model_name,
            )
    
    def embed_query(self, query: str) -> np.ndarray:
        """嵌入查询"""
        # 某些模型对查询有特殊前缀
        if "e5" in self.model_name.lower():
            query = f"query: {query}"
        return self.embed(query)[0]
    
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """嵌入文档"""
        # 某些模型对文档有特殊前缀
        if "e5" in self.model_name.lower():
            documents = [f"passage: {doc}" for doc in documents]
        return self.embed(documents)


class OpenAIEmbedder(BaseEmbedder):
    """
    OpenAI Embeddings API
    """
    
    MODELS = {
        "small": "text-embedding-3-small",   # 1536维
        "large": "text-embedding-3-large",   # 3072维
        "ada": "text-embedding-ada-002",     # 1536维 (旧版)
    }
    
    DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        batch_size: int = 100,
    ):
        """
        初始化 OpenAI Embedder
        
        Args:
            model_name: 模型名称
            api_key: API Key (不传则从环境变量读取)
            base_url: API Base URL (支持兼容 API)
            batch_size: 批处理大小
        """
        super().__init__(model_name)
        self.api_key = api_key
        self.base_url = base_url
        self.batch_size = batch_size
        self._client = None
    
    def _get_client(self):
        """获取 OpenAI 客户端"""
        if self._client is None:
            try:
                from openai import OpenAI
                
                kwargs = {}
                if self.api_key:
                    kwargs["api_key"] = self.api_key
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                
                self._client = OpenAI(**kwargs)
                
            except ImportError:
                raise EmbeddingError(
                    "Please install openai: pip install openai",
                    model=self.model_name,
                )
        
        return self._client
    
    @property
    def dimension(self) -> int:
        """返回向量维度"""
        return self.DIMENSIONS.get(self.model_name, 1536)
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """嵌入文本"""
        client = self._get_client()
        
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        
        # 分批处理
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            try:
                response = client.embeddings.create(
                    input=batch,
                    model=self.model_name,
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                raise EmbeddingError(
                    f"OpenAI API error: {e}",
                    model=self.model_name,
                )
        
        return np.array(all_embeddings)


class HuggingFaceEmbedder(BaseEmbedder):
    """
    Hugging Face Transformers Embedder
    直接使用 transformers 库
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        super().__init__(model_name)
        self.device = device or "cpu"
        self.normalize = normalize
        self._tokenizer = None
    
    def _load_model(self):
        if self._model is None:
            try:
                from transformers import AutoTokenizer, AutoModel
                import torch
                
                logger.info(f"Loading HuggingFace model: {self.model_name}")
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name)
                self._model.to(self.device)
                self._model.eval()
                
            except ImportError:
                raise EmbeddingError(
                    "Please install transformers: pip install transformers torch",
                    model=self.model_name,
                )
        
        return self._model
    
    @property
    def dimension(self) -> int:
        model = self._load_model()
        return model.config.hidden_size
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        import torch
        
        model = self._load_model()
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Embed
        with torch.no_grad():
            outputs = model(**encoded)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        embeddings = embeddings.cpu().numpy()
        
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
        
        return embeddings


class SiliconFlowEmbedder(BaseEmbedder):
    """
    SiliconFlow API Embedder
    免费提供 BGE-M3 等高质量中英文 Embedding 模型
    注册获取免费 API Key: https://siliconflow.cn/
    """
    
    MODELS = {
        "bge-m3": "BAAI/bge-m3",                    # 1024维，中英文
        "bge-large-zh": "BAAI/bge-large-zh-v1.5",  # 1024维，中文优化
        "bge-large-en": "BAAI/bge-large-en-v1.5",  # 1024维，英文优化
    }
    
    DIMENSIONS = {
        "BAAI/bge-m3": 1024,
        "BAAI/bge-large-zh-v1.5": 1024,
        "BAAI/bge-large-en-v1.5": 1024,
    }
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        api_key: Optional[str] = None,
        base_url: str = "https://api.siliconflow.cn/v1",
        batch_size: int = 32,
    ):
        """
        初始化 SiliconFlow Embedder
        
        Args:
            model_name: 模型名称 (默认 BGE-M3)
            api_key: API Key (不传则从 SILICONFLOW_API_KEY 环境变量读取)
            base_url: API 地址
            batch_size: 批处理大小
        """
        super().__init__(model_name)
        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
        self.base_url = base_url
        self.batch_size = batch_size
    
    @property
    def dimension(self) -> int:
        return self.DIMENSIONS.get(self.model_name, 1024)
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """嵌入文本"""
        if not self.api_key:
            raise EmbeddingError(
                "SiliconFlow API key not set. Set SILICONFLOW_API_KEY env var or pass api_key.",
                model=self.model_name,
            )
        
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        
        # 分批处理
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings)
    
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """嵌入一批文本"""
        url = f"{self.base_url}/embeddings"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        data = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float",
        }
        
        try:
            with httpx.Client(timeout=60) as client:
                response = client.post(url, json=data, headers=headers)
                response.raise_for_status()
                result = response.json()
            
            # 按 index 排序确保顺序正确
            embeddings = sorted(result["data"], key=lambda x: x["index"])
            return [e["embedding"] for e in embeddings]
            
        except httpx.HTTPError as e:
            raise EmbeddingError(
                f"SiliconFlow API error: {e}",
                model=self.model_name,
            )
        except Exception as e:
            raise EmbeddingError(
                f"Embedding failed: {e}",
                model=self.model_name,
            )


class JinaEmbedder(BaseEmbedder):
    """
    Jina AI Embeddings API
    提供免费额度的高质量 Embedding
    注册获取免费 API Key: https://jina.ai/embeddings/
    """
    
    MODELS = {
        "v3": "jina-embeddings-v3",           # 1024维，多语言
        "v2-base": "jina-embeddings-v2-base-en",  # 768维
    }
    
    DIMENSIONS = {
        "jina-embeddings-v3": 1024,
        "jina-embeddings-v2-base-en": 768,
    }
    
    def __init__(
        self,
        model_name: str = "jina-embeddings-v3",
        api_key: Optional[str] = None,
        batch_size: int = 32,
    ):
        super().__init__(model_name)
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        self.batch_size = batch_size
    
    @property
    def dimension(self) -> int:
        return self.DIMENSIONS.get(self.model_name, 1024)
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """嵌入文本"""
        if not self.api_key:
            raise EmbeddingError(
                "Jina API key not set. Set JINA_API_KEY env var or pass api_key.",
                model=self.model_name,
            )
        
        if isinstance(texts, str):
            texts = [texts]
        
        url = "https://api.jina.ai/v1/embeddings"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        data = {
            "model": self.model_name,
            "input": texts,
        }
        
        try:
            with httpx.Client(timeout=60) as client:
                response = client.post(url, json=data, headers=headers)
                response.raise_for_status()
                result = response.json()
            
            embeddings = sorted(result["data"], key=lambda x: x["index"])
            return np.array([e["embedding"] for e in embeddings])
            
        except Exception as e:
            raise EmbeddingError(f"Jina API error: {e}", model=self.model_name)


# 工厂函数
def get_embedder(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs,
) -> BaseEmbedder:
    """
    获取 Embedder 实例
    
    优先级: 函数参数 > .env 配置 > 默认值
    
    Args:
        provider: 提供商 (不传则从 .env 读取 EMBEDDING_PROVIDER)
            - "siliconflow": SiliconFlow API (免费，推荐 BGE-M3)
            - "jina": Jina AI API (有免费额度)
            - "sentence_transformers": 本地 SentenceTransformers
            - "openai": OpenAI API
            - "huggingface": 本地 HuggingFace Transformers
        model_name: 模型名称 (不传则从 .env 读取 EMBEDDING_MODEL_NAME)
        **kwargs: 额外参数
        
    Returns:
        Embedder 实例
    
    配置示例 (.env):
        EMBEDDING_PROVIDER=siliconflow
        EMBEDDING_MODEL_NAME=BAAI/bge-m3
        SILICONFLOW_API_KEY=your_key
    """
    # 从配置读取默认值
    from config import get_embedding_settings
    settings = get_embedding_settings()
    
    # 优先使用函数参数，否则用配置值
    provider = provider or settings.provider or "siliconflow"
    model_name = model_name or settings.model_name  # None 则使用各 Embedder 的默认模型
    
    if provider == "siliconflow":
        model_name = model_name or "BAAI/bge-m3"
        return SiliconFlowEmbedder(model_name, **kwargs)
    
    elif provider == "jina":
        model_name = model_name or "jina-embeddings-v3"
        return JinaEmbedder(model_name, **kwargs)
    
    elif provider == "sentence_transformers":
        model_name = model_name or "all-MiniLM-L6-v2"
        return SentenceTransformerEmbedder(model_name, **kwargs)
    
    elif provider == "openai":
        model_name = model_name or "text-embedding-3-small"
        return OpenAIEmbedder(model_name, **kwargs)
    
    elif provider == "huggingface":
        model_name = model_name or "BAAI/bge-small-en-v1.5"
        return HuggingFaceEmbedder(model_name, **kwargs)
    
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: siliconflow, jina, sentence_transformers, openai, huggingface")
