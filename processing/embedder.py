"""
Embedder
文本向量化模块 - 支持多种 Embedding 提供商
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union
import asyncio
import logging
import os

import httpx
import numpy as np

from utils.exceptions import EmbeddingError


logger = logging.getLogger(__name__)


def _ensure_text_list(texts: Union[str, List[str]]) -> List[str]:
    return [texts] if isinstance(texts, str) else list(texts)


def _batched(items: List[str], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _sorted_embedding_payload(result: dict[str, Any]) -> List[List[float]]:
    embeddings = sorted(result["data"], key=lambda x: x["index"])
    return [entry["embedding"] for entry in embeddings]


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

    @abstractmethod
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        同步嵌入文本

        Args:
            texts: 单个文本或文本列表

        Returns:
            向量数组 (n_texts, dimension)
        """

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
            except Exception as exc:
                raise EmbeddingError(
                    f"Failed to load model: {exc}",
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
        text_list = _ensure_text_list(texts)

        try:
            embeddings = model.encode(
                text_list,
                normalize_embeddings=self.normalize,
                show_progress_bar=len(text_list) > 100,
            )
            return np.array(embeddings)
        except Exception as exc:
            raise EmbeddingError(
                f"Embedding failed: {exc}",
                model=self.model_name,
            )

    def embed_query(self, query: str) -> np.ndarray:
        """嵌入查询"""
        if "e5" in self.model_name.lower():
            query = f"query: {query}"
        return self.embed(query)[0]

    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """嵌入文档"""
        if "e5" in self.model_name.lower():
            documents = [f"passage: {doc}" for doc in documents]
        return self.embed(documents)


class OpenAIEmbedder(BaseEmbedder):
    """
    OpenAI Embeddings API
    """

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
        text_list = _ensure_text_list(texts)
        all_embeddings: List[List[float]] = []

        for batch in _batched(text_list, self.batch_size):
            try:
                response = client.embeddings.create(
                    input=batch,
                    model=self.model_name,
                )
                all_embeddings.extend([item.embedding for item in response.data])
            except Exception as exc:
                raise EmbeddingError(
                    f"OpenAI API error: {exc}",
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
                from transformers import AutoModel, AutoTokenizer

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
        text_list = _ensure_text_list(texts)

        encoded = self._tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            embeddings = outputs.last_hidden_state.mean(dim=1)

        embeddings = embeddings.cpu().numpy()
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, 1e-12, None)

        return embeddings


class SiliconFlowEmbedder(BaseEmbedder):
    """
    SiliconFlow API Embedder
    免费提供 BGE-M3 等高质量中英文 Embedding 模型
    注册获取免费 API Key: https://siliconflow.cn/
    """

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

        text_list = _ensure_text_list(texts)
        all_embeddings: List[List[float]] = []
        for batch in _batched(text_list, self.batch_size):
            all_embeddings.extend(self._embed_batch(batch))

        return np.array(all_embeddings)

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """嵌入一批文本"""
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float",
        }

        try:
            with httpx.Client(timeout=60) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()
            return _sorted_embedding_payload(result)
        except httpx.HTTPError as exc:
            raise EmbeddingError(
                f"SiliconFlow API error: {exc}",
                model=self.model_name,
            )
        except Exception as exc:
            raise EmbeddingError(
                f"Embedding failed: {exc}",
                model=self.model_name,
            )


class JinaEmbedder(BaseEmbedder):
    """
    Jina AI Embeddings API
    提供免费额度的高质量 Embedding
    注册获取免费 API Key: https://jina.ai/embeddings/
    """

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

        text_list = _ensure_text_list(texts)
        all_embeddings: List[List[float]] = []
        for batch in _batched(text_list, self.batch_size):
            all_embeddings.extend(self._embed_batch(batch))

        return np.array(all_embeddings)

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        url = "https://api.jina.ai/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "input": texts,
        }

        try:
            with httpx.Client(timeout=60) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()
            return _sorted_embedding_payload(result)
        except Exception as exc:
            raise EmbeddingError(f"Jina API error: {exc}", model=self.model_name)


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
    from config import get_embedding_settings

    settings = get_embedding_settings()
    provider_name = (provider or settings.provider or "siliconflow").strip().lower()

    factory = {
        "siliconflow": (SiliconFlowEmbedder, "BAAI/bge-m3"),
        "jina": (JinaEmbedder, "jina-embeddings-v3"),
        "sentence_transformers": (SentenceTransformerEmbedder, "all-MiniLM-L6-v2"),
        "openai": (OpenAIEmbedder, "text-embedding-3-small"),
        "huggingface": (HuggingFaceEmbedder, "BAAI/bge-small-en-v1.5"),
    }

    if provider_name not in factory:
        supported = ", ".join(factory.keys())
        raise ValueError(f"Unknown provider: {provider_name}. Supported: {supported}")

    cls, default_model_name = factory[provider_name]
    return cls(model_name or settings.model_name or default_model_name, **kwargs)
