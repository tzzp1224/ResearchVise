"""
Vector Store
向量数据库存储 - 使用 Qdrant
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging
from dataclasses import dataclass

import numpy as np

from processing.chunker import DocumentChunk
from processing.embedder import BaseEmbedder, get_embedder
from utils.exceptions import VectorStoreError


logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """搜索结果"""
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float  # 相似度分数 (越高越相似)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score,
        }


class BaseVectorStore(ABC):
    """向量存储抽象基类"""
    
    def __init__(
        self,
        collection_name: str,
        embedder: Optional[BaseEmbedder] = None,
    ):
        self.collection_name = collection_name
        self.embedder = embedder or get_embedder()
    
    @abstractmethod
    def add(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """添加文档"""
        pass
    
    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """搜索相似文档"""
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """删除文档"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """清空集合"""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """返回文档数量"""
        pass
    
    def add_chunks(self, chunks: List[DocumentChunk]) -> List[str]:
        """添加文档块"""
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [chunk.id for chunk in chunks]
        return self.add(documents, metadatas, ids)
    
    def search_with_score(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[SearchResult]:
        """带分数阈值的搜索"""
        results = self.search(query, top_k)
        return [r for r in results if r.score >= score_threshold]


class QdrantVectorStore(BaseVectorStore):
    """
    Qdrant 向量存储
    
    优势:
    - 原生元数据过滤 (支持 year >= 2025, topic == "LLM" 等)
    - 内置持久化管理
    - 并发安全
    - 可内嵌运行 (无需额外服务)
    """
    
    def __init__(
        self,
        collection_name: str = "academic_research",
        persist_directory: Optional[str] = None,
        embedder: Optional[BaseEmbedder] = None,
        dimension: int = 1024,
        host: Optional[str] = None,
        port: int = 6333,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        初始化 Qdrant
        
        Args:
            collection_name: 集合名称
            persist_directory: 本地持久化目录 (None = 内存模式)
            embedder: Embedder 实例
            dimension: 向量维度
            host: Qdrant 服务地址 (用于远程连接)
            port: Qdrant 服务端口
            url: Qdrant Cloud URL
            api_key: Qdrant Cloud API Key
        """
        super().__init__(collection_name, embedder)
        self.persist_directory = persist_directory
        self.dimension = dimension
        self.host = host
        self.port = port
        self.url = url
        self.api_key = api_key
        
        self._client = None
        self._init_client()
    
    def _init_client(self):
        """初始化 Qdrant 客户端"""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            
            # 决定连接模式
            if self.url:
                # 远程 Qdrant Cloud
                self._client = QdrantClient(url=self.url, api_key=self.api_key)
                logger.info(f"Qdrant connected to cloud: {self.url}")
            elif self.host:
                # 远程 Qdrant 服务
                self._client = QdrantClient(host=self.host, port=self.port)
                logger.info(f"Qdrant connected to {self.host}:{self.port}")
            elif self.persist_directory:
                # 本地持久化模式
                persist_path = Path(self.persist_directory)
                persist_path.mkdir(parents=True, exist_ok=True)
                self._client = QdrantClient(path=str(persist_path))
                logger.info(f"Qdrant initialized with persistence at: {persist_path}")
            else:
                # 内存模式
                self._client = QdrantClient(":memory:")
                logger.info("Qdrant initialized in memory mode")
            
            # 创建集合 (如果不存在)
            collections = [c.name for c in self._client.get_collections().collections]
            if self.collection_name not in collections:
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
                )
                logger.info(f"Created collection '{self.collection_name}' with dimension {self.dimension}")
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")
                
        except ImportError:
            raise VectorStoreError("Please install qdrant-client: pip install qdrant-client")
    
    def add(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """添加文档 (自动计算嵌入)"""
        if not documents:
            return []
        embeddings = self.embedder.embed(documents)
        return self.add_with_embeddings(documents, embeddings.tolist(), metadatas, ids)
    
    def add_with_embeddings(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """添加带预计算嵌入的文档"""
        from qdrant_client.models import PointStruct
        
        if not documents:
            return []
        
        # 生成 ID
        if ids is None:
            import hashlib
            ids = [hashlib.md5(doc.encode()).hexdigest()[:16] for doc in documents]
        
        # 处理 metadata
        if metadatas is None:
            metadatas = [{} for _ in documents]
        
        # 构建 points
        points = []
        for i, (doc, embedding, metadata, id_) in enumerate(zip(documents, embeddings, metadatas, ids)):
            # 清理 metadata (Qdrant 不支持 None 值和复杂嵌套)
            clean_meta = {k: v for k, v in metadata.items() if v is not None}
            for k, v in list(clean_meta.items()):
                if isinstance(v, (list, dict)):
                    clean_meta[k] = str(v)
            
            clean_meta["_content"] = doc
            points.append(PointStruct(
                id=i if isinstance(id_, int) else hash(id_) & 0x7FFFFFFFFFFFFFFF,
                vector=embedding,
                payload={**clean_meta, "_id": id_},
            ))
        
        self._client.upsert(collection_name=self.collection_name, points=points)
        logger.info(f"Added {len(documents)} documents to Qdrant")
        return ids
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """搜索相似文档"""
        query_embedding = self.embedder.embed(query)
        return self.search_with_embedding(query_embedding[0].tolist(), top_k, filter)
    
    def search_with_embedding(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """
        使用预计算的嵌入搜索
        
        支持的过滤语法:
            - 精确匹配: {"source": "arxiv"}
            - 范围查询: {"year": {"$gte": 2024}}
            - 组合条件: {"source": "arxiv", "year": {"$gte": 2024}}
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
        
        # 构建过滤器
        qdrant_filter = None
        if filter:
            conditions = []
            for key, value in filter.items():
                if isinstance(value, dict):
                    # 范围查询: {"year": {"$gte": 2025}}
                    range_params = {}
                    for op, val in value.items():
                        if op == "$gte": range_params["gte"] = val
                        elif op == "$lte": range_params["lte"] = val
                        elif op == "$gt": range_params["gt"] = val
                        elif op == "$lt": range_params["lt"] = val
                    if range_params:
                        conditions.append(FieldCondition(key=key, range=Range(**range_params)))
                else:
                    # 精确匹配
                    conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            if conditions:
                qdrant_filter = Filter(must=conditions)
        
        # 搜索 (兼容新旧 API)
        try:
            results = self._client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True,
            ).points
        except AttributeError:
            # 旧版 qdrant-client 兼容
            results = self._client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True,
            )
        
        # 转换结果
        search_results = []
        for hit in results:
            payload = hit.payload or {}
            content = payload.pop("_content", "")
            original_id = payload.pop("_id", str(hit.id))
            search_results.append(SearchResult(
                id=original_id,
                content=content,
                metadata=payload,
                score=hit.score,
            ))
        return search_results
    
    def delete(self, ids: List[str]) -> None:
        """删除文档"""
        from qdrant_client.models import Filter, FieldCondition, MatchAny
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(must=[FieldCondition(key="_id", match=MatchAny(any=ids))]),
        )
        logger.info(f"Deleted {len(ids)} documents")
    
    def clear(self) -> None:
        """清空集合"""
        from qdrant_client.models import Distance, VectorParams
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
        )
        logger.info(f"Collection '{self.collection_name}' cleared")
    
    def count(self) -> int:
        """返回文档数量"""
        return self._client.get_collection(self.collection_name).points_count

    def close(self) -> None:
        """关闭底层客户端连接"""
        client = getattr(self, "_client", None)
        if client is None:
            return
        close_fn = getattr(client, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception as e:
                logger.debug(f"Failed to close Qdrant client: {e}")
    
    def get_all(self, limit: int = 1000) -> List[Dict]:
        """获取所有文档 (调试用)"""
        results = self._client.scroll(
            collection_name=self.collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        docs = []
        for point in results[0]:
            payload = point.payload or {}
            content = payload.pop("_content", "")
            original_id = payload.pop("_id", str(point.id))
            docs.append({"id": original_id, "content": content, "metadata": payload})
        return docs


# 工厂函数
def get_vector_store(
    collection_name: str = "academic_research",
    persist_directory: Optional[str] = None,
    embedder: Optional[BaseEmbedder] = None,
    **kwargs,
) -> QdrantVectorStore:
    """
    获取向量存储实例
    
    Args:
        collection_name: 集合名称
        persist_directory: 持久化目录 (不传则从 .env 读取)
        embedder: Embedder 实例
        **kwargs: 额外参数 (如 url, api_key 用于 Qdrant Cloud)
        
    Returns:
        QdrantVectorStore 实例
    """
    from config import get_storage_settings
    settings = get_storage_settings()
    persist_directory = persist_directory or settings.vector_db_path
    
    return QdrantVectorStore(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedder=embedder,
        **kwargs,
    )
