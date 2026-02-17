"""
RAG Tools
知识库检索和存储工具
"""
import asyncio
import hashlib
from typing import List, Dict, Any, Optional
import logging

from storage import QdrantVectorStore, get_vector_store
from processing import clean_text, chunk_document


logger = logging.getLogger(__name__)


# 全局向量存储实例 (延迟初始化)
_vector_store: Optional[QdrantVectorStore] = None


def get_knowledge_base() -> QdrantVectorStore:
    """获取知识库实例"""
    global _vector_store
    if _vector_store is None:
        _vector_store = get_vector_store(collection_name="research_knowledge")
    return _vector_store


def close_knowledge_base() -> None:
    """关闭并重置全局知识库实例，避免解释器退出时的析构噪音。"""
    global _vector_store
    if _vector_store is None:
        return
    try:
        close_fn = getattr(_vector_store, "close", None)
        if callable(close_fn):
            close_fn()
    except Exception as e:
        logger.debug(f"Failed to close knowledge base: {e}")
    finally:
        _vector_store = None


def _normalize_source_list(sources: Optional[List[str]]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for source in sources or []:
        value = str(source or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def _result_fingerprint(item: Dict[str, Any]) -> str:
    item_id = str(item.get("id", "")).strip()
    if item_id:
        return f"id:{item_id}"
    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        url = str(metadata.get("url", "")).strip()
        if url:
            return f"url:{url}"
    content = str(item.get("content", "")).strip()
    if content:
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:24]
        return f"content:{digest}"
    return ""


async def vector_search(
    query: str,
    top_k: int = 5,
    filter: Optional[Dict[str, Any]] = None,
    score_threshold: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    向量相似度搜索
    
    Args:
        query: 搜索查询
        top_k: 返回结果数
        filter: 元数据过滤条件
        score_threshold: 相似度阈值
        
    Returns:
        相似文档列表
    """
    store = get_knowledge_base()
    
    results = store.search(query, top_k=top_k, filter=filter)
    
    # 过滤低分结果
    filtered = [
        {
            "id": r.id,
            "content": r.content,
            "metadata": r.metadata,
            "score": r.score,
        }
        for r in results
        if r.score >= score_threshold
    ]
    
    logger.info(f"Vector search '{query[:50]}...': {len(filtered)} results (threshold={score_threshold})")
    return filtered


async def add_to_knowledge_base(
    documents: List[Dict[str, Any]],
    chunk_size: int = 500,
) -> int:
    """
    添加文档到知识库
    
    Args:
        documents: 文档列表 [{"content": "", "metadata": {}}]
        chunk_size: 分块大小
        
    Returns:
        添加的块数
    """
    store = get_knowledge_base()
    
    all_chunks = []
    for doc in documents:
        content = clean_text(doc.get("content", ""))
        metadata = doc.get("metadata", {})
        doc_id = doc.get("id", "")
        doc_type = doc.get("type", "unknown")
        
        # 分块
        chunks = chunk_document(
            content=content,
            doc_id=doc_id,
            doc_type=doc_type,
            metadata=metadata,
            chunk_size=chunk_size,
        )
        
        all_chunks.extend(chunks)
    
    if all_chunks:
        store.add_chunks(all_chunks)
        logger.info(f"Added {len(all_chunks)} chunks to knowledge base")
    
    return len(all_chunks)


async def hybrid_search(
    query: str,
    sources: Optional[List[str]] = None,
    year_filter: Optional[int] = None,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    混合搜索 - 结合元数据过滤
    
    Args:
        query: 搜索查询
        sources: 限定来源 (arxiv, huggingface, twitter, reddit, github)
        year_filter: 年份过滤 (>= 该年份)
        top_k: 返回结果数
        
    Returns:
        搜索结果
    """
    source_list = _normalize_source_list(sources)
    base_filter: Dict[str, Any] = {}
    if year_filter:
        base_filter["year"] = {"$gte": year_filter}

    if not source_list:
        return await vector_search(
            query=query,
            top_k=top_k,
            filter=base_filter if base_filter else None,
        )

    if len(source_list) == 1:
        single_filter = dict(base_filter)
        single_filter["source"] = source_list[0]
        return await vector_search(query=query, top_k=top_k, filter=single_filter)

    in_filter = dict(base_filter)
    in_filter["source"] = {"$in": source_list}
    try:
        return await vector_search(query=query, top_k=top_k, filter=in_filter)
    except Exception as exc:
        logger.warning(
            "Hybrid search IN filter failed; falling back to per-source fan-out: %s",
            exc,
            exc_info=True,
        )

    async def _search_by_source(source: str):
        source_filter = dict(base_filter)
        source_filter["source"] = source
        return await vector_search(query=query, top_k=top_k, filter=source_filter)

    fanout_results = await asyncio.gather(
        *[_search_by_source(source) for source in source_list],
        return_exceptions=True,
    )

    best_by_fingerprint: Dict[str, Dict[str, Any]] = {}
    for source, batch in zip(source_list, fanout_results):
        if isinstance(batch, Exception):
            logger.error(
                "Hybrid search fallback failed for source='%s': %s",
                source,
                batch,
                exc_info=True,
            )
            continue
        for item in batch:
            if not isinstance(item, dict):
                continue
            fingerprint = _result_fingerprint(item)
            if not fingerprint:
                continue
            current = best_by_fingerprint.get(fingerprint)
            current_score = float(current.get("score", 0.0) or 0.0) if current else float("-inf")
            next_score = float(item.get("score", 0.0) or 0.0)
            if next_score >= current_score:
                best_by_fingerprint[fingerprint] = item

    merged = list(best_by_fingerprint.values())
    merged.sort(key=lambda row: float(row.get("score", 0.0) or 0.0), reverse=True)
    return merged[: max(1, int(top_k))]


def create_rag_tools() -> List[Dict[str, Any]]:
    """
    创建 RAG 工具定义 (OpenAI function calling 格式)
    
    Returns:
        工具定义列表
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "vector_search",
                "description": "在知识库中搜索相关信息。用于回答用户问题时检索已收集的论文、讨论等。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索查询，应该是与问题相关的关键概念",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "返回结果数量",
                            "default": 5,
                        },
                        "filter": {
                            "type": "object",
                            "description": "元数据过滤条件，如 {\"source\": \"arxiv\"}",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "hybrid_search",
                "description": "混合搜索，支持按来源和年份过滤。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索查询",
                        },
                        "sources": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "限定来源: arxiv, huggingface, twitter, reddit, github",
                        },
                        "year_filter": {
                            "type": "integer",
                            "description": "年份过滤，返回该年份及之后的内容",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "返回结果数量",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
    ]


# 工具执行映射
RAG_TOOL_EXECUTORS = {
    "vector_search": vector_search,
    "hybrid_search": hybrid_search,
}


async def execute_rag_tool(name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
    """执行 RAG 工具"""
    executor = RAG_TOOL_EXECUTORS.get(name)
    if not executor:
        raise ValueError(f"Unknown RAG tool: {name}")
    return await executor(**arguments)
