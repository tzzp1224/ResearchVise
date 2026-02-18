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


_KB_COLLECTION_NAME = "research_knowledge"
_NAMESPACE_FIELD = "kb_namespace"


def _normalize_namespace(
    *,
    namespace: Optional[str] = None,
    topic_hash: Optional[str] = None,
) -> Optional[str]:
    raw = str(namespace or topic_hash or "").strip().lower()
    if not raw:
        return None
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw)
    cleaned = cleaned.strip("_-")
    if not cleaned:
        return None
    if len(cleaned) > 64:
        return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:32]
    return cleaned


def get_knowledge_base(
    *,
    namespace: Optional[str] = None,
    topic_hash: Optional[str] = None,
) -> QdrantVectorStore:
    """获取知识库实例（按需实例化，无全局单例）。"""
    _ = _normalize_namespace(namespace=namespace, topic_hash=topic_hash)
    return get_vector_store(collection_name=_KB_COLLECTION_NAME)


def close_knowledge_base(
    *,
    store: Optional[QdrantVectorStore] = None,
    namespace: Optional[str] = None,
    topic_hash: Optional[str] = None,
) -> None:
    """
    兼容保留的关闭接口。
    无全局实例可清理，仅关闭显式传入的当前会话 store。
    """
    _ = _normalize_namespace(namespace=namespace, topic_hash=topic_hash)
    if store is None:
        return
    try:
        close_fn = getattr(store, "close", None)
        if callable(close_fn):
            close_fn()
    except Exception as e:
        logger.debug(f"Failed to close knowledge base: {e}")


def _close_store_instance(store: Optional[QdrantVectorStore]) -> None:
    if store is None:
        return
    try:
        close_fn = getattr(store, "close", None)
        if callable(close_fn):
            close_fn()
    except Exception as e:
        logger.debug(f"Failed to close knowledge base store: {e}")


def _merge_namespace_filter(
    *,
    filter: Optional[Dict[str, Any]],
    namespace: Optional[str],
) -> Dict[str, Any]:
    merged = dict(filter or {})
    if namespace:
        merged[_NAMESPACE_FIELD] = namespace
    return merged


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
    score_threshold: float = 0.15,
    namespace: Optional[str] = None,
    topic_hash: Optional[str] = None,
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
    resolved_namespace = _normalize_namespace(namespace=namespace, topic_hash=topic_hash)
    store = get_knowledge_base(namespace=resolved_namespace)
    results = []
    try:
        filter_with_namespace = _merge_namespace_filter(filter=filter, namespace=resolved_namespace)
        results = store.search(query, top_k=top_k, filter=filter_with_namespace)
    finally:
        _close_store_instance(store)
    
    transformed = [
        {
            "id": r.id,
            "content": r.content,
            "metadata": r.metadata,
            "score": float(getattr(r, "score", 0.0) or 0.0),
        }
        for r in results
    ]
    max_score = max((float(item.get("score", 0.0) or 0.0) for item in transformed), default=0.0)
    if score_threshold <= 0:
        filtered = transformed
    else:
        filtered = [item for item in transformed if float(item.get("score", 0.0) or 0.0) >= score_threshold]

    if not filtered and transformed:
        # Fallback: avoid empty recall caused by overly strict threshold.
        filtered = transformed[: max(1, int(top_k))]

    logger.info(
        "Vector search '%s...': raw=%s filtered=%s max_score=%.4f threshold=%s namespace=%s",
        query[:50],
        len(transformed),
        len(filtered),
        max_score,
        score_threshold,
        resolved_namespace or "*",
    )
    return filtered


async def add_to_knowledge_base(
    documents: List[Dict[str, Any]],
    chunk_size: int = 500,
    namespace: Optional[str] = None,
    topic_hash: Optional[str] = None,
) -> int:
    """
    添加文档到知识库
    
    Args:
        documents: 文档列表 [{"content": "", "metadata": {}}]
        chunk_size: 分块大小
        
    Returns:
        添加的块数
    """
    resolved_namespace = _normalize_namespace(namespace=namespace, topic_hash=topic_hash)
    store = get_knowledge_base(namespace=resolved_namespace)
    
    all_chunks = []
    docs_count = len(documents or [])
    before_count: Optional[int] = None
    try:
        try:
            before_count = int(store.count() or 0)
        except Exception:
            before_count = None
        for doc in documents:
            content = clean_text(doc.get("content", ""))
            metadata = dict(doc.get("metadata", {}) or {})
            if resolved_namespace:
                metadata[_NAMESPACE_FIELD] = resolved_namespace
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
            persist_fn = getattr(store, "persist", None)
            if callable(persist_fn):
                persist_fn()
            after_count: Optional[int]
            try:
                after_count = int(store.count() or 0)
            except Exception:
                after_count = None
            logger.info(
                "Knowledge base write completed: docs=%s chunks=%s namespace=%s collection=%s before=%s after=%s",
                docs_count,
                len(all_chunks),
                resolved_namespace or "*",
                getattr(store, "collection_name", "unknown"),
                before_count if before_count is not None else "n/a",
                after_count if after_count is not None else "n/a",
            )
        else:
            logger.info(
                "Knowledge base write skipped: docs=%s chunks=0 namespace=%s",
                docs_count,
                resolved_namespace or "*",
            )
        return len(all_chunks)
    finally:
        _close_store_instance(store)


async def hybrid_search(
    query: str,
    sources: Optional[List[str]] = None,
    year_filter: Optional[int] = None,
    top_k: int = 10,
    score_threshold: float = 0.15,
    namespace: Optional[str] = None,
    topic_hash: Optional[str] = None,
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
            score_threshold=score_threshold,
            namespace=namespace,
            topic_hash=topic_hash,
        )

    if len(source_list) == 1:
        single_filter = dict(base_filter)
        single_filter["source"] = source_list[0]
        return await vector_search(
            query=query,
            top_k=top_k,
            filter=single_filter,
            score_threshold=score_threshold,
            namespace=namespace,
            topic_hash=topic_hash,
        )

    in_filter = dict(base_filter)
    in_filter["source"] = {"$in": source_list}
    try:
        return await vector_search(
            query=query,
            top_k=top_k,
            filter=in_filter,
            score_threshold=score_threshold,
            namespace=namespace,
            topic_hash=topic_hash,
        )
    except Exception as exc:
        logger.warning(
            "Hybrid search IN filter failed; falling back to per-source fan-out: %s",
            exc,
            exc_info=True,
        )

    async def _search_by_source(source: str):
        source_filter = dict(base_filter)
        source_filter["source"] = source
        return await vector_search(
            query=query,
            top_k=top_k,
            filter=source_filter,
            score_threshold=score_threshold,
            namespace=namespace,
            topic_hash=topic_hash,
        )

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
