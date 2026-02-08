"""
RAG Tools
知识库检索和存储工具
"""
from typing import List, Dict, Any, Optional
import logging

from storage import QdrantVectorStore, get_vector_store
from processing import clean_text, chunk_document, get_embedder


logger = logging.getLogger(__name__)


# 全局向量存储实例 (延迟初始化)
_vector_store: Optional[QdrantVectorStore] = None


def get_knowledge_base() -> QdrantVectorStore:
    """获取知识库实例"""
    global _vector_store
    if _vector_store is None:
        _vector_store = get_vector_store(collection_name="research_knowledge")
    return _vector_store


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
    filter_dict = {}
    
    if sources:
        # Qdrant 不直接支持 IN 查询，需要多次查询合并
        # 这里简化处理，只用第一个 source
        if len(sources) == 1:
            filter_dict["source"] = sources[0]
    
    if year_filter:
        filter_dict["year"] = {"$gte": year_filter}
    
    return await vector_search(
        query=query,
        top_k=top_k,
        filter=filter_dict if filter_dict else None,
    )


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
