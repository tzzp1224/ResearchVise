"""
Search Tools
搜索工具 - 包装 Phase 1 的 Scrapers
"""
from typing import List, Dict, Any, Optional
import asyncio
import logging
from functools import partial

from scrapers import (
    ArxivScraper,
    HuggingFaceScraper,
    TwitterScraper,
    RedditScraper,
    GitHubScraper,
)
from intelligence.state import SearchResult


logger = logging.getLogger(__name__)


async def arxiv_search(
    query: str,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """
    搜索 ArXiv 论文
    
    Args:
        query: 搜索关键词
        max_results: 最大结果数
        
    Returns:
        论文列表
    """
    scraper = ArxivScraper()
    papers = await scraper.search(query, max_results=max_results)
    
    results = []
    for paper in papers:
        results.append(SearchResult(
            id=f"arxiv_{paper.id}",
            source="arxiv",
            title=paper.title,
            content=paper.abstract or "",
            url=paper.url,
            metadata={
                "authors": [a.name for a in paper.authors],
                "published_date": paper.published_date.isoformat() if paper.published_date else None,
                "categories": paper.categories,
                "citation_count": paper.citation_count,
            },
        ).to_dict())
    
    logger.info(f"ArXiv search '{query}': found {len(results)} papers")
    return results


async def huggingface_search(
    query: str,
    search_type: str = "all",  # all, models, datasets, papers
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """
    搜索 HuggingFace
    
    Args:
        query: 搜索关键词
        search_type: 搜索类型 (all, models, datasets, papers)
        max_results: 最大结果数
        
    Returns:
        结果列表
    """
    scraper = HuggingFaceScraper()
    
    results = []
    
    if search_type in ["all", "models"]:
        models = await scraper.search_models(query, limit=max_results)
        for model in models:
            results.append(SearchResult(
                id=f"hf_model_{model.id}",
                source="huggingface",
                title=model.name,
                content=model.description or "",
                url=model.url,
                metadata={
                    "type": "model",
                    "downloads": model.downloads,
                    "likes": model.likes,
                    "tags": model.tags,
                    "pipeline_tag": model.pipeline_tag,
                },
            ).to_dict())
    
    if search_type in ["all", "datasets"]:
        datasets = await scraper.search_datasets(query, limit=max_results)
        for dataset in datasets:
            results.append(SearchResult(
                id=f"hf_dataset_{dataset.id}",
                source="huggingface",
                title=dataset.name,
                content=dataset.description or "",
                url=dataset.url,
                metadata={
                    "type": "dataset",
                    "downloads": dataset.downloads,
                    "likes": dataset.likes,
                    "tags": dataset.tags,
                },
            ).to_dict())
    
    logger.info(f"HuggingFace search '{query}' ({search_type}): found {len(results)} items")
    return results


async def twitter_search(
    query: str,
    max_results: int = 20,
) -> List[Dict[str, Any]]:
    """
    搜索 Twitter/X
    
    Args:
        query: 搜索关键词
        max_results: 最大结果数
        
    Returns:
        推文列表
    """
    try:
        scraper = TwitterScraper()
        posts = await scraper.search(query, max_results=max_results)
        
        results = []
        for post in posts:
            results.append(SearchResult(
                id=f"twitter_{post.id}",
                source="twitter",
                title=f"@{post.author}",
                content=post.content,
                url=post.url,
                metadata={
                    "author": post.author,
                    "likes": post.likes,
                    "retweets": post.retweets,
                    "replies": post.replies,
                    "created_at": post.created_at.isoformat() if post.created_at else None,
                },
            ).to_dict())
        
        logger.info(f"Twitter search '{query}': found {len(results)} tweets")
        return results
    except Exception as e:
        logger.warning(f"Twitter search failed: {e}")
        return []


async def reddit_search(
    query: str,
    subreddits: Optional[List[str]] = None,
    max_results: int = 20,
) -> List[Dict[str, Any]]:
    """
    搜索 Reddit
    
    Args:
        query: 搜索关键词
        subreddits: 限定的子版块 (默认 MachineLearning, LocalLLaMA)
        max_results: 最大结果数
        
    Returns:
        帖子列表
    """
    try:
        scraper = RedditScraper()
        subreddits = subreddits or ["MachineLearning", "LocalLLaMA", "artificial"]
        
        posts = await scraper.search(
            query, 
            subreddits=subreddits, 
            max_results=max_results
        )
        
        results = []
        for post in posts:
            results.append(SearchResult(
                id=f"reddit_{post.id}",
                source="reddit",
                title=post.title or f"r/{post.subreddit}",
                content=post.content,
                url=post.url,
                metadata={
                    "author": post.author,
                    "subreddit": post.subreddit,
                    "score": post.score,
                    "num_comments": post.num_comments,
                    "created_at": post.created_at.isoformat() if post.created_at else None,
                },
            ).to_dict())
        
        logger.info(f"Reddit search '{query}': found {len(results)} posts")
        return results
    except Exception as e:
        logger.warning(f"Reddit search failed: {e}")
        return []


async def github_search(
    query: str,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """
    搜索 GitHub 仓库
    
    Args:
        query: 搜索关键词
        max_results: 最大结果数
        
    Returns:
        仓库列表
    """
    try:
        scraper = GitHubScraper()
        repos = await scraper.search(query, max_results=max_results)
        
        results = []
        for repo in repos:
            results.append(SearchResult(
                id=f"github_{repo.id}",
                source="github",
                title=repo.name,
                content=repo.description or "",
                url=repo.url,
                metadata={
                    "owner": repo.owner,
                    "stars": repo.stars,
                    "forks": repo.forks,
                    "language": repo.language,
                    "topics": repo.topics,
                    "updated_at": repo.updated_at.isoformat() if repo.updated_at else None,
                },
            ).to_dict())
        
        logger.info(f"GitHub search '{query}': found {len(results)} repos")
        return results
    except Exception as e:
        logger.warning(f"GitHub search failed: {e}")
        return []


def create_search_tools() -> List[Dict[str, Any]]:
    """
    创建搜索工具定义 (OpenAI function calling 格式)
    
    Returns:
        工具定义列表
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "arxiv_search",
                "description": "搜索 ArXiv 学术论文。用于获取最新的学术研究成果、技术论文、模型架构细节等。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索关键词，可以是论文标题、作者、主题等",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "最大返回结果数",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "huggingface_search",
                "description": "搜索 HuggingFace 上的模型、数据集。用于了解开源模型、预训练权重、数据集信息。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索关键词",
                        },
                        "search_type": {
                            "type": "string",
                            "enum": ["all", "models", "datasets"],
                            "description": "搜索类型",
                            "default": "all",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "最大返回结果数",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "twitter_search",
                "description": "搜索 Twitter/X 上的讨论。用于了解社区反馈、实时热点、专家观点。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索关键词",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "最大返回结果数",
                            "default": 20,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "reddit_search",
                "description": "搜索 Reddit 上的讨论。用于了解社区深度讨论、使用经验、问题反馈。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索关键词",
                        },
                        "subreddits": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "限定的子版块",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "最大返回结果数",
                            "default": 20,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "github_search",
                "description": "搜索 GitHub 仓库。用于了解开源实现、代码质量、项目活跃度。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索关键词",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "最大返回结果数",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
    ]


# 工具执行映射
SEARCH_TOOL_EXECUTORS = {
    "arxiv_search": arxiv_search,
    "huggingface_search": huggingface_search,
    "twitter_search": twitter_search,
    "reddit_search": reddit_search,
    "github_search": github_search,
}


async def execute_search_tool(name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
    """执行搜索工具"""
    executor = SEARCH_TOOL_EXECUTORS.get(name)
    if not executor:
        raise ValueError(f"Unknown search tool: {name}")
    return await executor(**arguments)
