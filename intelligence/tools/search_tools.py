"""
Search Tools
搜索工具 - 包装 Phase 1 Scrapers，供 ReAct agent 调用。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set
import logging

from intelligence.state import SearchResult
from intelligence.tools.search_tool_specs import SEARCH_TOOL_DEFINITIONS
from scrapers import (
    ArxivScraper,
    GitHubScraper,
    HackerNewsScraper,
    HuggingFaceScraper,
    RedditScraper,
    SemanticScholarScraper,
    StackOverflowScraper,
    TwitterScraper,
)


logger = logging.getLogger(__name__)

_SOURCE_TO_TOOL = {
    "arxiv": "arxiv_search",
    "huggingface": "huggingface_search",
    "twitter": "twitter_search",
    "reddit": "reddit_search",
    "github": "github_search",
    "semantic_scholar": "semantic_scholar_search",
    "stackoverflow": "stackoverflow_search",
    "hackernews": "hackernews_search",
}

_TOOL_ORDER = [
    "arxiv_search",
    "semantic_scholar_search",
    "huggingface_search",
    "github_search",
    "stackoverflow_search",
    "hackernews_search",
    "reddit_search",
    "twitter_search",
]


def _iso(value: Any) -> Optional[str]:
    return value.isoformat() if value else None


def _result(
    *,
    id: str,
    source: str,
    title: str,
    content: str,
    url: Optional[str],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    return SearchResult(
        id=id,
        source=source,
        title=title,
        content=content,
        url=url,
        metadata=metadata,
    ).to_dict()


def _normalize_sources(allowed_sources: Optional[Iterable[str]]) -> Optional[Set[str]]:
    if allowed_sources is None:
        return None
    aliases = {
        "semantic-scholar": "semantic_scholar",
        "semantic scholar": "semantic_scholar",
        "hn": "hackernews",
        "stack-overflow": "stackoverflow",
        "stack overflow": "stackoverflow",
    }
    normalized: Set[str] = set()
    for item in allowed_sources:
        key = str(item or "").strip().lower().replace("-", "_")
        if not key:
            continue
        key = aliases.get(key, key)
        normalized.add(key)
    return normalized


def _allowed_tools(
    *,
    allowed_sources: Optional[Iterable[str]] = None,
    allowed_tool_names: Optional[Sequence[str]] = None,
) -> List[str]:
    if allowed_tool_names is not None:
        allowed = {str(item).strip() for item in allowed_tool_names if str(item).strip()}
        return [name for name in _TOOL_ORDER if name in allowed]

    normalized_sources = _normalize_sources(allowed_sources)
    if normalized_sources is None:
        return list(_TOOL_ORDER)

    allowed = {
        _SOURCE_TO_TOOL[source]
        for source in normalized_sources
        if source in _SOURCE_TO_TOOL
    }
    return [name for name in _TOOL_ORDER if name in allowed]


async def arxiv_search(
    query: str,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """搜索 ArXiv 论文。"""
    async with ArxivScraper() as scraper:
        papers = await scraper.search(query, max_results=max_results)

    results = [
        _result(
            id=f"arxiv_{paper.id}",
            source="arxiv",
            title=paper.title,
            content=paper.abstract or "",
            url=paper.url,
            metadata={
                "authors": [a.name for a in paper.authors],
                "published_date": _iso(paper.published_date),
                "categories": paper.categories,
                "citation_count": paper.citation_count,
            },
        )
        for paper in papers
    ]
    logger.info(f"ArXiv search '{query}': found {len(results)} papers")
    return results


async def semantic_scholar_search(
    query: str,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """搜索 Semantic Scholar。"""
    async with SemanticScholarScraper() as scraper:
        papers = await scraper.search(query, max_results=max_results)

    results = [
        _result(
            id=f"semantic_scholar_{paper.id}",
            source="semantic_scholar",
            title=paper.title,
            content=paper.abstract or "",
            url=paper.url,
            metadata={
                "authors": [a.name for a in paper.authors],
                "published_date": _iso(paper.published_date),
                "citation_count": paper.citation_count,
                "categories": paper.categories,
            },
        )
        for paper in papers
    ]
    logger.info(f"Semantic Scholar search '{query}': found {len(results)} papers")
    return results


async def huggingface_search(
    query: str,
    search_type: str = "all",
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """搜索 Hugging Face 模型与数据集。"""
    results: List[Dict[str, Any]] = []

    async with HuggingFaceScraper() as scraper:
        if search_type in ["all", "models"]:
            models = await scraper.search_models(query, max_results=max_results)
            results.extend(
                [
                    _result(
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
                    )
                    for model in models
                ]
            )

        if search_type in ["all", "datasets"]:
            datasets = await scraper.search_datasets(query, max_results=max_results)
            results.extend(
                [
                    _result(
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
                    )
                    for dataset in datasets
                ]
            )

    logger.info(f"HuggingFace search '{query}' ({search_type}): found {len(results)} items")
    return results


async def github_search(
    query: str,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """搜索 GitHub 仓库。"""
    try:
        async with GitHubScraper() as scraper:
            repos = await scraper.search(query, max_results=max_results)

        results = [
            _result(
                id=f"github_{repo.id}",
                source="github",
                title=repo.full_name,
                content=repo.description or "",
                url=repo.url,
                metadata={
                    "owner": repo.owner,
                    "stars": repo.stars,
                    "forks": repo.forks,
                    "language": repo.language,
                    "topics": repo.topics,
                    "updated_at": _iso(repo.updated_at),
                },
            )
            for repo in repos
        ]

        logger.info(f"GitHub search '{query}': found {len(results)} repos")
        return results
    except Exception as exc:
        logger.warning(f"GitHub search failed: {exc}")
        return []


async def stackoverflow_search(
    query: str,
    max_results: int = 10,
    tags: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """搜索 Stack Overflow。"""
    async with StackOverflowScraper() as scraper:
        questions = await scraper.search(query, max_results=max_results, tags=tags)

    results = [
        _result(
            id=f"stackoverflow_{item.id}",
            source="stackoverflow",
            title=item.title,
            content=item.body or "",
            url=item.url,
            metadata={
                "author": item.author,
                "tags": item.tags,
                "score": item.score,
                "view_count": item.view_count,
                "answer_count": item.answer_count,
                "is_answered": item.is_answered,
            },
        )
        for item in questions
    ]
    logger.info(f"StackOverflow search '{query}': found {len(results)} questions")
    return results


async def hackernews_search(
    query: str,
    max_results: int = 10,
    sort_by: str = "relevance",
) -> List[Dict[str, Any]]:
    """搜索 Hacker News 讨论。"""
    async with HackerNewsScraper() as scraper:
        items = await scraper.search(query, max_results=max_results, sort_by=sort_by)

    results = [
        _result(
            id=f"hackernews_{item.id}",
            source="hackernews",
            title=item.title,
            content=item.text or "",
            url=item.url or item.hn_url,
            metadata={
                "author": item.author,
                "points": item.points,
                "comment_count": item.comment_count,
                "created_at": _iso(item.created_at),
            },
        )
        for item in items
    ]
    logger.info(f"HackerNews search '{query}': found {len(results)} items")
    return results


async def twitter_search(
    query: str,
    max_results: int = 20,
) -> List[Dict[str, Any]]:
    """搜索 Twitter/X。"""
    try:
        async with TwitterScraper() as scraper:
            posts = await scraper.search(query, max_results=max_results)

        results = [
            _result(
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
                    "created_at": _iso(post.created_at),
                },
            )
            for post in posts
        ]

        logger.info(f"Twitter search '{query}': found {len(results)} tweets")
        return results
    except Exception as exc:
        logger.warning(f"Twitter search failed: {exc}")
        return []


async def reddit_search(
    query: str,
    subreddits: Optional[List[str]] = None,
    max_results: int = 20,
) -> List[Dict[str, Any]]:
    """搜索 Reddit。"""
    try:
        target_subreddits = subreddits or ["MachineLearning", "LocalLLaMA", "artificial"]
        async with RedditScraper() as scraper:
            posts = await scraper.search(
                query,
                subreddits=target_subreddits,
                max_results=max_results,
            )

        results = [
            _result(
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
                    "created_at": _iso(post.created_at),
                },
            )
            for post in posts
        ]

        logger.info(f"Reddit search '{query}': found {len(results)} posts")
        return results
    except Exception as exc:
        logger.warning(f"Reddit search failed: {exc}")
        return []


def create_search_tools(
    allowed_sources: Optional[Iterable[str]] = None,
    allowed_tool_names: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """
    创建搜索工具定义 (OpenAI function calling 格式)。
    """
    tool_names = _allowed_tools(
        allowed_sources=allowed_sources,
        allowed_tool_names=allowed_tool_names,
    )
    return [SEARCH_TOOL_DEFINITIONS[name] for name in tool_names if name in SEARCH_TOOL_DEFINITIONS]


SEARCH_TOOL_EXECUTORS = {
    "arxiv_search": arxiv_search,
    "huggingface_search": huggingface_search,
    "twitter_search": twitter_search,
    "reddit_search": reddit_search,
    "github_search": github_search,
    "semantic_scholar_search": semantic_scholar_search,
    "stackoverflow_search": stackoverflow_search,
    "hackernews_search": hackernews_search,
}


async def execute_search_tool(name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
    """执行搜索工具。"""
    executor = SEARCH_TOOL_EXECUTORS.get(name)
    if not executor:
        raise ValueError(f"Unknown search tool: {name}")
    return await executor(**arguments)
