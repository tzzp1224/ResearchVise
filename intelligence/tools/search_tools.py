"""
Search Tools
搜索工具 - 包装 Phase 1 Scrapers，供 ReAct agent 调用。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import html as html_lib
import hashlib
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set
import logging
import re
import xml.etree.ElementTree as ET

import httpx

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
_TOOL_TIMEOUT_SEC = 15

_SOURCE_TO_TOOL = {
    "arxiv": "arxiv_search",
    "arxiv_rss": "arxiv_rss_search",
    "huggingface": "huggingface_search",
    "twitter": "twitter_search",
    "reddit": "reddit_search",
    "github": "github_search",
    "semantic_scholar": "semantic_scholar_search",
    "openreview": "openreview_search",
    "stackoverflow": "stackoverflow_search",
    "hackernews": "hackernews_search",
}

_TOOL_ORDER = [
    "arxiv_search",
    "arxiv_rss_search",
    "openreview_search",
    "semantic_scholar_search",
    "huggingface_search",
    "github_search",
    "stackoverflow_search",
    "hackernews_search",
    "reddit_search",
    "twitter_search",
]

_CN_QUERY_EXPANSIONS = {
    "模型上下文协议": "model context protocol",
    "强化学习": "reinforcement learning",
    "深度强化学习": "deep reinforcement learning",
    "检索增强生成": "retrieval augmented generation",
    "向量数据库": "vector database",
    "知识图谱": "knowledge graph",
    "大语言模型": "large language model",
    "多模态": "multimodal model",
    "生产环境部署": "production deployment",
    "高可用": "high availability",
    "可观测性": "observability",
}

_RECENCY_HINTS = (
    "latest",
    "recent",
    "new",
    "newly",
    "today",
    "最近",
    "最新",
    "近期",
    "新发布",
    "刚发布",
)

_ARXIV_RSS_FEEDS = [
    "https://rss.arxiv.org/rss/cs.AI",
    "https://rss.arxiv.org/rss/cs.CL",
    "https://rss.arxiv.org/rss/cs.LG",
    "https://rss.arxiv.org/rss/cs.CV",
    "https://rss.arxiv.org/rss/stat.ML",
]


def _has_recency_intent(query: str) -> bool:
    text = str(query or "").strip().lower()
    if not text:
        return False
    if any(hint in text for hint in _RECENCY_HINTS):
        return True

    # Explicit year constraints (e.g. 2025/2026) usually imply freshness intent.
    year_hits = [int(x) for x in re.findall(r"(20\d{2})", text)]
    if not year_hits:
        return False
    current_year = datetime.now().year
    return any(year >= current_year - 1 for year in year_hits)


def _query_candidates(query: str, *, limit: int = 2) -> List[str]:
    base = str(query or "").strip()
    if not base:
        return []

    candidates: List[str] = [base]
    tokens = re.findall(r"[A-Za-z0-9#+-]+(?:\.[0-9]+)?", base)
    stopwords = {
        "production",
        "deployment",
        "design",
        "architecture",
        "analysis",
        "system",
        "systems",
        "for",
        "with",
        "and",
        "the",
        "in",
        "of",
        "to",
    }
    reduced = " ".join([token for token in tokens if token.lower() not in stopwords][:5]).strip()
    if reduced and reduced.lower() != base.lower():
        candidates.append(reduced)

    version_tokens = re.findall(r"\d+(?:\.\d+)+", base)
    if version_tokens:
        core = " ".join(
            [
                token
                for token in tokens
                if not re.fullmatch(r"\d+(?:\.\d+)+", token)
                and token.lower() not in stopwords
            ][:4]
        ).strip()
        for version in version_tokens:
            candidate = f"{core} {version}".strip() if core else version
            if candidate and candidate.lower() != base.lower():
                candidates.append(candidate)

    if re.search(r"[\u4e00-\u9fff]", base):
        translated = base
        for cn, en in sorted(_CN_QUERY_EXPANSIONS.items(), key=lambda pair: len(pair[0]), reverse=True):
            if cn in base:
                candidates.append(en)
                translated = translated.replace(cn, en)
        translated = re.sub(r"\s+", " ", translated).strip()
        if translated and translated.lower() != base.lower():
            candidates.append(translated)

    deduped: List[str] = []
    seen = set()
    for item in candidates:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= max(1, int(limit)):
            break
    return deduped


def _iso(value: Any) -> Optional[str]:
    return value.isoformat() if value else None


def _stable_id_suffix(value: Any, *, length: int = 16) -> str:
    text = str(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[: max(8, int(length))]


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


async def _run_with_timeout(coro, *, context: str, timeout_sec: int = _TOOL_TIMEOUT_SEC):
    timeout = max(3, int(timeout_sec))
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"{context} timed out after {timeout}s")
        return None


def _normalize_sources(allowed_sources: Optional[Iterable[str]]) -> Optional[Set[str]]:
    if allowed_sources is None:
        return None
    aliases = {
        "semantic-scholar": "semantic_scholar",
        "semantic scholar": "semantic_scholar",
        "open-review": "openreview",
        "open review": "openreview",
        "hn": "hackernews",
        "stack-overflow": "stackoverflow",
        "stack overflow": "stackoverflow",
        "arxiv_daily": "arxiv_rss",
        "arxiv-rss": "arxiv_rss",
        "arxiv rss": "arxiv_rss",
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
    try:
        async with ArxivScraper() as scraper:
            papers = []
            sort_by = "lastUpdatedDate" if _has_recency_intent(query) else None
            for candidate in _query_candidates(query):
                try:
                    found = await _run_with_timeout(
                        scraper.search(candidate, max_results=max_results, sort_by=sort_by),
                        context=f"ArXiv search '{candidate}'",
                    )
                except Exception as candidate_exc:
                    logger.warning(f"ArXiv search failed for '{candidate}': {candidate_exc}")
                    continue
                found = found or []
                if found:
                    papers = found
                    if candidate != query:
                        logger.info(f"ArXiv fallback query hit: '{candidate}'")
                    break
    except Exception as exc:
        logger.warning(f"ArXiv search failed: {exc}")
        return []

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
                "pdf_url": getattr(paper, "pdf_url", None),
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
    try:
        async with SemanticScholarScraper() as scraper:
            papers = []
            year_range = None
            if _has_recency_intent(query):
                current_year = datetime.now().year
                year_range = (max(2000, current_year - 2), current_year + 1)
            for candidate in _query_candidates(query):
                try:
                    found = await _run_with_timeout(
                        scraper.search(candidate, max_results=max_results, year_range=year_range),
                        context=f"Semantic Scholar search '{candidate}'",
                    )
                except Exception as candidate_exc:
                    logger.warning(f"Semantic Scholar search failed for '{candidate}': {candidate_exc}")
                    continue
                found = found or []
                if found:
                    papers = found
                    if candidate != query:
                        logger.info(f"Semantic Scholar fallback query hit: '{candidate}'")
                    break
    except Exception as exc:
        logger.warning(f"Semantic Scholar search failed: {exc}")
        return []

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
                "pdf_url": getattr(paper, "pdf_url", None),
            },
        )
        for paper in papers
    ]
    logger.info(f"Semantic Scholar search '{query}': found {len(results)} papers")
    return results


async def openreview_search(
    query: str,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """搜索 OpenReview 论文/评审条目。"""
    timeout = httpx.Timeout(12.0)
    params_base = {
        "type": "terms",
        "content": "all",
        "group": "all",
        "source": "all",
    }
    results: List[Dict[str, Any]] = []
    query_candidates = _query_candidates(query, limit=3)
    if _has_recency_intent(query):
        query_candidates.extend(
            [
                f"{candidate} {datetime.now().year}"
                for candidate in query_candidates
                if not re.search(r"(19|20)\d{2}", candidate)
            ]
        )

    async with httpx.AsyncClient(timeout=timeout) as client:
        for candidate in query_candidates:
            try:
                params = dict(params_base)
                params.update(
                    {
                        "term": candidate,
                        "offset": 0,
                        "limit": max(1, min(int(max_results), 20)),
                    }
                )
                response = await _run_with_timeout(
                    client.get("https://api2.openreview.net/notes/search", params=params),
                    context=f"OpenReview search '{candidate}'",
                )
                if response is None or getattr(response, "status_code", 500) != 200:
                    continue
                payload = response.json() if callable(getattr(response, "json", None)) else {}
                notes = list(payload.get("notes") or [])
                if not notes:
                    continue

                parsed: List[Dict[str, Any]] = []
                for note in notes[: max(1, min(int(max_results), 20))]:
                    if not isinstance(note, dict):
                        continue
                    content = dict(note.get("content") or {})
                    title_obj = content.get("title")
                    abstract_obj = content.get("abstract")
                    title = (
                        str(title_obj.get("value", "")).strip()
                        if isinstance(title_obj, dict)
                        else str(title_obj or "").strip()
                    )
                    abstract = (
                        str(abstract_obj.get("value", "")).strip()
                        if isinstance(abstract_obj, dict)
                        else str(abstract_obj or "").strip()
                    )
                    if not title:
                        continue
                    note_id = str(note.get("id") or "").strip()
                    note_url = f"https://openreview.net/forum?id={note_id}" if note_id else "https://openreview.net/"
                    venue_obj = content.get("venue")
                    venue = (
                        str(venue_obj.get("value", "")).strip()
                        if isinstance(venue_obj, dict)
                        else str(venue_obj or "").strip()
                    )
                    cdate = note.get("cdate")
                    published_date = ""
                    if isinstance(cdate, (int, float)) and cdate > 0:
                        try:
                            published_date = datetime.fromtimestamp(
                                float(cdate) / 1000.0,
                                tz=timezone.utc,
                            ).isoformat()
                        except Exception:
                            published_date = ""
                    parsed.append(
                        _result(
                            id=f"openreview_{note_id or _stable_id_suffix(title)}",
                            source="openreview",
                            title=title,
                            content=abstract,
                            url=note_url,
                            metadata={
                                "venue": venue,
                                "published_date": published_date,
                                "authors": content.get("authors"),
                            },
                        )
                    )
                if parsed:
                    results = parsed
                    if candidate != query:
                        logger.info(f"OpenReview fallback query hit: '{candidate}'")
                    break
            except Exception as exc:
                logger.warning(f"OpenReview search failed for '{candidate}': {exc}")
                continue

    logger.info(f"OpenReview search '{query}': found {len(results)} items")
    return results


def _rss_item_text(node: ET.Element, tag: str) -> str:
    child = node.find(tag)
    if child is None:
        return ""
    text = child.text or ""
    return html_lib.unescape(re.sub(r"<[^>]+>", " ", text)).strip()


async def arxiv_rss_search(
    query: str,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """通过 ArXiv Daily RSS 搜索近期更新论文。"""
    tokens = [token.lower() for token in re.findall(r"[a-zA-Z0-9]+(?:\.\d+)?|[\u4e00-\u9fff]+", query) if len(token) >= 2]
    if not tokens:
        tokens = [query.strip().lower()]
    max_results = max(1, min(int(max_results), 20))
    timeout = httpx.Timeout(12.0)
    rows: List[Dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [client.get(url) for url in _ARXIV_RSS_FEEDS]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    for feed_url, response in zip(_ARXIV_RSS_FEEDS, responses):
        if isinstance(response, Exception):
            continue
        if getattr(response, "status_code", 500) != 200:
            continue
        xml_text = str(getattr(response, "text", "") or "")
        if not xml_text:
            continue
        try:
            root = ET.fromstring(xml_text)
        except Exception:
            continue
        for item in root.findall(".//item"):
            title = _rss_item_text(item, "title")
            description = _rss_item_text(item, "description")
            link = _rss_item_text(item, "link")
            pub_date_raw = _rss_item_text(item, "pubDate")
            text_for_match = f"{title} {description}".lower()
            if tokens and not all(token in text_for_match for token in tokens[:2]):
                # Relax with partial hit when query is long.
                if not any(token in text_for_match for token in tokens):
                    continue
            published_date = ""
            if pub_date_raw:
                try:
                    published_date = parsedate_to_datetime(pub_date_raw).isoformat()
                except Exception:
                    published_date = pub_date_raw
            arxiv_id = ""
            if "arxiv.org/abs/" in link:
                arxiv_id = link.rstrip("/").split("/")[-1]
            rows.append(
                _result(
                    id=f"arxiv_rss_{arxiv_id or _stable_id_suffix((title, link))}",
                    source="arxiv_rss",
                    title=title,
                    content=description,
                    url=link,
                    metadata={
                        "published_date": published_date,
                        "feed_url": feed_url,
                        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else "",
                    },
                )
            )

    rows.sort(
        key=lambda item: str((item.get("metadata") or {}).get("published_date") or ""),
        reverse=True,
    )
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for item in rows:
        key = str(item.get("url", "")).strip() or str(item.get("id", "")).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= max_results:
            break
    logger.info(f"ArXiv RSS search '{query}': found {len(deduped)} items")
    return deduped


async def huggingface_search(
    query: str,
    search_type: str = "all",
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """搜索 Hugging Face 模型与数据集。"""
    try:
        async with HuggingFaceScraper() as scraper:
            for candidate in _query_candidates(query):
                results: List[Dict[str, Any]] = []
                if search_type in ["all", "models"]:
                    models = await _run_with_timeout(
                        scraper.search_models(candidate, max_results=max_results),
                        context=f"HuggingFace models search '{candidate}'",
                    )
                    models = models or []
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
                                    "repo_id": model.id,
                                    "downloads": getattr(model, "downloads", None),
                                    "likes": getattr(model, "likes", None),
                                    "tags": getattr(model, "tags", None),
                                    "pipeline_tag": getattr(model, "pipeline_tag", None),
                                },
                            )
                            for model in models
                        ]
                    )

                if search_type in ["all", "datasets"]:
                    datasets = await _run_with_timeout(
                        scraper.search_datasets(candidate, max_results=max_results),
                        context=f"HuggingFace datasets search '{candidate}'",
                    )
                    datasets = datasets or []
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
                                    "repo_id": dataset.id,
                                    "downloads": dataset.downloads,
                                    "likes": getattr(dataset, "likes", None),
                                    "tags": getattr(dataset, "tags", None),
                                },
                            )
                            for dataset in datasets
                        ]
                    )
                if results:
                    if candidate != query:
                        logger.info(f"HuggingFace fallback query hit: '{candidate}'")
                    logger.info(f"HuggingFace search '{query}' ({search_type}): found {len(results)} items")
                    return results
    except Exception as exc:
        logger.warning(f"HuggingFace search failed: {exc}")
        return []

    logger.info(f"HuggingFace search '{query}' ({search_type}): found 0 items")
    return []


async def github_search(
    query: str,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """搜索 GitHub 仓库 + Issues/Discussions。"""
    try:
        async with GitHubScraper() as scraper:
            repos = []
            discussions = []
            repo_sort = "updated" if _has_recency_intent(query) else "stars"
            for candidate in _query_candidates(query):
                repos_found, discussions_found = await asyncio.gather(
                    _run_with_timeout(
                        scraper.search_repos(candidate, max_results=max_results, sort=repo_sort),
                        context=f"GitHub repo search '{candidate}'",
                    ),
                    _run_with_timeout(
                        scraper.search_discussions(
                            candidate, max_results=max(1, int(max_results // 2))
                        ),
                        context=f"GitHub issue search '{candidate}'",
                    ),
                )
                repos_found = repos_found or []
                discussions_found = discussions_found or []
                if repos_found or discussions_found:
                    repos = repos_found
                    discussions = discussions_found
                    if candidate != query:
                        logger.info(f"GitHub fallback query hit: '{candidate}'")
                    break

        repo_results = [
            _result(
                id=f"github_{repo.id}",
                source="github",
                title=repo.full_name,
                content=repo.description or "",
                url=repo.url,
                metadata={
                    "owner": repo.owner,
                    "repo_full_name": repo.full_name,
                    "stars": repo.stars,
                    "forks": repo.forks,
                    "language": repo.language,
                    "topics": repo.topics,
                    "updated_at": _iso(repo.updated_at),
                },
            )
            for repo in repos
        ]

        discussion_results = [
            _result(
                id=f"github_issue_{post.id}",
                source="github",
                title=(str(post.content).splitlines()[0].strip() if str(post.content).strip() else "GitHub discussion"),
                content=post.content or "",
                url=post.url,
                metadata={
                    "type": "discussion",
                    "author": post.author,
                    "comments": post.comments,
                    "created_at": _iso(post.created_at),
                    **(post.extra or {}),
                },
            )
            for post in discussions
        ]

        results = repo_results + discussion_results
        logger.info(
            f"GitHub search '{query}': found {len(repo_results)} repos and {len(discussion_results)} discussions"
        )
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
    try:
        async with StackOverflowScraper() as scraper:
            questions = []
            for candidate in _query_candidates(query):
                found = await _run_with_timeout(
                    scraper.search(candidate, max_results=max_results, tags=tags),
                    context=f"StackOverflow search '{candidate}'",
                )
                found = found or []
                if found:
                    questions = found
                    if candidate != query:
                        logger.info(f"StackOverflow fallback query hit: '{candidate}'")
                    break
    except Exception as exc:
        logger.warning(f"StackOverflow search failed: {exc}")
        return []

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
    try:
        async with HackerNewsScraper() as scraper:
            items = []
            for candidate in _query_candidates(query):
                found = await _run_with_timeout(
                    scraper.search(candidate, max_results=max_results, sort_by=sort_by),
                    context=f"HackerNews search '{candidate}'",
                )
                found = found or []
                if found:
                    items = found
                    if candidate != query:
                        logger.info(f"HackerNews fallback query hit: '{candidate}'")
                    break
    except Exception as exc:
        logger.warning(f"HackerNews search failed: {exc}")
        return []

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
    timeout_sec = max(3, int(_TOOL_TIMEOUT_SEC))

    def _as_int(value: Any, default: int = 0) -> int:
        try:
            if value in (None, ""):
                return default
            return int(value)
        except (TypeError, ValueError):
            return default

    try:
        async with TwitterScraper() as scraper:
            posts = await asyncio.wait_for(
                scraper.search(query, max_results=max_results),
                timeout=timeout_sec,
            )
        if posts is None:
            posts = []
        if not isinstance(posts, list):
            raise TypeError(
                f"Twitter search expected list result, got {type(posts).__name__}"
            )
    except asyncio.TimeoutError:
        logger.warning(f"Twitter search '{query}' timed out after {timeout_sec}s")
        return []
    except (AttributeError, RuntimeError, TypeError, ValueError, httpx.HTTPError) as exc:
        logger.error(f"Twitter search failed for query '{query}': {exc}", exc_info=True)
        return []

    results: List[Dict[str, Any]] = []
    for post in posts:
        post_id_for_log = getattr(post, "id", "unknown")
        try:
            extra = getattr(post, "extra", {}) or {}
            if not isinstance(extra, dict):
                raise TypeError(f"unexpected extra type: {type(extra).__name__}")

            post_id = str(getattr(post, "id", extra.get("id", ""))).strip()
            if not post_id:
                raise ValueError("missing twitter post id")

            author = str(getattr(post, "author", extra.get("author", "")) or "").strip()
            if not author:
                author = "unknown"
            title = author if author.startswith("@") else f"@{author}"

            content = str(getattr(post, "content", extra.get("content", "")) or "")
            url = getattr(post, "url", extra.get("url"))
            likes = _as_int(getattr(post, "likes", extra.get("likes", 0)))
            reposts = _as_int(
                getattr(
                    post,
                    "reposts",
                    extra.get("reposts", extra.get("retweets", extra.get("retweet_count", 0))),
                )
            )
            comments = _as_int(
                getattr(
                    post,
                    "comments",
                    extra.get("comments", extra.get("replies", extra.get("reply_count", 0))),
                )
            )
            created_at_raw = getattr(post, "created_at", extra.get("created_at"))
            created_at = _iso(created_at_raw) if hasattr(created_at_raw, "isoformat") else None
            if created_at is None and created_at_raw is not None:
                created_at = str(created_at_raw)

            results.append(
                _result(
                    id=f"twitter_{post_id}",
                    source="twitter",
                    title=title,
                    content=content,
                    url=url,
                    metadata={
                        "author": author,
                        "likes": likes,
                        "reposts": reposts,
                        "comments": comments,
                        "retweets": reposts,
                        "replies": comments,
                        "created_at": created_at,
                    },
                )
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            logger.error(
                "Twitter post parse failed for post_id=%s query='%s': %s",
                post_id_for_log,
                query,
                exc,
                exc_info=True,
            )
            continue

    logger.info(f"Twitter search '{query}': found {len(results)} tweets")
    return results


async def reddit_search(
    query: str,
    subreddits: Optional[List[str]] = None,
    max_results: int = 20,
) -> List[Dict[str, Any]]:
    """搜索 Reddit。"""
    timeout_sec = max(3, int(_TOOL_TIMEOUT_SEC))

    def _as_int(value: Any, default: int = 0) -> int:
        try:
            if value in (None, ""):
                return default
            return int(value)
        except (TypeError, ValueError):
            return default

    try:
        target_subreddits = subreddits or ["MachineLearning", "LocalLLaMA", "artificial"]
        async with RedditScraper() as scraper:
            posts = await asyncio.wait_for(
                scraper.search(
                    query,
                    subreddits=target_subreddits,
                    max_results=max_results,
                ),
                timeout=timeout_sec,
            )
        if posts is None:
            posts = []
        if not isinstance(posts, list):
            raise TypeError(
                f"Reddit search expected list result, got {type(posts).__name__}"
            )
    except asyncio.TimeoutError:
        logger.warning(f"Reddit search '{query}' timed out after {timeout_sec}s")
        return []
    except (AttributeError, RuntimeError, TypeError, ValueError, httpx.HTTPError) as exc:
        logger.error(f"Reddit search failed for query '{query}': {exc}", exc_info=True)
        return []

    results: List[Dict[str, Any]] = []
    for post in posts:
        post_id_for_log = getattr(post, "id", "unknown")
        try:
            extra = getattr(post, "extra", {}) or {}
            if not isinstance(extra, dict):
                raise TypeError(f"unexpected extra type: {type(extra).__name__}")

            post_id = str(getattr(post, "id", extra.get("id", ""))).strip()
            if not post_id:
                raise ValueError("missing reddit post id")

            content = str(getattr(post, "content", extra.get("content", "")) or "")
            subreddit = str(getattr(post, "subreddit", extra.get("subreddit", "")) or "").strip()
            title = str(getattr(post, "title", extra.get("title", "")) or "").strip()
            if not title:
                first_line = content.splitlines()[0].strip() if content else ""
                title = first_line or (f"r/{subreddit}" if subreddit else "Reddit discussion")

            author = str(getattr(post, "author", extra.get("author", "")) or "").strip()
            if not author:
                author = "[unknown]"
            url = getattr(post, "url", extra.get("url"))
            score = _as_int(getattr(post, "score", getattr(post, "likes", extra.get("score", extra.get("likes", 0)))))
            num_comments = _as_int(
                getattr(
                    post,
                    "num_comments",
                    getattr(post, "comments", extra.get("num_comments", extra.get("comments", 0))),
                )
            )
            created_at_raw = getattr(post, "created_at", extra.get("created_at"))
            created_at = _iso(created_at_raw) if hasattr(created_at_raw, "isoformat") else None
            if created_at is None and created_at_raw is not None:
                created_at = str(created_at_raw)

            results.append(
                _result(
                    id=f"reddit_{post_id}",
                    source="reddit",
                    title=title,
                    content=content,
                    url=url,
                    metadata={
                        "author": author,
                        "subreddit": subreddit or None,
                        "score": score,
                        "likes": score,
                        "num_comments": num_comments,
                        "comments": num_comments,
                        "created_at": created_at,
                    },
                )
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            logger.error(
                "Reddit post parse failed for post_id=%s query='%s': %s",
                post_id_for_log,
                query,
                exc,
                exc_info=True,
            )
            continue

    logger.info(f"Reddit search '{query}': found {len(results)} posts")
    return results


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
    "arxiv_rss_search": arxiv_rss_search,
    "huggingface_search": huggingface_search,
    "twitter_search": twitter_search,
    "reddit_search": reddit_search,
    "github_search": github_search,
    "semantic_scholar_search": semantic_scholar_search,
    "openreview_search": openreview_search,
    "stackoverflow_search": stackoverflow_search,
    "hackernews_search": hackernews_search,
}


async def execute_search_tool(name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
    """执行搜索工具。"""
    executor = SEARCH_TOOL_EXECUTORS.get(name)
    if not executor:
        raise ValueError(f"Unknown search tool: {name}")
    return await executor(**arguments)
