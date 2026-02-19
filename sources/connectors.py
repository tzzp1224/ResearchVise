"""Tiered source connectors for v2 ingestion."""

from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import html as html_lib
import os
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

import httpx

from config import get_settings
from core import RawItem
from scrapers.hackernews_scraper import HackerNewsScraper
from scrapers.huggingface_scraper import HuggingFaceScraper
from scrapers.social.github_scraper import GitHubScraper

try:
    import trafilatura
except Exception:  # pragma: no cover - optional runtime dependency
    trafilatura = None  # type: ignore[assignment]

try:
    from readability import Document
except Exception:  # pragma: no cover - optional runtime dependency
    Document = None  # type: ignore[assignment]

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover - optional runtime dependency
    BeautifulSoup = None  # type: ignore[assignment]

try:
    from newspaper import Article
except Exception:  # pragma: no cover - optional runtime dependency
    Article = None  # type: ignore[assignment]


_HN_FIREBASE = "https://hacker-news.firebaseio.com/v0"
_HN_ALGOLIA_SEARCH = "https://hn.algolia.com/api/v1/search"

_TOPIC_EXPANSION_TERMS = [
    "agentic",
    "autonomous agent",
    "multi-agent",
    "tool calling",
    "function calling",
    "orchestration",
    "workflow",
    "mcp",
    "langchain",
    "langgraph",
    "autogen",
    "crewai",
    "openhands",
    "swe-agent",
    "rag agent",
]


def _normalize_queries(topic: str, *, expanded: bool = False, queries: Optional[List[str]] = None) -> List[str]:
    provided = [str(item or "").strip() for item in list(queries or []) if str(item or "").strip()]
    if provided:
        deduped: List[str] = []
        seen = set()
        for query in provided:
            key = query.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(query)
        return deduped
    return _topic_queries(topic, expanded=expanded)


def _matches_topic_constraints(
    *,
    text: str,
    must_include_terms: Optional[List[str]] = None,
    must_exclude_terms: Optional[List[str]] = None,
) -> bool:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return False

    include_tokens = [str(item or "").strip().lower() for item in list(must_include_terms or []) if str(item or "").strip()]
    exclude_tokens = [str(item or "").strip().lower() for item in list(must_exclude_terms or []) if str(item or "").strip()]

    if include_tokens and not any(token in lowered for token in include_tokens):
        return False
    if exclude_tokens and any(token in lowered for token in exclude_tokens):
        return False
    return True


def _parse_window_days(time_window: str) -> int:
    token = str(time_window or "").strip().lower()
    if not token:
        return 3
    if token == "today":
        return 3
    if token.endswith("d"):
        try:
            return max(1, int(token[:-1]))
        except Exception:
            return 3
    if token.endswith("h"):
        try:
            hours = int(token[:-1])
            return max(1, int((hours + 23) // 24))
        except Exception:
            return 1
    if token in {"past_week", "last_week", "weekly"}:
        return 7
    if token in {"past_month", "monthly"}:
        return 30
    return 3


def _topic_queries(topic: str, *, expanded: bool = False) -> List[str]:
    seed = str(topic or "").strip()
    if not seed:
        return []
    queries: List[str] = [seed]
    lowered = seed.lower()
    if "agent" in lowered or "copilot" in lowered or "assistant" in lowered:
        queries.extend(["agent framework", "agent orchestration", "tool-calling agent"])
    if expanded:
        queries.extend(_TOPIC_EXPANSION_TERMS)
    deduped: List[str] = []
    seen = set()
    for query in queries:
        token = str(query or "").strip()
        key = token.lower()
        if not token or key in seen:
            continue
        seen.add(key)
        deduped.append(token)
    return deduped


def _parse_datetime(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None

    normalized = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass

    try:
        dt2 = parsedate_to_datetime(text)
        if dt2.tzinfo is None:
            dt2 = dt2.replace(tzinfo=timezone.utc)
        return dt2.astimezone(timezone.utc)
    except Exception:
        return None


def _coalesce_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _safe_truncate(text: str, max_len: int = 9000) -> str:
    value = str(text or "")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t\f\v]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value).strip()
    if len(value) <= max_len:
        return value
    return value[: max_len - 3].rstrip() + "..."


def _strip_html(value: str) -> str:
    text = str(value or "")
    text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html_lib.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


async def _http_get_json(url: str, *, headers: Optional[Dict[str, str]] = None) -> Any:
    timeout = httpx.Timeout(12.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()


async def _http_get_text(url: str, *, headers: Optional[Dict[str, str]] = None) -> str:
    timeout = httpx.Timeout(12.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return str(response.text or "")


def _github_headers(token: Optional[str]) -> Dict[str, str]:
    headers: Dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "AcademicResearchAgent/2.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


async def _fetch_github_readme(repo_full_name: str, token: Optional[str]) -> str:
    headers = _github_headers(token)
    raw_headers = dict(headers)
    raw_headers["Accept"] = "application/vnd.github.raw"

    api_url = f"https://api.github.com/repos/{repo_full_name}/readme"
    try:
        text = await _http_get_text(api_url, headers=raw_headers)
        if len(text.strip()) >= 80:
            return _safe_truncate(text, max_len=7000)
    except Exception:
        pass

    fallback_url = f"https://raw.githubusercontent.com/{repo_full_name}/HEAD/README.md"
    try:
        text = await _http_get_text(fallback_url, headers={"User-Agent": "AcademicResearchAgent/2.0"})
        if len(text.strip()) >= 80:
            return _safe_truncate(text, max_len=7000)
    except Exception:
        pass

    return ""


async def _fetch_hf_card(repo_id: str, *, repo_type: str = "model") -> str:
    base = "datasets/" if repo_type == "dataset" else ""
    card_url = f"https://huggingface.co/{base}{repo_id}/raw/main/README.md"
    try:
        text = await _http_get_text(card_url, headers={"User-Agent": "AcademicResearchAgent/2.0"})
        cleaned = _safe_truncate(text, max_len=7000)
        if len(cleaned) >= 120:
            return cleaned
    except Exception:
        return ""
    return ""


def _extract_with_trafilatura(html: str, url: str) -> Tuple[str, Optional[str]]:
    if trafilatura is None:
        return "", "trafilatura_unavailable"
    try:
        extracted = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
            output_format="txt",
        )
        text = _safe_truncate(str(extracted or ""), max_len=7000)
        if len(text) >= 180:
            return text, None
        return "", "trafilatura_empty"
    except Exception as exc:
        return "", f"trafilatura_error:{exc}"


def _extract_with_readability(html: str) -> Tuple[str, Optional[str]]:
    if Document is None:
        return "", "readability_unavailable"
    try:
        summary = Document(html).summary()
        text = _safe_truncate(_strip_html(summary), max_len=7000)
        if len(text) >= 180:
            return text, None
        return "", "readability_empty"
    except Exception as exc:
        return "", f"readability_error:{exc}"


def _extract_with_newspaper(html: str, url: str) -> Tuple[str, Optional[str]]:
    if Article is None:
        return "", "newspaper3k_unavailable"
    try:
        article = Article(url=url or "https://example.com")
        article.set_html(html)
        article.parse()
        text = _safe_truncate(str(article.text or ""), max_len=7000)
        if len(text) >= 180:
            return text, None
        return "", "newspaper3k_empty"
    except Exception as exc:
        return "", f"newspaper3k_error:{exc}"


def _extract_with_fallback(html: str) -> Tuple[str, Optional[str]]:
    if BeautifulSoup is not None:
        try:
            soup = BeautifulSoup(html, "lxml")
            article = soup.find("article")
            node = article if article else soup
            paragraphs = [tag.get_text(" ", strip=True) for tag in node.find_all(["p", "li"]) if tag.get_text(" ", strip=True)]
            text = "\n".join(paragraphs)
            cleaned = _safe_truncate(_strip_html(text), max_len=7000)
            if len(cleaned) >= 120:
                return cleaned, None
        except Exception:
            pass

    body_match = re.search(r"<article[^>]*>(.*?)</article>", html, flags=re.IGNORECASE | re.DOTALL)
    candidate = body_match.group(1) if body_match else html
    text = _safe_truncate(_strip_html(candidate), max_len=7000)
    if len(text) >= 80:
        return text, None
    return _safe_truncate(_strip_html(html), max_len=3000), "fallback_short_body"


def _extract_article_content(html: str, *, url: str) -> Tuple[str, str, Optional[str]]:
    text, error = _extract_with_trafilatura(html, url)
    if text:
        return text, "trafilatura", None

    errors = [error] if error else []

    text, error = _extract_with_readability(html)
    if text:
        return text, "readability", None
    if error:
        errors.append(error)

    text, error = _extract_with_newspaper(html, url)
    if text:
        return text, "newspaper3k", None
    if error:
        errors.append(error)

    fallback_text, fallback_error = _extract_with_fallback(html)
    if fallback_error:
        errors.append(fallback_error)

    combined_error = " | ".join([item for item in errors if item]) or None
    return fallback_text, "fallback", combined_error


async def _hn_fetch_item(item_id: str) -> Dict[str, Any]:
    return await _http_get_json(f"{_HN_FIREBASE}/item/{item_id}.json")


async def _hn_fetch_top_comments(item: Dict[str, Any], *, max_comments: int = 3) -> List[str]:
    kids = list(item.get("kids") or [])[: max(0, int(max_comments))]
    snippets: List[str] = []
    for kid in kids:
        try:
            payload = await _hn_fetch_item(str(kid))
            if not payload or payload.get("deleted") or payload.get("dead"):
                continue
            text = _safe_truncate(_strip_html(str(payload.get("text") or "")), max_len=360)
            if text:
                snippets.append(text)
        except Exception:
            continue
    return snippets


async def fetch_github_trending(
    max_results: int = 20,
    *,
    language: Optional[str] = None,
    since: str = "weekly",
) -> List[RawItem]:
    """Fetch GitHub trending repositories with README/metadata enrichment (Tier A)."""
    async with GitHubScraper() as scraper:
        repos = await scraper.get_trending(language=language, since=since)

    token = str(get_settings().github.token or os.getenv("GITHUB_TOKEN") or "").strip() or None
    items: List[RawItem] = []
    for repo in repos[: max(1, int(max_results))]:
        readme = await _fetch_github_readme(repo.full_name, token)
        body = _safe_truncate(
            "\n\n".join(
                [
                    _coalesce_text(repo.description, ""),
                    f"Stars: {int(repo.stars or 0)} | Forks: {int(repo.forks or 0)} | LastPush: {repo.updated_at or 'unknown'}",
                    readme,
                ]
            ),
            max_len=9000,
        )
        items.append(
            RawItem(
                id=f"github_repo_{repo.id}",
                source="github",
                title=repo.full_name,
                url=repo.url,
                body=body,
                author=repo.owner,
                published_at=repo.updated_at or repo.created_at,
                tier="A",
                metadata={
                    "stars": int(repo.stars or 0),
                    "forks": int(repo.forks or 0),
                    "watchers": int(repo.watchers or 0),
                    "last_push": str(repo.updated_at or ""),
                    "language": repo.language,
                    "topics": list(repo.topics or []),
                    "item_type": "repo",
                    "extraction_method": "github_api_readme" if readme else "github_api_metadata",
                    "extraction_failed": bool(not readme),
                    "readme_len": len(readme),
                },
            )
        )
    return items


async def fetch_github_topic_search(
    topic: str,
    time_window: str = "today",
    limit: int = 20,
    *,
    expanded: bool = False,
    queries: Optional[List[str]] = None,
    must_include_terms: Optional[List[str]] = None,
    must_exclude_terms: Optional[List[str]] = None,
) -> List[RawItem]:
    """Topic-first GitHub repo retrieval for live mode."""
    query_list = _normalize_queries(topic, expanded=expanded, queries=queries)
    if not query_list:
        return []

    max_items = max(1, int(limit))
    per_query = max(2, int(max_items / max(1, len(queries))) + 1)
    window_days = _parse_window_days(time_window)
    cutoff = datetime.now(timezone.utc).timestamp() - float(window_days) * 86400.0

    token = str(get_settings().github.token or os.getenv("GITHUB_TOKEN") or "").strip() or None
    repos_by_full_name: Dict[str, Tuple[Any, str]] = {}
    async with GitHubScraper() as scraper:
        for query in query_list:
            repos = await scraper.search_repos(query=query, max_results=per_query, sort="updated", order="desc")
            for repo in list(repos or []):
                full_name = str(getattr(repo, "full_name", "") or "").strip()
                if not full_name:
                    continue
                relevance_text = " ".join(
                    [
                        full_name,
                        str(getattr(repo, "description", "") or ""),
                        " ".join([str(tag) for tag in list(getattr(repo, "topics", []) or [])]),
                    ]
                )
                if not _matches_topic_constraints(
                    text=relevance_text,
                    must_include_terms=must_include_terms,
                    must_exclude_terms=must_exclude_terms,
                ):
                    continue
                updated_dt = _parse_datetime(getattr(repo, "updated_at", None) or getattr(repo, "created_at", None))
                if updated_dt and updated_dt.timestamp() < cutoff:
                    continue
                if full_name not in repos_by_full_name:
                    repos_by_full_name[full_name] = (repo, query)
                if len(repos_by_full_name) >= max_items * 2:
                    break
            if len(repos_by_full_name) >= max_items * 2:
                break

    items: List[RawItem] = []
    for full_name, (repo, query) in list(repos_by_full_name.items())[:max_items]:
        readme = await _fetch_github_readme(full_name, token)
        body = _safe_truncate(
            "\n\n".join(
                [
                    _coalesce_text(getattr(repo, "description", ""), ""),
                    (
                        f"Stars: {int(getattr(repo, 'stars', 0) or 0)} | "
                        f"Forks: {int(getattr(repo, 'forks', 0) or 0)} | "
                        f"LastPush: {getattr(repo, 'updated_at', '') or 'unknown'}"
                    ),
                    readme,
                ]
            ),
            max_len=9000,
        )
        items.append(
            RawItem(
                id=f"github_topic_{getattr(repo, 'id', full_name)}",
                source="github",
                title=full_name,
                url=_coalesce_text(getattr(repo, "url", ""), f"https://github.com/{full_name}"),
                body=body,
                author=getattr(repo, "owner", None),
                published_at=_parse_datetime(getattr(repo, "updated_at", None) or getattr(repo, "created_at", None)),
                tier="A",
                metadata={
                    "stars": int(getattr(repo, "stars", 0) or 0),
                    "forks": int(getattr(repo, "forks", 0) or 0),
                    "watchers": int(getattr(repo, "watchers", 0) or 0),
                    "last_push": str(getattr(repo, "updated_at", "") or ""),
                    "language": getattr(repo, "language", None),
                    "topics": list(getattr(repo, "topics", []) or []),
                    "item_type": "repo",
                    "retrieval_mode": "topic_search",
                    "source_query": query,
                    "window_days": window_days,
                    "search_endpoint": "github_search_repositories",
                    "extraction_method": "github_topic_search_readme" if readme else "github_topic_search_metadata",
                    "extraction_failed": bool(not readme),
                    "readme_len": len(readme),
                },
            )
        )
    return items


async def fetch_github_releases(
    repo_full_names: List[str],
    max_results_per_repo: int = 2,
    *,
    token: Optional[str] = None,
) -> List[RawItem]:
    """Fetch latest GitHub releases for selected repositories (Tier A)."""
    normalized_repos = [str(item or "").strip() for item in list(repo_full_names or []) if str(item or "").strip()]
    if not normalized_repos:
        return []

    auth_token = token or str(get_settings().github.token or os.getenv("GITHUB_TOKEN") or "").strip() or None
    headers = _github_headers(auth_token)

    items: List[RawItem] = []
    for repo in normalized_repos:
        try:
            url = f"https://api.github.com/repos/{repo}/releases"
            payload = await _http_get_json(url, headers=headers)
        except Exception:
            continue

        for release in list(payload or [])[: max(1, int(max_results_per_repo))]:
            rel_name = _coalesce_text(release.get("name"), release.get("tag_name"), "Release")
            rel_url = _coalesce_text(release.get("html_url"), f"https://github.com/{repo}/releases")
            body = _safe_truncate(
                _coalesce_text(
                    release.get("body"),
                    release.get("name"),
                    release.get("tag_name"),
                    "No release notes available.",
                ),
                max_len=9000,
            )
            items.append(
                RawItem(
                    id=f"github_release_{repo}_{release.get('id') or release.get('tag_name') or rel_name}",
                    source="github",
                    title=f"{repo} {rel_name}",
                    url=rel_url,
                    body=body,
                    author=_coalesce_text((release.get("author") or {}).get("login"), repo.split("/", 1)[0]),
                    published_at=_parse_datetime(release.get("published_at") or release.get("created_at")),
                    tier="A",
                    metadata={
                        "repo": repo,
                        "tag_name": release.get("tag_name"),
                        "prerelease": bool(release.get("prerelease", False)),
                        "item_type": "release",
                        "extraction_method": "github_release_notes",
                        "extraction_failed": False,
                    },
                )
            )
    return items


async def fetch_huggingface_trending(max_results: int = 20) -> List[RawItem]:
    """Fetch Hugging Face models/datasets with card markdown enrichment (Tier A)."""
    async with HuggingFaceScraper() as scraper:
        models = await scraper.search_models(query="", max_results=max(1, int(max_results)))
        dataset_max = max(1, int(max_results // 4))
        datasets = await scraper.search_datasets(query="", max_results=dataset_max)

    items: List[RawItem] = []

    for model in models[: max(1, int(max_results))]:
        card = await _fetch_hf_card(model.id, repo_type="model")
        body = _safe_truncate(
            "\n\n".join(
                [
                    _coalesce_text(model.description, ""),
                    f"Downloads: {int(model.downloads or 0)} | Likes: {int(model.likes or 0)} | LastModified: {model.updated_at or 'unknown'}",
                    card,
                ]
            ),
            max_len=9000,
        )
        items.append(
            RawItem(
                id=f"hf_model_{model.id}",
                source="huggingface",
                title=model.name or model.id,
                url=model.url,
                body=body,
                author=model.author,
                published_at=model.updated_at or model.created_at,
                tier="A",
                metadata={
                    "repo_id": model.id,
                    "downloads": int(model.downloads or 0),
                    "likes": int(model.likes or 0),
                    "tags": list(model.tags or []),
                    "last_modified": str(model.updated_at or ""),
                    "item_type": "model",
                    "extraction_method": "hf_card" if card else "hf_metadata",
                    "extraction_failed": bool(not card),
                    "card_len": len(card),
                },
            )
        )

    for dataset in datasets[:dataset_max]:
        card = await _fetch_hf_card(dataset.id, repo_type="dataset")
        body = _safe_truncate(
            "\n\n".join(
                [
                    _coalesce_text(dataset.description, ""),
                    f"Downloads: {int(dataset.downloads or 0)} | DatasetID: {dataset.id}",
                    card,
                ]
            ),
            max_len=9000,
        )
        items.append(
            RawItem(
                id=f"hf_dataset_{dataset.id}",
                source="huggingface",
                title=f"dataset/{dataset.name or dataset.id}",
                url=dataset.url,
                body=body,
                author=dataset.author,
                published_at=_parse_datetime((dataset.extra or {}).get("last_modified")),
                tier="A",
                metadata={
                    "repo_id": dataset.id,
                    "downloads": int(dataset.downloads or 0),
                    "tags": list(dataset.tags or []),
                    "item_type": "dataset",
                    "extraction_method": "hf_card" if card else "hf_metadata",
                    "extraction_failed": bool(not card),
                    "card_len": len(card),
                },
            )
        )

    return items


async def fetch_huggingface_search(
    topic: str,
    time_window: str = "today",
    limit: int = 20,
    *,
    expanded: bool = False,
    queries: Optional[List[str]] = None,
    must_include_terms: Optional[List[str]] = None,
    must_exclude_terms: Optional[List[str]] = None,
) -> List[RawItem]:
    """Topic-first Hugging Face retrieval using model/dataset search."""
    query_list = _normalize_queries(topic, expanded=expanded, queries=queries)
    if not query_list:
        return []

    max_items = max(1, int(limit))
    per_query = max(2, int(max_items / max(1, len(queries))) + 1)
    window_days = _parse_window_days(time_window)
    cutoff = datetime.now(timezone.utc).timestamp() - float(window_days) * 86400.0

    by_key: Dict[str, Tuple[str, Any, str]] = {}
    async with HuggingFaceScraper() as scraper:
        for query in query_list:
            models = await scraper.search_models(query=query, max_results=per_query, sort="lastModified")
            for model in list(models or []):
                model_id = str(getattr(model, "id", "") or "").strip()
                if not model_id:
                    continue
                relevance_text = " ".join(
                    [
                        str(getattr(model, "name", "") or ""),
                        str(getattr(model, "description", "") or ""),
                        " ".join([str(tag) for tag in list(getattr(model, "tags", []) or [])]),
                    ]
                )
                if not _matches_topic_constraints(
                    text=relevance_text,
                    must_include_terms=must_include_terms,
                    must_exclude_terms=must_exclude_terms,
                ):
                    continue
                updated_dt = _parse_datetime(getattr(model, "updated_at", None) or getattr(model, "created_at", None))
                if updated_dt and updated_dt.timestamp() < cutoff:
                    continue
                key = f"model:{model_id}"
                if key not in by_key:
                    by_key[key] = ("model", model, query)
            datasets = await scraper.search_datasets(query=query, max_results=max(1, per_query // 2))
            for dataset in list(datasets or []):
                dataset_id = str(getattr(dataset, "id", "") or "").strip()
                if not dataset_id:
                    continue
                relevance_text = " ".join(
                    [
                        str(getattr(dataset, "name", "") or ""),
                        str(getattr(dataset, "description", "") or ""),
                        " ".join([str(tag) for tag in list(getattr(dataset, "tags", []) or [])]),
                    ]
                )
                if not _matches_topic_constraints(
                    text=relevance_text,
                    must_include_terms=must_include_terms,
                    must_exclude_terms=must_exclude_terms,
                ):
                    continue
                updated_dt = _parse_datetime((getattr(dataset, "extra", {}) or {}).get("last_modified"))
                if updated_dt and updated_dt.timestamp() < cutoff:
                    continue
                key = f"dataset:{dataset_id}"
                if key not in by_key:
                    by_key[key] = ("dataset", dataset, query)
            if len(by_key) >= max_items * 2:
                break

    items: List[RawItem] = []
    for key, (item_type, payload, query) in list(by_key.items())[:max_items]:
        if item_type == "model":
            model = payload
            card = await _fetch_hf_card(model.id, repo_type="model")
            body = _safe_truncate(
                "\n\n".join(
                    [
                        _coalesce_text(getattr(model, "description", ""), ""),
                        (
                            f"Downloads: {int(getattr(model, 'downloads', 0) or 0)} | "
                            f"Likes: {int(getattr(model, 'likes', 0) or 0)} | "
                            f"LastModified: {getattr(model, 'updated_at', '') or 'unknown'}"
                        ),
                        card,
                    ]
                ),
                max_len=9000,
            )
            items.append(
                RawItem(
                    id=f"hf_search_{key}",
                    source="huggingface",
                    title=getattr(model, "name", None) or getattr(model, "id", "hf-model"),
                    url=getattr(model, "url", f"https://huggingface.co/{model.id}"),
                    body=body,
                    author=getattr(model, "author", None),
                    published_at=_parse_datetime(getattr(model, "updated_at", None) or getattr(model, "created_at", None)),
                    tier="A",
                    metadata={
                        "repo_id": getattr(model, "id", ""),
                        "downloads": int(getattr(model, "downloads", 0) or 0),
                        "likes": int(getattr(model, "likes", 0) or 0),
                        "tags": list(getattr(model, "tags", []) or []),
                        "last_modified": str(getattr(model, "updated_at", "") or ""),
                        "item_type": "model",
                        "retrieval_mode": "topic_search",
                        "source_query": query,
                        "window_days": window_days,
                        "search_endpoint": "huggingface_search_models",
                        "extraction_method": "hf_card" if card else "hf_metadata",
                        "extraction_failed": bool(not card),
                        "card_len": len(card),
                    },
                )
            )
        else:
            dataset = payload
            card = await _fetch_hf_card(dataset.id, repo_type="dataset")
            body = _safe_truncate(
                "\n\n".join(
                    [
                        _coalesce_text(getattr(dataset, "description", ""), ""),
                        f"Downloads: {int(getattr(dataset, 'downloads', 0) or 0)} | DatasetID: {dataset.id}",
                        card,
                    ]
                ),
                max_len=9000,
            )
            items.append(
                RawItem(
                    id=f"hf_search_{key}",
                    source="huggingface",
                    title=f"dataset/{getattr(dataset, 'name', None) or dataset.id}",
                    url=getattr(dataset, "url", f"https://huggingface.co/datasets/{dataset.id}"),
                    body=body,
                    author=getattr(dataset, "author", None),
                    published_at=_parse_datetime((getattr(dataset, "extra", {}) or {}).get("last_modified")),
                    tier="A",
                    metadata={
                        "repo_id": getattr(dataset, "id", ""),
                        "downloads": int(getattr(dataset, "downloads", 0) or 0),
                        "tags": list(getattr(dataset, "tags", []) or []),
                        "item_type": "dataset",
                        "retrieval_mode": "topic_search",
                        "source_query": query,
                        "window_days": window_days,
                        "search_endpoint": "huggingface_search_datasets",
                        "extraction_method": "hf_card" if card else "hf_metadata",
                        "extraction_failed": bool(not card),
                        "card_len": len(card),
                    },
                )
            )

    return items


async def fetch_hackernews_top(max_results: int = 20) -> List[RawItem]:
    """Fetch Hacker News top stories + optional top comments (Tier A)."""
    async with HackerNewsScraper() as scraper:
        stories = await scraper.get_front_page(max_results=max(1, int(max_results)))

    items: List[RawItem] = []
    for story in stories[: max(1, int(max_results))]:
        comments = []
        try:
            payload = await _hn_fetch_item(str(story.id))
            comments = await _hn_fetch_top_comments(payload, max_comments=3)
        except Exception:
            comments = []

        body = _safe_truncate(
            "\n\n".join(
                [
                    _coalesce_text(story.text, ""),
                    f"Points: {int(story.points or 0)} | Comments: {int(story.comment_count or 0)} | ItemType: {story.item_type}",
                    "\n".join([f"Top comment: {line}" for line in comments]),
                ]
            ),
            max_len=9000,
        )
        items.append(
            RawItem(
                id=f"hn_{story.id}",
                source="hackernews",
                title=story.title,
                url=_coalesce_text(story.url, story.hn_url),
                body=body,
                author=story.author,
                published_at=story.created_at,
                tier="A",
                metadata={
                    "points": int(story.points or 0),
                    "comment_count": int(story.comment_count or 0),
                    "hn_url": story.hn_url,
                    "item_type": story.item_type or "story",
                    "top_comments": comments,
                    "extraction_method": "hn_story_plus_comments",
                    "extraction_failed": False,
                },
            )
        )
    return items


async def fetch_hackernews_search(
    topic: str,
    time_window: str = "today",
    limit: int = 20,
    *,
    expanded: bool = False,
    queries: Optional[List[str]] = None,
    must_include_terms: Optional[List[str]] = None,
    must_exclude_terms: Optional[List[str]] = None,
) -> List[RawItem]:
    """Topic-first Hacker News recall via Algolia query search."""
    query_list = _normalize_queries(topic, expanded=expanded, queries=queries)
    if not query_list:
        return []

    max_items = max(1, int(limit))
    per_query = max(2, int(max_items / max(1, len(queries))) + 1)
    window_days = _parse_window_days(time_window)
    if window_days <= 1:
        hn_range = "last_24h"
    elif window_days <= 7:
        hn_range = "past_week"
    elif window_days <= 30:
        hn_range = "past_month"
    else:
        hn_range = "past_year"

    by_id: Dict[str, Tuple[Any, str]] = {}
    async with HackerNewsScraper() as scraper:
        for query in query_list:
            stories = await scraper.search(query=query, max_results=per_query, sort_by="relevance", time_range=hn_range)
            for story in list(stories or []):
                sid = str(getattr(story, "id", "") or "").strip()
                if not sid:
                    continue
                relevance_text = " ".join(
                    [
                        str(getattr(story, "title", "") or ""),
                        str(getattr(story, "text", "") or ""),
                    ]
                )
                if not _matches_topic_constraints(
                    text=relevance_text,
                    must_include_terms=must_include_terms,
                    must_exclude_terms=must_exclude_terms,
                ):
                    continue
                if sid not in by_id:
                    by_id[sid] = (story, query)
            if len(by_id) >= max_items * 2:
                break

    items: List[RawItem] = []
    for sid, (story, query) in list(by_id.items())[:max_items]:
        comments: List[str] = []
        try:
            payload = await _hn_fetch_item(sid)
            comments = await _hn_fetch_top_comments(payload, max_comments=2)
        except Exception:
            comments = []
        body = _safe_truncate(
            "\n\n".join(
                [
                    _coalesce_text(getattr(story, "text", ""), ""),
                    f"Points: {int(getattr(story, 'points', 0) or 0)} | Comments: {int(getattr(story, 'comment_count', 0) or 0)}",
                    "\n".join([f"Top comment: {line}" for line in comments]),
                ]
            ),
            max_len=9000,
        )
        items.append(
            RawItem(
                id=f"hn_search_{sid}",
                source="hackernews",
                title=getattr(story, "title", None) or "HN Story",
                url=_coalesce_text(getattr(story, "url", ""), getattr(story, "hn_url", "")),
                body=body,
                author=getattr(story, "author", None),
                published_at=getattr(story, "created_at", None),
                tier="A",
                metadata={
                    "points": int(getattr(story, "points", 0) or 0),
                    "comment_count": int(getattr(story, "comment_count", 0) or 0),
                    "hn_url": getattr(story, "hn_url", ""),
                    "item_type": getattr(story, "item_type", "story") or "story",
                    "top_comments": comments,
                    "retrieval_mode": "topic_search",
                    "source_query": query,
                    "window_days": window_days,
                    "search_endpoint": _HN_ALGOLIA_SEARCH,
                    "extraction_method": "hn_algolia_search",
                    "extraction_failed": False,
                },
            )
        )
    return items


def _rss_text(node: ET.Element, tag: str) -> str:
    try:
        child = node.find(tag)
        return str((child.text if child is not None else "") or "").strip()
    except Exception:
        return ""


def _find_creator(node: ET.Element) -> str:
    explicit = _rss_text(node, "author")
    if explicit:
        return explicit
    for child in list(node):
        tag = str(child.tag or "")
        if tag.endswith("creator"):
            return str(child.text or "").strip()
    return ""


async def fetch_rss_feed(feed_url: str, max_results: int = 20) -> List[RawItem]:
    """Fetch RSS entries (Tier B) and attempt full-article extraction for each entry."""
    xml_text = await _http_get_text(feed_url)
    root = ET.fromstring(xml_text)

    items: List[RawItem] = []
    for entry in root.findall(".//item")[: max(1, int(max_results))]:
        title = _rss_text(entry, "title") or "Untitled"
        link = _rss_text(entry, "link")
        description = _safe_truncate(_strip_html(_rss_text(entry, "description")), max_len=4000)
        author = _find_creator(entry)
        published = _parse_datetime(_rss_text(entry, "pubDate"))
        guid = _rss_text(entry, "guid") or link or title

        body = description
        method = "rss_description"
        extraction_error = None
        extraction_failed = False

        if link:
            try:
                html = await _http_get_text(link, headers={"User-Agent": "AcademicResearchAgent/2.0"})
                extracted, method, extraction_error = _extract_article_content(html, url=link)
                if extracted and len(extracted) > len(description):
                    body = extracted
            except Exception as exc:
                extraction_failed = True
                extraction_error = f"rss_article_fetch_error:{exc}"
                method = "fallback"

        items.append(
            RawItem(
                id=f"rss_{hashlib.sha1(guid.encode('utf-8')).hexdigest()[:12]}",
                source="rss",
                title=title,
                url=link,
                body=_safe_truncate(body, max_len=9000),
                author=author or None,
                published_at=published,
                tier="B",
                metadata={
                    "feed_url": feed_url,
                    "item_type": "rss_entry",
                    "extraction_method": method,
                    "extraction_error": extraction_error,
                    "extraction_failed": extraction_failed,
                },
            )
        )
    return items


async def fetch_web_article(url: str) -> List[RawItem]:
    """Fetch and extract a web article as Tier B fallback content."""
    html = await _http_get_text(url, headers={"User-Agent": "AcademicResearchAgent/2.0"})

    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    title = _strip_html(title_match.group(1)) if title_match else "Web Article"

    body, method, extraction_error = _extract_article_content(html, url=url)

    item = RawItem(
        id=f"web_{hashlib.sha1(url.encode('utf-8')).hexdigest()[:12]}",
        source="web_article",
        title=title or "Web Article",
        url=url,
        body=_safe_truncate(body, max_len=9000),
        author=None,
        published_at=None,
        tier="B",
        metadata={
            "item_type": "web_article",
            "extraction_method": method,
            "extraction_error": extraction_error,
            "extraction_failed": bool(extraction_error and method == "fallback"),
        },
    )
    return [item]
