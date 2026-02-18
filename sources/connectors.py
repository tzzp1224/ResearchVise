"""Tiered source connectors for v2 ingestion."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

import httpx

from core import RawItem
from scrapers.hackernews_scraper import HackerNewsScraper
from scrapers.huggingface_scraper import HuggingFaceScraper
from scrapers.social.github_scraper import GitHubScraper


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
        return None


def _coalesce_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _strip_html(value: str) -> str:
    text = str(value or "")
    text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


async def _http_get_json(url: str, *, headers: Optional[Dict[str, str]] = None) -> Any:
    timeout = httpx.Timeout(12.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()


async def _http_get_text(url: str, *, headers: Optional[Dict[str, str]] = None) -> str:
    timeout = httpx.Timeout(12.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return str(response.text or "")


async def fetch_github_trending(
    max_results: int = 20,
    *,
    language: Optional[str] = None,
    since: str = "weekly",
) -> List[RawItem]:
    """Fetch GitHub trending repositories (Tier A)."""
    async with GitHubScraper() as scraper:
        repos = await scraper.get_trending(language=language, since=since)

    items: List[RawItem] = []
    for repo in repos[: max(1, int(max_results))]:
        body = _coalesce_text(
            repo.description,
            f"Stars: {repo.stars} | Forks: {repo.forks} | Language: {repo.language or 'unknown'}",
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
                    "language": repo.language,
                    "topics": list(repo.topics or []),
                    "item_type": "repo",
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

    headers: Dict[str, str] = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

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
            items.append(
                RawItem(
                    id=f"github_release_{repo}_{release.get('id') or release.get('tag_name') or rel_name}",
                    source="github",
                    title=f"{repo} {rel_name}",
                    url=rel_url,
                    body=_coalesce_text(release.get("body"), release.get("name"), release.get("tag_name")),
                    author=_coalesce_text((release.get("author") or {}).get("login"), repo.split("/", 1)[0]),
                    published_at=_parse_datetime(release.get("published_at") or release.get("created_at")),
                    tier="A",
                    metadata={
                        "repo": repo,
                        "tag_name": release.get("tag_name"),
                        "prerelease": bool(release.get("prerelease", False)),
                        "item_type": "release",
                    },
                )
            )
    return items


async def fetch_huggingface_trending(max_results: int = 20) -> List[RawItem]:
    """Fetch Hugging Face trending models (Tier A)."""
    async with HuggingFaceScraper() as scraper:
        models = await scraper.search_models(query="", max_results=max(1, int(max_results)))

    items: List[RawItem] = []
    for model in models[: max(1, int(max_results))]:
        items.append(
            RawItem(
                id=f"hf_model_{model.id}",
                source="huggingface",
                title=model.name or model.id,
                url=model.url,
                body=_coalesce_text(model.description, f"Downloads: {model.downloads} | Likes: {model.likes}"),
                author=model.author,
                published_at=model.updated_at or model.created_at,
                tier="A",
                metadata={
                    "repo_id": model.id,
                    "downloads": int(model.downloads or 0),
                    "likes": int(model.likes or 0),
                    "tags": list(model.tags or []),
                    "item_type": "model",
                },
            )
        )
    return items


async def fetch_hackernews_top(max_results: int = 20) -> List[RawItem]:
    """Fetch Hacker News top stories (Tier A community-technical signal)."""
    async with HackerNewsScraper() as scraper:
        stories = await scraper.get_front_page(max_results=max(1, int(max_results)))

    items: List[RawItem] = []
    for story in stories[: max(1, int(max_results))]:
        items.append(
            RawItem(
                id=f"hn_{story.id}",
                source="hackernews",
                title=story.title,
                url=_coalesce_text(story.url, story.hn_url),
                body=_coalesce_text(story.text, story.title),
                author=story.author,
                published_at=story.created_at,
                tier="A",
                metadata={
                    "points": int(story.points or 0),
                    "comment_count": int(story.comment_count or 0),
                    "hn_url": story.hn_url,
                    "item_type": "story",
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
    """Fetch RSS entries (Tier B fallback channel)."""
    xml_text = await _http_get_text(feed_url)
    root = ET.fromstring(xml_text)

    items: List[RawItem] = []
    for entry in root.findall(".//item")[: max(1, int(max_results))]:
        title = _rss_text(entry, "title") or "Untitled"
        link = _rss_text(entry, "link")
        description = _strip_html(_rss_text(entry, "description"))
        author = _find_creator(entry)
        published = _parse_datetime(_rss_text(entry, "pubDate"))
        guid = _rss_text(entry, "guid") or link or title

        items.append(
            RawItem(
                id=f"rss_{hashlib.sha1(guid.encode('utf-8')).hexdigest()[:12]}",
                source="rss",
                title=title,
                url=link,
                body=description,
                author=author or None,
                published_at=published,
                tier="B",
                metadata={
                    "feed_url": feed_url,
                    "item_type": "rss_entry",
                },
            )
        )
    return items


async def fetch_web_article(url: str) -> List[RawItem]:
    """Fetch and extract a web article as Tier B fallback content."""
    html = await _http_get_text(url)

    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    title = _strip_html(title_match.group(1)) if title_match else "Web Article"

    body_match = re.search(r"<article[^>]*>(.*?)</article>", html, flags=re.IGNORECASE | re.DOTALL)
    candidate = body_match.group(1) if body_match else html
    body = _strip_html(candidate)
    body = body[:6000]

    item = RawItem(
        id=f"web_{hashlib.sha1(url.encode('utf-8')).hexdigest()[:12]}",
        source="web_article",
        title=title or "Web Article",
        url=url,
        body=body,
        author=None,
        published_at=None,
        tier="B",
        metadata={
            "item_type": "web_article",
            "extracted": True,
        },
    )
    return [item]
