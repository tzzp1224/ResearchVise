"""Normalization stage for v2 raw connector outputs."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from core import Citation, NormalizedItem, RawItem
from pipeline_v2.sanitize import is_allowed_citation_url, is_valid_http_url, normalize_url, sanitize_markdown


_TIER_A_SOURCES = {
    "github",
    "huggingface",
    "hackernews",
    "arxiv",
    "semantic_scholar",
    "openreview",
    "arxiv_rss",
}
_SG_TZ = ZoneInfo("Asia/Singapore")
_QUICKSTART_MARKERS = ("quickstart", "getting started", "install", "installation", "usage", "run ")
_BENCH_MARKERS = ("benchmark", "result", "eval", "evaluation", "metric", "latency", "throughput", "accuracy")
_IMAGE_URL_RE = re.compile(r"(https?://[^\s)\"'>]+?\.(?:png|jpg|jpeg|gif|webp|svg))", re.IGNORECASE)


def _parse_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
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


def _compact_text(value: Any, max_len: int = 300) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= max_len else text[: max_len - 3].rstrip() + "..."


def _count_links(body: str, url: str) -> int:
    links = set()
    for link in re.findall(r"https?://[^\s)\]>]+", str(body or "")):
        token = normalize_url(link)
        if token and is_valid_http_url(token):
            links.add(token)
    normalized_url = normalize_url(str(url or "").strip())
    if normalized_url and is_valid_http_url(normalized_url):
        links.add(normalized_url)
    return len(links)


def _published_recency_days(published: Optional[datetime]) -> Optional[float]:
    if not published:
        return None
    dt = published if published.tzinfo else published.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() / 86400.0)
    return round(age_days, 2)


def _publish_or_update_time(item: RawItem, metadata: Dict[str, Any]) -> Optional[datetime]:
    candidates = [
        metadata.get("publish_or_update_time"),
        metadata.get("last_push"),
        metadata.get("updated_at"),
        metadata.get("pushed_at"),
        metadata.get("last_modified"),
        metadata.get("release_published_at"),
        metadata.get("published_at"),
        item.published_at,
    ]
    for candidate in candidates:
        parsed = _parse_datetime(candidate)
        if parsed is not None:
            return parsed
    return None


def _update_recency_days(dt: Optional[datetime]) -> Optional[float]:
    if dt is None:
        return None
    target = dt.astimezone(_SG_TZ) if dt.tzinfo else dt.replace(tzinfo=timezone.utc).astimezone(_SG_TZ)
    now = datetime.now(_SG_TZ)
    delta = max(0.0, (now - target).total_seconds() / 86400.0)
    return round(delta, 2)


def _count_non_badge_images(raw_text: str) -> int:
    count = 0
    for url in _IMAGE_URL_RE.findall(str(raw_text or "")):
        lowered = str(url).lower()
        if "img.shields.io" in lowered or "badge" in lowered:
            continue
        count += 1
    return count


def _is_high_quality_link(url: str) -> bool:
    value = normalize_url(str(url or "")).lower()
    if not is_valid_http_url(value):
        return False
    host = str(urlparse(value).netloc or "").strip().lower()
    path = str(urlparse(value).path or "").strip().lower()
    if not host:
        return False
    if host in {"img.shields.io", "buymeacoffee.com", "github.com"} and "/sponsors" in value:
        return False
    if host in {"github.com", "www.github.com"}:
        return "/issues/" in path or "/releases/" in path or path.count("/") >= 2
    if host.endswith("huggingface.co") or host.endswith("arxiv.org") or host.endswith("openreview.net"):
        return True
    if host in {"news.ycombinator.com"} and "item?id=" in value:
        return True
    if "docs" in host or "/docs" in path or "paper" in path or "demo" in path:
        return True
    return False


def _to_raw_item(raw: Any) -> RawItem:
    if isinstance(raw, RawItem):
        return raw
    if isinstance(raw, dict):
        payload = dict(raw)
        if not payload.get("id"):
            seed = f"{payload.get('source', '')}|{payload.get('title', '')}|{payload.get('url', '')}"
            payload["id"] = f"raw_{hashlib.sha1(seed.encode('utf-8')).hexdigest()[:12]}"
        return RawItem(**payload)
    raise TypeError("raw must be RawItem or dict")


def _infer_tier(source: str, explicit: Optional[str]) -> str:
    if explicit in {"A", "B"}:
        return explicit
    return "A" if source in _TIER_A_SOURCES else "B"


def extract_citations(item: Any) -> List[Citation]:
    """Extract citations from metadata/body/url with deterministic dedup."""
    raw = _to_raw_item(item)
    metadata = dict(raw.metadata or {})
    body = str(raw.body or "")
    _clean_md, clean_text, _sanitize_stats = sanitize_markdown(body)

    citations: List[Citation] = []

    for entry in list(metadata.get("citations") or []):
        if isinstance(entry, dict):
            url = normalize_url(str(entry.get("url") or "").strip())
            if url and not is_allowed_citation_url(url):
                continue
            citations.append(
                Citation(
                    title=_compact_text(entry.get("title") or raw.title, max_len=140),
                    url=url,
                    snippet=_compact_text(entry.get("snippet") or "", max_len=220),
                    source=str(entry.get("source") or raw.source),
                )
            )
        elif isinstance(entry, str) and entry.strip().startswith("http"):
            url = normalize_url(entry.strip())
            if not is_allowed_citation_url(url):
                continue
            citations.append(
                Citation(
                    title=_compact_text(raw.title, max_len=140),
                    url=url,
                    snippet="",
                    source=raw.source,
                )
            )

    md_links = re.findall(r"\[([^\]]+)\]\((https?://[^)]+)\)", body)
    for title, link in md_links:
        normalized_link = normalize_url(str(link).strip())
        if not is_allowed_citation_url(normalized_link):
            continue
        citations.append(
            Citation(
                title=_compact_text(title, max_len=140),
                url=normalized_link,
                snippet=_compact_text(clean_text, max_len=220),
                source=raw.source,
            )
        )

    plain_urls = re.findall(r"https?://[^\s)\]>]+", body)
    for link in plain_urls[:6]:
        normalized_link = normalize_url(str(link).strip())
        if not is_allowed_citation_url(normalized_link):
            continue
        citations.append(
            Citation(
                title=_compact_text(raw.title, max_len=140),
                url=normalized_link,
                snippet=_compact_text(clean_text, max_len=220),
                source=raw.source,
            )
        )

    normalized_raw_url = normalize_url(str(raw.url or "").strip())
    if normalized_raw_url and is_allowed_citation_url(normalized_raw_url):
        citations.append(
            Citation(
                title=_compact_text(raw.title, max_len=140),
                url=normalized_raw_url,
                snippet=_compact_text(clean_text, max_len=220),
                source=raw.source,
            )
        )

    deduped: List[Citation] = []
    seen = set()
    for citation in citations:
        url = str(citation.url or "").strip()
        key = url or f"{citation.title}|{citation.snippet}"
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(citation)
    return deduped


def content_hash(item: Any) -> str:
    """Stable content hash for exact deduplication."""
    raw = _to_raw_item(item)
    _clean_md, clean_text, _sanitize_stats = sanitize_markdown(str(raw.body or ""))
    payload = "|".join(
        [
            str(raw.source or "").strip().lower(),
            str(raw.title or "").strip().lower(),
            str(raw.url or "").strip().lower(),
            re.sub(r"\s+", " ", str(clean_text or "").strip().lower()),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _infer_credibility(
    *,
    tier: str,
    source: str,
    author: Optional[str],
    published_at: Optional[datetime],
    citations: List[Citation],
    body_len: int,
    link_count: int,
) -> str:
    citation_count = len(list(citations or []))
    source_base = 0.62 if tier == "A" else 0.36
    if source in {"github", "huggingface"}:
        source_base += 0.07
    if source in {"hackernews"}:
        source_base += 0.03
    if source in {"rss", "web_article"}:
        source_base -= 0.08

    score = source_base
    score += min(0.22, 0.05 * float(citation_count))
    score += 0.05 if author else 0.0
    score += 0.05 if published_at else 0.0
    score += 0.05 if int(body_len) >= 500 else 0.0
    score += 0.04 if int(link_count) >= 2 else 0.0

    density = float(citation_count) / max(1.0, float(body_len) / 600.0)
    score += 0.05 if density >= 1.2 else (0.02 if density >= 0.5 else -0.03)

    if score >= 0.78:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def normalize(raw: Any) -> NormalizedItem:
    """Normalize connector output into canonical NormalizedItem."""
    item = _to_raw_item(raw)
    tier = _infer_tier(item.source, item.tier)
    citations = extract_citations(item)
    digest = content_hash(item)
    published = _parse_datetime(item.published_at)
    clean_md, clean_text, sanitize_stats = sanitize_markdown(str(item.body or ""))

    metadata: Dict[str, Any] = dict(item.metadata or {})
    body_text = str(clean_text or "").strip()
    body_len = len(re.sub(r"\s+", " ", body_text))
    link_count = max(_count_links(body_text, str(item.url or "")), len(citations))
    published_recency = _published_recency_days(published)
    publish_or_update = _publish_or_update_time(item, metadata)
    update_recency = _update_recency_days(publish_or_update)
    raw_len = int(float((sanitize_stats or {}).get("raw_len", 0) or 0))
    clean_len = int(float((sanitize_stats or {}).get("clean_len", len(body_text)) or 0))
    content_density = round(float(clean_len) / float(max(1, raw_len)), 4)
    lowered_body = body_text.lower()
    quickstart = any(marker in lowered_body for marker in _QUICKSTART_MARKERS)
    has_bench = any(marker in lowered_body for marker in _BENCH_MARKERS)
    non_badge_images = _count_non_badge_images(str(item.body or ""))
    evidence_quality_links = len(
        {
            str(citation.url or "").strip()
            for citation in list(citations or [])
            if _is_high_quality_link(str(citation.url or "").strip())
        }
    )

    metadata["credibility"] = _infer_credibility(
        tier=tier,
        source=item.source,
        author=item.author,
        published_at=published,
        citations=citations,
        body_len=body_len,
        link_count=link_count,
    )
    metadata["source_channel"] = "tier_a" if tier == "A" else "tier_b"
    metadata["citation_count"] = len(citations)
    metadata["body_len"] = body_len
    metadata["link_count"] = link_count
    metadata["published_recency"] = published_recency
    metadata["quality_metrics"] = {
        "body_len": body_len,
        "citation_count": len(citations),
        "published_recency": published_recency,
        "link_count": link_count,
    }
    metadata["publish_or_update_time"] = publish_or_update.isoformat() if publish_or_update else None
    metadata["update_recency_days"] = update_recency
    metadata["quality_signals"] = {
        "content_density": content_density,
        "has_quickstart": bool(quickstart),
        "has_results_or_bench": bool(has_bench),
        "has_images_non_badge": bool(non_badge_images > 0),
        "images_non_badge_count": int(non_badge_images),
        "publish_or_update_time": metadata["publish_or_update_time"],
        "update_recency_days": update_recency,
        "evidence_links_quality": int(evidence_quality_links),
    }
    metadata["clean_text"] = clean_text
    metadata["sanitize"] = sanitize_stats

    return NormalizedItem(
        id=str(item.id).strip(),
        source=str(item.source).strip(),
        title=_compact_text(item.title, max_len=220) or "Untitled",
        url=normalize_url(str(item.url or "").strip()),
        author=str(item.author).strip() if item.author else None,
        published_at=published,
        body_md=clean_md or body_text,
        citations=citations,
        tier=tier,  # type: ignore[arg-type]
        lang=str(metadata.get("lang") or "en").strip() or "en",
        hash=digest,
        metadata=metadata,
    )
