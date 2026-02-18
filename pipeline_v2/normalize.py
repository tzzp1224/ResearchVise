"""Normalization stage for v2 raw connector outputs."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import re
from typing import Any, Dict, List, Optional

from core import Citation, NormalizedItem, RawItem
from pipeline_v2.sanitize import is_allowed_citation_url, sanitize_markdown


_TIER_A_SOURCES = {
    "github",
    "huggingface",
    "hackernews",
    "arxiv",
    "semantic_scholar",
    "openreview",
    "arxiv_rss",
}


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
        token = str(link).strip().rstrip(".,;:")
        if token:
            links.add(token)
    if str(url or "").strip():
        links.add(str(url).strip())
    return len(links)


def _published_recency_days(published: Optional[datetime]) -> Optional[float]:
    if not published:
        return None
    dt = published if published.tzinfo else published.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() / 86400.0)
    return round(age_days, 2)


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
            url = str(entry.get("url") or "").strip()
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
            url = entry.strip()
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
        if not is_allowed_citation_url(str(link).strip()):
            continue
        citations.append(
            Citation(
                title=_compact_text(title, max_len=140),
                url=str(link).strip(),
                snippet=_compact_text(clean_text, max_len=220),
                source=raw.source,
            )
        )

    plain_urls = re.findall(r"https?://[^\s)\]>]+", body)
    for link in plain_urls[:6]:
        if not is_allowed_citation_url(str(link).strip()):
            continue
        citations.append(
            Citation(
                title=_compact_text(raw.title, max_len=140),
                url=str(link).strip(),
                snippet=_compact_text(clean_text, max_len=220),
                source=raw.source,
            )
        )

    if raw.url and is_allowed_citation_url(str(raw.url).strip()):
        citations.append(
            Citation(
                title=_compact_text(raw.title, max_len=140),
                url=str(raw.url).strip(),
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
    metadata["clean_text"] = clean_text
    metadata["sanitize"] = sanitize_stats

    return NormalizedItem(
        id=str(item.id).strip(),
        source=str(item.source).strip(),
        title=_compact_text(item.title, max_len=220) or "Untitled",
        url=str(item.url or "").strip(),
        author=str(item.author).strip() if item.author else None,
        published_at=published,
        body_md=clean_md or body_text,
        citations=citations,
        tier=tier,  # type: ignore[arg-type]
        lang=str(metadata.get("lang") or "en").strip() or "en",
        hash=digest,
        metadata=metadata,
    )
