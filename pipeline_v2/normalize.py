"""Normalization stage for v2 raw connector outputs."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import re
from typing import Any, Dict, List, Optional

from core import Citation, NormalizedItem, RawItem


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

    citations: List[Citation] = []

    for entry in list(metadata.get("citations") or []):
        if isinstance(entry, dict):
            citations.append(
                Citation(
                    title=_compact_text(entry.get("title") or raw.title, max_len=140),
                    url=str(entry.get("url") or "").strip(),
                    snippet=_compact_text(entry.get("snippet") or "", max_len=220),
                    source=str(entry.get("source") or raw.source),
                )
            )
        elif isinstance(entry, str) and entry.strip().startswith("http"):
            citations.append(
                Citation(
                    title=_compact_text(raw.title, max_len=140),
                    url=entry.strip(),
                    snippet="",
                    source=raw.source,
                )
            )

    md_links = re.findall(r"\[([^\]]+)\]\((https?://[^)]+)\)", body)
    for title, link in md_links:
        citations.append(
            Citation(
                title=_compact_text(title, max_len=140),
                url=str(link).strip(),
                snippet=_compact_text(body, max_len=220),
                source=raw.source,
            )
        )

    plain_urls = re.findall(r"https?://[^\s)\]>]+", body)
    for link in plain_urls[:6]:
        citations.append(
            Citation(
                title=_compact_text(raw.title, max_len=140),
                url=str(link).strip(),
                snippet=_compact_text(body, max_len=220),
                source=raw.source,
            )
        )

    if raw.url:
        citations.append(
            Citation(
                title=_compact_text(raw.title, max_len=140),
                url=str(raw.url).strip(),
                snippet=_compact_text(raw.body, max_len=220),
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
    payload = "|".join(
        [
            str(raw.source or "").strip().lower(),
            str(raw.title or "").strip().lower(),
            str(raw.url or "").strip().lower(),
            re.sub(r"\s+", " ", str(raw.body or "").strip().lower()),
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
) -> str:
    if tier == "A":
        return "high"
    # Tier B is strictly medium/low per PRD.
    if author and published_at and len(citations) >= 2:
        return "medium"
    if source in {"rss", "web_article"} and len(citations) >= 2:
        return "medium"
    return "low"


def normalize(raw: Any) -> NormalizedItem:
    """Normalize connector output into canonical NormalizedItem."""
    item = _to_raw_item(raw)
    tier = _infer_tier(item.source, item.tier)
    citations = extract_citations(item)
    digest = content_hash(item)
    published = _parse_datetime(item.published_at)

    metadata: Dict[str, Any] = dict(item.metadata or {})
    metadata["credibility"] = _infer_credibility(
        tier=tier,
        source=item.source,
        author=item.author,
        published_at=published,
        citations=citations,
    )
    metadata["source_channel"] = "tier_a" if tier == "A" else "tier_b"
    metadata["citation_count"] = len(citations)

    return NormalizedItem(
        id=str(item.id).strip(),
        source=str(item.source).strip(),
        title=_compact_text(item.title, max_len=220) or "Untitled",
        url=str(item.url or "").strip(),
        author=str(item.author).strip() if item.author else None,
        published_at=published,
        body_md=str(item.body or "").strip(),
        citations=citations,
        tier=tier,  # type: ignore[arg-type]
        lang=str(metadata.get("lang") or "en").strip() or "en",
        hash=digest,
        metadata=metadata,
    )
