"""Deduplication and clustering for normalized items."""

from __future__ import annotations

import hashlib
import math
import re
from typing import List, Sequence

from core import CanonicalItem, Citation, NormalizedItem


def dedup_exact(items: Sequence[NormalizedItem]) -> List[NormalizedItem]:
    """Remove exact duplicates by content hash."""
    unique: List[NormalizedItem] = []
    seen = set()
    for item in items:
        key = str(item.hash or "").strip() or str(item.id)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+(?:\.[0-9]+)?|[\u4e00-\u9fff]+", str(text or "").lower())
    return [token for token in tokens if len(token) > 1]


def embed(items: Sequence[NormalizedItem], dimensions: int = 32) -> List[List[float]]:
    """Deterministic local embedding via hashed token bag-of-words."""
    dims = max(8, int(dimensions))
    vectors: List[List[float]] = []

    for item in items:
        vec = [0.0] * dims
        text = f"{item.title}\n{item.body_md}\n{item.source}"
        for token in _tokenize(text):
            digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % dims
            vec[idx] += 1.0

        norm = math.sqrt(sum(value * value for value in vec))
        if norm > 0:
            vec = [value / norm for value in vec]
        vectors.append(vec)

    return vectors


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        return 0.0
    return float(sum(x * y for x, y in zip(a, b)))


def _centroid(vectors: Sequence[Sequence[float]]) -> List[float]:
    if not vectors:
        return []
    dims = len(vectors[0])
    merged = [0.0] * dims
    for vec in vectors:
        for i, value in enumerate(vec):
            merged[i] += float(value)
    count = float(len(vectors))
    merged = [value / count for value in merged]
    norm = math.sqrt(sum(value * value for value in merged))
    if norm > 0:
        merged = [value / norm for value in merged]
    return merged


def cluster(
    items: Sequence[NormalizedItem],
    embeddings: Sequence[Sequence[float]],
    similarity_threshold: float = 0.86,
) -> List[List[NormalizedItem]]:
    """Greedy cosine clustering over precomputed embeddings."""
    if len(items) != len(embeddings):
        raise ValueError("items and embeddings length must match")

    threshold = max(0.0, min(1.0, float(similarity_threshold)))
    grouped_items: List[List[NormalizedItem]] = []
    grouped_vectors: List[List[List[float]]] = []

    for item, vector in zip(items, embeddings):
        best_idx = -1
        best_score = -1.0
        for idx, vectors in enumerate(grouped_vectors):
            center = _centroid(vectors)
            score = _cosine(center, vector)
            if score > best_score:
                best_idx = idx
                best_score = score

        if best_idx >= 0 and best_score >= threshold:
            grouped_items[best_idx].append(item)
            grouped_vectors[best_idx].append(list(vector))
        else:
            grouped_items.append([item])
            grouped_vectors.append([list(vector)])

    return grouped_items


def _citation_key(citation: Citation) -> str:
    url = str(citation.url or "").strip()
    return url or f"{citation.title}|{citation.snippet}"


def _anchor_score(item: NormalizedItem) -> float:
    tier_score = 1.0 if item.tier == "A" else 0.0
    citation_score = min(1.0, len(item.citations) / 4.0)
    text_score = min(1.0, len(item.body_md or "") / 4000.0)
    return 0.6 * tier_score + 0.25 * citation_score + 0.15 * text_score


def merge_cluster(cluster_items: Sequence[NormalizedItem]) -> CanonicalItem:
    """Merge a cluster into one canonical representative item."""
    if not cluster_items:
        raise ValueError("cluster_items must not be empty")

    ordered = sorted(cluster_items, key=_anchor_score, reverse=True)
    anchor = ordered[0]

    merged_citations: List[Citation] = []
    seen = set()
    for item in ordered:
        for citation in item.citations:
            key = _citation_key(citation)
            if not key or key in seen:
                continue
            seen.add(key)
            merged_citations.append(citation)

    longest_body = max([str(item.body_md or "") for item in ordered], key=len)
    sources = sorted({str(item.source).strip() for item in ordered if str(item.source).strip()})

    return CanonicalItem(
        id=anchor.id,
        source=anchor.source,
        title=anchor.title,
        url=anchor.url,
        author=anchor.author,
        published_at=anchor.published_at,
        body_md=longest_body,
        citations=merged_citations,
        tier=anchor.tier,
        lang=anchor.lang,
        hash=anchor.hash,
        metadata=dict(anchor.metadata),
        cluster_size=len(ordered),
        merged_ids=[item.id for item in ordered],
        alias_titles=[item.title for item in ordered[1:]],
        source_set=sources,
    )
