"""Explainable scoring and ranking for canonical items."""

from __future__ import annotations

from datetime import datetime, timezone
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

from core import NormalizedItem, RankedItem


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _quality_metrics(item: NormalizedItem) -> Tuple[int, int, Optional[float], int]:
    metadata = dict(item.metadata or {})
    body_len = int(_to_float(metadata.get("body_len"), len(str(item.body_md or ""))))
    citation_count = int(_to_float(metadata.get("citation_count"), len(item.citations)))
    raw_recency = metadata.get("published_recency")
    published_recency: Optional[float]
    if raw_recency in (None, "", "unknown"):
        published_recency = None
    else:
        published_recency = _to_float(raw_recency)
    link_count = int(_to_float(metadata.get("link_count"), 0.0))
    return body_len, citation_count, published_recency, link_count


def _is_short_announcement(item: NormalizedItem) -> bool:
    text = f"{item.title}\n{item.body_md}".lower()
    markers = [
        "release",
        "changelog",
        "security advisory",
        "hotfix",
        "patch",
        "version",
        "v0.",
        "v1.",
        "v2.",
        "announce",
    ]
    return any(marker in text for marker in markers)


def score_novelty(item: NormalizedItem) -> float:
    """Novelty favors recency and release-like artifacts."""
    published = item.published_at
    if published is None:
        recency = 0.45
    else:
        dt = published
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        days_old = max(0.0, (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() / 86400.0)
        recency = max(0.0, 1.0 - days_old / 45.0)

    body = str(item.body_md or "")
    token_diversity = len(set(re.findall(r"[a-zA-Z0-9_]+", body.lower()))) / max(
        1.0, len(re.findall(r"[a-zA-Z0-9_]+", body.lower()))
    )
    release_bonus = 0.12 if str(item.metadata.get("item_type", "")).lower() == "release" else 0.0
    return _clamp01(0.62 * recency + 0.26 * token_diversity + release_bonus)


def score_talkability(item: NormalizedItem) -> float:
    """Talkability reflects community traction and shareability."""
    metadata = dict(item.metadata or {})
    stars = float(metadata.get("stars", 0) or 0)
    downloads = float(metadata.get("downloads", 0) or 0)
    likes = float(metadata.get("likes", 0) or 0)
    points = float(metadata.get("points", 0) or 0)
    comments = float(metadata.get("comment_count", metadata.get("comments", 0)) or 0)

    engagement = stars + 0.35 * downloads + 1.2 * likes + 1.4 * points + 1.6 * comments
    engagement_score = _clamp01(math.log10(1.0 + max(0.0, engagement)) / 4.0)

    title = str(item.title or "")
    punch = 0.12 if any(marker in title for marker in [":", "?", "vs", "VS", "对比"]) else 0.0
    return _clamp01(engagement_score + punch)


def score_credibility(item: NormalizedItem) -> float:
    """Credibility favors Tier A, citations, and explicit source quality labels."""
    base = 0.78 if item.tier == "A" else 0.34
    citation_bonus = min(0.22, 0.06 * len(item.citations))
    label = str(item.metadata.get("credibility", "") or "").lower().strip()

    if label == "high":
        label_adj = 0.12
    elif label == "medium":
        label_adj = 0.03
    elif label == "low":
        label_adj = -0.08
    else:
        label_adj = 0.0

    return _clamp01(base + citation_bonus + label_adj)


def score_visual_assets(item: NormalizedItem) -> float:
    """Visual asset score reflects render-friendly evidence density."""
    metadata = dict(item.metadata or {})
    flags = 0.0
    for key in ("has_image", "has_video", "has_diagram", "demo_url"):
        value = metadata.get(key)
        if value:
            flags += 0.2

    text = f"{item.title}\n{item.body_md}".lower()
    keyword_hits = sum(
        1
        for keyword in ("diagram", "chart", "benchmark", "architecture", "timeline", "screenshot", "demo")
        if keyword in text
    )
    text_bonus = min(0.45, 0.08 * keyword_hits)

    return _clamp01(0.25 + flags + text_bonus)


def _weighted_total(scores: Dict[str, float]) -> float:
    return _clamp01(
        0.25 * scores["novelty"]
        + 0.30 * scores["talkability"]
        + 0.30 * scores["credibility"]
        + 0.15 * scores["visual_assets"]
    )


def rank_items(
    items: Sequence[NormalizedItem],
    *,
    tier_b_top3_talkability_threshold: float = 0.88,
    min_body_len_for_top_picks: int = 300,
) -> List[RankedItem]:
    """Rank items with explainable subscores and Tier B top3 gate."""
    staged = []
    for item in items:
        body_len, citation_count, published_recency, link_count = _quality_metrics(item)
        short_announcement = _is_short_announcement(item)
        quality_eligible = bool(
            body_len >= int(min_body_len_for_top_picks)
            or (short_announcement and citation_count >= 1 and link_count >= 1)
        )

        scores = {
            "novelty": score_novelty(item),
            "talkability": score_talkability(item),
            "credibility": score_credibility(item),
            "visual_assets": score_visual_assets(item),
        }
        total = _weighted_total(scores)

        if citation_count <= 0:
            total = _clamp01(total * 0.76)
        if published_recency is None:
            total = _clamp01(total * 0.82)
        if not quality_eligible:
            total = _clamp01(total * 0.72)

        # Tier B default exclusion from top3 can be overridden by exceptional talkability.
        if item.tier == "B" and scores["talkability"] >= float(tier_b_top3_talkability_threshold):
            margin = (scores["talkability"] - float(tier_b_top3_talkability_threshold)) / max(
                1e-6, 1.0 - float(tier_b_top3_talkability_threshold)
            )
            total = _clamp01(total + 0.08 + 0.12 * _clamp01(margin))
        staged.append(
            (
                item,
                scores,
                total,
                body_len,
                citation_count,
                published_recency,
                link_count,
                quality_eligible,
                short_announcement,
            )
        )

    staged.sort(key=lambda row: row[2], reverse=True)

    # Top picks must pass body/evidence gate unless short announcement with evidence.
    top: List[tuple] = []
    deferred_tier_b: List[tuple] = []
    deferred_quality_gate: List[tuple] = []
    for row in staged:
        item, scores, _total, _body_len, _citation_count, _published_recency, _link_count, quality_eligible, _short_announcement = row
        if not quality_eligible:
            deferred_quality_gate.append(row)
            continue
        if len(top) < 3 and item.tier == "B" and scores["talkability"] < float(tier_b_top3_talkability_threshold):
            deferred_tier_b.append(row)
            continue
        top.append(row)

    ordered = top + deferred_tier_b + deferred_quality_gate
    ranked: List[RankedItem] = []
    for idx, (
        item,
        scores,
        total,
        body_len,
        citation_count,
        published_recency,
        link_count,
        quality_eligible,
        short_announcement,
    ) in enumerate(ordered, start=1):
        reasons = [
            f"novelty={scores['novelty']:.2f}",
            f"talkability={scores['talkability']:.2f}",
            f"credibility={scores['credibility']:.2f}",
            f"visual_assets={scores['visual_assets']:.2f}",
            f"tier={item.tier}",
            f"quality.body_len={body_len}",
            f"quality.citation_count={citation_count}",
            f"quality.published_recency_days={published_recency if published_recency is not None else 'unknown'}",
            f"quality.link_count={link_count}",
            f"quality.top_pick_gate={'pass' if quality_eligible else 'deferred'}",
        ]
        if short_announcement:
            reasons.append("quality.short_announcement=true")
        if citation_count <= 0:
            reasons.append("penalty.no_citations")
        if published_recency is None:
            reasons.append("penalty.missing_published_time")
        if not quality_eligible:
            reasons.append(f"penalty.body_len_lt_{int(min_body_len_for_top_picks)}")
        if item.tier == "B" and idx <= 3 and scores["talkability"] >= float(tier_b_top3_talkability_threshold):
            reasons.append("tier_b_top3_override=talkability")

        ranked.append(
            RankedItem(
                rank=idx,
                item=item,
                total_score=total,
                novelty_score=scores["novelty"],
                talkability_score=scores["talkability"],
                credibility_score=scores["credibility"],
                visual_assets_score=scores["visual_assets"],
                reasons=reasons,
            )
        )

    return ranked
