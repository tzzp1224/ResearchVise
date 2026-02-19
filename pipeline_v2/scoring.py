"""Explainable scoring and ranking for canonical items."""

from __future__ import annotations

from datetime import datetime, timezone
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

from core import NormalizedItem, RankedItem
from pipeline_v2.topic_profile import TopicProfile


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


def _quality_signals(item: NormalizedItem) -> Dict[str, object]:
    metadata = dict(item.metadata or {})
    payload = dict(metadata.get("quality_signals") or {})
    return {
        "content_density": _to_float(payload.get("content_density"), 0.0),
        "has_quickstart": bool(payload.get("has_quickstart")),
        "has_results_or_bench": bool(payload.get("has_results_or_bench")),
        "has_images_non_badge": bool(payload.get("has_images_non_badge")),
        "evidence_links_quality": int(_to_float(payload.get("evidence_links_quality"), 0)),
        "update_recency_days": (
            None
            if payload.get("update_recency_days") in (None, "", "unknown")
            else _to_float(payload.get("update_recency_days"))
        ),
        "publish_or_update_time": payload.get("publish_or_update_time"),
    }


def _quality_signal_boost(item: NormalizedItem) -> float:
    signals = _quality_signals(item)
    boost = 0.0

    density = float(signals["content_density"])
    if density >= 0.2:
        boost += 0.05
    elif density >= 0.12:
        boost += 0.02
    else:
        boost -= 0.03

    if bool(signals["has_quickstart"]):
        boost += 0.03
    if bool(signals["has_results_or_bench"]):
        boost += 0.03
    if bool(signals["has_images_non_badge"]):
        boost += 0.02

    evidence_links = int(signals["evidence_links_quality"])
    if evidence_links >= 3:
        boost += 0.06
    elif evidence_links >= 1:
        boost += 0.03
    else:
        boost -= 0.04

    recency = signals["update_recency_days"]
    if recency is None:
        boost -= 0.02
    else:
        recency_value = float(recency)
        if recency_value <= 7:
            boost += 0.06
        elif recency_value <= 30:
            boost += 0.03
        elif recency_value >= 120:
            boost -= 0.05

    return max(-0.15, min(0.2, boost))


_TOPIC_STOP_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "for",
    "to",
    "of",
    "in",
    "on",
    "with",
    "today",
    "latest",
    "news",
}

_AGENT_GENERIC_TERMS = {"agent", "agents", "assistant", "copilot"}
_AGENT_HIGH_VALUE_TERMS = {
    "mcp",
    "model context protocol",
    "tool calling",
    "function calling",
    "orchestration",
    "langgraph",
    "autogen",
    "crewai",
    "agent eval",
    "agent benchmark",
    "agent runtime",
    "inspect",
}


def _topic_tokens(topic: Optional[str]) -> List[str]:
    parts = re.findall(r"[a-z0-9]+", str(topic or "").lower())
    tokens = [token for token in parts if token not in _TOPIC_STOP_WORDS and len(token) >= 2]
    deduped: List[str] = []
    seen = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


def _has_token(text: str, token: str) -> bool:
    return re.search(rf"\b{re.escape(token)}\b", text) is not None


def _contains_term(text: str, term: str) -> bool:
    payload = str(text or "").lower()
    token = str(term or "").strip().lower()
    if not payload or not token:
        return False
    pattern = r"\b" + re.escape(token).replace(r"\ ", r"\s+") + r"\b"
    return re.search(pattern, payload) is not None


def _contains_negated_term(text: str, term: str) -> bool:
    payload = str(text or "").lower()
    token = str(term or "").strip().lower()
    if not payload or not token:
        return False
    term_pattern = re.escape(token).replace(r"\ ", r"\s+")
    pattern = r"\b(?:no|not|without|lack|lacks|lacking)\s+(?:[a-z0-9_-]+\s+){0,2}" + term_pattern + r"\b"
    return re.search(pattern, payload) is not None


def _topic_text(item: NormalizedItem) -> str:
    metadata = dict(item.metadata or {})
    return " ".join(
        [
            str(item.title or ""),
            str(metadata.get("clean_text") or item.body_md or ""),
            " ".join(str(tag) for tag in list(metadata.get("topics") or [])),
            str(metadata.get("item_type") or ""),
        ]
    ).lower()


def evaluate_relevance(
    item: NormalizedItem,
    topic: Optional[str],
    *,
    topic_profile: Optional[TopicProfile] = None,
) -> Dict[str, object]:
    """Deterministic relevance with topic profile hard gate + soft scoring."""
    tokens = _topic_tokens(topic)
    if not tokens:
        return {
            "score": 1.0,
            "hard_pass": True,
            "hard_terms": [],
            "boost_terms": [],
            "penalty_terms": [],
        }

    profile = topic_profile or TopicProfile.for_topic(str(topic or ""))
    topic_text = _topic_text(item)
    hard_terms = profile.matched_hard_terms(topic_text)
    hard_pass = profile.hard_match_pass(topic_text)
    if not hard_pass:
        return {
            "score": 0.0,
            "hard_pass": False,
            "hard_terms": hard_terms,
            "boost_terms": [],
            "penalty_terms": [],
        }

    matched = sum(1 for token in tokens if _has_token(topic_text, token))
    coverage = float(matched) / float(max(1, len(tokens)))

    phrase = " ".join(tokens)
    phrase_bonus = 0.2 if phrase and phrase in topic_text else 0.0

    # Token proxies for typical "AI agent" semantics while staying deterministic.
    proxy_bonus = 0.0
    if "agent" in tokens and any(
        hint in topic_text for hint in ("tool calling", "mcp", "workflow", "orchestrat", "assistant", "autonomous")
    ):
        proxy_bonus += 0.12
    if "ai" in tokens and any(hint in topic_text for hint in ("llm", "model", "inference", "reasoning")):
        proxy_bonus += 0.08

    density = min(1.0, topic_text.count(tokens[0]) / 4.0) if tokens else 0.0
    base = 0.1 if coverage > 0 else 0.0
    boost, penalty, boost_terms, penalty_terms = profile.soft_adjustment(topic_text)
    score = _clamp01(base + 0.65 * coverage + 0.15 * density + phrase_bonus + proxy_bonus + boost - penalty)

    agent_non_generic_hits: List[str] = []
    agent_high_value_hits: List[str] = []
    agent_score_cap = ""
    if str(profile.key or "").strip().lower() == "ai_agent":
        for term in hard_terms:
            lowered = str(term).strip().lower()
            if lowered and lowered not in _AGENT_GENERIC_TERMS and not _contains_negated_term(topic_text, lowered):
                agent_non_generic_hits.append(term)
        for term in _AGENT_HIGH_VALUE_TERMS:
            if _contains_term(topic_text, term) and not _contains_negated_term(topic_text, term):
                agent_high_value_hits.append(term)

        has_high_value = bool(agent_high_value_hits)
        high_value_count = len(
            {str(term).strip().lower() for term in agent_high_value_hits if str(term).strip()}
        )
        non_generic_count = len({str(term).strip().lower() for term in agent_non_generic_hits if str(term).strip()})
        if score >= 0.8 and not (has_high_value or non_generic_count >= 2):
            score = 0.79
            agent_score_cap = "cap_lt_0.8_requires_non_generic_or_high_value"
        elif score >= 0.9 and not (has_high_value and non_generic_count >= 2):
            score = 0.89
            agent_score_cap = "cap_lt_0.9_requires_high_value_and_non_generic_depth"

        signals = _quality_signals(item)
        metadata = dict(item.metadata or {})
        stars = int(_to_float(metadata.get("stars"), 0.0))
        forks = int(_to_float(metadata.get("forks"), 0.0))
        points = int(_to_float(metadata.get("points"), 0.0))
        comments = int(_to_float(metadata.get("comment_count", metadata.get("comments", 0)), 0.0))
        downloads = int(_to_float(metadata.get("downloads"), 0.0))
        cross_source_corroborated = bool(metadata.get("cross_source_corroborated")) or int(
            _to_float(metadata.get("cross_source_corroboration_count"), 0.0)
        ) >= 2
        has_engagement_signal = bool(
            stars >= 30
            or forks >= 5
            or points >= 20
            or comments >= 8
            or downloads >= 1000
            or cross_source_corroborated
        )
        body_len = int(_to_float((item.metadata or {}).get("body_len"), len(str(item.body_md or ""))))
        has_verifiable_content_signal = bool(
            float(signals["content_density"]) >= 0.14
            or bool(signals["has_quickstart"])
            or bool(signals["has_results_or_bench"])
            or re.search(r"\b(quickstart|benchmark|result|usage|cli|api|workflow)\b", topic_text)
        )
        if (
            has_high_value
            and (high_value_count >= 2 or non_generic_count >= 2)
            and has_verifiable_content_signal
            and float(signals["content_density"]) >= 0.18
            and body_len >= 500
            and has_engagement_signal
            and score >= 0.94
        ):
            score = 1.0
        else:
            score = min(score, 0.93)
            if not has_engagement_signal and score >= 0.89:
                agent_score_cap = "cap_lt_1.0_requires_engagement_or_cross_source"
            elif not agent_score_cap and score >= 0.93:
                agent_score_cap = "cap_lt_1.0_requires_high_value_plus_verifiable_content"

    return {
        "score": score,
        "hard_pass": hard_pass,
        "hard_terms": hard_terms,
        "boost_terms": boost_terms,
        "penalty_terms": penalty_terms,
        "agent_non_generic_hits": agent_non_generic_hits,
        "agent_high_value_hits": agent_high_value_hits,
        "agent_score_cap": agent_score_cap,
    }


def score_relevance(
    item: NormalizedItem,
    topic: Optional[str],
    *,
    topic_profile: Optional[TopicProfile] = None,
) -> float:
    payload = evaluate_relevance(item, topic, topic_profile=topic_profile)
    return float(payload.get("score", 0.0) or 0.0)


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
    topic: Optional[str] = None,
    topic_profile: Optional[TopicProfile] = None,
    relevance_threshold: float = 0.55,
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
        relevance_payload = evaluate_relevance(item, topic, topic_profile=topic_profile)
        relevance_score = float(relevance_payload.get("score", 0.0) or 0.0)
        hard_match_pass = bool(relevance_payload.get("hard_pass", True))
        hard_terms = [str(value) for value in list(relevance_payload.get("hard_terms") or []) if str(value).strip()]
        boost_terms = [str(value) for value in list(relevance_payload.get("boost_terms") or []) if str(value).strip()]
        penalty_terms = [str(value) for value in list(relevance_payload.get("penalty_terms") or []) if str(value).strip()]
        agent_non_generic_hits = [
            str(value) for value in list(relevance_payload.get("agent_non_generic_hits") or []) if str(value).strip()
        ]
        agent_high_value_hits = [
            str(value) for value in list(relevance_payload.get("agent_high_value_hits") or []) if str(value).strip()
        ]
        agent_score_cap = str(relevance_payload.get("agent_score_cap") or "").strip()
        cross_source_bonus = max(0.0, min(0.08, _to_float((item.metadata or {}).get("cross_source_bonus"), 0.0)))
        repeat_penalty = max(0.0, min(0.16, _to_float((item.metadata or {}).get("recent_topic_repeat_penalty"), 0.0)))
        repeat_count = int(max(0.0, _to_float((item.metadata or {}).get("recent_topic_pick_count"), 0.0)))
        corroboration_sources = [
            str(value).strip()
            for value in list((item.metadata or {}).get("cross_source_corroboration_sources") or [])
            if str(value).strip()
        ]

        metadata = dict(item.metadata or {})
        metadata["topic_hard_match_pass"] = bool(hard_match_pass)
        metadata["topic_hard_match_terms"] = hard_terms
        metadata["topic_soft_boost_terms"] = boost_terms
        metadata["topic_soft_penalty_terms"] = penalty_terms
        item.metadata = metadata
        gate_on_relevance = bool(topic and float(relevance_threshold) > 0.0)
        relevance_eligible = bool(relevance_score >= float(relevance_threshold)) if gate_on_relevance else True

        scores = {
            "novelty": score_novelty(item),
            "talkability": score_talkability(item),
            "credibility": score_credibility(item),
            "visual_assets": score_visual_assets(item),
        }
        total = _weighted_total(scores)
        total = _clamp01(total * (0.5 + 0.5 * relevance_score))
        quality_boost = _quality_signal_boost(item)

        if citation_count <= 0:
            total = _clamp01(total * 0.76)
        if published_recency is None:
            total = _clamp01(total * 0.82)
        if not quality_eligible:
            total = _clamp01(total * 0.72)
        if gate_on_relevance and not relevance_eligible:
            total = _clamp01(total * 0.45)
        else:
            total = _clamp01(total + quality_boost)
        if cross_source_bonus > 0:
            total = _clamp01(total + cross_source_bonus)
        if repeat_penalty > 0:
            total = _clamp01(total - repeat_penalty)

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
                relevance_score,
                relevance_eligible,
                quality_boost,
                hard_match_pass,
                hard_terms,
                boost_terms,
                penalty_terms,
                agent_non_generic_hits,
                agent_high_value_hits,
                agent_score_cap,
                cross_source_bonus,
                repeat_penalty,
                repeat_count,
                corroboration_sources,
            )
        )

    staged.sort(key=lambda row: row[2], reverse=True)

    # Top picks must pass body/evidence gate unless short announcement with evidence.
    top: List[tuple] = []
    deferred_tier_b: List[tuple] = []
    deferred_quality_gate: List[tuple] = []
    deferred_relevance_gate: List[tuple] = []
    for row in staged:
        (
            item,
            scores,
            _total,
            _body_len,
            _citation_count,
            _published_recency,
            _link_count,
            quality_eligible,
            _short_announcement,
            relevance_score,
            relevance_eligible,
            _quality_boost,
            _hard_match_pass,
            _hard_terms,
            _boost_terms,
            _penalty_terms,
            _agent_non_generic_hits,
            _agent_high_value_hits,
            _agent_score_cap,
            _cross_source_bonus,
            _repeat_penalty,
            _repeat_count,
            _corroboration_sources,
        ) = row
        if topic and float(relevance_threshold) > 0.0 and not relevance_eligible:
            deferred_relevance_gate.append(row)
            continue
        if not quality_eligible:
            deferred_quality_gate.append(row)
            continue
        if len(top) < 3 and item.tier == "B" and scores["talkability"] < float(tier_b_top3_talkability_threshold):
            deferred_tier_b.append(row)
            continue
        top.append(row)

    ordered = top + deferred_tier_b + deferred_quality_gate + deferred_relevance_gate
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
        relevance_score,
        relevance_eligible,
        quality_boost,
        hard_match_pass,
        hard_terms,
        boost_terms,
        penalty_terms,
        agent_non_generic_hits,
        agent_high_value_hits,
        agent_score_cap,
        cross_source_bonus,
        repeat_penalty,
        repeat_count,
        corroboration_sources,
    ) in enumerate(ordered, start=1):
        reasons = [
            f"novelty={scores['novelty']:.2f}",
            f"talkability={scores['talkability']:.2f}",
            f"credibility={scores['credibility']:.2f}",
            f"visual_assets={scores['visual_assets']:.2f}",
            f"relevance={relevance_score:.2f}",
            f"tier={item.tier}",
            f"quality.body_len={body_len}",
            f"quality.citation_count={citation_count}",
            f"quality.published_recency_days={published_recency if published_recency is not None else 'unknown'}",
            f"quality.link_count={link_count}",
            f"quality.top_pick_gate={'pass' if quality_eligible else 'deferred'}",
            f"quality.relevance_gate={'pass' if relevance_eligible else 'deferred'}",
            f"topic.hard_match={'pass' if hard_match_pass else 'deferred'}",
            f"topic.hard_terms={','.join(hard_terms) if hard_terms else 'none'}",
            f"quality.boost={quality_boost:.2f}",
        ]
        if boost_terms:
            reasons.append(f"topic.soft_boost_terms={','.join(boost_terms)}")
        if penalty_terms:
            reasons.append(f"topic.soft_penalty_terms={','.join(penalty_terms)}")
        if agent_non_generic_hits:
            reasons.append(f"topic.agent_non_generic_hits={','.join(agent_non_generic_hits)}")
        if agent_high_value_hits:
            reasons.append(f"topic.agent_high_value_hits={','.join(agent_high_value_hits)}")
        if agent_score_cap:
            reasons.append(f"topic.agent_score_cap={agent_score_cap}")
        if cross_source_bonus > 0:
            reasons.append(f"cross_source.corroboration_bonus={cross_source_bonus:.2f}")
            if corroboration_sources:
                reasons.append(f"cross_source.sources={','.join(corroboration_sources)}")
        if repeat_penalty > 0:
            reasons.append(f"penalty.recent_topic_repeat={repeat_penalty:.2f}")
            reasons.append(f"recent_topic.pick_count={repeat_count}")
        signals = _quality_signals(item)
        reasons.extend(
            [
                f"quality.signal.density={float(signals['content_density']):.3f}",
                f"quality.signal.quickstart={bool(signals['has_quickstart'])}",
                f"quality.signal.results={bool(signals['has_results_or_bench'])}",
                f"quality.signal.evidence_links={int(signals['evidence_links_quality'])}",
                (
                    "quality.signal.update_recency_days="
                    f"{signals['update_recency_days'] if signals['update_recency_days'] is not None else 'unknown'}"
                ),
            ]
        )
        if short_announcement:
            reasons.append("quality.short_announcement=true")
        if citation_count <= 0:
            reasons.append("penalty.no_citations")
        if published_recency is None:
            reasons.append("penalty.missing_published_time")
        if not quality_eligible:
            reasons.append(f"penalty.body_len_lt_{int(min_body_len_for_top_picks)}")
        if topic and float(relevance_threshold) > 0.0 and not relevance_eligible:
            reasons.append(f"penalty.relevance_lt_{float(relevance_threshold):.2f}")
        if topic and not hard_match_pass:
            reasons.append("topic.hard_gate_fail")
        if int(signals["evidence_links_quality"]) <= 0:
            reasons.append("penalty.no_evidence_links")
        if signals["update_recency_days"] is not None and float(signals["update_recency_days"]) >= 120:
            reasons.append("penalty.too_old")
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
                relevance_score=relevance_score,
                reasons=reasons,
            )
        )

    return ranked
