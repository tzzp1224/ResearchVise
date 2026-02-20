"""Explainable scoring and ranking for canonical items."""

from __future__ import annotations

from datetime import datetime, timezone
import math
import re
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from core import NormalizedItem, RankedItem
from pipeline_v2.topic_profile import TopicProfile
from pipeline_v2.topic_intent import TopicIntent


_HOT_SCORE_WEIGHTS = {
    "trend": 0.46,
    "content_worth": 0.29,
    "relevance": 0.25,
    "penalty": 0.42,
}


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
    "computer use",
    "browser agent",
    "agent memory",
    "planning",
    "plan-and-execute",
    "task decomposition",
}
_AGENT_EVIDENCE_SIGNAL_TERMS = {
    "quickstart",
    "demo",
    "benchmark",
    "tool calling",
    "function calling",
    "orchestration",
    "mcp",
    "langgraph",
    "autogen",
    "crewai",
    "agent runtime",
}


def _parse_window_days(window: Optional[str]) -> int:
    token = str(window or "").strip().lower()
    if not token:
        return 1
    if token == "today":
        return 1
    if token.endswith("d"):
        try:
            return max(1, int(token[:-1]))
        except Exception:
            return 3
    if token.endswith("h"):
        try:
            return max(1, int((int(token[:-1]) + 23) // 24))
        except Exception:
            return 1
    if token in {"past_week", "last_week", "weekly"}:
        return 7
    if token in {"past_month", "monthly"}:
        return 30
    return 3


def _activity_signals(item: NormalizedItem, *, recall_window: Optional[str]) -> Dict[str, float]:
    metadata = dict(item.metadata or {})
    quality = _quality_signals(item)
    update_days_raw = quality.get("update_recency_days")
    update_days = float(update_days_raw) if update_days_raw not in (None, "", "unknown") else None
    stars = float(metadata.get("stars", 0) or 0)
    forks = float(metadata.get("forks", 0) or 0)
    points = float(metadata.get("points", 0) or 0)
    comments = float(metadata.get("comment_count", metadata.get("comments", 0)) or 0)
    downloads = float(metadata.get("downloads", 0) or 0)
    engagement = stars + forks * 2.5 + points * 3.0 + comments * 2.5 + downloads / 220.0
    engagement_score = _clamp01(math.log10(1.0 + max(0.0, engagement)) / 3.3)

    recent_update_signal = 0.0
    if update_days is not None:
        if update_days <= 2:
            recent_update_signal = 1.0
        elif update_days <= 7:
            recent_update_signal = 0.85
        elif update_days <= 14:
            recent_update_signal = 0.65
        elif update_days <= 30:
            recent_update_signal = 0.4
        else:
            recent_update_signal = 0.15

    window_days = _parse_window_days(recall_window)
    # Keep star separation meaningful in 7d mode; avoid saturating mid-star repos to ~1.0.
    stars_delta_proxy = _clamp01((math.log10(1.0 + max(0.0, stars)) / 4.2) * (0.45 + 0.55 * recent_update_signal))

    boost = 0.0
    if window_days >= 7:
        if update_days is not None and update_days <= 7:
            boost += 0.07
        elif update_days is not None and update_days > 21:
            boost -= 0.04
    if engagement_score >= 0.62:
        boost += 0.03
    elif engagement_score <= 0.2:
        boost -= 0.02
    if window_days >= 7:
        boost += (stars_delta_proxy - 0.55) * 0.08
        if stars_delta_proxy < 0.35:
            boost -= 0.03
    else:
        boost += (stars_delta_proxy - 0.5) * 0.04

    return {
        "recent_update_signal": round(float(recent_update_signal), 4),
        "stars_delta_proxy": round(float(stars_delta_proxy), 4),
        "engagement_signal": round(float(engagement_score), 4),
        "boost": round(float(max(-0.1, min(0.12, boost))), 4),
        "window_days": float(window_days),
    }


def _parse_iso_datetime(value: object) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _days_since(value: object) -> Optional[float]:
    dt = _parse_iso_datetime(value)
    if dt is None:
        return None
    return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0)


def _content_worth_score(item: NormalizedItem) -> float:
    signals = _quality_signals(item)
    text = _topic_text(item)
    score = 0.0

    density = float(signals["content_density"])
    if density >= 0.24:
        score += 0.24
    elif density >= 0.16:
        score += 0.18
    elif density >= 0.1:
        score += 0.12
    else:
        score += 0.05

    if bool(signals["has_quickstart"]):
        score += 0.20
    if bool(signals["has_results_or_bench"]):
        score += 0.18
    if bool(signals["has_images_non_badge"]):
        score += 0.12

    evidence_links = int(signals["evidence_links_quality"])
    if evidence_links >= 3:
        score += 0.18
    elif evidence_links >= 1:
        score += 0.11
    else:
        score += 0.03

    if re.search(r"\b(what|how|demo|usage|quickstart|screenshot|showcase|example)\b", text):
        score += 0.11
    if re.search(r"\b(cli|api|sdk|install|run|deploy|workflow)\b", text):
        score += 0.08

    return _clamp01(score)


def _anti_dominance_penalty(item: NormalizedItem) -> float:
    metadata = dict(item.metadata or {})
    intent_is_infra = bool(metadata.get("intent_is_infra"))
    infra_exception_event = bool(metadata.get("infra_exception_event"))
    intent_is_handbook = bool(metadata.get("intent_is_handbook"))
    intent_partition = str(metadata.get("intent_partition") or "").strip().lower()
    repeat_penalty = max(0.0, min(0.2, _to_float(metadata.get("recent_topic_repeat_penalty"), 0.0)))
    score = float(repeat_penalty)
    if intent_is_infra and not infra_exception_event:
        score += 0.82
    elif intent_is_infra and infra_exception_event:
        score += 0.34
    if intent_is_handbook or intent_partition == "background":
        score += 0.76
    return _clamp01(score)


def _bonus_scores(item: NormalizedItem, *, trend_score: float) -> Dict[str, float]:
    metadata = dict(item.metadata or {})
    cross_source_count = int(_to_float(metadata.get("cross_source_corroboration_count"), 0.0))
    cross_source_corroborated = bool(metadata.get("cross_source_corroborated")) or cross_source_count >= 2
    cross_source_bonus = 0.06 if cross_source_corroborated else 0.0

    points = _to_float(metadata.get("points"), 0.0)
    comments = _to_float(metadata.get("comment_count", metadata.get("comments", 0)), 0.0)
    hn_discussion_bonus = 0.0
    if str(item.source or "").strip().lower() == "hackernews" or points > 0 or comments > 0:
        hn_discussion_bonus = min(0.1, (points / 320.0) * 0.06 + (comments / 140.0) * 0.04)

    downloads = _to_float(metadata.get("downloads"), 0.0)
    likes = _to_float(metadata.get("likes"), 0.0)
    hf_rank = _to_float(metadata.get("search_rank"), 0.0)
    hf_pool = _to_float(metadata.get("search_pool_size"), 0.0)
    hf_trending_bonus = 0.0
    if str(item.source or "").strip().lower() == "huggingface" or downloads > 0:
        hf_trending_bonus = min(0.08, (downloads / 9000.0) * 0.05 + (likes / 400.0) * 0.03)
        if hf_rank > 0 and hf_pool > 1:
            hf_trending_bonus = min(0.08, hf_trending_bonus + max(0.0, 0.03 * (1.0 - (hf_rank - 1.0) / max(1.0, hf_pool - 1.0))))

    bonus_total = max(0.0, min(0.24, cross_source_bonus + hn_discussion_bonus + hf_trending_bonus))
    single_source_hot = bool(not cross_source_corroborated and float(trend_score) >= 0.72)
    return {
        "cross_source": round(float(cross_source_bonus), 4),
        "hn_discussion": round(float(hn_discussion_bonus), 4),
        "hf_trending": round(float(hf_trending_bonus), 4),
        "total": round(float(bonus_total), 4),
        "single_source_hot": float(1.0 if single_source_hot else 0.0),
    }


def _trend_score(item: NormalizedItem, *, activity: Mapping[str, float], recall_window: Optional[str]) -> float:
    metadata = dict(item.metadata or {})
    trend_signal = _clamp01(_to_float(metadata.get("trend_signal_score"), 0.0))
    recent_update = _clamp01(_to_float(activity.get("recent_update_signal"), 0.0))
    stars_proxy = _clamp01(_to_float(activity.get("stars_delta_proxy"), 0.0))
    base = _clamp01(0.55 * trend_signal + 0.25 * recent_update + 0.2 * stars_proxy)

    recency_bonus = 0.0
    created_days = _days_since(metadata.get("created_at"))
    pushed_days = _days_since(metadata.get("updated_at") or metadata.get("last_push"))
    release_days = _days_since(metadata.get("release_published_at"))
    window_days = max(1, _parse_window_days(recall_window))
    if created_days is not None and created_days <= min(window_days, 7):
        recency_bonus += 0.07
    if pushed_days is not None and pushed_days <= min(window_days, 7):
        recency_bonus += 0.08
    if release_days is not None and release_days <= min(window_days, 7):
        recency_bonus += 0.1
    return _clamp01(base + recency_bonus)


def _hot_score_breakdown(
    item: NormalizedItem,
    *,
    relevance_score: float,
    activity: Mapping[str, float],
    recall_window: Optional[str],
) -> Dict[str, object]:
    trend_score = _trend_score(item, activity=activity, recall_window=recall_window)
    content_worth_score = _content_worth_score(item)
    relevance_component = _clamp01(relevance_score)
    penalty_score = _anti_dominance_penalty(item)
    bonus = _bonus_scores(item, trend_score=trend_score)
    bonus_total = _to_float(bonus.get("total"), 0.0)

    final_hot_score = _clamp01(
        _HOT_SCORE_WEIGHTS["trend"] * trend_score
        + _HOT_SCORE_WEIGHTS["content_worth"] * content_worth_score
        + _HOT_SCORE_WEIGHTS["relevance"] * relevance_component
        - _HOT_SCORE_WEIGHTS["penalty"] * penalty_score
        + bonus_total
    )
    return {
        "relevance_score": round(float(relevance_component), 4),
        "trend_score": round(float(trend_score), 4),
        "content_worth_score": round(float(content_worth_score), 4),
        "anti_dominance_penalty": round(float(penalty_score), 4),
        "bonus": {
            "cross_source": round(float(_to_float(bonus.get("cross_source"), 0.0)), 4),
            "hn_discussion": round(float(_to_float(bonus.get("hn_discussion"), 0.0)), 4),
            "hf_trending": round(float(_to_float(bonus.get("hf_trending"), 0.0)), 4),
            "total": round(float(bonus_total), 4),
        },
        "single_source_hot": bool(_to_float(bonus.get("single_source_hot"), 0.0) > 0.0),
        "final_hot_score": round(float(final_hot_score), 4),
        "weights": dict(_HOT_SCORE_WEIGHTS),
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
    payload = re.sub(r"[-_/]+", " ", str(text or "").lower())
    token = str(term or "").strip().lower()
    if not payload or not token:
        return False
    pattern = r"\b" + re.escape(token).replace(r"\ ", r"[\s\-_]+") + r"\b"
    return re.search(pattern, payload) is not None


def _contains_negated_term(text: str, term: str) -> bool:
    payload = re.sub(r"[-_/]+", " ", str(text or "").lower())
    token = str(term or "").strip().lower()
    if not payload or not token:
        return False
    term_pattern = re.escape(token).replace(r"\ ", r"[\s\-_]+")
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
        metadata = dict(item.metadata or {})
        bucket_hits = [
            str(value).strip()
            for value in list(metadata.get("bucket_hits") or [])
            if str(value).strip()
        ]
        has_bucket_signal = bool(bucket_hits)
        for term in hard_terms:
            lowered = str(term).strip().lower()
            if lowered and lowered not in _AGENT_GENERIC_TERMS and not _contains_negated_term(topic_text, lowered):
                agent_non_generic_hits.append(term)
        for term in _AGENT_HIGH_VALUE_TERMS:
            if _contains_term(topic_text, term) and not _contains_negated_term(topic_text, term):
                agent_high_value_hits.append(term)

        has_high_value = bool(agent_high_value_hits)
        high_value_count = len({str(term).strip().lower() for term in agent_high_value_hits if str(term).strip()})
        non_generic_count = len({str(term).strip().lower() for term in agent_non_generic_hits if str(term).strip()})
        evidence_signal_hits = [
            term
            for term in list(_AGENT_EVIDENCE_SIGNAL_TERMS)
            if _contains_term(topic_text, term) and not _contains_negated_term(topic_text, term)
        ]
        has_evidence_signal = bool(evidence_signal_hits)
        has_semantic_depth = bool(
            high_value_count >= 2
            or (high_value_count >= 1 and has_evidence_signal)
            or non_generic_count >= 2
            or has_bucket_signal
        )
        if score >= 0.75 and not has_semantic_depth:
            score = 0.74
            agent_score_cap = "cap_lt_0.75_requires_high_value_or_depth"
        elif score > 0.85 and not (high_value_count >= 1 and has_evidence_signal):
            score = 0.85
            agent_score_cap = "cap_0.85_requires_high_value_plus_evidence_signal"
        elif score > 0.9 and not (high_value_count >= 2 and has_evidence_signal):
            score = 0.9
            agent_score_cap = "cap_0.90_requires_two_high_value_terms_plus_evidence_signal"

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
        update_recency_days = signals["update_recency_days"]
        has_verifiable_content_signal = bool(
            float(signals["content_density"]) >= 0.18
            or bool(signals["has_quickstart"])
            or bool(signals["has_results_or_bench"])
            or re.search(r"\b(quickstart|benchmark|result|usage|cli|api|workflow)\b", topic_text)
        )
        has_update_or_hot_signal = bool(
            (update_recency_days is not None and float(update_recency_days) <= 7.0)
            or has_engagement_signal
        )
        if (
            high_value_count >= 2
            and has_verifiable_content_signal
            and has_update_or_hot_signal
            and float(signals["content_density"]) >= 0.18
            and body_len >= 500
            and score >= 0.93
        ):
            score = 1.0
        else:
            if high_value_count >= 2 and (has_verifiable_content_signal or has_update_or_hot_signal):
                score = min(score, 0.9)
                if not agent_score_cap and score >= 0.9:
                    agent_score_cap = "cap_0.90_requires_full_triplet_for_1.0"
            else:
                score = min(score, 0.85)
                if not agent_score_cap and score >= 0.85:
                    agent_score_cap = "cap_0.85_requires_strong_high_value_and_activity_signals"

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
    topic_intent: Optional[TopicIntent] = None,
    relevance_threshold: float = 0.55,
    recall_window: Optional[str] = None,
) -> List[RankedItem]:
    """Rank items with explainable subscores and Tier B top3 gate."""
    staged = []
    window_days = _parse_window_days(recall_window)
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
        if topic_intent is not None:
            topic_intent.classify_and_annotate(item)
        cross_source_bonus = max(0.0, min(0.08, _to_float((item.metadata or {}).get("cross_source_bonus"), 0.0)))
        repeat_penalty = max(0.0, min(0.16, _to_float((item.metadata or {}).get("recent_topic_repeat_penalty"), 0.0)))
        if window_days >= 7:
            # Keep anti-repetition, but avoid overwhelming true trending/high-signal repos in weekly mode.
            repeat_penalty = min(0.08, repeat_penalty * 0.5)
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
        metadata["topic_agent_non_generic_hits"] = list(agent_non_generic_hits)
        metadata["topic_agent_high_value_hits"] = list(agent_high_value_hits)
        metadata["topic_agent_score_cap"] = agent_score_cap
        item.metadata = metadata
        gate_on_relevance = bool(topic and float(relevance_threshold) > 0.0)
        relevance_eligible = bool(relevance_score >= float(relevance_threshold)) if gate_on_relevance else True

        scores = {
            "novelty": score_novelty(item),
            "talkability": score_talkability(item),
            "credibility": score_credibility(item),
            "visual_assets": score_visual_assets(item),
        }
        activity = _activity_signals(item, recall_window=recall_window)
        metadata = dict(item.metadata or {})
        metadata["recent_update_signal"] = float(activity["recent_update_signal"])
        metadata["stars_delta_proxy"] = float(activity["stars_delta_proxy"])
        metadata["activity_signal_boost"] = float(activity["boost"])
        item.metadata = metadata

        hot_breakdown = _hot_score_breakdown(
            item,
            relevance_score=float(relevance_score),
            activity=activity,
            recall_window=recall_window,
        )
        metadata = dict(item.metadata or {})
        metadata["hot_score_breakdown"] = dict(hot_breakdown)
        metadata["single_source_hot"] = bool(hot_breakdown.get("single_source_hot"))
        metadata["final_hot_score"] = float(hot_breakdown.get("final_hot_score", 0.0) or 0.0)
        metadata["relevance_score_component"] = float(hot_breakdown.get("relevance_score", 0.0) or 0.0)
        metadata["trend_score_component"] = float(hot_breakdown.get("trend_score", 0.0) or 0.0)
        metadata["content_worth_score_component"] = float(hot_breakdown.get("content_worth_score", 0.0) or 0.0)
        metadata["anti_dominance_penalty_component"] = float(hot_breakdown.get("anti_dominance_penalty", 0.0) or 0.0)
        metadata["bonus_total_component"] = float(dict(hot_breakdown.get("bonus") or {}).get("total", 0.0) or 0.0)
        item.metadata = metadata

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
        total = _clamp01(total + float(activity["boost"]))
        if cross_source_bonus > 0:
            total = _clamp01(total + cross_source_bonus)
        if repeat_penalty > 0:
            total = _clamp01(total - repeat_penalty)

        intent_metadata = dict(item.metadata or {})
        trend_signal_score = _clamp01(_to_float(intent_metadata.get("trend_signal_score"), 0.0))
        trend_reasons = [str(value).strip() for value in list(intent_metadata.get("trend_signal_reasons") or []) if str(value).strip()]
        trend_proxy_used = bool(intent_metadata.get("trend_signal_proxy_used"))
        intent_hot_candidate = bool(intent_metadata.get("intent_hot_candidate"))
        intent_is_infra = bool(intent_metadata.get("intent_is_infra"))
        intent_is_handbook = bool(intent_metadata.get("intent_is_handbook"))
        infra_exception_event = bool(intent_metadata.get("infra_exception_event"))
        intent_bucket = str(intent_metadata.get("intent_bucket") or "").strip().lower()
        evidence_alignment_proxy = _clamp01(
            float(_quality_signals(item)["content_density"]) * 1.4
            + (0.2 if bool(_quality_signals(item)["has_quickstart"]) else 0.0)
            + (0.2 if bool(_quality_signals(item)["has_results_or_bench"]) else 0.0)
            + min(0.3, float(_quality_signals(item)["evidence_links_quality"]) * 0.1)
            + (0.2 if bool((item.metadata or {}).get("cross_source_corroborated")) else 0.0)
        )

        if topic_intent is not None and topic_intent.hot_new_agents_mode:
            total = _clamp01(float(hot_breakdown.get("final_hot_score", 0.0) or 0.0))

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
                activity,
                trend_signal_score,
                evidence_alignment_proxy,
                trend_reasons,
                trend_proxy_used,
                intent_hot_candidate,
                intent_is_infra,
                intent_is_handbook,
                infra_exception_event,
                intent_bucket,
                hot_breakdown,
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
            _activity,
            _trend_signal_score,
            _evidence_alignment_proxy,
            _trend_reasons,
            _trend_proxy_used,
            _intent_hot_candidate,
            _intent_is_infra,
            _intent_is_handbook,
            _infra_exception_event,
            _intent_bucket,
            _hot_breakdown,
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
        activity,
        trend_signal_score,
        evidence_alignment_proxy,
        trend_reasons,
        trend_proxy_used,
        intent_hot_candidate,
        intent_is_infra,
        intent_is_handbook,
        infra_exception_event,
        intent_bucket,
        hot_breakdown,
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
            f"activity.recent_update_signal={float(activity['recent_update_signal']):.2f}",
            f"activity.stars_delta_proxy={float(activity['stars_delta_proxy']):.2f}",
            f"activity.boost={float(activity['boost']):.2f}",
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
        if trend_signal_score > 0:
            reasons.append(f"trend.signal={trend_signal_score:.2f}")
            reasons.append(f"trend.evidence_alignment_proxy={evidence_alignment_proxy:.2f}")
        if trend_reasons:
            reasons.append(f"trend.reasons={','.join(trend_reasons[:3])}")
        if trend_proxy_used:
            reasons.append("trend.proxy=search_rank_position")
        if intent_hot_candidate:
            reasons.append("intent.hot_candidate=true")
        if intent_is_infra:
            reasons.append("intent.infra=true")
        if intent_is_handbook:
            reasons.append("intent.handbook=true")
        if infra_exception_event:
            reasons.append("infra_exception_event=true")
        if intent_bucket:
            reasons.append(f"intent.bucket={intent_bucket}")
        hot_bonus_payload = dict(hot_breakdown.get("bonus") or {})
        reasons.extend(
            [
                f"hot.relevance_score={float(hot_breakdown.get('relevance_score', 0.0) or 0.0):.2f}",
                f"hot.trend_score={float(hot_breakdown.get('trend_score', 0.0) or 0.0):.2f}",
                f"hot.content_worth_score={float(hot_breakdown.get('content_worth_score', 0.0) or 0.0):.2f}",
                f"hot.anti_dominance_penalty={float(hot_breakdown.get('anti_dominance_penalty', 0.0) or 0.0):.2f}",
                f"hot.bonus.cross_source={float(hot_bonus_payload.get('cross_source', 0.0) or 0.0):.2f}",
                f"hot.bonus.hn_discussion={float(hot_bonus_payload.get('hn_discussion', 0.0) or 0.0):.2f}",
                f"hot.bonus.hf_trending={float(hot_bonus_payload.get('hf_trending', 0.0) or 0.0):.2f}",
                f"hot.bonus.total={float(hot_bonus_payload.get('total', 0.0) or 0.0):.2f}",
                f"hot.final_score={float(hot_breakdown.get('final_hot_score', 0.0) or 0.0):.2f}",
            ]
        )
        if bool(hot_breakdown.get("single_source_hot")):
            reasons.append("single_source_hot=true")
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
