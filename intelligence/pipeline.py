"""
Research Pipeline
关键词驱动的端到端研究流水线（Phase 1 -> 4 + 可选视频生成）
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import json
import logging
import re

from aggregator import DataAggregator
from config import get_research_cache_settings, get_retrieval_settings
from intelligence.agents import (
    AnalystAgent,
    ChatAgent,
    ContentAgent,
    CriticAgent,
    PlannerAgent,
    SearchAgent,
)
from intelligence.llm import BaseLLM, get_llm
from intelligence.pipeline_helpers import (
    aggregated_result_to_search_results,
    evaluate_output_depth,
    evaluate_research_quality,
    normalize_one_pager_content,
    normalize_one_pager_resources,
    normalize_timeline_dates,
    normalize_video_brief,
)
from intelligence.tools.rag_tools import add_to_knowledge_base, close_knowledge_base
from intelligence.tools.deep_content_enricher import enrich_search_results_deep
from outputs import export_research_outputs
from outputs.video_generator import BaseVideoGenerator, create_video_generator
from storage import ResearchArtifactStore


logger = logging.getLogger(__name__)

_RESULT_SNAPSHOT_FILE = "research_result_snapshot.json"
_ARTIFACT_SCHEMA_VERSION = "2026-02-08-slidev-v2"

_AGGREGATOR_FLAG_TO_SOURCE = {
    "enable_arxiv": "arxiv",
    "enable_huggingface": "huggingface",
    "enable_twitter": "twitter",
    "enable_reddit": "reddit",
    "enable_github": "github",
    "enable_semantic_scholar": "semantic_scholar",
    "enable_stackoverflow": "stackoverflow",
    "enable_hackernews": "hackernews",
}

_SOURCE_ALIASES = {
    "semantic-scholar": "semantic_scholar",
    "semantic scholar": "semantic_scholar",
    "stack-overflow": "stackoverflow",
    "stack overflow": "stackoverflow",
    "hn": "hackernews",
}

_CATEGORY_HINTS = {
    "architecture": ["architecture", "design", "mechanism", "framework", "架构", "机制"],
    "performance": ["benchmark", "latency", "throughput", "speed", "性能", "吞吐", "延迟"],
    "training": ["train", "dataset", "fine-tune", "optimization", "训练", "数据集"],
    "comparison": ["compare", "versus", "vs", "trade-off", "对比", "取舍"],
    "limitation": ["limitation", "risk", "issue", "failure", "局限", "风险", "问题"],
    "deployment": ["deploy", "production", "infra", "monitor", "部署", "生产", "监控"],
}

_SOURCE_TYPE_MAP = {
    "arxiv": "paper",
    "arxiv_rss": "paper",
    "openreview": "paper",
    "semantic_scholar": "paper",
    "huggingface": "code",
    "github": "code",
    "stackoverflow": "social",
    "hackernews": "social",
    "reddit": "social",
    "twitter": "social",
}

_TOPIC_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "into",
    "towards",
    "using",
    "based",
    "model",
    "models",
    "system",
    "systems",
    "framework",
    "study",
    "analysis",
    "learning",
    "技术",
    "研究",
    "系统",
    "方法",
    "应用",
}

_VERSION_PATTERN = re.compile(r"\d+(?:\.\d+)+")


async def _emit_progress(
    progress_callback: Optional[Callable[[Dict[str, Any]], Any]],
    *,
    event: str,
    **payload: Any,
) -> None:
    if progress_callback is None:
        return
    message: Dict[str, Any] = {"event": event}
    message.update(payload)
    try:
        result = progress_callback(message)
        if inspect.isawaitable(result):
            await result
    except Exception:
        logger.debug("Progress callback failed", exc_info=True)


def _normalize_sources(values: Optional[List[str]]) -> List[str]:
    if not values:
        return []
    normalized: List[str] = []
    seen = set()
    for item in values:
        key = str(item or "").strip().lower().replace("-", "_")
        if not key:
            continue
        key = _SOURCE_ALIASES.get(key, key)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    return normalized


def _sources_from_aggregator_kwargs(kwargs: Dict[str, Any]) -> List[str]:
    sources: List[str] = []
    for flag, source in _AGGREGATOR_FLAG_TO_SOURCE.items():
        if bool(kwargs.get(flag, False)):
            sources.append(source)
    return sorted(sources)


def _aggregator_kwargs_from_sources(sources: List[str]) -> Dict[str, bool]:
    allowed = set(_normalize_sources(sources))
    if not allowed:
        return {}
    return {
        flag: source in allowed for flag, source in _AGGREGATOR_FLAG_TO_SOURCE.items()
    }


def _summarize_search_results(search_results: List[Dict[str, Any]]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for item in search_results:
        source = str(item.get("source", "")).strip() or "unknown"
        summary[source] = summary.get(source, 0) + 1
    summary["total"] = len(search_results)
    return summary


def _deduplicate_search_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique: List[Dict[str, Any]] = []
    seen = set()
    for item in results:
        rid = str(item.get("id") or "").strip()
        url = str(item.get("url") or "").strip()
        title = str(item.get("title") or "").strip().lower()
        key = rid or url or title
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _covered_sources(search_results: List[Dict[str, Any]]) -> List[str]:
    return sorted(
        {
            str(item.get("source", "")).strip()
            for item in search_results
            if str(item.get("source", "")).strip()
        }
    )


def _should_enrich_search_results(
    *,
    search_results: List[Dict[str, Any]],
    resolved_sources: List[str],
    max_results_per_source: int,
) -> bool:
    if not search_results:
        return True
    covered = set(_covered_sources(search_results))
    requested = set(_normalize_sources(resolved_sources))
    covered_requested = covered.intersection(requested) if requested else covered
    min_results = max(10, int(max_results_per_source) * 2)
    min_sources = min(4, len(requested)) if requested else 3
    return len(search_results) < min_results or len(covered_requested) < min_sources


def _topic_signals(topic: str, query_rewrites: Optional[List[str]] = None) -> List[str]:
    raw = [str(topic or "").strip()]
    raw.extend([str(item).strip() for item in (query_rewrites or []) if str(item).strip()])
    combined = " ".join([item for item in raw if item]).strip().lower()
    if not combined:
        return []

    phrases: List[str] = []
    if "reinforcement learning" in combined:
        phrases.append("reinforcement learning")
    if "model context protocol" in combined:
        phrases.append("model context protocol")
    if "强化学习" in combined:
        phrases.append("强化学习")
    if "mcp" in combined:
        phrases.append("mcp")

    tokens = re.findall(r"[a-z0-9\u4e00-\u9fff#+-]+(?:\.[0-9]+)?", combined)
    for token in tokens:
        if len(token) < 2 or token in _TOPIC_STOPWORDS:
            continue
        phrases.append(token)

    deduped: List[str] = []
    seen = set()
    for item in phrases:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped[:24]


def _topic_version_markers(topic: str, query_rewrites: Optional[List[str]] = None) -> List[str]:
    raw = [str(topic or "").strip()]
    raw.extend([str(item).strip() for item in (query_rewrites or []) if str(item).strip()])
    combined = " ".join([item for item in raw if item]).strip().lower()
    if not combined:
        return []
    markers = _VERSION_PATTERN.findall(combined)
    deduped: List[str] = []
    seen = set()
    for marker in markers:
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(marker)
    return deduped[:6]


def _extract_versions(text: str) -> List[str]:
    values = _VERSION_PATTERN.findall(str(text or "").lower())
    deduped: List[str] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _result_topic_score(
    item: Dict[str, Any],
    topic_signals: List[str],
    *,
    version_markers: Optional[List[str]] = None,
) -> int:
    if not topic_signals:
        return 0
    text = f"{item.get('title', '')} {item.get('content', '')}".lower()
    score = 0
    for signal in topic_signals:
        if signal and signal in text:
            if len(signal) >= 8 or " " in signal:
                score += 3
            else:
                score += 1
    if re.search(r"\d", text):
        score += 1

    expected_versions = list(version_markers or [])
    if expected_versions:
        item_versions = set(_extract_versions(text))
        matched_versions = item_versions.intersection(set(expected_versions))
        if matched_versions:
            score += 5 * len(matched_versions)
        elif item_versions:
            # Strongly down-rank explicit version mismatches (e.g. query=2.5 but result=1.5).
            score -= 6
        else:
            score -= 1
    return score


def _prune_off_topic_results(
    *,
    search_results: List[Dict[str, Any]],
    topic: str,
    query_rewrites: Optional[List[str]] = None,
    min_keep: int = 10,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    version_markers = _topic_version_markers(topic, query_rewrites=query_rewrites)
    if len(search_results) <= max(4, min_keep) and not version_markers:
        return search_results, {"removed": 0, "kept": len(search_results)}

    signals = _topic_signals(topic, query_rewrites=query_rewrites)
    if not signals:
        return search_results, {"removed": 0, "kept": len(search_results)}

    candidate_pool = list(search_results)
    version_filtered = 0
    version_strict_applied = False
    if version_markers:
        expected = set(version_markers)
        narrowed: List[Dict[str, Any]] = []
        for item in candidate_pool:
            text = f"{item.get('title', '')} {item.get('content', '')}"
            versions = set(_extract_versions(text))
            if versions and versions.isdisjoint(expected):
                version_filtered += 1
                continue
            narrowed.append(item)
        # Prefer strict version filtering whenever it keeps at least one candidate.
        if narrowed:
            candidate_pool = narrowed
            version_strict_applied = True
            min_keep = min(max(1, int(min_keep)), len(candidate_pool))

    scored: List[tuple[int, int, Dict[str, Any]]] = []
    for idx, item in enumerate(candidate_pool):
        scored.append(
            (
                _result_topic_score(item, signals, version_markers=version_markers),
                idx,
                item,
            )
        )

    # 先保留高相关结果
    primary = [row for row in scored if row[0] >= 2]

    # 保底：每个来源至少保留一个最高分
    best_by_source: Dict[str, tuple[int, int, Dict[str, Any]]] = {}
    for row in scored:
        score, _, item = row
        source = str(item.get("source", "")).strip()
        current = best_by_source.get(source)
        if current is None or score > current[0]:
            best_by_source[source] = row

    selected = list(primary)
    selected_ids = {id(row[2]) for row in selected}
    for row in best_by_source.values():
        if id(row[2]) not in selected_ids:
            selected.append(row)
            selected_ids.add(id(row[2]))

    if len(selected) < min_keep:
        for row in sorted(scored, key=lambda x: x[0], reverse=True):
            if id(row[2]) in selected_ids:
                continue
            if version_markers and row[0] < 0:
                continue
            selected.append(row)
            selected_ids.add(id(row[2]))
            if len(selected) >= min_keep:
                break

    selected_sorted = sorted(selected, key=lambda x: x[1])
    filtered = [row[2] for row in selected_sorted]
    removed = max(0, len(search_results) - len(filtered))
    return filtered, {
        "removed": removed,
        "kept": len(filtered),
        "signals": signals[:8],
        "version_markers": version_markers,
        "version_filtered": int(version_filtered),
        "version_strict_applied": bool(version_strict_applied),
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _exception_text(exc: Exception) -> str:
    text = str(exc).strip()
    if text:
        return text
    return exc.__class__.__name__


def _infer_fact_category(text: str) -> str:
    lowered = str(text or "").lower()
    for category, keywords in _CATEGORY_HINTS.items():
        if any(keyword in lowered for keyword in keywords):
            return category
    return "community"


def _compact_text(text: Any, max_len: int = 220) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3] + "..."


def _first_sentence(text: Any, max_len: int = 200) -> str:
    cleaned = _compact_text(text, max_len=max_len * 2)
    if not cleaned:
        return ""
    parts = re.split(r"[。.!?；;\n]+", cleaned)
    for part in parts:
        part = _compact_text(part, max_len=max_len)
        if part:
            return part
    return _compact_text(cleaned, max_len=max_len)


def _year_from_metadata(metadata: Dict[str, Any]) -> str:
    for key in ("published_date", "created_at", "updated_at", "year"):
        value = str((metadata or {}).get(key, "")).strip()
        match = re.search(r"(19|20)\d{2}", value)
        if match:
            return match.group(0)
    return ""


def _search_result_fact_priority(item: Dict[str, Any]) -> float:
    source = str(item.get("source", "")).strip().lower()
    metadata = dict(item.get("metadata", {}) or {})
    title = str(item.get("title", "")).strip()
    content = str(item.get("content", "")).strip()
    combined = f"{title} {content}".strip()

    score = {
        "arxiv": 8.0,
        "semantic_scholar": 7.8,
        "openreview": 7.5,
        "arxiv_rss": 7.2,
        "github": 6.5,
        "huggingface": 6.0,
        "stackoverflow": 4.6,
        "hackernews": 4.1,
        "reddit": 3.6,
        "twitter": 3.2,
    }.get(source, 3.0)

    if source in {"arxiv", "semantic_scholar", "openreview", "arxiv_rss"}:
        score += min(4.0, _safe_float(metadata.get("citation_count"), 0.0) / 120.0)
    elif source == "github":
        score += min(3.0, _safe_float(metadata.get("stars"), 0.0) / 900.0)
    elif source == "huggingface":
        score += min(2.5, _safe_float(metadata.get("downloads"), 0.0) / 20000.0)
    elif source == "stackoverflow":
        score += min(1.2, _safe_float(metadata.get("score"), 0.0) / 40.0)
    elif source == "hackernews":
        score += min(1.2, _safe_float(metadata.get("points"), 0.0) / 120.0)

    if re.search(r"\d", combined):
        score += 1.1
    if len(combined) >= 180:
        score += 0.7
    if any(token in combined.lower() for token in ("benchmark", "latency", "throughput", "cost", "ablation", "deploy", "风险", "对比", "部署")):
        score += 0.8
    return score


def _balanced_fact_candidates(
    search_results: List[Dict[str, Any]],
    *,
    limit: int,
) -> List[Dict[str, Any]]:
    if not search_results or limit <= 0:
        return []

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in search_results:
        source = str(item.get("source", "")).strip().lower() or "unknown"
        grouped.setdefault(source, []).append(item)

    source_order = sorted(
        grouped.keys(),
        key=lambda src: (
            -_search_result_fact_priority(grouped[src][0]),
            src,
        ),
    )
    for source in source_order:
        grouped[source] = sorted(
            grouped[source],
            key=_search_result_fact_priority,
            reverse=True,
        )

    selected: List[Dict[str, Any]] = []
    cursor = 0
    while len(selected) < limit:
        progressed = False
        for source in source_order:
            items = grouped.get(source, [])
            if cursor >= len(items):
                continue
            selected.append(items[cursor])
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
        cursor += 1
    return selected


def _heuristic_facts_from_search_results(
    search_results: List[Dict[str, Any]],
    *,
    limit: int = 12,
) -> List[Dict[str, Any]]:
    facts: List[Dict[str, Any]] = []
    seen = set()
    candidates = _balanced_fact_candidates(
        search_results,
        limit=max(limit * 3, min(len(search_results), 36)),
    )
    for item in candidates:
        source = str(item.get("source", "")).strip().lower()
        title = str(item.get("title", "")).strip()
        content = str(item.get("content", "")).strip().replace("\n", " ")
        metadata = dict(item.get("metadata", {}) or {})
        if not title and not content:
            continue

        year = _year_from_metadata(metadata)
        snippet = _first_sentence(content, max_len=180)
        if source == "github":
            stars = metadata.get("stars")
            language = str(metadata.get("language", "")).strip()
            stars_text = f", {stars} stars" if stars not in (None, "") else ""
            language_text = f", language={language}" if language else ""
            base = title or "GitHub repository"
            detail = snippet or _compact_text(content, max_len=160)
            claim = (
                f"Repository {base}{stars_text}{language_text} provides implementation details: {detail}"
            )
        elif source in {"arxiv", "semantic_scholar"}:
            paper = title or "paper"
            year_text = f" ({year})" if year else ""
            detail = snippet or _compact_text(content, max_len=180)
            claim = f"Paper {paper}{year_text} reports: {detail}"
        elif source == "stackoverflow":
            claim = f"StackOverflow discussion '{title}' highlights: {snippet or _compact_text(content, max_len=170)}"
        elif source == "hackernews":
            claim = f"HackerNews thread '{title}' highlights: {snippet or _compact_text(content, max_len=170)}"
        elif source == "huggingface":
            downloads = metadata.get("downloads")
            typ = metadata.get("type")
            d_text = f", downloads={downloads}" if downloads not in (None, "") else ""
            t_text = f", type={typ}" if typ else ""
            claim = f"HuggingFace asset {title}{t_text}{d_text}: {snippet or _compact_text(content, max_len=170)}"
        else:
            claim = f"{title}. {snippet}".strip(". ").strip()

        claim = _compact_text(claim, max_len=280)
        if not claim:
            continue
        key = claim.lower()
        if key in seen:
            continue
        seen.add(key)

        source_type = _SOURCE_TYPE_MAP.get(source, "unknown")
        base_conf = {
            "paper": 0.72,
            "code": 0.65,
            "social": 0.56,
            "unknown": 0.5,
        }.get(source_type, 0.5)
        if len(claim) >= 90:
            base_conf += 0.03
        confidence = float(max(0.45, min(base_conf, 0.9)))

        evidence_id = str(item.get("id", "")).strip()
        fact = {
            "id": f"heur_fact_{len(facts) + 1}",
            "claim": claim[:280],
            "evidence": [evidence_id] if evidence_id else [],
            "confidence": confidence,
            "source_type": source_type,
            "category": _infer_fact_category(f"{title} {content}"),
        }
        facts.append(fact)
        if len(facts) >= limit:
            break
    return facts


def _fact_claim_key(value: Any) -> str:
    normalized = _compact_text(value, max_len=320).lower()
    return re.sub(r"[\W_]+", "", normalized)


def _normalize_fact_dict(raw: Dict[str, Any], *, fallback_idx: int) -> Optional[Dict[str, Any]]:
    claim = _compact_text(raw.get("claim", ""), max_len=280)
    if not claim:
        return None
    evidence = [
        str(item).strip()
        for item in list(raw.get("evidence") or [])
        if str(item).strip()
    ]
    evidence = list(dict.fromkeys(evidence))
    category = str(raw.get("category", "")).strip().lower() or _infer_fact_category(claim)
    source_type = str(raw.get("source_type", "")).strip().lower() or "unknown"
    confidence = max(0.35, min(0.98, _safe_float(raw.get("confidence"), 0.55)))
    fact_id = str(raw.get("id", "")).strip() or f"fact_auto_{fallback_idx}"
    return {
        "id": fact_id,
        "claim": claim,
        "evidence": evidence,
        "confidence": confidence,
        "source_type": source_type,
        "category": category,
    }


def _merge_facts_with_search_results(
    *,
    facts: List[Dict[str, Any]],
    search_results: List[Dict[str, Any]],
    min_total: int,
    max_total: int,
) -> List[Dict[str, Any]]:
    if max_total <= 0:
        return []

    required_categories = {"architecture", "performance", "training", "comparison", "limitation", "deployment"}
    normalized: List[Dict[str, Any]] = []
    seen_claims = set()

    def _append(raw: Dict[str, Any], *, fallback_idx: int) -> bool:
        item = _normalize_fact_dict(raw, fallback_idx=fallback_idx)
        if not item:
            return False
        key = _fact_claim_key(item.get("claim", ""))
        if not key or key in seen_claims:
            return False
        seen_claims.add(key)
        normalized.append(item)
        return True

    for idx, fact in enumerate(facts, start=1):
        _append(fact, fallback_idx=idx)

    supplemental = _heuristic_facts_from_search_results(
        search_results,
        limit=max(max_total * 2, min_total + 8),
    )

    covered = {str(item.get("category", "")).strip().lower() for item in normalized}
    missing = [category for category in sorted(required_categories) if category not in covered]
    next_idx = len(normalized) + 1

    if missing:
        for category in missing:
            matched = False
            for item in supplemental:
                if str(item.get("category", "")).strip().lower() != category:
                    continue
                if _append(item, fallback_idx=next_idx):
                    next_idx += 1
                    matched = True
                if matched:
                    break
            if len(normalized) >= max_total:
                break

    for item in supplemental:
        if len(normalized) >= max_total:
            break
        if len(normalized) >= min_total:
            break
        if _append(item, fallback_idx=next_idx):
            next_idx += 1

    if len(normalized) > max_total:
        def _fact_priority(entry: Dict[str, Any]) -> float:
            conf = _safe_float(entry.get("confidence"), 0.5)
            evidence_count = len(list(entry.get("evidence") or []))
            category = str(entry.get("category", "")).strip().lower()
            return conf + 0.08 * min(4, evidence_count) + (0.12 if category in required_categories else 0.0)

        normalized = sorted(normalized, key=_fact_priority, reverse=True)[:max_total]

    for idx, item in enumerate(normalized, start=1):
        if not str(item.get("id", "")).strip():
            item["id"] = f"fact_auto_{idx}"
    return normalized


def _ensure_generated_defaults(
    *,
    topic: str,
    generated: Dict[str, Any],
    facts: List[Dict[str, Any]],
    knowledge_gaps: List[str],
) -> Dict[str, Any]:
    output = dict(generated or {})
    category_order = ["architecture", "deployment", "performance", "comparison", "limitation", "training", "community"]
    category_title = {
        "architecture": "架构与核心机制",
        "deployment": "生产部署路径",
        "performance": "性能与容量指标",
        "comparison": "替代方案对比",
        "limitation": "风险与失效边界",
        "training": "训练与数据依赖",
        "community": "社区实践反馈",
    }
    category_visual = {
        "architecture": "system architecture diagram, request and tool call flow",
        "deployment": "production deployment topology, service mesh and runtime controls",
        "performance": "benchmark dashboard, latency and throughput charts",
        "comparison": "side-by-side comparison matrix with trade-off annotations",
        "limitation": "risk heatmap, attack surface graph, mitigation checklist",
        "training": "training pipeline and data lineage diagram",
        "community": "community usage trends and issue taxonomy board",
    }

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for fact in facts:
        category = str(fact.get("category", "other")).strip().lower() or "other"
        grouped.setdefault(category, []).append(fact)

    def _claim(item: Dict[str, Any]) -> str:
        return _compact_text(item.get("claim", ""), max_len=220)

    ordered_facts: List[Dict[str, Any]] = []
    for category in category_order:
        ordered_facts.extend(grouped.get(category, []))
    for category, items in grouped.items():
        if category not in category_order:
            ordered_facts.extend(items)

    top_claims = [_claim(item) for item in ordered_facts if _claim(item)]
    key_findings = top_claims[:6] or [f"{topic} 当前证据不足，建议补充工程实现与性能指标。"]

    if not output.get("timeline"):
        events: List[Dict[str, Any]] = []
        phase = 1
        for category in category_order:
            items = grouped.get(category, [])
            if not items:
                continue
            claim = _claim(items[0])
            if not claim:
                continue
            events.append(
                {
                    "date": f"Phase-{phase}",
                    "title": category_title.get(category, category),
                    "description": claim,
                    "importance": max(1, 6 - phase),
                    "source_refs": list(items[0].get("evidence", []) or []),
                }
            )
            phase += 1
            if phase > 6:
                break
        if not events:
            events.append(
                {
                    "date": "Phase-1",
                    "title": "核心结论",
                    "description": key_findings[0],
                    "importance": 3,
                    "source_refs": [],
                }
            )
        output["timeline"] = events

    if not output.get("one_pager"):
        strengths = [_claim(item) for item in grouped.get("architecture", [])[:2] + grouped.get("performance", [])[:1] if _claim(item)]
        weaknesses = [_claim(item) for item in grouped.get("limitation", [])[:3] if _claim(item)]
        deep_dive = [
            _claim(item)
            for item in (
                grouped.get("architecture", [])[:2]
                + grouped.get("deployment", [])[:1]
                + grouped.get("training", [])[:1]
            )
            if _claim(item)
        ]
        implementation_notes = [
            "定义上线 SLO 与错误预算，并将其映射到报警阈值。",
            "拆分关键调用链路（鉴权/路由/执行/回写），逐段做可观测性埋点。",
            "使用灰度发布 + 自动回滚条件，避免一次性全量切换。",
        ]
        if grouped.get("deployment"):
            first_deploy_claim = _claim(grouped["deployment"][0])
            if first_deploy_claim:
                implementation_notes.insert(0, first_deploy_claim)

        output["one_pager"] = {
            "title": f"{topic} One-Pager",
            "executive_summary": key_findings[0],
            "key_findings": key_findings,
            "metrics": {},
            "strengths": strengths[:3],
            "weaknesses": weaknesses[:3] or [gap for gap in knowledge_gaps[:2]],
            "technical_deep_dive": deep_dive[:4] or key_findings[:3],
            "implementation_notes": implementation_notes[:4],
            "risks_and_mitigations": weaknesses[:3]
            or ["证据覆盖不足 -> 增加跨来源交叉验证并补充可复现实验。"],
            "resources": [],
        }

    if not output.get("video_brief"):
        segments: List[Dict[str, Any]] = []
        for category in category_order:
            items = grouped.get(category, [])
            if not items:
                continue
            claims = [_claim(item) for item in items if _claim(item)]
            if not claims:
                continue
            segments.append(
                {
                    "title": category_title.get(category, category),
                    "content": claims[0],
                    "talking_points": claims[:3],
                    "duration_sec": 45,
                    "visual_prompt": category_visual.get(
                        category,
                        "technical explainer scene with architecture and metrics overlays",
                    ),
                }
            )
            if len(segments) >= 5:
                break

        if not segments:
            segments = [
                {
                    "title": "核心问题与边界",
                    "content": key_findings[0],
                    "talking_points": key_findings[:3],
                    "duration_sec": 45,
                    "visual_prompt": "technical architecture board and KPI dashboard",
                }
            ]

        output["video_brief"] = {
            "title": f"{topic} Video Brief",
            "duration_estimate": "3-6 minutes",
            "hook": f"{topic} 在生产环境中，哪些设计会直接决定可用性与成本？",
            "target_audience": "platform / research engineers",
            "visual_style": "evidence-driven technical explainer",
            "segments": segments,
            "conclusion": "优先固化可验证指标，再逐步放量。",
            "call_to_action": "按模块执行灰度验证，并记录回归指标。",
        }

    return output


def _result_snapshot_path(output_dir: Path) -> Path:
    return output_dir / _RESULT_SNAPSHOT_FILE


def _write_result_snapshot(output_dir: Path, payload: Dict[str, Any]) -> Path:
    snapshot_path = _result_snapshot_path(output_dir)
    snapshot_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return snapshot_path


def _read_result_snapshot(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _build_artifact_summary_text(topic: str, result: Dict[str, Any]) -> str:
    lines = [str(topic or "").strip()]
    one_pager = result.get("one_pager") or {}
    summary = str(one_pager.get("executive_summary", "")).strip()
    if summary:
        lines.append(summary)

    key_findings = [
        str(item).strip()
        for item in (one_pager.get("key_findings") or [])
        if str(item).strip()
    ]
    if key_findings:
        lines.extend(key_findings[:6])

    facts = result.get("facts") or []
    for fact in facts[:8]:
        claim = str((fact or {}).get("claim", "")).strip()
        if claim:
            lines.append(claim)

    return "\n".join([line for line in lines if line]).strip() or str(topic or "").strip()


def _video_artifact_exists(payload: Dict[str, Any]) -> bool:
    artifact = payload.get("video_artifact") or {}
    output_path = str(artifact.get("output_path") or "").strip()
    if not output_path:
        return False
    return Path(output_path).exists()


def _search_results_to_documents(search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    documents: List[Dict[str, Any]] = []
    for item in search_results:
        metadata = dict(item.get("metadata", {}) or {})
        metadata["source"] = str(item.get("source", "")).strip() or metadata.get("source", "unknown")
        url = str(item.get("url", "")).strip()
        if url:
            metadata["url"] = url
        documents.append(
            {
                "id": str(item.get("id", "")).strip(),
                "content": f"{item.get('title', '')}\n\n{item.get('content', '')}",
                "metadata": metadata,
                "type": "search_result",
            }
        )
    return documents


async def _warmup_knowledge_base_from_search_results(search_results: List[Dict[str, Any]]) -> None:
    if not search_results:
        return
    documents = _search_results_to_documents(search_results)
    await add_to_knowledge_base(documents)


def _load_video_flow(metadata_path: Path) -> Dict[str, Any]:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if isinstance(metadata, dict) and isinstance(metadata.get("flow"), dict):
            return metadata["flow"]
    except Exception:
        pass
    return {}


def _sync_video_brief_with_flow(
    *,
    video_brief: Optional[Dict[str, Any]],
    flow: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not video_brief:
        return video_brief

    timeline = list(flow.get("slide_timeline") or [])
    if not timeline:
        return video_brief

    normalized = dict(video_brief)
    segments = [dict(item) for item in list(normalized.get("segments") or []) if isinstance(item, dict)]
    if not segments:
        segments = []

    synced_segments: List[Dict[str, Any]] = []
    for idx, slide in enumerate(timeline):
        base = dict(segments[idx]) if idx < len(segments) else {}
        item = dict(base)
        item["duration_sec"] = max(1, int(slide.get("duration_sec", item.get("duration_sec", 0)) or 1))
        item["start_sec"] = max(0, int(slide.get("start_sec", 0) or 0))
        slide_title = str(slide.get("title", "")).strip() or f"Segment {idx + 1}"
        if not str(item.get("title", "")).strip():
            item["title"] = slide_title
        if not str(item.get("content", "")).strip():
            item["content"] = f"Technical walkthrough for {slide_title}."
        points = [str(p).strip() for p in list(item.get("talking_points") or []) if str(p).strip()]
        if not points:
            points = [slide_title]
        item["talking_points"] = points
        if not str(item.get("visual_prompt", "")).strip():
            item["visual_prompt"] = f"technical explainer visual for {slide_title}"
        synced_segments.append(item)

    normalized["segments"] = synced_segments
    total_duration = sum(max(1, int(item.get("duration_sec", 0) or 1)) for item in synced_segments)
    if total_duration > 0:
        minutes, seconds = divmod(total_duration, 60)
        normalized["duration_estimate"] = f"{minutes}m {seconds:02d}s"
    return normalized


async def _generate_video_artifact(
    *,
    topic: str,
    output_dir: Path,
    generated: Dict[str, Any],
    facts: List[Dict[str, Any]],
    search_results: List[Dict[str, Any]],
    video_generator: Optional[BaseVideoGenerator],
    progress_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    generator = video_generator or create_video_generator(provider="slidev")

    try:
        await _emit_progress(
            progress_callback,
            event="video_generation_started",
            topic=topic,
            output_dir=str(output_dir),
        )
        artifact = await generator.generate(
            topic=topic,
            out_dir=output_dir,
            video_brief=generated.get("video_brief"),
            one_pager=generated.get("one_pager"),
            facts=facts,
            search_results=search_results,
        )
        flow = _load_video_flow(Path(artifact.metadata_path))
        payload: Dict[str, Any] = {
            "provider": artifact.provider,
            "output_path": str(artifact.output_path),
            "metadata_path": str(artifact.metadata_path),
        }
        for key in ("narration_audio_path", "narration_script_path", "plan_md", "plan_json"):
            value = flow.get(key)
            if value:
                payload[key] = str(value)
        if isinstance(flow.get("slide_timeline"), list):
            payload["slide_timeline"] = flow.get("slide_timeline")
        if flow.get("estimated_duration_sec") is not None:
            payload["estimated_duration_sec"] = int(flow.get("estimated_duration_sec"))
        await _emit_progress(
            progress_callback,
            event="video_generation_completed",
            video_artifact=payload,
        )
        return payload, None
    except Exception as exc:
        message = str(exc)
        logger.warning(f"Video generation failed, keeping document outputs: {message}")
        await _emit_progress(
            progress_callback,
            event="video_generation_failed",
            error=message,
        )
        return None, message


def _timeline_event_count(timeline: Any) -> int:
    if isinstance(timeline, dict):
        return len((timeline.get("events") or []))
    return len(timeline or [])


def _normalize_generated_outputs(
    *,
    topic: str,
    generated: Dict[str, Any],
    facts: List[Dict[str, Any]],
    knowledge_gaps: List[str],
    search_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    output = _ensure_generated_defaults(
        topic=topic,
        generated=generated,
        facts=facts,
        knowledge_gaps=list(knowledge_gaps or []),
    )
    output["timeline"] = normalize_timeline_dates(
        topic=topic,
        timeline=output.get("timeline"),
        facts=facts,
        search_results=search_results,
    )
    output["one_pager"] = normalize_one_pager_content(
        topic=topic,
        one_pager=output.get("one_pager"),
        facts=facts,
        search_results=search_results,
    )
    output["one_pager"] = normalize_one_pager_resources(
        one_pager=output.get("one_pager"),
        facts=facts,
        search_results=search_results,
    )
    output["video_brief"] = normalize_video_brief(
        topic=topic,
        video_brief=output.get("video_brief"),
        facts=facts,
        search_results=search_results,
    )
    return output


async def _run_analysis_stage(
    *,
    topic: str,
    search_results: List[Dict[str, Any]],
    llm: BaseLLM,
    enable_knowledge_indexing: bool,
    analysis_timeout_sec: int,
    progress_callback: Optional[Callable[[Dict[str, Any]], Any]],
) -> Dict[str, Any]:
    await _emit_progress(
        progress_callback,
        event="analysis_started",
        topic=topic,
        search_results_count=len(search_results),
    )
    analyst = AnalystAgent(
        llm=llm,
        enable_knowledge_indexing=enable_knowledge_indexing,
    )
    try:
        analysis = await asyncio.wait_for(
            analyst.analyze(topic, search_results),
            timeout=max(10, int(analysis_timeout_sec)),
        )
    except Exception as exc:
        error_text = _exception_text(exc)
        fallback_facts = _heuristic_facts_from_search_results(search_results)
        await _emit_progress(
            progress_callback,
            event="analysis_failed",
            error=error_text,
            fallback_facts_count=len(fallback_facts),
        )
        analysis = {
            "facts": fallback_facts,
            "knowledge_gaps": [f"analysis_stage_failed: {error_text}"],
        }
        await _emit_progress(
            progress_callback,
            event="analysis_fallback_applied",
            facts_count=len(fallback_facts),
        )

    raw_facts = list(analysis.get("facts", []))
    target_min_facts = max(8, min(18, int(len(search_results) * 0.5) + 4))
    target_max_facts = max(target_min_facts, min(30, max(16, len(search_results) + 8)))
    facts = _merge_facts_with_search_results(
        facts=raw_facts,
        search_results=search_results,
        min_total=target_min_facts,
        max_total=target_max_facts,
    )
    analysis["facts"] = facts
    if len(facts) > len(raw_facts):
        await _emit_progress(
            progress_callback,
            event="analysis_facts_enriched",
            original_facts_count=len(raw_facts),
            enriched_facts_count=len(facts),
            added_facts=max(0, len(facts) - len(raw_facts)),
        )
    await _emit_progress(
        progress_callback,
        event="analysis_completed",
        facts_count=len(facts),
        knowledge_gaps_count=len(list(analysis.get("knowledge_gaps", []))),
    )
    return analysis


async def _run_content_generation_stage(
    *,
    topic: str,
    facts: List[Dict[str, Any]],
    knowledge_gaps: List[str],
    search_results: List[Dict[str, Any]],
    llm: BaseLLM,
    content_timeout_sec: int,
    progress_callback: Optional[Callable[[Dict[str, Any]], Any]],
) -> Dict[str, Any]:
    await _emit_progress(
        progress_callback,
        event="content_generation_started",
        facts_count=len(facts),
        search_results_count=len(search_results),
    )
    content_agent = ContentAgent(
        llm=llm,
        request_timeout_sec=max(10, min(int(content_timeout_sec), 60)),
    )
    try:
        generated = await content_agent.generate(
            topic,
            facts,
            search_results=search_results,
        )
    except Exception as exc:
        await _emit_progress(
            progress_callback,
            event="content_generation_failed",
            error=_exception_text(exc),
        )
        generated = {}

    normalized = _normalize_generated_outputs(
        topic=topic,
        generated=generated,
        facts=facts,
        knowledge_gaps=knowledge_gaps,
        search_results=search_results,
    )
    await _emit_progress(
        progress_callback,
        event="content_generation_completed",
        timeline_events=_timeline_event_count(normalized.get("timeline")),
    )
    return normalized


async def _export_documents_stage(
    *,
    topic: str,
    output_dir: Path,
    generated: Dict[str, Any],
    facts: List[Dict[str, Any]],
    search_results: List[Dict[str, Any]],
    progress_callback: Optional[Callable[[Dict[str, Any]], Any]],
) -> Dict[str, str]:
    await _emit_progress(
        progress_callback,
        event="documents_export_started",
        output_dir=str(output_dir),
    )
    written_files = export_research_outputs(
        output_dir,
        topic=topic,
        timeline=generated.get("timeline"),
        one_pager=generated.get("one_pager"),
        video_brief=generated.get("video_brief"),
        facts=facts,
        search_results=search_results,
        write_report=True,
    )
    payload = {k: str(v) for k, v in written_files.items()}
    await _emit_progress(
        progress_callback,
        event="documents_exported",
        output_dir=str(output_dir),
        written_files=payload,
    )
    return payload


async def _run_critic_stage(
    *,
    enabled: bool,
    critic_threshold: float,
    facts: List[Dict[str, Any]],
    search_results: List[Dict[str, Any]],
    one_pager: Optional[Dict[str, Any]],
    video_brief: Optional[Dict[str, Any]],
    knowledge_gaps: List[str],
    critic_timeout_sec: int,
    progress_callback: Optional[Callable[[Dict[str, Any]], Any]],
) -> tuple[Optional[Dict[str, Any]], Optional[bool], List[str]]:
    if not enabled:
        return None, None, []

    await _emit_progress(
        progress_callback,
        event="critic_started",
        threshold=critic_threshold,
    )
    critic = CriticAgent(quality_threshold=critic_threshold)
    try:
        critic_result = await asyncio.wait_for(
            critic.evaluate(
                facts=facts,
                search_results=search_results,
                one_pager=one_pager,
                video_brief=video_brief,
                knowledge_gaps=knowledge_gaps,
            ),
            timeout=max(8, int(critic_timeout_sec)),
        )
    except Exception as exc:
        await _emit_progress(
            progress_callback,
            event="critic_failed",
            error=_exception_text(exc),
        )
        return None, None, []

    quality_metrics = critic_result.get("quality_metrics")
    quality_gate_pass = bool(critic_result.get("pass", False))
    quality_recommendations = list(critic_result.get("recommendations", []))
    await _emit_progress(
        progress_callback,
        event="critic_completed",
        quality_metrics=quality_metrics,
        quality_gate_pass=quality_gate_pass,
        quality_recommendations=quality_recommendations,
    )
    return quality_metrics, quality_gate_pass, quality_recommendations


async def _refresh_documents_after_video_sync(
    *,
    topic: str,
    output_dir: Path,
    generated: Dict[str, Any],
    facts: List[Dict[str, Any]],
    search_results: List[Dict[str, Any]],
    video_artifact: Dict[str, Any],
    progress_callback: Optional[Callable[[Dict[str, Any]], Any]],
) -> Dict[str, str]:
    flow = _load_video_flow(Path(str(video_artifact.get("metadata_path", "")).strip()))
    synced_video_brief = _sync_video_brief_with_flow(
        video_brief=generated.get("video_brief"),
        flow=flow,
    )
    if not synced_video_brief:
        return {}

    generated["video_brief"] = synced_video_brief
    refreshed_written = export_research_outputs(
        output_dir,
        topic=topic,
        timeline=generated.get("timeline"),
        one_pager=generated.get("one_pager"),
        video_brief=generated.get("video_brief"),
        facts=facts,
        search_results=search_results,
        write_report=True,
    )
    payload = {k: str(v) for k, v in refreshed_written.items()}
    await _emit_progress(
        progress_callback,
        event="documents_reexported_after_video",
        output_dir=str(output_dir),
        written_files=payload,
    )
    return payload


async def run_research_from_search_results(
    *,
    topic: str,
    search_results: List[Dict[str, Any]],
    llm: Optional[BaseLLM] = None,
    out_dir: Optional[Path] = None,
    generate_video: bool = False,
    video_generator: Optional[BaseVideoGenerator] = None,
    enable_knowledge_indexing: bool = True,
    enable_critic_gate: bool = True,
    critic_threshold: float = 0.65,
    analysis_timeout_sec: int = 60,
    content_timeout_sec: int = 75,
    critic_timeout_sec: int = 30,
    progress_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
) -> Dict[str, Any]:
    """
    使用已准备好的搜索结果执行 Phase 3 + Phase 4。
    """
    owns_llm = llm is None
    llm = llm or get_llm()

    try:
        analysis = await _run_analysis_stage(
            topic=topic,
            search_results=search_results,
            llm=llm,
            enable_knowledge_indexing=enable_knowledge_indexing,
            analysis_timeout_sec=analysis_timeout_sec,
            progress_callback=progress_callback,
        )
        facts = list(analysis.get("facts", []))
        knowledge_gaps = list(analysis.get("knowledge_gaps", []) or [])

        generated = await _run_content_generation_stage(
            topic=topic,
            facts=facts,
            knowledge_gaps=knowledge_gaps,
            search_results=search_results,
            llm=llm,
            content_timeout_sec=content_timeout_sec,
            progress_callback=progress_callback,
        )

        output_dir = out_dir or Path("data") / "outputs" / (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{topic.replace(' ', '_')}"
        )
        written_files_payload = await _export_documents_stage(
            topic=topic,
            output_dir=output_dir,
            generated=generated,
            facts=facts,
            search_results=search_results,
            progress_callback=progress_callback,
        )

        depth = evaluate_output_depth(
            facts=facts,
            one_pager=generated.get("one_pager"),
            video_brief=generated.get("video_brief"),
        )

        quality_metrics, quality_gate_pass, quality_recommendations = await _run_critic_stage(
            enabled=enable_critic_gate,
            critic_threshold=critic_threshold,
            facts=facts,
            search_results=search_results,
            one_pager=generated.get("one_pager"),
            video_brief=generated.get("video_brief"),
            knowledge_gaps=knowledge_gaps,
            critic_timeout_sec=critic_timeout_sec,
            progress_callback=progress_callback,
        )

        video_artifact = None
        video_error = None
        if generate_video:
            video_artifact, video_error = await _generate_video_artifact(
                topic=topic,
                output_dir=output_dir,
                generated=generated,
                facts=facts,
                search_results=search_results,
                video_generator=video_generator,
                progress_callback=progress_callback,
            )
            if video_artifact:
                refreshed_payload = await _refresh_documents_after_video_sync(
                    topic=topic,
                    output_dir=output_dir,
                    generated=generated,
                    facts=facts,
                    search_results=search_results,
                    video_artifact=video_artifact,
                    progress_callback=progress_callback,
                )
                if refreshed_payload:
                    written_files_payload = refreshed_payload

        result_payload = {
            "topic": topic,
            "search_results_count": len(search_results),
            "search_results": search_results,
            "facts": facts,
            "knowledge_gaps": knowledge_gaps,
            "timeline": generated.get("timeline"),
            "one_pager": generated.get("one_pager"),
            "video_brief": generated.get("video_brief"),
            "depth_assessment": depth,
            "quality_metrics": quality_metrics,
            "quality_gate_pass": quality_gate_pass,
            "quality_recommendations": quality_recommendations,
            "output_dir": str(output_dir),
            "written_files": written_files_payload,
            "video_artifact": video_artifact,
            "video_error": video_error,
            "cache_hit": False,
        }
        await _emit_progress(
            progress_callback,
            event="phase4_completed",
            output_dir=result_payload.get("output_dir"),
            search_results_count=result_payload.get("search_results_count", 0),
            facts_count=len(facts),
            quality_gate_pass=result_payload.get("quality_gate_pass"),
        )
        return result_payload
    finally:
        if enable_knowledge_indexing:
            close_knowledge_base()
        if owns_llm:
            try:
                await llm.aclose()
            except Exception:
                logger.debug("LLM close skipped", exc_info=True)


def _default_planner_output(topic: str, reason: str = "planner skipped") -> Dict[str, Any]:
    return {
        "is_technical": True,
        "normalized_topic": topic,
        "query_rewrites": [topic],
        "research_questions": [],
        "search_plan": [],
        "reason": reason,
    }


def _build_blocked_payload(
    *,
    input_topic: str,
    planned_topic: str,
    planner_output: Dict[str, Any],
    reason: str,
    critic_threshold: float,
) -> Dict[str, Any]:
    quality = evaluate_research_quality(
        facts=[],
        search_results=[],
        one_pager=None,
        video_brief=None,
        knowledge_gaps=[reason],
        threshold=critic_threshold,
    )
    return {
        "topic": planned_topic,
        "input_topic": input_topic,
        "blocked": True,
        "blocked_reason": reason,
        "planner": planner_output,
        "search_results_count": 0,
        "search_results": [],
        "facts": [],
        "knowledge_gaps": [f"Planner blocked request: {reason}"],
        "timeline": None,
        "one_pager": None,
        "video_brief": None,
        "depth_assessment": evaluate_output_depth(
            facts=[],
            one_pager=None,
            video_brief=None,
        ),
        "quality_metrics": quality,
        "quality_gate_pass": False,
        "quality_recommendations": list(quality.get("recommendations", [])),
        "output_dir": None,
        "written_files": {},
        "video_artifact": None,
        "video_error": None,
        "aggregated_summary": {"total": 0},
        "search_trace": [],
        "search_coverage": {},
        "search_strategy": "blocked",
        "cache_hit": False,
    }


async def _run_planner_stage(
    *,
    topic: str,
    user_query: Optional[str],
    enable_planner: bool,
    planner_timeout_sec: int,
    llm: BaseLLM,
    critic_threshold: float,
    progress_callback: Optional[Callable[[Dict[str, Any]], Any]],
) -> tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]:
    planned_topic = topic
    planner_output = _default_planner_output(topic)
    if not enable_planner:
        return planned_topic, planner_output, None

    await _emit_progress(
        progress_callback,
        event="planner_started",
        topic=topic,
        user_query=user_query,
    )
    planner = PlannerAgent(llm=llm)
    try:
        planner_output = await asyncio.wait_for(
            planner.plan(topic=topic, user_query=user_query),
            timeout=max(5, int(planner_timeout_sec)),
        )
    except Exception as exc:
        await _emit_progress(
            progress_callback,
            event="planner_failed",
            error=str(exc),
        )
        planner_output = _default_planner_output(topic, reason=f"planner_failed: {exc}")

    normalized_topic = str(planner_output.get("normalized_topic") or "").strip()
    if normalized_topic:
        planned_topic = normalized_topic
    await _emit_progress(
        progress_callback,
        event="planner_completed",
        planner=planner_output,
        planned_topic=planned_topic,
    )

    if bool(planner_output.get("is_technical", True)):
        return planned_topic, planner_output, None

    reason = str(planner_output.get("reason") or "non-technical request").strip()
    blocked_payload = _build_blocked_payload(
        input_topic=topic,
        planned_topic=planned_topic,
        planner_output=planner_output,
        reason=reason,
        critic_threshold=critic_threshold,
    )
    await _emit_progress(
        progress_callback,
        event="research_blocked",
        blocked_reason=reason,
        planner=planner_output,
    )
    await _emit_progress(
        progress_callback,
        event="research_finished",
        output_dir=None,
        cache_hit=False,
        quality_gate_pass=False,
        blocked=True,
    )
    return planned_topic, planner_output, blocked_payload


async def _attempt_cache_reuse(
    *,
    planned_topic: str,
    input_topic: str,
    planner_output: Dict[str, Any],
    cache_cfg: Any,
    cache_min_quality: float,
    cache_require_quality_gate: bool,
    cache_min_facts: int,
    generate_video: bool,
    enable_knowledge_indexing: bool,
    progress_callback: Optional[Callable[[Dict[str, Any]], Any]],
) -> Optional[Dict[str, Any]]:
    store = None
    try:
        await _emit_progress(
            progress_callback,
            event="cache_lookup_started",
            planned_topic=planned_topic,
            similarity_threshold=_safe_float(cache_cfg.similarity_threshold, 0.82),
            top_k=max(1, int(cache_cfg.top_k)),
        )
        store = ResearchArtifactStore(collection_name=cache_cfg.collection_name)
        candidates = store.find_similar(
            query=planned_topic,
            score_threshold=_safe_float(cache_cfg.similarity_threshold, 0.82),
            top_k=max(1, int(cache_cfg.top_k)),
        )
        for candidate in candidates:
            if str(candidate.get("artifact_schema_version", "")).strip() != _ARTIFACT_SCHEMA_VERSION:
                continue
            if _safe_float(candidate.get("quality_score"), 0.0) < cache_min_quality:
                continue
            if cache_require_quality_gate and not bool(candidate.get("quality_gate_pass", False)):
                continue
            if (
                "search_results_count" in candidate
                and _safe_int(candidate.get("search_results_count"), 0) <= 0
            ):
                continue
            snapshot_path = Path(str(candidate.get("snapshot_path", "")).strip())
            if not snapshot_path.exists():
                continue

            cached = _read_result_snapshot(snapshot_path)
            if not cached:
                continue
            if cache_min_facts and len(cached.get("facts") or []) < cache_min_facts:
                continue
            if generate_video and bool(cache_cfg.require_video_for_video_request):
                if not _video_artifact_exists(cached):
                    continue

            cached_results = list(cached.get("search_results") or [])
            cached_facts = list(cached.get("facts") or [])
            cached["timeline"] = normalize_timeline_dates(
                topic=str(cached.get("topic") or planned_topic),
                timeline=cached.get("timeline"),
                facts=cached_facts,
                search_results=cached_results,
            )
            cached["one_pager"] = normalize_one_pager_content(
                topic=str(cached.get("topic") or planned_topic),
                one_pager=cached.get("one_pager"),
                facts=cached_facts,
                search_results=cached_results,
            )
            cached["one_pager"] = normalize_one_pager_resources(
                one_pager=cached.get("one_pager"),
                facts=cached_facts,
                search_results=cached_results,
            )
            cached["video_brief"] = normalize_video_brief(
                topic=str(cached.get("topic") or planned_topic),
                video_brief=cached.get("video_brief"),
                facts=cached_facts,
                search_results=cached_results,
            )
            artifact_meta_path = str((cached.get("video_artifact") or {}).get("metadata_path", "")).strip()
            if artifact_meta_path:
                flow = _load_video_flow(Path(artifact_meta_path))
                cached["video_brief"] = _sync_video_brief_with_flow(
                    video_brief=cached.get("video_brief"),
                    flow=flow,
                )

            output_dir_text = str(cached.get("output_dir", "")).strip()
            if output_dir_text:
                refreshed_written = export_research_outputs(
                    output_dir_text,
                    topic=str(cached.get("topic") or planned_topic),
                    timeline=cached.get("timeline"),
                    one_pager=cached.get("one_pager"),
                    video_brief=cached.get("video_brief"),
                    facts=cached_facts,
                    search_results=cached_results,
                    write_report=True,
                )
                cached["written_files"] = {k: str(v) for k, v in refreshed_written.items()}

            cached["input_topic"] = input_topic
            cached["topic"] = cached.get("topic") or planned_topic
            cached["planner"] = planner_output
            cached["cache_hit"] = True
            cached["cache_score"] = _safe_float(candidate.get("score"), 0.0)
            cached["cache_matched_topic"] = str(candidate.get("topic", "")).strip() or cached["topic"]
            cached["search_strategy"] = "cache_reuse"
            cached.setdefault("search_trace", [])
            cached.setdefault("search_coverage", {})
            cached.setdefault("aggregated_summary", {"total": cached.get("search_results_count", 0)})
            await _emit_progress(
                progress_callback,
                event="cache_hit",
                cache_score=cached["cache_score"],
                cache_matched_topic=cached["cache_matched_topic"],
                output_dir=cached.get("output_dir"),
            )
            if enable_knowledge_indexing:
                try:
                    await _warmup_knowledge_base_from_search_results(cached_results)
                except Exception as exc:
                    logger.warning(f"Cache hit KB warmup skipped: {exc}")
                finally:
                    close_knowledge_base()
            await _emit_progress(
                progress_callback,
                event="research_finished",
                output_dir=cached.get("output_dir"),
                cache_hit=True,
                quality_gate_pass=cached.get("quality_gate_pass"),
            )
            return cached

        await _emit_progress(
            progress_callback,
            event="cache_miss",
            planned_topic=planned_topic,
        )
        return None
    except Exception as exc:
        logger.warning(f"Research artifact cache lookup skipped: {exc}")
        await _emit_progress(
            progress_callback,
            event="cache_lookup_failed",
            error=str(exc),
        )
        return None
    finally:
        if store is not None:
            store.close()


async def _collect_search_results_stage(
    *,
    llm: BaseLLM,
    planned_topic: str,
    user_query: Optional[str],
    planner_output: Dict[str, Any],
    max_results_per_source: int,
    use_agentic_search: bool,
    resolved_sources: List[str],
    aggregator_kwargs: Dict[str, Any],
    show_progress: bool,
    search_max_iterations: int,
    search_tool_timeout_sec: int,
    react_thought_timeout_sec: int,
    search_time_budget_sec: int,
    search_stage_timeout_sec: int,
    seed_search_results: Optional[List[Dict[str, Any]]],
    progress_callback: Optional[Callable[[Dict[str, Any]], Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], str]:
    search_results: List[Dict[str, Any]] = []
    aggregated_summary: Dict[str, Any] = {}
    search_trace: List[Dict[str, Any]] = []
    search_coverage: Dict[str, Any] = {}
    search_strategy = "aggregator"
    seed_results = _deduplicate_search_results(list(seed_search_results or []))

    if use_agentic_search:
        await _emit_progress(
            progress_callback,
            event="search_started",
            strategy="react_agent",
            max_iterations=max(1, int(search_max_iterations)),
            tool_timeout_sec=max(3, int(search_tool_timeout_sec)),
            react_thought_timeout_sec=max(4, int(react_thought_timeout_sec)),
            time_budget_sec=max(20, int(search_time_budget_sec)),
            planned_topic=planned_topic,
        )
        search_agent = SearchAgent(
            llm=llm,
            max_iterations=max(1, int(search_max_iterations)),
            min_total_results=max(8, max_results_per_source * 2),
            allowed_sources=resolved_sources or None,
            progress_callback=progress_callback,
        )
        try:
            search_output = await asyncio.wait_for(
                search_agent.run(
                    {
                        "topic": planned_topic,
                        "user_query": user_query,
                        "query_rewrites": planner_output.get("query_rewrites", []),
                        "research_questions": planner_output.get("research_questions", []),
                        "search_plan": planner_output.get("search_plan", []),
                        "max_results_per_source": max_results_per_source,
                        "tool_timeout_sec": max(3, int(search_tool_timeout_sec)),
                        "react_thought_timeout_sec": max(4, int(react_thought_timeout_sec)),
                        "time_budget_sec": max(20, int(search_time_budget_sec)),
                    }
                ),
                timeout=max(20, int(search_stage_timeout_sec)),
            )
            search_results = list(search_output.get("search_results", []))
            search_trace = list(search_output.get("search_trace", []))
            search_coverage = dict(search_output.get("coverage", {}))
            search_strategy = str(search_output.get("strategy", "react_agent"))
            aggregated_summary = _summarize_search_results(search_results)
            disabled_tools = list(search_output.get("disabled_tools", []))
            await _emit_progress(
                progress_callback,
                event="search_completed",
                strategy=search_strategy,
                aggregated_summary=aggregated_summary,
                search_coverage=search_coverage,
                trace_steps=len(search_trace),
                disabled_tools=disabled_tools,
            )
        except Exception as exc:
            await _emit_progress(
                progress_callback,
                event="search_failed",
                error=str(exc),
                strategy="react_agent",
            )

    if not search_results:
        await _emit_progress(
            progress_callback,
            event="search_fallback_started",
            strategy="aggregator",
            planned_topic=planned_topic,
        )
        fallback_kwargs = dict(aggregator_kwargs)
        if resolved_sources:
            fallback_kwargs.update(_aggregator_kwargs_from_sources(resolved_sources))
        async with DataAggregator(**fallback_kwargs) as aggregator:
            aggregated = await aggregator.aggregate(
                topic=planned_topic,
                max_results_per_source=max_results_per_source,
                show_progress=show_progress,
            )
        search_results = aggregated_result_to_search_results(aggregated)
        aggregated_summary = aggregated.summary()
        if search_trace:
            search_strategy = f"{search_strategy}+aggregator_fallback"
        else:
            search_strategy = "aggregator"
            search_coverage = {}
        await _emit_progress(
            progress_callback,
            event="search_fallback_completed",
            strategy=search_strategy,
            aggregated_summary=aggregated_summary,
        )
    elif search_coverage.get("missing_dimensions") is not None and _should_enrich_search_results(
        search_results=search_results,
        resolved_sources=resolved_sources,
        max_results_per_source=max_results_per_source,
    ):
        covered_now = set(_covered_sources(search_results))
        missing_sources = [source for source in resolved_sources if source not in covered_now]
        enrich_sources = missing_sources or list(resolved_sources)
        await _emit_progress(
            progress_callback,
            event="search_enrichment_started",
            strategy=search_strategy,
            covered_sources=sorted(covered_now),
            enrich_sources=enrich_sources,
        )
        enrich_kwargs = dict(aggregator_kwargs)
        if enrich_sources:
            enrich_kwargs.update(_aggregator_kwargs_from_sources(enrich_sources))
        async with DataAggregator(**enrich_kwargs) as aggregator:
            enriched = await aggregator.aggregate(
                topic=planned_topic,
                max_results_per_source=max_results_per_source,
                show_progress=show_progress,
            )
        enriched_results = aggregated_result_to_search_results(enriched)
        merged_results = _deduplicate_search_results(search_results + enriched_results)
        added = max(0, len(merged_results) - len(search_results))
        search_results = merged_results
        aggregated_summary = _summarize_search_results(search_results)
        search_strategy = f"{search_strategy}+aggregator_enrich"
        search_coverage = {
            "result_count": len(search_results),
            "source_coverage": _covered_sources(search_results),
        }
        await _emit_progress(
            progress_callback,
            event="search_enrichment_completed",
            strategy=search_strategy,
            aggregated_summary=aggregated_summary,
            added_results=added,
            source_coverage=search_coverage.get("source_coverage", []),
        )

    if seed_results:
        before = len(search_results)
        search_results = _deduplicate_search_results(seed_results + search_results)
        merged = len(search_results)
        aggregated_summary = _summarize_search_results(search_results)
        search_strategy = f"{search_strategy}+seed_input" if search_strategy else "seed_input"
        await _emit_progress(
            progress_callback,
            event="seed_input_merged",
            seed_count=len(seed_results),
            added=max(0, merged - before),
            merged_total=merged,
        )

    return search_results, aggregated_summary, search_trace, search_coverage, search_strategy


async def _post_process_search_results(
    *,
    search_results: List[Dict[str, Any]],
    aggregated_summary: Dict[str, Any],
    planned_topic: str,
    query_rewrites: List[str],
    max_results_per_source: int,
    retrieval_cfg: Any,
    progress_callback: Optional[Callable[[Dict[str, Any]], Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    deep_enrichment_summary: Dict[str, Any] = {
        "attempted": 0,
        "enriched": 0,
        "sources": {},
    }
    filter_summary: Dict[str, Any] = {"removed": 0, "kept": len(search_results)}

    results = list(search_results)
    summary = dict(aggregated_summary)
    if results:
        results, filter_summary = _prune_off_topic_results(
            search_results=results,
            topic=planned_topic,
            query_rewrites=query_rewrites,
            min_keep=max(8, int(max_results_per_source) * 2),
        )
        if int(filter_summary.get("removed", 0)) > 0:
            summary = _summarize_search_results(results)
            await _emit_progress(
                progress_callback,
                event="search_filter_completed",
                removed=int(filter_summary.get("removed", 0)),
                kept=int(filter_summary.get("kept", len(results))),
                signals=filter_summary.get("signals", []),
                version_markers=filter_summary.get("version_markers", []),
                version_filtered=int(filter_summary.get("version_filtered", 0)),
                aggregated_summary=summary,
            )

    if retrieval_cfg.deep_enrichment_enabled and results:
        await _emit_progress(
            progress_callback,
            event="deep_enrichment_started",
            candidate_count=len(results),
            max_items_per_source=int(retrieval_cfg.deep_max_items_per_source),
            concurrency=int(retrieval_cfg.deep_concurrency),
        )
        try:
            results, deep_enrichment_summary = await enrich_search_results_deep(
                results,
                max_items_per_source=max(1, int(retrieval_cfg.deep_max_items_per_source)),
                concurrency=max(1, int(retrieval_cfg.deep_concurrency)),
                timeout_sec=max(4.0, float(retrieval_cfg.deep_request_timeout_sec)),
                max_pdf_pages=max(1, int(retrieval_cfg.deep_max_pdf_pages)),
                max_chars_per_item=max(1200, int(retrieval_cfg.deep_max_chars_per_item)),
            )
            summary = _summarize_search_results(results)
            await _emit_progress(
                progress_callback,
                event="deep_enrichment_completed",
                summary=deep_enrichment_summary,
                aggregated_summary=summary,
            )
        except Exception as exc:
            deep_enrichment_summary = {
                "attempted": 0,
                "enriched": 0,
                "sources": {},
                "error": str(exc),
            }
            await _emit_progress(
                progress_callback,
                event="deep_enrichment_failed",
                error=str(exc),
            )

    return results, summary, filter_summary, deep_enrichment_summary


async def run_research_end_to_end(
    *,
    topic: str,
    max_results_per_source: int = 10,
    llm: Optional[BaseLLM] = None,
    out_dir: Optional[Path] = None,
    show_progress: bool = True,
    generate_video: bool = False,
    video_generator: Optional[BaseVideoGenerator] = None,
    enable_knowledge_indexing: bool = True,
    aggregator_kwargs: Optional[Dict[str, Any]] = None,
    user_query: Optional[str] = None,
    allowed_sources: Optional[List[str]] = None,
    use_agentic_search: bool = True,
    enable_planner: bool = True,
    enable_critic_gate: bool = True,
    critic_threshold: float = 0.65,
    planner_timeout_sec: int = 20,
    search_max_iterations: int = 8,
    search_tool_timeout_sec: int = 12,
    react_thought_timeout_sec: int = 9,
    search_time_budget_sec: int = 90,
    search_stage_timeout_sec: int = 120,
    analysis_timeout_sec: int = 60,
    content_timeout_sec: int = 45,
    critic_timeout_sec: int = 25,
    allow_cache_hit: bool = True,
    seed_search_results: Optional[List[Dict[str, Any]]] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
) -> Dict[str, Any]:
    """
    主生产线入口：执行端到端研究工作流并产出可发布工件。

    数据流（主链路）：
    Planner -> Search -> Analysis -> Content -> Critic -> Export

    说明：
    - Search 阶段包含 agentic tool-calling 与聚合器 fallback。
    - Export 后可选执行视频生成、缓存快照与知识库索引。
    """
    owns_llm = llm is None
    llm = llm or get_llm()
    aggregator_kwargs = dict(aggregator_kwargs or {})

    async def _close_owned_llm() -> None:
        if not owns_llm:
            return
        try:
            await llm.aclose()
        except Exception:
            logger.debug("LLM close skipped", exc_info=True)
    await _emit_progress(
        progress_callback,
        event="research_started",
        input_topic=topic,
        generate_video=generate_video,
    )

    resolved_sources = _normalize_sources(allowed_sources)
    if not resolved_sources:
        resolved_sources = _sources_from_aggregator_kwargs(aggregator_kwargs)

    planned_topic, planner_output, blocked_payload = await _run_planner_stage(
        topic=topic,
        user_query=user_query,
        enable_planner=enable_planner,
        planner_timeout_sec=planner_timeout_sec,
        llm=llm,
        critic_threshold=critic_threshold,
        progress_callback=progress_callback,
    )
    if blocked_payload is not None:
        await _close_owned_llm()
        return blocked_payload

    cache_cfg = get_research_cache_settings()
    retrieval_cfg = get_retrieval_settings()
    cache_enabled = bool(cache_cfg.enabled)
    cache_min_quality = _safe_float(cache_cfg.min_quality_score, default=0.0)
    cache_require_quality_gate = bool(getattr(cache_cfg, "require_quality_gate_pass", False))
    cache_min_facts = max(0, _safe_int(getattr(cache_cfg, "min_facts_count", 0), 0))

    if cache_enabled and allow_cache_hit:
        cached = await _attempt_cache_reuse(
            planned_topic=planned_topic,
            input_topic=topic,
            planner_output=planner_output,
            cache_cfg=cache_cfg,
            cache_min_quality=cache_min_quality,
            cache_require_quality_gate=cache_require_quality_gate,
            cache_min_facts=cache_min_facts,
            generate_video=generate_video,
            enable_knowledge_indexing=enable_knowledge_indexing,
            progress_callback=progress_callback,
        )
        if cached is not None:
            await _close_owned_llm()
            return cached

    elif cache_enabled and not allow_cache_hit:
        await _emit_progress(
            progress_callback,
            event="cache_bypassed",
            planned_topic=planned_topic,
            reason="allow_cache_hit=false",
        )

    search_results, aggregated_summary, search_trace, search_coverage, search_strategy = (
        await _collect_search_results_stage(
            llm=llm,
            planned_topic=planned_topic,
            user_query=user_query,
            planner_output=planner_output,
            max_results_per_source=max_results_per_source,
            use_agentic_search=use_agentic_search,
            resolved_sources=resolved_sources,
            aggregator_kwargs=aggregator_kwargs,
            show_progress=show_progress,
            search_max_iterations=search_max_iterations,
            search_tool_timeout_sec=search_tool_timeout_sec,
            react_thought_timeout_sec=react_thought_timeout_sec,
            search_time_budget_sec=search_time_budget_sec,
            search_stage_timeout_sec=search_stage_timeout_sec,
            seed_search_results=seed_search_results,
            progress_callback=progress_callback,
        )
    )

    search_results, aggregated_summary, filter_summary, deep_enrichment_summary = (
        await _post_process_search_results(
            search_results=search_results,
            aggregated_summary=aggregated_summary,
            planned_topic=planned_topic,
            query_rewrites=list(planner_output.get("query_rewrites", []) or []),
            max_results_per_source=max_results_per_source,
            retrieval_cfg=retrieval_cfg,
            progress_callback=progress_callback,
        )
    )

    result = await run_research_from_search_results(
        topic=planned_topic,
        search_results=search_results,
        llm=llm,
        out_dir=out_dir,
        generate_video=generate_video,
        video_generator=video_generator,
        enable_knowledge_indexing=enable_knowledge_indexing,
        enable_critic_gate=enable_critic_gate,
        critic_threshold=critic_threshold,
        analysis_timeout_sec=max(10, int(analysis_timeout_sec)),
        content_timeout_sec=max(12, int(content_timeout_sec)),
        critic_timeout_sec=max(8, int(critic_timeout_sec)),
        progress_callback=progress_callback,
    )
    result["input_topic"] = topic
    result["search_results"] = search_results
    result["planner"] = planner_output
    result["aggregated_summary"] = aggregated_summary
    result["search_filter"] = filter_summary
    result["deep_enrichment"] = deep_enrichment_summary
    result["search_trace"] = search_trace
    result["search_coverage"] = search_coverage
    result["search_strategy"] = search_strategy
    result["cache_hit"] = False

    output_dir_text = str(result.get("output_dir") or "").strip()
    snapshot_path: Optional[Path] = None
    if output_dir_text:
        output_dir_path = Path(output_dir_text)
        try:
            output_dir_path.mkdir(parents=True, exist_ok=True)
            snapshot_path = _write_result_snapshot(output_dir_path, result)
            result["result_snapshot_path"] = str(snapshot_path)
        except Exception as exc:
            logger.warning(f"Failed to write result snapshot: {exc}")

    if cache_enabled and snapshot_path is not None:
        store = None
        try:
            store = ResearchArtifactStore(collection_name=cache_cfg.collection_name)
            manifest_path = str((result.get("written_files") or {}).get("manifest_json", "")).strip() or None
            video_output_path = str((result.get("video_artifact") or {}).get("output_path", "")).strip() or None
            store.index_artifact(
                topic=planned_topic,
                summary_text=_build_artifact_summary_text(planned_topic, result),
                output_dir=output_dir_text,
                snapshot_path=str(snapshot_path),
                manifest_path=manifest_path,
                video_output_path=video_output_path,
                quality_score=_safe_float((result.get("quality_metrics") or {}).get("overall_score"), 0.0),
                quality_gate_pass=bool(result.get("quality_gate_pass", False)),
                search_results_count=int(result.get("search_results_count", 0)),
                artifact_schema_version=_ARTIFACT_SCHEMA_VERSION,
            )
        except Exception as exc:
            logger.warning(f"Research artifact cache index skipped: {exc}")
        finally:
            if store is not None:
                store.close()
    await _emit_progress(
        progress_callback,
        event="research_finished",
        output_dir=result.get("output_dir"),
        cache_hit=False,
        quality_gate_pass=result.get("quality_gate_pass"),
    )
    await _close_owned_llm()
    return result


async def chat_over_knowledge_base(
    *,
    question: str,
    llm: Optional[BaseLLM] = None,
    top_k: int = 6,
    sources: Optional[List[str]] = None,
    year_filter: Optional[int] = None,
    use_hybrid: bool = True,
) -> Dict[str, Any]:
    """
    基于向量知识库进行对话问答。
    """
    try:
        chat_agent = ChatAgent(llm=llm or get_llm(), default_top_k=top_k)
        return await chat_agent.ask(
            question=question,
            top_k=top_k,
            sources=sources,
            year_filter=year_filter,
            use_hybrid=use_hybrid,
        )
    finally:
        close_knowledge_base()


async def chat_over_kb(
    *,
    question: str,
    llm: Optional[BaseLLM] = None,
    top_k: int = 6,
    sources: Optional[List[str]] = None,
    year_filter: Optional[int] = None,
    use_hybrid: bool = True,
) -> Dict[str, Any]:
    """chat_over_knowledge_base 的简写别名。"""
    return await chat_over_knowledge_base(
        question=question,
        llm=llm,
        top_k=top_k,
        sources=sources,
        year_filter=year_filter,
        use_hybrid=use_hybrid,
    )
