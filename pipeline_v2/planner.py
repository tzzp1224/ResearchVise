"""Deterministic retrieval planning for topic-first live recall."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Protocol

from pipeline_v2.topic_profile import TopicProfile


_DEFAULT_EXCLUDES = [
    "vscode theme",
    "color theme",
    "wallpaper",
    "icon pack",
    "ui template",
    "landing page",
]


@dataclass(frozen=True)
class PlanPhase:
    phase: str
    window: str
    expanded_queries: bool
    limit_multiplier: int


@dataclass(frozen=True)
class RetrievalPlan:
    topic: str
    profile_key: str
    base_queries: List[str]
    expanded_queries: List[str]
    source_weights: Dict[str, float]
    source_limits: Dict[str, Dict[str, int]]
    time_window_policy: List[PlanPhase]
    must_include_terms: List[str]
    must_exclude_terms: List[str]
    source_filters: Dict[str, Dict[str, List[str]]]
    query_buckets: Dict[str, List[str]]
    bucket_queries_by_source: Dict[str, Dict[str, List[str]]]

    def _expanded_for_phase(self, phase: str) -> bool:
        for rule in self.time_window_policy:
            if rule.phase == phase:
                return bool(rule.expanded_queries)
        return False

    def queries_for_phase(self, phase: str, *, source: Optional[str] = None) -> List[str]:
        if source:
            return self.source_queries_for_phase(source=source, phase=phase)
        expanded = self._expanded_for_phase(phase)
        return list(self.expanded_queries if expanded else self.base_queries)

    def bucket_queries_for_phase(self, phase: str, *, source: Optional[str] = None) -> Dict[str, List[str]]:
        expanded = self._expanded_for_phase(phase)
        if source:
            source_key = str(source or "").strip().lower()
            bucket_map = dict(self.bucket_queries_by_source.get(source_key) or {})
            payload: Dict[str, List[str]] = {}
            for bucket_name, queries in bucket_map.items():
                values = list(queries or [])
                if not values:
                    continue
                payload[bucket_name] = list(values if expanded else values[:1])
            return payload

        payload: Dict[str, List[str]] = {}
        for source_key in sorted(self.bucket_queries_by_source.keys()):
            bucket_map = self.bucket_queries_for_phase(phase, source=source_key)
            for bucket_name, queries in bucket_map.items():
                merged = list(payload.get(bucket_name) or []) + list(queries or [])
                payload[bucket_name] = _dedupe(merged)
        return payload

    def source_queries_for_phase(self, *, source: str, phase: str) -> List[str]:
        source_key = str(source or "").strip().lower()
        bucket_map = self.bucket_queries_for_phase(phase, source=source_key)
        flattened: List[str] = [self.topic] if self.topic else []
        for queries in bucket_map.values():
            flattened.extend(list(queries or []))
        if not flattened:
            flattened.extend(self.queries_for_phase(phase))
        return _dedupe(flattened)

    def bucket_for_query(self, query: str, *, source: Optional[str] = None) -> Optional[str]:
        token = str(query or "").strip().lower()
        if not token:
            return None

        if source:
            source_key = str(source or "").strip().lower()
            bucket_map = dict(self.bucket_queries_by_source.get(source_key) or {})
            for bucket_name, queries in bucket_map.items():
                if token in {str(item).strip().lower() for item in list(queries or []) if str(item).strip()}:
                    return str(bucket_name)

        for source_key, bucket_map in dict(self.bucket_queries_by_source or {}).items():
            _ = source_key
            for bucket_name, queries in dict(bucket_map or {}).items():
                if token in {str(item).strip().lower() for item in list(queries or []) if str(item).strip()}:
                    return str(bucket_name)
        return None

    def window_for_phase(self, phase: str, fallback: str = "today") -> str:
        for rule in self.time_window_policy:
            if rule.phase == phase:
                return str(rule.window or fallback)
        return fallback

    def source_limit_for_phase(self, *, source: str, phase: str, fallback: int) -> int:
        source_key = str(source or "").strip().lower()
        phase_key = str(phase or "base").strip().lower() or "base"
        source_map = dict(self.source_limits or {}).get(source_key) or {}
        value = int(source_map.get(phase_key, fallback) or fallback)
        return max(1, value)

    def model_dump(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "profile_key": self.profile_key,
            "base_queries": list(self.base_queries),
            "expanded_queries": list(self.expanded_queries),
            "source_weights": dict(self.source_weights),
            "source_limits": {str(k): dict(v) for k, v in dict(self.source_limits).items()},
            "time_window_policy": [
                {
                    "phase": rule.phase,
                    "window": rule.window,
                    "expanded_queries": bool(rule.expanded_queries),
                    "limit_multiplier": int(rule.limit_multiplier),
                }
                for rule in list(self.time_window_policy)
            ],
            "must_include_terms": list(self.must_include_terms),
            "must_exclude_terms": list(self.must_exclude_terms),
            "source_filters": {
                str(source): {str(name): list(values or []) for name, values in dict(filters or {}).items()}
                for source, filters in dict(self.source_filters or {}).items()
            },
            "query_buckets": {str(name): list(values or []) for name, values in dict(self.query_buckets or {}).items()},
            "bucket_queries_by_source": {
                str(source): {str(bucket): list(values or []) for bucket, values in dict(bucket_map or {}).items()}
                for source, bucket_map in dict(self.bucket_queries_by_source or {}).items()
            },
        }


class PlannerProtocol(Protocol):
    def plan(self, topic: str, *, time_window: Optional[str] = None) -> RetrievalPlan:
        ...


class ResearchPlanner:
    """Deterministic planner with topic profile + bucket decomposition."""

    def plan(self, topic: str, *, time_window: Optional[str] = None) -> RetrievalPlan:
        return _build_plan(topic, time_window=time_window)


class LLMPlanner(ResearchPlanner):
    """Stub planner slot for future LLM-backed planning."""

    def __init__(self, *, enabled: bool = False) -> None:
        self.enabled = bool(enabled)

    def plan(self, topic: str, *, time_window: Optional[str] = None) -> RetrievalPlan:
        # Placeholder: currently falls back to deterministic planner.
        return super().plan(topic, time_window=time_window)


def _dedupe(seq: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in list(seq or []):
        item = str(raw or "").strip()
        key = item.lower()
        if not item or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _normalize_window(time_window: Optional[str]) -> str:
    token = str(time_window or "").strip().lower()
    if token in {"today", "24h", "1d", "3d", "7d", "30d"}:
        return token
    if token.endswith("h") or token.endswith("d"):
        return token
    return "today"


def _build_plan(topic: str, *, time_window: Optional[str] = None) -> RetrievalPlan:
    topic_text = str(topic or "").strip()
    profile = TopicProfile.for_topic(topic_text)
    normalized_window = _normalize_window(time_window)

    source_weights = {
        "github": 0.45,
        "hackernews": 0.30,
        "huggingface": 0.25,
    }
    source_limits = {
        "github": {"base": 10, "limit_x2": 20, "window_3d": 24, "window_7d": 24, "query_expanded": 28},
        "hackernews": {"base": 8, "limit_x2": 16, "window_3d": 18, "window_7d": 18, "query_expanded": 20},
        "huggingface": {"base": 8, "limit_x2": 14, "window_3d": 16, "window_7d": 16, "query_expanded": 18},
    }

    policy = [
        PlanPhase(phase="base", window=normalized_window, expanded_queries=False, limit_multiplier=1),
        PlanPhase(phase="limit_x2", window=normalized_window, expanded_queries=False, limit_multiplier=2),
        PlanPhase(phase="window_3d", window="3d", expanded_queries=False, limit_multiplier=2),
        PlanPhase(phase="window_7d", window="7d", expanded_queries=False, limit_multiplier=2),
        PlanPhase(phase="query_expanded", window="7d", expanded_queries=True, limit_multiplier=2),
    ]

    bucket_terms = {bucket.name: list(bucket.terms) for bucket in list(profile.buckets or ())}
    bucket_queries_by_source: Dict[str, Dict[str, List[str]]] = {"github": {}, "hackernews": {}, "huggingface": {}}
    for source in list(bucket_queries_by_source.keys()):
        for bucket in list(profile.buckets or ()):
            queries = list(bucket.queries_for_source(source, expanded=True))
            if not queries:
                continue
            bucket_queries_by_source[source][bucket.name] = _dedupe(queries)

    planner = RetrievalPlan(
        topic=topic_text,
        profile_key=profile.key,
        base_queries=[],
        expanded_queries=[],
        source_weights=source_weights,
        source_limits=source_limits,
        time_window_policy=policy,
        must_include_terms=_dedupe(list(profile.hard_include_any)),
        must_exclude_terms=_dedupe(
            list(_DEFAULT_EXCLUDES)
            + [str(term) for term in list(profile.soft_penalty.keys())]
            + [
                str(term)
                for source_filters in list(profile.source_filters.values())
                for term in list(source_filters.get("must_exclude_any") or ())
            ]
        ),
        source_filters={
            str(source): {
                "must_include_any": _dedupe(list(filters.get("must_include_any") or [])),
                "must_exclude_any": _dedupe(list(filters.get("must_exclude_any") or [])),
            }
            for source, filters in dict(profile.source_filters or {}).items()
        },
        query_buckets=bucket_terms,
        bucket_queries_by_source=bucket_queries_by_source,
    )

    base_queries: List[str] = [topic_text] if topic_text else []
    expanded_queries: List[str] = [topic_text] if topic_text else []
    for source in ["github", "huggingface", "hackernews"]:
        base_queries.extend(planner.source_queries_for_phase(source=source, phase="base"))
        expanded_queries.extend(planner.source_queries_for_phase(source=source, phase="query_expanded"))

    object.__setattr__(planner, "base_queries", _dedupe([item for item in base_queries if len(item) >= 3]))
    object.__setattr__(planner, "expanded_queries", _dedupe([item for item in expanded_queries if len(item) >= 3]))

    if not planner.base_queries and topic_text:
        object.__setattr__(planner, "base_queries", [topic_text])
    if not planner.expanded_queries:
        object.__setattr__(planner, "expanded_queries", list(planner.base_queries))

    return planner


def build_retrieval_plan(topic: str, *, time_window: Optional[str] = None) -> RetrievalPlan:
    """Build deterministic query/source/window policy for topic-first retrieval."""
    return ResearchPlanner().plan(topic, time_window=time_window)
