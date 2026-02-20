"""Deterministic topic profile for hard relevance gating and bucket coverage."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Mapping, Sequence, Tuple


_STOPWORDS = {
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
    "latest",
    "today",
    "news",
}


def _dedupe(values: Sequence[str]) -> List[str]:
    output: List[str] = []
    seen = set()
    for raw in list(values or []):
        token = str(raw or "").strip()
        key = token.lower()
        if not token or key in seen:
            continue
        seen.add(key)
        output.append(token)
    return output


def _topic_tokens(topic: str) -> List[str]:
    parts = re.findall(r"[a-z0-9]+", str(topic or "").lower())
    return _dedupe([token for token in parts if token not in _STOPWORDS and len(token) >= 2])


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


@dataclass(frozen=True)
class TopicBucket:
    name: str
    terms: Tuple[str, ...]
    queries_by_source: Mapping[str, Tuple[str, ...]]

    def queries_for_source(self, source: str, *, expanded: bool) -> List[str]:
        key = str(source or "").strip().lower()
        values = list(self.queries_by_source.get(key) or ())
        if not values:
            values = list(self.queries_by_source.get("default") or ())
        if not values:
            return []
        if expanded:
            return _dedupe(values)
        return _dedupe(values[:1])


@dataclass(frozen=True)
class TopicProfile:
    topic: str
    key: str
    hard_include_any: Tuple[str, ...]
    soft_boost: Mapping[str, float]
    soft_penalty: Mapping[str, float]
    source_filters: Mapping[str, Mapping[str, Tuple[str, ...]]]
    buckets: Tuple[TopicBucket, ...]
    requires_hard_gate: bool = False
    high_value_terms: Tuple[str, ...] = ()
    generic_terms: Tuple[str, ...] = ()
    evidence_terms: Tuple[str, ...] = ()
    hot_new_priority_terms: Tuple[str, ...] = ()
    infra_dominance_terms: Tuple[str, ...] = ()

    @property
    def minimum_bucket_coverage(self) -> int:
        return 2 if len(self.buckets) >= 2 else 1

    def matched_hard_terms(self, text: str) -> List[str]:
        return [term for term in list(self.hard_include_any or ()) if _contains_term(text, term)]

    def hard_match_pass(self, text: str) -> bool:
        if not self.requires_hard_gate:
            return True
        if not self.hard_include_any:
            return True
        hard_hits = self.matched_hard_terms(text)
        if self.key != "ai_agent":
            return len(hard_hits) > 0

        high_value_hits = self.matched_high_value_terms(text)
        high_value_count = len(
            {str(value).strip().lower() for value in list(high_value_hits or []) if str(value).strip()}
        )
        if high_value_count >= 2:
            return True
        if high_value_count >= 1 and self.has_evidence_signal(text):
            return True
        return False

    def matched_high_value_terms(self, text: str) -> List[str]:
        return [
            term
            for term in list(self.high_value_terms or ())
            if _contains_term(text, term) and not _contains_negated_term(text, term)
        ]

    def has_evidence_signal(self, text: str) -> bool:
        if not self.evidence_terms:
            return False
        return any(
            _contains_term(text, term) and not _contains_negated_term(text, term)
            for term in list(self.evidence_terms or ())
        )

    def matched_soft_boost_terms(self, text: str) -> List[str]:
        matches: List[str] = []
        for term in list(self.soft_boost.keys()):
            if _contains_term(text, term):
                matches.append(term)
        return _dedupe(matches)

    def matched_soft_penalty_terms(self, text: str) -> List[str]:
        matches: List[str] = []
        for term in list(self.soft_penalty.keys()):
            if _contains_term(text, term):
                matches.append(term)
        return _dedupe(matches)

    def soft_adjustment(self, text: str) -> Tuple[float, float, List[str], List[str]]:
        boost_terms = self.matched_soft_boost_terms(text)
        penalty_terms = self.matched_soft_penalty_terms(text)
        boost = sum(float(self.soft_boost.get(term, 0.0) or 0.0) for term in boost_terms)
        penalty = sum(float(self.soft_penalty.get(term, 0.0) or 0.0) for term in penalty_terms)
        return min(0.3, boost), min(0.4, penalty), boost_terms, penalty_terms

    def bucket_hits(self, text: str) -> List[str]:
        payload = str(text or "")
        hits: List[str] = []
        for bucket in list(self.buckets or ()):  # deterministic order
            if any(_contains_term(payload, term) for term in list(bucket.terms or ())):
                hits.append(bucket.name)
        return _dedupe(hits)

    def bucket_for_query(self, query: str, *, source: str | None = None) -> str | None:
        token = str(query or "").strip().lower()
        if not token:
            return None
        source_key = str(source or "").strip().lower()
        for bucket in list(self.buckets or ()):
            sources = [source_key] if source_key else list({*bucket.queries_by_source.keys(), "default"})
            for item_source in sources:
                for candidate in list(bucket.queries_by_source.get(item_source) or ()):  # type: ignore[arg-type]
                    if token == str(candidate).strip().lower():
                        return bucket.name
        return None

    def source_bucket_queries(self, source: str, *, expanded: bool) -> Dict[str, List[str]]:
        payload: Dict[str, List[str]] = {}
        for bucket in list(self.buckets or ()):
            queries = bucket.queries_for_source(source, expanded=expanded)
            if queries:
                payload[bucket.name] = _dedupe(queries)
        return payload

    def source_queries(self, source: str, *, expanded: bool) -> List[str]:
        bucket_map = self.source_bucket_queries(source, expanded=expanded)
        flattened: List[str] = []
        for values in bucket_map.values():
            flattened.extend(list(values or []))
        if self.topic:
            flattened.insert(0, self.topic)
        return _dedupe(flattened)

    @classmethod
    def for_topic(cls, topic: str) -> "TopicProfile":
        raw_topic = str(topic or "").strip()
        lowered = raw_topic.lower()
        tokens = _topic_tokens(raw_topic)

        is_agent = any(
            token in lowered
            for token in (
                "agent",
                "agentic",
                "multi-agent",
                "copilot",
                "assistant",
                "orchestration",
                "tool calling",
            )
        )

        if is_agent:
            buckets: Tuple[TopicBucket, ...] = (
                TopicBucket(
                    name="Frameworks",
                    terms=("langgraph", "autogen", "crewai", "semantic kernel", "agent framework"),
                    queries_by_source={
                        "github": (
                            "langgraph agent framework",
                            "autogen multi-agent framework",
                            "crewai agent framework",
                            "semantic kernel agent runtime",
                        ),
                        "huggingface": (
                            "langgraph agent framework",
                            "autogen agent toolkit",
                            "crewai agents",
                        ),
                        "hackernews": (
                            "langgraph in production",
                            "autogen multi-agent lessons",
                        ),
                    },
                ),
                TopicBucket(
                    name="Protocols & tool use",
                    terms=("mcp", "model context protocol", "tool calling", "function calling", "tool use"),
                    queries_by_source={
                        "github": (
                            "model context protocol mcp tool calling",
                            "function calling agent runtime",
                        ),
                        "huggingface": (
                            "tool calling function calling agent",
                            "mcp model context protocol",
                        ),
                        "hackernews": (
                            "mcp tool calling",
                            "function calling agents",
                        ),
                    },
                ),
                TopicBucket(
                    name="Ops & runtime",
                    terms=("agent runtime", "orchestration", "workflow automation", "control plane", "runbook"),
                    queries_by_source={
                        "github": (
                            "agent runtime orchestration",
                            "workflow automation agent control plane",
                        ),
                        "huggingface": (
                            "agent runtime workflow automation",
                            "agent orchestration runtime",
                        ),
                        "hackernews": (
                            "agent runtime orchestration lessons",
                            "workflow automation agent incidents",
                        ),
                    },
                ),
                TopicBucket(
                    name="Evaluation",
                    terms=("agent eval", "agent benchmark", "inspect", "trace", "evaluation"),
                    queries_by_source={
                        "github": (
                            "agent eval benchmark inspect",
                            "agent tracing evaluation",
                        ),
                        "huggingface": (
                            "agent eval benchmark",
                            "agent inspect dataset",
                        ),
                        "hackernews": (
                            "agent eval benchmark discussion",
                            "agent inspect failures",
                        ),
                    },
                ),
            )
            return cls(
                topic=raw_topic,
                key="ai_agent",
                hard_include_any=(
                    "agent",
                    "agentic",
                    "multi-agent",
                    "orchestration",
                    "tool calling",
                    "function calling",
                    "mcp",
                    "langgraph",
                    "autogen",
                    "crewai",
                    "agent framework",
                    "agent runtime",
                ),
                soft_boost={
                    "mcp": 0.10,
                    "tool calling": 0.08,
                    "function calling": 0.08,
                    "orchestration": 0.08,
                    "agent runtime": 0.07,
                    "langgraph": 0.08,
                    "autogen": 0.08,
                    "crewai": 0.08,
                    "agent eval": 0.07,
                    "benchmark": 0.04,
                },
                soft_penalty={
                    "clip": 0.16,
                    "vit": 0.16,
                    "vision transformer": 0.16,
                    "vision-language": 0.14,
                    "diffusion": 0.12,
                    "image classification": 0.16,
                    "text-to-image": 0.10,
                    "embedding": 0.10,
                    "retriever": 0.14,
                    "retrieval": 0.10,
                    "reranker": 0.14,
                    "colbert": 0.18,
                    "foundation model": 0.08,
                    "reasoning model": 0.08,
                    "vision-language model": 0.16,
                    "vl model": 0.16,
                },
                source_filters={
                    "github": {
                        "must_include_any": (
                            "agent",
                            "orchestration",
                            "tool calling",
                            "mcp",
                            "langgraph",
                            "autogen",
                            "crewai",
                        ),
                        "must_exclude_any": (
                            "vscode theme",
                            "ui theme",
                            "wallpaper",
                            "icon pack",
                        ),
                    },
                    "huggingface": {
                        "must_include_any": (
                            "agent",
                            "tool calling",
                            "mcp",
                            "agent eval",
                            "langgraph",
                            "autogen",
                            "crewai",
                        ),
                        "must_exclude_any": (
                            "clip",
                            "vit",
                            "diffusion",
                            "image classification",
                            "text-to-image",
                            "vision-language",
                            "vision model",
                            "embedding model",
                            "retriever",
                            "reranker",
                            "colbert",
                        ),
                    },
                    "hackernews": {
                        "must_include_any": (
                            "agent",
                            "orchestration",
                            "tool calling",
                            "mcp",
                            "function calling",
                        ),
                        "must_exclude_any": (
                            "vscode theme",
                            "css theme",
                            "ui showcase",
                        ),
                    },
                },
                buckets=buckets,
                requires_hard_gate=True,
                high_value_terms=(
                    "mcp",
                    "model context protocol",
                    "tool calling",
                    "function calling",
                    "orchestration",
                    "agent runtime",
                    "langgraph",
                    "autogen",
                    "crewai",
                    "agent framework",
                    "agent eval",
                    "agent benchmark",
                ),
                generic_terms=("agent", "agents", "assistant", "copilot", "evaluation"),
                evidence_terms=(
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
                    "runtime",
                    "workflow",
                ),
                hot_new_priority_terms=(
                    "vertical agent",
                    "agent app",
                    "mcp server",
                    "browser agent",
                    "computer use",
                    "demo",
                    "tool calling",
                    "agent eval",
                ),
                infra_dominance_terms=(
                    "langchain",
                    "langgraph",
                    "autogen",
                    "crewai",
                    "llamaindex",
                    "agent framework",
                    "agent sdk",
                ),
            )

        core_terms = tuple(_dedupe(tokens[:6] or ([raw_topic] if raw_topic else [])))
        bucket = TopicBucket(
            name="Core",
            terms=core_terms,
            queries_by_source={
                "default": tuple(_dedupe([raw_topic] + [" ".join(tokens[:2]).strip(), " ".join(tokens[:3]).strip()])),
            },
        )
        include_terms = tuple(core_terms)
        return cls(
            topic=raw_topic,
            key="generic",
            hard_include_any=include_terms,
            soft_boost={},
            soft_penalty={},
            source_filters={
                "github": {"must_include_any": include_terms, "must_exclude_any": ()},
                "huggingface": {"must_include_any": include_terms, "must_exclude_any": ()},
                "hackernews": {"must_include_any": include_terms, "must_exclude_any": ()},
            },
            buckets=(bucket,),
            requires_hard_gate=False,
            high_value_terms=(),
            generic_terms=(),
            evidence_terms=(),
        )
