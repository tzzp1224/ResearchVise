"""Deterministic evidence audit for ranked candidates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
import re
from typing import Any, Dict, Iterable, List, Mapping, Protocol, Sequence
from urllib.parse import urlparse

from core import NormalizedItem
from pipeline_v2.sanitize import canonicalize_url, is_allowed_citation_url
from pipeline_v2.topic_profile import TopicProfile


VERDICT_PASS = "pass"
VERDICT_DOWNGRADE = "downgrade"
VERDICT_REJECT = "reject"

_TECH_TOKENS = (
    "cli",
    "api",
    "sdk",
    "mcp",
    "tool calling",
    "orchestration",
    "workflow",
    "deploy",
    "rollback",
    "benchmark",
    "quickstart",
    "usage",
)
_COMMAND_RE = re.compile(r"\b(pip\s+install|npm\s+i|npm\s+install|curl\s+|docker\s+run|python\s+\S+)\b", re.IGNORECASE)
_HYPE_RE = re.compile(
    r"\b(best|most|revolutionary|amazing|ultimate|world[- ]class|ever built|game[- ]changing)\b",
    re.IGNORECASE,
)
_CV_OFFTOPIC_TOKENS = (
    "clip",
    "vit",
    "vision transformer",
    "vision-language",
    "diffusion",
    "image classification",
    "text-to-image",
    "embedding",
    "retriever",
    "reranker",
    "colbert",
)
_AGENT_HIGH_VALUE_TOKENS = (
    "mcp",
    "tool calling",
    "function calling",
    "orchestration",
    "langgraph",
    "autogen",
    "crewai",
    "agent eval",
    "benchmark",
    "computer use",
    "browser agent",
    "agent runtime",
    "agent memory",
)
_AGENT_GENERIC_TOKENS = ("agent", "agents", "assistant", "copilot")
_HANDBOOK_TOKENS = (
    "handbook",
    "curated",
    "resources",
    "awesome list",
    "roadmap",
    "learning path",
)
_AGENT_RUNTIME_TOOLING_TOKENS = (
    "agent runtime",
    "framework",
    "orchestration",
    "tool calling",
    "function calling",
    "mcp",
    "langgraph",
    "autogen",
    "crewai",
    "sdk",
    "cli",
    "tooling",
)


@dataclass
class AuditRecord:
    item_id: str
    title: str
    source: str
    rank: int
    verdict: str
    reasons: List[str]
    used_evidence_urls: List[str]
    evidence_domains: List[str]
    citation_duplicate_prefix_ratio: float
    evidence_links_quality: int
    body_len: int
    min_body_len: int
    publish_or_update_time: str
    machine_action: Dict[str, str]
    evidence_target_alignment: float = 0.0
    evidence_alignment_hits: int = 0
    evidence_alignment_total: int = 0
    link_heavy: bool = False

    def model_dump(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "title": self.title,
            "source": self.source,
            "rank": self.rank,
            "verdict": self.verdict,
            "reasons": list(self.reasons),
            "used_evidence_urls": list(self.used_evidence_urls),
            "evidence_domains": list(self.evidence_domains),
            "citation_duplicate_prefix_ratio": round(float(self.citation_duplicate_prefix_ratio), 4),
            "evidence_links_quality": int(self.evidence_links_quality),
            "body_len": int(self.body_len),
            "min_body_len": int(self.min_body_len),
            "publish_or_update_time": self.publish_or_update_time or "",
            "machine_action": dict(self.machine_action or {}),
            "evidence_target_alignment": round(float(self.evidence_target_alignment), 4),
            "evidence_alignment_hits": int(self.evidence_alignment_hits),
            "evidence_alignment_total": int(self.evidence_alignment_total),
            "link_heavy": bool(self.link_heavy),
        }


@dataclass
class AuditSelection:
    selected_rows: List[Any]
    records: List[AuditRecord]

    def by_item_id(self) -> Dict[str, AuditRecord]:
        return {record.item_id: record for record in list(self.records)}

    def report_payload(self, *, topic: str | None, top_k: int, selected_phase: str | None = None) -> Dict[str, Any]:
        selected_ids = [str(getattr(row.item, "id", "")) for row in list(self.selected_rows or []) if getattr(row, "item", None)]
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "topic": str(topic or "").strip() or None,
            "requested_top_k": int(top_k),
            "selected_phase": str(selected_phase or "").strip() or None,
            "final_top_item_ids": selected_ids,
            "records": [record.model_dump() for record in list(self.records or [])],
        }


def _collect_evidence_urls(item: NormalizedItem) -> List[str]:
    def _is_low_value_evidence(url: str) -> bool:
        parsed = urlparse(str(url or ""))
        host = str(parsed.netloc or "").strip().lower()
        path = str(parsed.path or "").strip().lower()
        if host == "github.com":
            if path.startswith("/user-attachments/"):
                return True
            if path.endswith(".git"):
                return True
            if "/assets/" in path and "/releases/" not in path:
                return True
        if host in {"proxy.example.com"}:
            return True
        return False

    def _evidence_rank(url: str, *, source_hint: str) -> float:
        parsed = urlparse(str(url or ""))
        host = str(parsed.netloc or "").strip().lower()
        path = str(parsed.path or "").strip().lower()
        source = str(source_hint or "").strip().lower()

        score = 0.0
        if host in {"arxiv.org", "openreview.net", "docs.python.org"}:
            score += 10.0
        if host in {"github.com"}:
            score += 8.0
            if "/releases/" in path:
                score += 2.0
            if "/wiki" in path or "/issues/" in path:
                score += 1.2
        if host.endswith("huggingface.co"):
            score += 8.0
            if "/datasets/" in path or "/models/" in path:
                score += 1.2
        if "docs" in host or "/docs" in path or "/documentation" in path:
            score += 7.0
        if host in {"news.ycombinator.com"} and "item" in parsed.query:
            score += 6.0

        if source == "github" and host == "github.com":
            score += 1.0
        if source == "huggingface" and host.endswith("huggingface.co"):
            score += 1.0

        if host in {
            "x.com",
            "twitter.com",
            "mobile.twitter.com",
            "xiaohongshu.com",
            "www.xiaohongshu.com",
            "weibo.com",
            "www.weibo.com",
            "t.co",
            "refactoringenglish.com",
        }:
            score -= 4.5
        if _is_low_value_evidence(url):
            score -= 6.0
        return score

    urls: List[str] = []
    metadata = dict(item.metadata or {})
    for raw in list(metadata.get("evidence_links") or []):
        token = canonicalize_url(str(raw or ""))
        if token and is_allowed_citation_url(token) and not _is_low_value_evidence(token) and token not in urls:
            urls.append(token)
    for citation in list(item.citations or []):
        token = canonicalize_url(str(citation.url or ""))
        if token and is_allowed_citation_url(token) and not _is_low_value_evidence(token) and token not in urls:
            urls.append(token)
    item_url = canonicalize_url(str(item.url or ""))
    if item_url and is_allowed_citation_url(item_url) and not _is_low_value_evidence(item_url) and item_url not in urls:
        urls.append(item_url)
    ranked = sorted(
        list(urls),
        key=lambda token: (_evidence_rank(token, source_hint=str(item.source or "")), token),
        reverse=True,
    )
    return ranked[:12]


def _source_min_body_len(source: str) -> int:
    key = str(source or "").strip().lower()
    if key in {"github", "huggingface"}:
        return 400
    if key in {"hackernews", "web_article"}:
        return 600
    return 500


def _citation_duplicate_ratio(item: NormalizedItem) -> float:
    prefixes: List[str] = []
    domains: List[str] = []
    path_roots: List[str] = []
    for citation in list(item.citations or []):
        snippet = re.sub(r"\s+", " ", str(citation.snippet or citation.title or "")).strip().lower()
        if not snippet:
            continue
        snippet_tokens = [token for token in re.findall(r"[a-z0-9]+", snippet) if len(token) >= 3]
        prefix = " ".join(snippet_tokens[:8]) if snippet_tokens else snippet[:48]
        prefixes.append(prefix)
        url = canonicalize_url(str(citation.url or "").strip())
        parsed = urlparse(url) if url else urlparse("")
        host = str(parsed.netloc or "").strip().lower()
        path = str(parsed.path or "").strip().lower()
        if host:
            domains.append(host)
        path_parts = [part for part in path.split("/") if part]
        path_root = "/".join(path_parts[:2]) if path_parts else ""
        if path_root:
            path_roots.append(path_root)
    if len(prefixes) < 2:
        return 0.0

    unique_prefixes = len(set(prefixes))
    ratio = 1.0 - float(unique_prefixes) / float(max(1, len(prefixes)))
    domain_diversity = float(len(set(domains))) / float(max(1, len(domains)))
    root_diversity = float(len(set(path_roots))) / float(max(1, len(path_roots)))
    # Keep duplicate detection strong for same-entity link lists, but soften when
    # snippets come from clearly distinct domains/pages.
    if domain_diversity >= 0.7 and root_diversity >= 0.7:
        ratio *= 0.8
    elif domain_diversity >= 0.5 and root_diversity >= 0.5:
        ratio *= 0.9
    return max(0.0, min(1.0, ratio))


def _looks_marketing(text: str) -> bool:
    payload = str(text or "").strip()
    if not payload:
        return False
    return bool(_HYPE_RE.search(payload))


def _has_verifiable_point(text: str) -> bool:
    value = str(text or "").strip().lower()
    if not value:
        return False
    if _COMMAND_RE.search(value):
        return True
    if any(token in value for token in _TECH_TOKENS):
        return True
    if re.search(r"\b\d+(\.\d+)?(%|ms|s|x)?\b", value):
        return True
    return False


def _domains(urls: Iterable[str]) -> List[str]:
    out: List[str] = []
    for url in list(urls or []):
        host = str(urlparse(str(url or "")).netloc or "").strip().lower()
        if host.startswith("www."):
            host = host[4:]
        if host and host not in out:
            out.append(host)
    return out


def _repo_self_only(urls: Sequence[str]) -> bool:
    normalized = [canonicalize_url(str(url or "")) for url in list(urls or []) if str(url or "").strip()]
    domains = _domains(normalized)
    if len(domains) != 1:
        return False
    host = domains[0]
    if host not in {"github.com", "huggingface.co"}:
        return len({str(urlparse(url).path or "") for url in normalized}) <= 1

    paths = [str(urlparse(url).path or "").lower() for url in normalized]
    has_release = any("/releases/" in path for path in paths)
    has_docs = any("/docs" in path or "/wiki" in path for path in paths)
    has_demo = any("demo" in path for path in paths)
    has_path_diversity = len(set(paths)) >= 2
    return not (has_release or has_docs or has_demo or has_path_diversity)


def _has_high_trust_technical_evidence(urls: Sequence[str]) -> bool:
    for raw in list(urls or []):
        token = canonicalize_url(str(raw or "").strip())
        if not token:
            continue
        parsed = urlparse(token)
        host = str(parsed.netloc or "").strip().lower()
        path = str(parsed.path or "").strip().lower()
        if host == "github.com" and (path.startswith("/user-attachments/") or path.endswith(".git")):
            continue
        if host in {"arxiv.org", "openreview.net", "docs.python.org"}:
            return True
        if host in {"github.com"} and (
            (path.count("/") >= 2 and "/blob/" not in path)
            or "/releases/" in path
            or "/wiki" in path
            or "/issues/" in path
        ):
            return True
        if host.endswith("huggingface.co") and ("/datasets/" in path or "/models/" in path):
            return True
        if "docs" in host or "/docs" in path or "/documentation" in path:
            return True
    return False


def _has_non_repo_technical_evidence(urls: Sequence[str]) -> bool:
    for raw in list(urls or []):
        token = canonicalize_url(str(raw or "").strip())
        if not token:
            continue
        parsed = urlparse(token)
        host = str(parsed.netloc or "").strip().lower()
        path = str(parsed.path or "").strip().lower()
        if host in {"arxiv.org", "openreview.net", "docs.python.org"}:
            return True
        if host == "github.com" and ("/releases/" in path or "/wiki" in path or "/issues/" in path):
            return True
        if host.endswith("huggingface.co") and ("/datasets/" in path or "/models/" in path):
            return True
        if "docs" in host or "/docs" in path or "/documentation" in path:
            return True
    return False


def _is_evergreen_source(source: str) -> bool:
    return str(source or "").strip().lower() in {"arxiv", "semantic_scholar", "openreview"}


def _parse_datetime_token(value: str) -> datetime | None:
    token = str(value or "").strip()
    if not token:
        return None
    try:
        return datetime.fromisoformat(token.replace("Z", "+00:00"))
    except Exception:
        return None


def _item_age_hours(item: NormalizedItem, *, publish_or_update_time: str) -> float | None:
    dt = item.published_at
    if dt is None:
        dt = _parse_datetime_token(publish_or_update_time)
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - dt.astimezone(timezone.utc)
    return max(0.0, float(delta.total_seconds()) / 3600.0)


def _reason_code(reason: str) -> str:
    token = str(reason or "").strip()
    if not token:
        return "unspecified"
    if ":" in token:
        return token.split(":", 1)[0].strip().lower() or "unspecified"
    return token.lower()


def _human_reason(reasons: Sequence[str]) -> str:
    values = [str(reason).strip() for reason in list(reasons or []) if str(reason).strip()]
    if not values:
        return "passed evidence audit checks"
    return "; ".join(values)


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    payload = re.sub(r"[-_/]+", " ", str(text or "").lower())
    for term in list(terms or []):
        token = str(term or "").strip().lower()
        if not token:
            continue
        if re.search(r"\b" + re.escape(token).replace(r"\ ", r"[\s\-_]+") + r"\b", payload):
            return True
    return False


def _tokenize(text: str) -> List[str]:
    return [token for token in re.findall(r"[a-z0-9]+", str(text or "").lower()) if len(token) >= 2]


def _item_alignment_tokens(item: NormalizedItem) -> set[str]:
    metadata = dict(item.metadata or {})
    tokens = set(_tokenize(str(item.title or "")))
    tokens.update(_tokenize(str(metadata.get("repo_id") or "")))
    tokens.update(_tokenize(str(metadata.get("source_query") or "")))
    for value in list(metadata.get("topics") or []):
        tokens.update(_tokenize(str(value or "")))
    canonical_url = canonicalize_url(str(item.url or "").strip())
    if canonical_url:
        parsed = urlparse(canonical_url)
        tokens.update(_tokenize(str(parsed.path or "")))
    return {token for token in tokens if token not in {"github", "com", "huggingface", "co"}}


def _url_tokens(url: str) -> set[str]:
    parsed = urlparse(str(url or ""))
    combined = " ".join([str(parsed.netloc or ""), str(parsed.path or ""), str(parsed.query or "")])
    return set(_tokenize(combined))


def _is_same_repo_reference(url: str, item: NormalizedItem) -> bool:
    target = canonicalize_url(str(item.url or "").strip())
    probe = canonicalize_url(str(url or "").strip())
    if not target or not probe:
        return False
    parsed_target = urlparse(target)
    parsed_probe = urlparse(probe)
    target_host = str(parsed_target.netloc or "").strip().lower()
    probe_host = str(parsed_probe.netloc or "").strip().lower()
    if target_host != probe_host:
        return False
    target_parts = [part for part in str(parsed_target.path or "").split("/") if part]
    probe_parts = [part for part in str(parsed_probe.path or "").split("/") if part]
    if len(target_parts) < 2 or len(probe_parts) < 2:
        return False
    return target_parts[0].lower() == probe_parts[0].lower() and target_parts[1].lower() == probe_parts[1].lower()


def _evidence_alignment_scores(
    *,
    item: NormalizedItem,
    urls: Sequence[str],
    clean_text: str,
    topic: str,
) -> List[float]:
    item_tokens = _item_alignment_tokens(item)
    topic_tokens = set(_tokenize(topic))
    topic_tokens.update(
        {
            token
            for term in list(_AGENT_HIGH_VALUE_TOKENS)
            for token in list(_tokenize(term))
        }
    )
    body_tokens = set(_tokenize(clean_text))
    scores: List[float] = []
    for raw in list(urls or []):
        token = canonicalize_url(str(raw or "").strip())
        if not token:
            continue
        url_terms = _url_tokens(token)
        overlap_item = len(url_terms.intersection(item_tokens))
        overlap_topic = len(url_terms.intersection(topic_tokens))
        overlap_body = len(url_terms.intersection(body_tokens))
        score = 0.0
        if _is_same_repo_reference(token, item):
            score += 0.65
        if overlap_item >= 3:
            score += 0.45
        elif overlap_item == 2:
            score += 0.35
        elif overlap_item == 1:
            score += 0.2
        if overlap_topic >= 2:
            score += 0.25
        elif overlap_topic == 1:
            score += 0.15
        if overlap_body >= 2:
            score += 0.12
        score = max(0.0, min(1.0, score))
        scores.append(score)
    return scores


class EvidenceAuditor:
    """Rule-based quality/evidence auditor for candidate gating."""

    def __init__(
        self,
        *,
        min_evidence_links_quality: int = 2,
        duplicate_ratio_threshold: float = 0.6,
        topic: str | None = None,
        topic_profile: TopicProfile | None = None,
    ) -> None:
        self._min_evidence_links_quality = max(1, int(min_evidence_links_quality))
        self._duplicate_ratio_threshold = float(max(0.0, min(1.0, duplicate_ratio_threshold)))
        self._topic = str(topic or "").strip()
        self._topic_profile = topic_profile
        try:
            self._hn_min_points = max(0, int(float(os.getenv("ARA_V2_AUDIT_HN_MIN_POINTS", "5"))))
        except Exception:
            self._hn_min_points = 5
        try:
            self._hn_min_comments = max(0, int(float(os.getenv("ARA_V2_AUDIT_HN_MIN_COMMENTS", "2"))))
        except Exception:
            self._hn_min_comments = 2
        try:
            self._hn_recent_hours = max(1, int(float(os.getenv("ARA_V2_AUDIT_HN_RECENT_HOURS", "24"))))
        except Exception:
            self._hn_recent_hours = 24

    def _is_agent_topic(self) -> bool:
        if self._topic_profile and str(self._topic_profile.key or "").strip().lower() == "ai_agent":
            return True
        return "agent" in str(self._topic or "").lower()

    @staticmethod
    def _promote_verdict(current: str, target: str) -> str:
        order = {VERDICT_PASS: 0, VERDICT_DOWNGRADE: 1, VERDICT_REJECT: 2}
        return target if order.get(target, 0) >= order.get(current, 0) else current

    def _apply_rule(
        self,
        *,
        verdict: str,
        reasons: List[str],
        target: str,
        reason: str,
    ) -> str:
        updated = self._promote_verdict(verdict, target)
        if reason not in reasons:
            reasons.append(reason)
        return updated

    def audit_row(self, row: Any, rank: int) -> AuditRecord:
        item = row.item
        metadata = dict(item.metadata or {})
        source_key = str(item.source or "").strip().lower()
        signals = dict(metadata.get("quality_signals") or {})
        body_len = int(float(metadata.get("body_len", len(str(item.body_md or ""))) or 0))
        relevance_score = float(getattr(row, "relevance_score", 0.0) or 0.0)
        hard_gate_fail = not bool(metadata.get("topic_hard_match_pass", True))
        cross_source_count = int(float(metadata.get("cross_source_corroboration_count", 0) or 0))
        cross_source_corroborated = bool(metadata.get("cross_source_corroborated")) or cross_source_count >= 2
        min_body_len = _source_min_body_len(item.source)
        evidence_links_quality = int(float(signals.get("evidence_links_quality", 0) or 0))
        if source_key == "hackernews":
            hn_points_seed = int(float(metadata.get("points", 0) or 0))
            hn_comments_seed = int(float(metadata.get("comment_count", metadata.get("comments", 0)) or 0))
            if hn_points_seed >= max(40, self._hn_min_points * 8) or hn_comments_seed >= max(16, self._hn_min_comments * 8):
                evidence_links_quality = max(1, evidence_links_quality)
        effective_min_evidence = int(self._min_evidence_links_quality)
        if source_key == "hackernews":
            effective_min_evidence = 1
        elif source_key == "huggingface" and self._is_agent_topic():
            effective_min_evidence = max(1, int(self._min_evidence_links_quality) - 1)
        publish_or_update_time = str(signals.get("publish_or_update_time") or metadata.get("publish_or_update_time") or "").strip()
        age_hours = _item_age_hours(item, publish_or_update_time=publish_or_update_time)
        duplicate_ratio = _citation_duplicate_ratio(item)
        urls = _collect_evidence_urls(item)
        domains = _domains(urls)
        clean_text = str(metadata.get("clean_text") or item.body_md or "")
        alignment_scores = _evidence_alignment_scores(
            item=item,
            urls=urls,
            clean_text=clean_text,
            topic=str(self._topic or ""),
        )
        evidence_target_alignment = (
            float(sum(alignment_scores)) / float(max(1, len(alignment_scores))) if alignment_scores else 0.0
        )
        evidence_alignment_hits = int(sum(1 for value in list(alignment_scores or []) if float(value) >= 0.55))
        link_heavy = bool(len(urls) >= 4 or evidence_links_quality >= 4)

        verdict = VERDICT_PASS
        reasons: List[str] = []

        if relevance_score <= 0.0:
            verdict = self._apply_rule(
                verdict=verdict,
                reasons=reasons,
                target=VERDICT_REJECT,
                reason="topic_relevance_zero",
            )

        if hard_gate_fail:
            verdict = self._apply_rule(
                verdict=verdict,
                reasons=reasons,
                target=VERDICT_REJECT,
                reason="topic_hard_gate_fail",
            )

        if body_len <= 0:
            verdict = self._apply_rule(
                verdict=verdict,
                reasons=reasons,
                target=VERDICT_REJECT,
                reason="body_len_zero",
            )

        if body_len < min_body_len:
            verdict = self._apply_rule(
                verdict=verdict,
                reasons=reasons,
                target=VERDICT_DOWNGRADE,
                reason=f"body_len_lt_min:{body_len}<{min_body_len}",
            )

        if evidence_links_quality < effective_min_evidence:
            if evidence_links_quality <= 0:
                target = VERDICT_REJECT
            else:
                target = VERDICT_DOWNGRADE
            verdict = self._apply_rule(
                verdict=verdict,
                reasons=reasons,
                target=target,
                reason=f"evidence_links_quality_lt_min:{evidence_links_quality}<{effective_min_evidence}",
            )

        if link_heavy and evidence_target_alignment < 0.35:
            alignment_target = VERDICT_REJECT if evidence_alignment_hits <= 0 else VERDICT_DOWNGRADE
            verdict = self._apply_rule(
                verdict=verdict,
                reasons=reasons,
                target=alignment_target,
                reason="link_heavy_low_alignment",
            )
        elif len(urls) >= 3 and evidence_target_alignment < 0.22 and evidence_alignment_hits <= 1:
            verdict = self._apply_rule(
                verdict=verdict,
                reasons=reasons,
                target=VERDICT_DOWNGRADE,
                reason="evidence_alignment_weak",
            )

        if duplicate_ratio > self._duplicate_ratio_threshold:
            duplicate_target = VERDICT_REJECT
            if len(domains) >= 2 and evidence_links_quality >= effective_min_evidence:
                duplicate_target = VERDICT_DOWNGRADE
            verdict = self._apply_rule(
                verdict=verdict,
                reasons=reasons,
                target=duplicate_target,
                reason=f"citation_duplicate_prefix_ratio_gt_threshold:{duplicate_ratio:.2f}>{self._duplicate_ratio_threshold:.2f}",
            )

        if not publish_or_update_time and not _is_evergreen_source(item.source):
            hf_acceptable_missing_time = bool(
                source_key == "huggingface"
                and bool(metadata.get("deep_fetch_applied"))
                and body_len >= min_body_len
                and evidence_links_quality >= max(1, effective_min_evidence - 1)
            )
            if not hf_acceptable_missing_time:
                verdict = self._apply_rule(
                    verdict=verdict,
                    reasons=reasons,
                    target=VERDICT_DOWNGRADE,
                    reason="missing_publish_or_update_time",
                )

        head_text = " ".join(re.split(r"[.!?;\n]+", clean_text)[:3]).strip()
        if _looks_marketing(head_text) and not _has_verifiable_point(clean_text):
            verdict = self._apply_rule(
                verdict=verdict,
                reasons=reasons,
                target=VERDICT_DOWNGRADE,
                reason="marketing_declaration_without_verifiable_points",
            )

        if source_key != "hackernews" and _repo_self_only(urls):
            verdict = self._apply_rule(
                verdict=verdict,
                reasons=reasons,
                target=VERDICT_DOWNGRADE,
                reason="single_domain_repo_self_evidence",
            )

        text = " ".join([str(item.title or ""), clean_text]).lower()
        handbook_like = _contains_any(text, _HANDBOOK_TOKENS)
        has_runtime_tooling_signal = _contains_any(text, _AGENT_RUNTIME_TOOLING_TOKENS)
        if self._is_agent_topic() and handbook_like and not has_runtime_tooling_signal:
            verdict = self._apply_rule(
                verdict=verdict,
                reasons=reasons,
                target=VERDICT_DOWNGRADE,
                reason="handbook_like_for_agent_topic",
            )
        hard_terms = [str(value).strip().lower() for value in list(metadata.get("topic_hard_match_terms") or []) if str(value).strip()]
        agent_non_generic_hits = [
            str(value).strip().lower()
            for value in list(metadata.get("topic_agent_non_generic_hits") or [])
            if str(value).strip()
        ]
        agent_high_value_hits = [
            str(value).strip().lower()
            for value in list(metadata.get("topic_agent_high_value_hits") or [])
            if str(value).strip()
        ]
        bucket_hits = [str(value).strip() for value in list(metadata.get("bucket_hits") or []) if str(value).strip()]
        text_has_high_value = _contains_any(text, _AGENT_HIGH_VALUE_TOKENS)
        text_has_generic_agent = _contains_any(text, _AGENT_GENERIC_TOKENS)
        has_agent_semantic_depth = bool(
            agent_high_value_hits
            or agent_non_generic_hits
            or bucket_hits
            or text_has_high_value
            or (
                hard_terms
                and any(token not in _AGENT_GENERIC_TOKENS for token in hard_terms)
            )
        )
        if (
            not has_agent_semantic_depth
            and text_has_generic_agent
            and (
                bool(signals.get("has_quickstart"))
                or bool(signals.get("has_results_or_bench"))
                or evidence_links_quality >= max(2, effective_min_evidence)
            )
        ):
            has_agent_semantic_depth = True
        if self._is_agent_topic() and not has_agent_semantic_depth:
            semantic_target = VERDICT_REJECT if source_key == "hackernews" else VERDICT_DOWNGRADE
            verdict = self._apply_rule(
                verdict=verdict,
                reasons=reasons,
                target=semantic_target,
                reason="agent_semantic_weak",
            )
        has_quickstart = bool(signals.get("has_quickstart"))
        has_results = bool(signals.get("has_results_or_bench"))
        has_multi_domain_evidence = len(domains) >= 2
        has_release_note = any("/releases/" in str(urlparse(url).path or "").lower() for url in urls)

        if source_key == "hackernews":
            points = int(float(metadata.get("points", 0) or 0))
            comments = int(float(metadata.get("comment_count", metadata.get("comments", 0)) or 0))
            low_engagement = points < self._hn_min_points and comments < self._hn_min_comments
            high_discussion = bool(
                points >= max(40, self._hn_min_points * 8)
                or comments >= max(16, self._hn_min_comments * 8)
            )
            has_external_high_trust = bool(
                _has_high_trust_technical_evidence(urls)
                and any(str(domain).strip().lower() != "news.ycombinator.com" for domain in list(domains or []))
            )
            has_strong_signal = bool(
                (has_multi_domain_evidence and (has_quickstart or has_results or evidence_links_quality >= 3))
                or high_discussion
                or cross_source_corroborated
            )
            if low_engagement and not has_strong_signal:
                if age_hours is not None and age_hours <= float(self._hn_recent_hours) and (
                    has_external_high_trust or cross_source_corroborated
                ):
                    verdict = self._apply_rule(
                        verdict=verdict,
                        reasons=reasons,
                        target=VERDICT_DOWNGRADE,
                        reason=f"hn_low_engagement_recent:{points}/{comments}",
                    )
                else:
                    verdict = self._apply_rule(
                        verdict=verdict,
                        reasons=reasons,
                        target=VERDICT_REJECT,
                        reason=f"hn_low_engagement:{points}/{comments}",
                    )
            if (
                verdict == VERDICT_PASS
                and evidence_links_quality < self._min_evidence_links_quality
                and not high_discussion
                and not has_external_high_trust
                and not cross_source_corroborated
            ):
                verdict = self._apply_rule(
                    verdict=verdict,
                    reasons=reasons,
                    target=VERDICT_DOWNGRADE,
                    reason=f"hn_evidence_single_link_weak:{evidence_links_quality}",
                )

        if source_key == "github":
            stars = int(float(metadata.get("stars", 0) or 0))
            forks = int(float(metadata.get("forks", 0) or 0))
            retrieval_window_days = int(float(metadata.get("retrieval_window_days", 0) or 0))
            canonical_item_url = canonicalize_url(str(item.url or "").strip())
            parsed_item = urlparse(canonical_item_url) if canonical_item_url else None
            repo_root_path = ""
            if parsed_item and str(parsed_item.netloc or "").strip().lower().endswith("github.com"):
                parts = [part for part in str(parsed_item.path or "").split("/") if part]
                if len(parts) >= 2:
                    repo_root_path = f"/{parts[0].lower()}/{parts[1].lower()}"
            benchmark_evidence_paths: List[str] = []
            benchmark_evidence_domains: List[str] = []
            for url in list(urls or []):
                token = canonicalize_url(str(url or "").strip())
                if not token or token == canonical_item_url:
                    continue
                parsed = urlparse(token)
                host = str(parsed.netloc or "").strip().lower()
                path = str(parsed.path or "").strip().lower()
                if repo_root_path and host.endswith("github.com") and (path == repo_root_path or path == f"{repo_root_path}/"):
                    continue
                benchmark_evidence_paths.append(path)
                benchmark_evidence_domains.append(host)
            has_benchmark_evidence = any(
                token in path for path in benchmark_evidence_paths for token in ("bench", "eval", "leaderboard", "result")
            ) or any(
                domain in {"arxiv.org", "openreview.net", "paperswithcode.com"}
                for domain in benchmark_evidence_domains
            )
            has_benchmark = bool(has_results and has_benchmark_evidence)
            has_external_discussion = any(
                domain in {"news.ycombinator.com", "reddit.com", "www.reddit.com", "lobste.rs"}
                for domain in list(domains or [])
            )
            traction_emerging = bool(stars >= 15 or forks >= 3)
            traction_strong = bool(stars >= 50 or forks >= 10)
            if (
                stars < 30
                and forks < 5
                and not has_release_note
                and not has_benchmark
                and not cross_source_corroborated
                and not has_external_discussion
            ):
                target = VERDICT_REJECT if stars < 10 and forks < 2 else VERDICT_DOWNGRADE
                verdict = self._apply_rule(
                    verdict=verdict,
                    reasons=reasons,
                    target=target,
                    reason=f"github_low_signal_repo:{stars}/{forks}",
                )
            if (
                verdict == VERDICT_PASS
                and not traction_emerging
                and not cross_source_corroborated
                and not has_external_discussion
                and not has_release_note
                and not has_benchmark
            ):
                target = VERDICT_REJECT if stars < 8 and forks < 2 and not has_external_discussion else VERDICT_DOWNGRADE
                verdict = self._apply_rule(
                    verdict=verdict,
                    reasons=reasons,
                    target=target,
                    reason=f"github_traction_weak:{stars}/{forks}",
                )
            elif verdict == VERDICT_PASS and not traction_strong and not (
                has_release_note or has_benchmark or cross_source_corroborated or has_external_discussion
            ):
                verdict = self._apply_rule(
                    verdict=verdict,
                    reasons=reasons,
                    target=VERDICT_DOWNGRADE,
                    reason=f"github_traction_unproven:{stars}/{forks}",
                )

            if (
                self._is_agent_topic()
                and retrieval_window_days >= 7
                and stars < 800
                and forks < 80
                and not has_release_note
                and not has_benchmark
                and not cross_source_corroborated
                and not has_external_discussion
            ):
                verdict = self._apply_rule(
                    verdict=verdict,
                    reasons=reasons,
                    target=VERDICT_DOWNGRADE,
                    reason=f"github_weekly_trending_signal_weak:{stars}/{forks}",
                )

        if source_key == "huggingface" and self._is_agent_topic():
            has_cv_token = _contains_any(text, _CV_OFFTOPIC_TOKENS)
            has_agent_high_value = _contains_any(text, _AGENT_HIGH_VALUE_TOKENS)
            if has_cv_token and not has_agent_high_value:
                verdict = self._apply_rule(
                    verdict=verdict,
                    reasons=reasons,
                    target=VERDICT_REJECT,
                    reason="hf_cv_offtopic_for_agent_topic",
                )
            if not has_agent_high_value and evidence_links_quality < self._min_evidence_links_quality:
                verdict = self._apply_rule(
                    verdict=verdict,
                    reasons=reasons,
                    target=VERDICT_DOWNGRADE,
                    reason="hf_agent_signal_weak",
                )

        if verdict == VERDICT_DOWNGRADE and cross_source_corroborated and evidence_links_quality >= 1:
            hard_reject_reasons = {
                "topic_relevance_zero",
                "topic_hard_gate_fail",
                "body_len_zero",
                "hf_cv_offtopic_for_agent_topic",
                "hn_low_engagement",
                "agent_semantic_weak",
                "link_heavy_low_alignment",
                "handbook_like_for_agent_topic",
                "evidence_alignment_weak",
            }
            reason_codes = {_reason_code(reason) for reason in list(reasons or [])}
            if not reason_codes.intersection(hard_reject_reasons) and "evidence_high_trust_missing" not in reason_codes:
                verdict = VERDICT_PASS
                if "cross_source_corroboration_bonus" not in reasons:
                    reasons.append("cross_source_corroboration_bonus")

        if source_key in {"github", "huggingface"} and verdict != VERDICT_REJECT:
            if not _has_high_trust_technical_evidence(urls):
                verdict = self._apply_rule(
                    verdict=verdict,
                    reasons=reasons,
                    target=VERDICT_DOWNGRADE,
                    reason="evidence_high_trust_missing",
                )

        preferred_codes = [
            "link_heavy_low_alignment",
            "handbook_like_for_agent_topic",
            "evidence_alignment_weak",
        ]
        reason_codes = [_reason_code(reason) for reason in list(reasons or [])]
        primary_code = _reason_code(reasons[0] if reasons else "")
        for code in preferred_codes:
            if code in reason_codes:
                primary_code = code
                break
        machine_action = {
            "action": verdict,
            "reason_code": primary_code if verdict != VERDICT_PASS else "ok",
            "human_reason": _human_reason(reasons),
        }

        return AuditRecord(
            item_id=item.id,
            title=item.title,
            source=item.source,
            rank=int(rank),
            verdict=verdict,
            reasons=reasons,
            used_evidence_urls=urls,
            evidence_domains=domains,
            citation_duplicate_prefix_ratio=duplicate_ratio,
            evidence_links_quality=evidence_links_quality,
            body_len=body_len,
            min_body_len=min_body_len,
            publish_or_update_time=publish_or_update_time,
            machine_action=machine_action,
            evidence_target_alignment=float(evidence_target_alignment),
            evidence_alignment_hits=int(evidence_alignment_hits),
            evidence_alignment_total=int(len(alignment_scores)),
            link_heavy=bool(link_heavy),
        )

    def audit_rows(self, *, ranked_rows: Sequence[Any]) -> List[AuditRecord]:
        records: List[AuditRecord] = []
        for idx, row in enumerate(list(ranked_rows or []), start=1):
            records.append(self.audit_row(row, rank=idx))
        return records

    def audit_and_select(
        self,
        *,
        initial_picks: Sequence[Any],
        ranked_rows: Sequence[Any],
        top_count: int,
    ) -> AuditSelection:
        top_n = max(1, int(top_count))
        records = self.audit_rows(ranked_rows=ranked_rows)
        records_by_id: Dict[str, AuditRecord] = {record.item_id: record for record in list(records)}

        selected: List[Any] = []
        selected_ids = set()

        def _try_add(row: Any) -> None:
            item_id = str(getattr(row.item, "id", "") or "").strip()
            if not item_id or item_id in selected_ids:
                return
            record = records_by_id.get(item_id)
            if record is None:
                return
            if record.verdict == VERDICT_REJECT:
                return
            # downgrade candidates are acceptable only if reasons are preserved.
            if record.verdict == VERDICT_DOWNGRADE:
                reason_token = "audit.downgrade:" + ",".join(record.reasons or ["unspecified"])
                reasons = list(getattr(row, "reasons", []) or [])
                if reason_token not in reasons:
                    reasons.append(reason_token)
                row.reasons = reasons
            selected.append(row)
            selected_ids.add(item_id)

        for row in list(initial_picks or []):
            _try_add(row)

        if len(selected) < top_n:
            for row in list(ranked_rows or []):
                item_id = str(getattr(row.item, "id", "") or "").strip()
                if not item_id or item_id in selected_ids:
                    continue
                record = records_by_id.get(item_id)
                if record and record.verdict == VERDICT_PASS:
                    _try_add(row)
                if len(selected) >= top_n:
                    break

        if len(selected) < top_n:
            for row in list(ranked_rows or []):
                item_id = str(getattr(row.item, "id", "") or "").strip()
                if not item_id or item_id in selected_ids:
                    continue
                record = records_by_id.get(item_id)
                if record and record.verdict == VERDICT_DOWNGRADE:
                    _try_add(row)
                if len(selected) >= top_n:
                    break

        return AuditSelection(selected_rows=selected[:top_n], records=records)


class EvidenceAuditorProtocol(Protocol):
    def audit_rows(self, *, ranked_rows: Sequence[Any]) -> List[AuditRecord]:
        ...

    def audit_and_select(
        self,
        *,
        initial_picks: Sequence[Any],
        ranked_rows: Sequence[Any],
        top_count: int,
    ) -> AuditSelection:
        ...


class LLMEvidenceAuditor(EvidenceAuditor):
    """Stub slot for future LLM-backed evidence auditing."""

    def __init__(self, *, enabled: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.enabled = bool(enabled)


def summarize_top_pick_verdicts(
    records: Sequence[AuditRecord],
    top_item_ids: Sequence[str],
) -> Dict[str, Mapping[str, Any]]:
    """Helper for validator/reporting: verdict map for top picks only."""
    wanted = {str(item).strip() for item in list(top_item_ids or []) if str(item).strip()}
    payload: Dict[str, Mapping[str, Any]] = {}
    for record in list(records or []):
        if record.item_id not in wanted:
            continue
        payload[record.item_id] = {
            "verdict": record.verdict,
            "reasons": list(record.reasons),
            "citation_duplicate_prefix_ratio": round(float(record.citation_duplicate_prefix_ratio), 4),
        }
    return payload
