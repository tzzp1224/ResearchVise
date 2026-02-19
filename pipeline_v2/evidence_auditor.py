"""Deterministic evidence audit for ranked candidates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Any, Dict, Iterable, List, Mapping, Protocol, Sequence
from urllib.parse import urlparse

from core import NormalizedItem
from pipeline_v2.sanitize import canonicalize_url, is_allowed_citation_url


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
    urls: List[str] = []
    metadata = dict(item.metadata or {})
    for raw in list(metadata.get("evidence_links") or []):
        token = canonicalize_url(str(raw or ""))
        if token and is_allowed_citation_url(token) and token not in urls:
            urls.append(token)
    for citation in list(item.citations or []):
        token = canonicalize_url(str(citation.url or ""))
        if token and is_allowed_citation_url(token) and token not in urls:
            urls.append(token)
    item_url = canonicalize_url(str(item.url or ""))
    if item_url and is_allowed_citation_url(item_url) and item_url not in urls:
        urls.append(item_url)
    return urls[:12]


def _source_min_body_len(source: str) -> int:
    key = str(source or "").strip().lower()
    if key in {"github", "huggingface"}:
        return 400
    if key in {"hackernews", "web_article"}:
        return 600
    return 500


def _citation_duplicate_ratio(item: NormalizedItem) -> float:
    snippets = [
        re.sub(r"\s+", " ", str(citation.snippet or citation.title or "")).strip().lower()
        for citation in list(item.citations or [])
        if str(citation.snippet or citation.title or "").strip()
    ]
    if len(snippets) < 2:
        return 0.0
    prefixes = [snippet[:48] for snippet in snippets]
    unique_prefixes = len(set(prefixes))
    ratio = 1.0 - float(unique_prefixes) / float(max(1, len(prefixes)))
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


def _is_evergreen_source(source: str) -> bool:
    return str(source or "").strip().lower() in {"arxiv", "semantic_scholar", "openreview"}


class EvidenceAuditor:
    """Rule-based quality/evidence auditor for candidate gating."""

    def __init__(self, *, min_evidence_links_quality: int = 2, duplicate_ratio_threshold: float = 0.6) -> None:
        self._min_evidence_links_quality = max(1, int(min_evidence_links_quality))
        self._duplicate_ratio_threshold = float(max(0.0, min(1.0, duplicate_ratio_threshold)))

    def audit_row(self, row: Any, rank: int) -> AuditRecord:
        item = row.item
        metadata = dict(item.metadata or {})
        signals = dict(metadata.get("quality_signals") or {})
        body_len = int(float(metadata.get("body_len", len(str(item.body_md or ""))) or 0))
        min_body_len = _source_min_body_len(item.source)
        evidence_links_quality = int(float(signals.get("evidence_links_quality", 0) or 0))
        publish_or_update_time = str(signals.get("publish_or_update_time") or metadata.get("publish_or_update_time") or "").strip()
        duplicate_ratio = _citation_duplicate_ratio(item)
        urls = _collect_evidence_urls(item)
        domains = _domains(urls)

        verdict = VERDICT_PASS
        reasons: List[str] = []

        if body_len < min_body_len:
            verdict = VERDICT_DOWNGRADE
            reasons.append(f"body_len_lt_min:{body_len}<{min_body_len}")

        if evidence_links_quality < self._min_evidence_links_quality:
            if evidence_links_quality <= 0:
                verdict = VERDICT_REJECT
            elif verdict != VERDICT_REJECT:
                verdict = VERDICT_DOWNGRADE
            reasons.append(
                f"evidence_links_quality_lt_min:{evidence_links_quality}<{self._min_evidence_links_quality}"
            )

        if duplicate_ratio > self._duplicate_ratio_threshold:
            verdict = VERDICT_REJECT
            reasons.append(
                f"citation_duplicate_prefix_ratio_gt_threshold:{duplicate_ratio:.2f}>{self._duplicate_ratio_threshold:.2f}"
            )

        if not publish_or_update_time and not _is_evergreen_source(item.source):
            if verdict != VERDICT_REJECT:
                verdict = VERDICT_DOWNGRADE
            reasons.append("missing_publish_or_update_time")

        clean_text = str(metadata.get("clean_text") or item.body_md or "")
        head_text = " ".join(re.split(r"[.!?;\n]+", clean_text)[:3]).strip()
        if _looks_marketing(head_text) and not _has_verifiable_point(clean_text):
            if verdict != VERDICT_REJECT:
                verdict = VERDICT_DOWNGRADE
            reasons.append("marketing_declaration_without_verifiable_points")

        if _repo_self_only(urls):
            if verdict != VERDICT_REJECT:
                verdict = VERDICT_DOWNGRADE
            reasons.append("single_domain_repo_self_evidence")

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
        )

    def audit_and_select(
        self,
        *,
        initial_picks: Sequence[Any],
        ranked_rows: Sequence[Any],
        top_count: int,
    ) -> AuditSelection:
        top_n = max(1, int(top_count))
        records: List[AuditRecord] = []
        records_by_id: Dict[str, AuditRecord] = {}

        for idx, row in enumerate(list(ranked_rows or []), start=1):
            record = self.audit_row(row, rank=idx)
            records.append(record)
            records_by_id[record.item_id] = record

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
