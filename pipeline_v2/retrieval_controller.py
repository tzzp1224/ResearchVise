"""Audit-driven retrieval selection controller for multi-phase recall."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence


@dataclass
class AttemptOutcome:
    selected_rows: List[Any]
    selected_verdicts: Dict[str, str]
    selected_downgrade_reasons: Dict[str, List[str]]
    candidate_count: int
    pass_count: int
    downgrade_count: int
    reject_count: int
    pass_ratio: float
    top_picks_min_evidence_quality: float
    bucket_coverage: int
    source_coverage: int
    used_downgrade: bool
    all_selected_downgrade: bool
    max_downgrade_allowed: int
    downgrade_cap_reached: bool


@dataclass
class ExpansionDecision:
    should_expand: bool
    reasons: List[str]


class SelectionController:
    """Select top picks from audit output and decide expansion triggers."""

    def __init__(
        self,
        *,
        requested_top_k: int,
        min_pass_ratio: float = 0.3,
        min_evidence_quality: float = 2.0,
        min_bucket_coverage: int = 2,
        min_source_coverage: int = 2,
        max_downgrade_ratio: float = 0.34,
        max_downgrade_count: int | None = None,
    ) -> None:
        self._requested_top_k = max(1, int(requested_top_k))
        self._min_pass_ratio = max(0.0, min(1.0, float(min_pass_ratio)))
        self._min_evidence_quality = float(max(0.0, min_evidence_quality))
        self._min_bucket_coverage = max(1, int(min_bucket_coverage))
        self._min_source_coverage = max(1, int(min_source_coverage))
        ratio = max(0.0, min(1.0, float(max_downgrade_ratio)))
        default_cap = max(1, int(self._requested_top_k * ratio))
        if max_downgrade_count is None:
            self._max_downgrade_count = default_cap
        else:
            self._max_downgrade_count = max(0, int(max_downgrade_count))

    @staticmethod
    def _item_id(row: Any) -> str:
        return str((getattr(row, "item", None).id if getattr(row, "item", None) else "") or "").strip()

    @staticmethod
    def _item_source(row: Any) -> str:
        return str((getattr(row, "item", None).source if getattr(row, "item", None) else "") or "").strip().lower()

    @staticmethod
    def _item_buckets(row: Any) -> List[str]:
        item = getattr(row, "item", None)
        metadata = dict((getattr(item, "metadata", None) or {}) if item else {})
        return [str(value).strip() for value in list(metadata.get("bucket_hits") or []) if str(value).strip()]

    @staticmethod
    def _item_metadata(row: Any) -> Mapping[str, Any]:
        item = getattr(row, "item", None)
        return dict((getattr(item, "metadata", None) or {}) if item else {})

    @staticmethod
    def _relevance(row: Any) -> float:
        try:
            return float(getattr(row, "relevance_score", 0.0) or 0.0)
        except Exception:
            return 0.0

    @staticmethod
    def _body_len(row: Any) -> int:
        metadata = SelectionController._item_metadata(row)
        raw = metadata.get("body_len")
        if raw not in (None, ""):
            try:
                return max(0, int(float(raw)))
            except Exception:
                pass
        item = getattr(row, "item", None)
        body = str((getattr(item, "body_md", None) if item else "") or "")
        return max(0, len(body.strip()))

    @staticmethod
    def _hard_match_pass(row: Any) -> bool:
        metadata = SelectionController._item_metadata(row)
        return bool(metadata.get("topic_hard_match_pass", True))

    def _is_selection_eligible(self, row: Any, *, min_relevance: float) -> bool:
        if self._relevance(row) <= 0.0:
            return False
        if self._relevance(row) + 1e-9 < float(max(0.0, min_relevance)):
            return False
        if not self._hard_match_pass(row):
            return False
        if self._body_len(row) <= 0:
            return False
        return True

    def _select_diverse(
        self,
        rows: Sequence[Any],
        *,
        target_count: int,
        preselected: Sequence[Any] | None = None,
    ) -> List[Any]:
        selected: List[Any] = list(preselected or [])
        selected_ids = {self._item_id(row) for row in selected if self._item_id(row)}
        used_sources = {self._item_source(row) for row in selected if self._item_source(row)}
        used_buckets = {bucket for row in selected for bucket in self._item_buckets(row)}

        def _add(row: Any) -> None:
            item_id = self._item_id(row)
            if not item_id or item_id in selected_ids:
                return
            selected.append(row)
            selected_ids.add(item_id)
            source = self._item_source(row)
            if source:
                used_sources.add(source)
            used_buckets.update(self._item_buckets(row))

        if self._min_bucket_coverage > 1:
            for row in list(rows or []):
                if len(selected) >= target_count:
                    break
                item_id = self._item_id(row)
                if not item_id or item_id in selected_ids:
                    continue
                row_buckets = self._item_buckets(row)
                if not row_buckets:
                    continue
                if not any(bucket not in used_buckets for bucket in row_buckets):
                    continue
                _add(row)
                if len(used_buckets) >= self._min_bucket_coverage and len(selected) >= 1:
                    break

        for row in list(rows or []):
            if len(selected) >= target_count:
                break
            item_id = self._item_id(row)
            if not item_id or item_id in selected_ids:
                continue
            source = self._item_source(row)
            if source and source in used_sources:
                continue
            _add(row)

        if len([token for token in used_sources if token]) < int(self._min_source_coverage):
            return selected[:target_count]

        for row in list(rows or []):
            if len(selected) >= target_count:
                break
            item_id = self._item_id(row)
            if not item_id or item_id in selected_ids:
                continue
            _add(row)

        return selected[:target_count]

    def _fill_with_downgrade(
        self,
        *,
        selected: Sequence[Any],
        downgrade_rows: Sequence[Any],
        target_count: int,
        max_to_add: int,
    ) -> List[Any]:
        picks: List[Any] = list(selected or [])
        selected_ids = {self._item_id(row) for row in picks if self._item_id(row)}
        used_sources = {self._item_source(row) for row in picks if self._item_source(row)}
        used_buckets = {bucket for row in picks for bucket in self._item_buckets(row)}
        base_count = len(picks)

        def _add(row: Any) -> None:
            item_id = self._item_id(row)
            if not item_id or item_id in selected_ids:
                return
            picks.append(row)
            selected_ids.add(item_id)
            source = self._item_source(row)
            if source:
                used_sources.add(source)
            used_buckets.update(self._item_buckets(row))

        # Stage 1: prefer rows that increase source diversity.
        for row in list(downgrade_rows or []):
            if len(picks) >= target_count:
                break
            if (len(picks) - base_count) >= max_to_add:
                break
            item_id = self._item_id(row)
            if not item_id or item_id in selected_ids:
                continue
            source = self._item_source(row)
            if source and source not in used_sources:
                _add(row)
                if len([token for token in used_sources if token]) >= int(self._min_source_coverage):
                    break

        # If we still do not satisfy source coverage, stop and return shortage.
        if len([token for token in used_sources if token]) < int(self._min_source_coverage):
            if not picks:
                for row in list(downgrade_rows or []):
                    if (len(picks) - base_count) >= max_to_add:
                        break
                    item_id = self._item_id(row)
                    if item_id and item_id not in selected_ids:
                        _add(row)
                        break
            return picks[:target_count]

        # Stage 2: with source coverage satisfied, continue normal diverse fill.
        remaining = [row for row in list(downgrade_rows or []) if self._item_id(row) not in selected_ids]
        filled = self._select_diverse(
            remaining,
            target_count=min(target_count, base_count + max_to_add),
            preselected=picks,
        )
        return filled[:target_count]

    def evaluate(
        self,
        *,
        ranked_rows: Sequence[Any],
        audit_records: Sequence[Any],
        allow_downgrade_fill: bool,
        min_relevance_for_selection: float = 0.0,
    ) -> AttemptOutcome:
        records_by_id = {
            str(getattr(record, "item_id", "") or "").strip(): record
            for record in list(audit_records or [])
            if str(getattr(record, "item_id", "") or "").strip()
        }

        ranked_list = [row for row in list(ranked_rows or []) if self._item_id(row)]
        eligible_rows = [
            row
            for row in ranked_list
            if self._is_selection_eligible(row, min_relevance=float(min_relevance_for_selection))
        ]
        pass_rows = [
            row for row in eligible_rows if str(getattr(records_by_id.get(self._item_id(row)), "verdict", "")).lower() == "pass"
        ]
        downgrade_rows = [
            row for row in eligible_rows if str(getattr(records_by_id.get(self._item_id(row)), "verdict", "")).lower() == "downgrade"
        ]

        selected = self._select_diverse(pass_rows, target_count=self._requested_top_k)
        used_downgrade = False
        max_downgrade_allowed = max(0, min(self._max_downgrade_count, self._requested_top_k - len(selected)))
        if allow_downgrade_fill and len(selected) < self._requested_top_k:
            used_downgrade = True
            selected = self._fill_with_downgrade(
                selected=selected,
                downgrade_rows=downgrade_rows,
                target_count=self._requested_top_k,
                max_to_add=max_downgrade_allowed,
            )

        selected_verdicts: Dict[str, str] = {}
        selected_downgrade_reasons: Dict[str, List[str]] = {}
        evidence_qualities: List[float] = []
        selected_buckets = set()
        selected_sources = set()
        all_selected_downgrade = bool(selected)
        selected_downgrade_total = 0
        for row in list(selected or []):
            item_id = self._item_id(row)
            record = records_by_id.get(item_id)
            verdict = str(getattr(record, "verdict", "reject") or "reject").strip().lower()
            selected_verdicts[item_id] = verdict
            if verdict != "downgrade":
                all_selected_downgrade = False
            if verdict == "downgrade":
                selected_downgrade_total += 1
                selected_downgrade_reasons[item_id] = [
                    str(reason) for reason in list(getattr(record, "reasons", []) or []) if str(reason).strip()
                ]
            evidence_qualities.append(float(getattr(record, "evidence_links_quality", 0) or 0))
            selected_sources.add(self._item_source(row))
            selected_buckets.update(self._item_buckets(row))

        pass_count = 0
        downgrade_count = 0
        reject_count = 0
        eligible_ids = {self._item_id(row) for row in list(eligible_rows or []) if self._item_id(row)}
        for record in list(audit_records or []):
            item_id = str(getattr(record, "item_id", "") or "").strip()
            if not item_id or item_id not in eligible_ids:
                continue
            verdict = str(getattr(record, "verdict", "") or "").strip().lower()
            if verdict == "pass":
                pass_count += 1
            elif verdict == "downgrade":
                downgrade_count += 1
            else:
                reject_count += 1

        candidate_count = len(eligible_rows)
        pass_ratio = float(pass_count) / float(max(1, candidate_count))
        min_evidence_quality = min(evidence_qualities) if evidence_qualities else 0.0
        downgrade_cap_reached = bool(
            allow_downgrade_fill
            and len(selected) < self._requested_top_k
            and selected_downgrade_total >= max_downgrade_allowed
        )

        return AttemptOutcome(
            selected_rows=selected,
            selected_verdicts=selected_verdicts,
            selected_downgrade_reasons=selected_downgrade_reasons,
            candidate_count=candidate_count,
            pass_count=pass_count,
            downgrade_count=downgrade_count,
            reject_count=reject_count,
            pass_ratio=pass_ratio,
            top_picks_min_evidence_quality=float(min_evidence_quality),
            bucket_coverage=int(len([bucket for bucket in selected_buckets if bucket])),
            source_coverage=int(len([source for source in selected_sources if source])),
            used_downgrade=bool(used_downgrade),
            all_selected_downgrade=bool(all_selected_downgrade),
            max_downgrade_allowed=int(max_downgrade_allowed),
            downgrade_cap_reached=bool(downgrade_cap_reached),
        )

    def expansion_decision(self, *, outcome: AttemptOutcome) -> ExpansionDecision:
        reasons: List[str] = []

        if int(outcome.pass_count) < int(self._requested_top_k):
            reasons.append(f"pass_count_lt_{self._requested_top_k}")
        if float(outcome.pass_ratio) < float(self._min_pass_ratio):
            reasons.append(f"pass_ratio_lt_{self._min_pass_ratio:.2f}")
        if float(outcome.top_picks_min_evidence_quality) < float(self._min_evidence_quality):
            reasons.append(f"top_picks_min_evidence_quality_lt_{self._min_evidence_quality:.1f}")
        if int(outcome.bucket_coverage) < int(self._min_bucket_coverage):
            reasons.append(f"bucket_coverage_lt_{self._min_bucket_coverage}")
        if int(outcome.source_coverage) < int(self._min_source_coverage):
            reasons.append(f"source_coverage_lt_{self._min_source_coverage}")

        return ExpansionDecision(
            should_expand=bool(reasons),
            reasons=reasons,
        )
