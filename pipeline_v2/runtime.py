"""End-to-end v2 pipeline runtime from RunRequest to artifacts."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import re
from time import perf_counter
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence
from urllib.parse import urlparse

from core import Artifact, ArtifactType, RawItem, RenderStatus, RunMode, RunRequest
from orchestrator import RunOrchestrator
from pipeline_v2.data_mode import resolve_data_mode, should_allow_smoke
from pipeline_v2.dedup_cluster import cluster, dedup_exact, embed, merge_cluster
from pipeline_v2.evidence_auditor import EvidenceAuditor, EvidenceAuditorProtocol, LLMEvidenceAuditor
from pipeline_v2.normalize import normalize
from pipeline_v2.notification import notify_user, post_to_web, send_email
from pipeline_v2.planner import LLMPlanner, ResearchPlanner, RetrievalPlan
from pipeline_v2.prompt_compiler import compile_storyboard
from pipeline_v2.retrieval_controller import SelectionController
from pipeline_v2.report_export import export_package, generate_onepager, generate_thumbnail
from pipeline_v2.scoring import rank_items
from pipeline_v2.script_generator import build_facts, generate_script
from pipeline_v2.sanitize import is_allowed_citation_url, normalize_url
from pipeline_v2.storyboard_generator import auto_fix_storyboard, script_to_storyboard, validate_storyboard
from pipeline_v2.topic_profile import TopicProfile
from render import align_subtitles, mix_bgm, tts_generate
from render.manager import RenderManager
from sources import connectors

import httpx

logger = logging.getLogger(__name__)

ConnectorFunc = Callable[..., Any]


class RunCancelledError(RuntimeError):
    """Raised when cancellation has been requested for an active run."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")  # type: ignore[no-any-return]
    if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass
class RunExecutionResult:
    run_id: str
    output_dir: str
    top_item_ids: List[str]
    render_job_id: Optional[str]
    data_mode: str


@dataclass
class RecallProfile:
    limit_multiplier: int = 1
    window: str = "today"
    expanded_queries: bool = False
    phase: str = "base"


class RunPipelineRuntime:
    """Worker runtime that executes queued runs and dispatches render jobs."""

    def __init__(
        self,
        *,
        orchestrator: RunOrchestrator,
        render_manager: Optional[RenderManager] = None,
        output_root: str | Path = "data/outputs/v2_runs",
        connector_overrides: Optional[Mapping[str, ConnectorFunc]] = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._render_manager = render_manager or RenderManager()
        self._output_root = Path(output_root)
        self._output_root.mkdir(parents=True, exist_ok=True)
        self._connector_overrides = dict(connector_overrides or {})

    def run_next(self) -> Optional[RunExecutionResult]:
        """Process one queued run request end-to-end (excluding render worker)."""
        picked = self._orchestrator.dequeue_next_run()
        if not picked:
            return None
        run_id, request = picked
        logger.info("run_start run_id=%s mode=%s", run_id, request.mode.value)

        try:
            result = asyncio.run(self._execute_run(run_id, request))
            self._orchestrator.mark_run_completed(run_id)
            self._orchestrator.append_event(run_id, "run_completed", "Run completed successfully")
            logger.info("run_completed run_id=%s", run_id)
            return result
        except RunCancelledError as exc:
            self._orchestrator.mark_run_canceled(run_id, str(exc))
            self._orchestrator.append_event(run_id, "run_canceled", str(exc))
            logger.info("run_canceled run_id=%s reason=%s", run_id, exc)
            return None
        except Exception as exc:
            self._orchestrator.mark_run_failed(run_id, str(exc))
            self._orchestrator.append_event(run_id, "run_failed", str(exc))
            logger.exception("run_failed run_id=%s error=%s", run_id, exc)
            return None

    def process_next_render(self) -> Optional[RenderStatus]:
        """Process one queued render job and attach MP4 artifact to its run."""
        status = self._render_manager.process_next()
        if not status:
            return None
        if status.output_path:
            self._record_artifact(
                status.run_id,
                Artifact(
                    type=ArtifactType.MP4,
                    path=status.output_path,
                    metadata={
                        "render_job_id": status.render_job_id,
                        "retry_count": status.retry_count,
                        "failed_shot_indices": list(status.failed_shot_indices or []),
                        "valid_mp4": status.valid_mp4,
                        "probe_error": status.probe_error,
                    },
                ),
            )
        self._orchestrator.append_event(
            status.run_id,
            "render_completed",
            f"render_job_id={status.render_job_id} output={status.output_path or ''}",
        )
        return status

    def confirm_render(self, render_job_id: str, *, approved: bool = True) -> Optional[RenderStatus]:
        return self._render_manager.confirm_render(render_job_id, approved=approved)

    def cancel_render(self, render_job_id: str) -> bool:
        return self._render_manager.cancel_render(render_job_id)

    def get_render_status(self, render_job_id: str) -> Optional[RenderStatus]:
        return self._render_manager.poll_render(render_job_id)

    def get_run_bundle(self, run_id: str) -> Dict[str, Any]:
        """Get status + artifacts + render state bundle for API responses."""
        status = self._orchestrator.get_run_status(run_id)
        artifacts = self._orchestrator.list_artifacts(run_id)
        events = self._orchestrator.list_events(run_id)
        render_job_id = self._orchestrator.get_render_job(run_id)
        render_status = self._render_manager.poll_render(render_job_id) if render_job_id else None
        return {
            "run_id": run_id,
            "status": status.model_dump(mode="json") if status else None,
            "artifacts": [artifact.model_dump(mode="json") for artifact in artifacts],
            "events": events,
            "render_job_id": render_job_id,
            "render_status": render_status.model_dump(mode="json") if render_status else None,
        }

    async def _execute_run(self, run_id: str, request: RunRequest) -> RunExecutionResult:
        self._ensure_not_canceled(run_id)
        out_dir = self._output_root / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        self._orchestrator.append_event(run_id, "run_started", f"output_dir={out_dir}")

        data_mode = resolve_data_mode(budget=dict(request.budget or {}))
        if data_mode == "smoke" and not should_allow_smoke(budget=dict(request.budget or {})):
            data_mode = "live"
            self._orchestrator.append_event(run_id, "data_mode_forced_live", "smoke mode requires explicit request")

        topic_value = str(request.topic or "").strip()
        topic_profile: Optional[TopicProfile] = None
        retrieval_plan: Optional[RetrievalPlan] = None
        if data_mode == "live" and topic_value:
            topic_profile = TopicProfile.for_topic(topic_value)
            planner = self._resolve_research_planner(request)
            retrieval_plan = planner.plan(topic_value, time_window=request.time_window)

        top_count = self._top_n(request)
        controller = self._build_selection_controller(request, top_count=top_count)
        auditor = self._resolve_evidence_auditor(request, topic_profile=topic_profile)
        recall_profiles = self._recall_profiles(request, data_mode=data_mode, retrieval_plan=retrieval_plan)
        budget = dict(request.budget or {})
        max_recall_attempts_raw = budget.get(
            "max_recall_attempts",
            os.getenv("ARA_V2_MAX_RECALL_ATTEMPTS", len(recall_profiles)),
        )
        try:
            max_recall_attempts = max(1, int(float(max_recall_attempts_raw)))
        except Exception:
            max_recall_attempts = max(1, int(len(recall_profiles)))
        if len(recall_profiles) > max_recall_attempts:
            recall_profiles = list(recall_profiles)[:max_recall_attempts]
        attempts: List[Dict[str, Any]] = []
        selected: Optional[Dict[str, Any]] = None
        quality_triggered_any = False

        for idx, profile in enumerate(recall_profiles, start=1):
            self._ensure_not_canceled(run_id)
            collect_diag: Dict[str, Any] = {}
            raw_items = await self._collect_raw_items(
                request,
                data_mode=data_mode,
                profile=profile,
                retrieval_plan=retrieval_plan,
                topic_profile=topic_profile,
                diagnostics=collect_diag,
            )
            run_context_candidate = self._build_run_context(
                data_mode=data_mode,
                raw_items=raw_items,
                run_id=run_id,
                topic=request.topic,
            )

            normalized = [normalize(item) for item in raw_items]
            unique_items = dedup_exact(normalized)
            embeddings = embed(unique_items)
            grouped = cluster(unique_items, embeddings)
            merged = [merge_cluster(group) for group in grouped]
            rank_pack = self._rank_with_relevance_gate(
                merged=merged,
                topic=request.topic,
                topic_profile=topic_profile,
                data_mode=data_mode,
                top_count=top_count,
            )
            ranked = list(rank_pack["ranked"] or [])
            relevance_eligible = list(rank_pack["relevance_eligible"] or [])
            audit_records = auditor.audit_rows(ranked_rows=ranked)

            pass_first_outcome = controller.evaluate(
                ranked_rows=relevance_eligible,
                audit_records=audit_records,
                allow_downgrade_fill=False,
                min_relevance_for_selection=float(rank_pack["relevance_threshold"]),
            )
            expansion_decision = controller.expansion_decision(outcome=pass_first_outcome)
            quality_reasons = [str(value).strip() for value in list(expansion_decision.reasons or []) if str(value).strip()]
            has_next_attempt = idx < len(recall_profiles)
            quality_triggered_expansion = bool(expansion_decision.should_expand and has_next_attempt)
            quality_triggered_any = bool(quality_triggered_any or quality_triggered_expansion)

            final_outcome = pass_first_outcome
            if not quality_triggered_expansion and not has_next_attempt:
                final_outcome = controller.evaluate(
                    ranked_rows=relevance_eligible,
                    audit_records=audit_records,
                    allow_downgrade_fill=True,
                    min_relevance_for_selection=float(rank_pack["relevance_threshold"]),
                )

            picks = list(final_outcome.selected_rows or [])
            quality_snapshot = self._attempt_quality_snapshot(
                ranked=ranked,
                picks=picks,
                topic_profile=topic_profile,
                audit_records=audit_records,
                selection_outcome=final_outcome,
            )

            selected_downgrade_count = int(
                sum(1 for verdict in list((final_outcome.selected_verdicts or {}).values()) if str(verdict).strip().lower() == "downgrade")
            )
            selected_pass_count = int(
                sum(1 for verdict in list((final_outcome.selected_verdicts or {}).values()) if str(verdict).strip().lower() == "pass")
            )

            attempt_payload = {
                "attempt": idx,
                "phase": profile.phase,
                "window": str((collect_diag.get("profile") or {}).get("window") or profile.window),
                "query_set": "expanded" if bool(profile.expanded_queries) else "base",
                "profile": dict(collect_diag.get("profile") or {}),
                "raw_items": len(raw_items),
                "candidate_count": int(final_outcome.candidate_count),
                "top_picks_count": len(picks),
                "filtered_by_relevance": max(0, len(ranked) - len(relevance_eligible)),
                "relevance_threshold_used": float(rank_pack["relevance_threshold"]),
                "relaxation_steps": int(rank_pack["relaxation_steps"]),
                "connector_calls": list(collect_diag.get("connector_calls") or []),
                "queries": list(collect_diag.get("queries") or []),
                "must_include_terms": list(collect_diag.get("must_include_terms") or []),
                "must_exclude_terms": list(collect_diag.get("must_exclude_terms") or []),
                "deep_fetch_applied": bool(collect_diag.get("deep_fetch_applied")),
                "deep_fetch_count": int(collect_diag.get("deep_fetch_count", 0) or 0),
                "deep_fetch_details": list(collect_diag.get("deep_fetch_details") or []),
                "bucket_queries": dict(collect_diag.get("bucket_queries") or {}),
                "bucket_hits_summary": dict(quality_snapshot.get("bucket_hits_summary") or {}),
                "bucket_coverage": int(quality_snapshot.get("bucket_coverage", 0) or 0),
                "source_coverage": int(quality_snapshot.get("source_coverage", 0) or 0),
                "hard_match_terms_used": list(quality_snapshot.get("hard_match_terms_used") or []),
                "hard_match_pass_count": int(quality_snapshot.get("hard_match_pass_count", 0) or 0),
                "top_picks_min_relevance": float(quality_snapshot.get("top_picks_min_relevance", 0.0) or 0.0),
                "top_picks_hard_match_count": int(quality_snapshot.get("top_picks_hard_match_count", 0) or 0),
                "top_picks_min_evidence_quality": float(quality_snapshot.get("top_picks_min_evidence_quality", 0.0) or 0.0),
                "pass_count": int(final_outcome.pass_count),
                "downgrade_count": int(final_outcome.downgrade_count),
                "reject_count": int(final_outcome.reject_count),
                "pass_ratio": float(final_outcome.pass_ratio),
                "selected_pass_count": selected_pass_count,
                "selected_downgrade_count": selected_downgrade_count,
                "selected_source_coverage": int(quality_snapshot.get("source_coverage", 0) or 0),
                "selected_all_downgrade": bool(final_outcome.all_selected_downgrade),
                "quality_trigger_reasons": list(quality_reasons),
                "quality_triggered_expansion": bool(quality_triggered_expansion),
                "next_phase_reason": list(quality_reasons),
                "expansion_applied": idx > 1,
            }
            attempts.append(attempt_payload)

            if quality_triggered_expansion:
                continue

            selected = {
                "raw_items": raw_items,
                "run_context": run_context_candidate,
                "normalized": normalized,
                "merged": merged,
                "rank_pack": rank_pack,
                "profile": profile,
                "attempt": idx,
                "quality_snapshot": quality_snapshot,
                "quality_reasons": quality_reasons,
                "selection_outcome": final_outcome,
                "audit_records": audit_records,
            }
            if picks:
                break

        if selected is None:
            raise RuntimeError("no relevant ranked items available")

        raw_items = list(selected["raw_items"] or [])
        run_context = dict(selected["run_context"] or {})
        normalized = list(selected["normalized"] or [])
        merged = list(selected["merged"] or [])
        rank_pack = dict(selected["rank_pack"] or {})
        ranked = list(rank_pack.get("ranked") or [])
        relevance_eligible = list(rank_pack.get("relevance_eligible") or [])
        profile = selected["profile"]
        selection_outcome = selected["selection_outcome"]
        picks = list(selection_outcome.selected_rows or [])
        audit_records = list(selected.get("audit_records") or [])
        records_by_id = {
            str(record.item_id or "").strip(): record
            for record in list(audit_records or [])
            if str(record.item_id or "").strip()
        }
        selected_quality_reasons = [str(value) for value in list(selected.get("quality_reasons") or []) if str(value).strip()]
        if not selected_quality_reasons and quality_triggered_any:
            for item in attempts:
                if bool(item.get("quality_triggered_expansion")):
                    selected_quality_reasons = [
                        str(value) for value in list(item.get("quality_trigger_reasons") or []) if str(value).strip()
                    ]
                    if selected_quality_reasons:
                        break

        base_relevance_threshold = float(rank_pack.get("base_relevance_threshold", 0.55 if data_mode == "live" else 0.0))
        relevance_threshold = float(rank_pack.get("relevance_threshold", base_relevance_threshold))
        relaxation_steps = int(rank_pack.get("relaxation_steps", 0))
        quality_snapshot = dict(selected.get("quality_snapshot") or {})

        if not picks:
            rescued_pick = self._shortage_rescue_pick(
                ranked=ranked,
                relevance_eligible=relevance_eligible,
                audit_records=audit_records,
            )
            if rescued_pick is None:
                raise RuntimeError("no candidates survived evidence audit")

            rescue_id = str((getattr(rescued_pick, "item", None).id if getattr(rescued_pick, "item", None) else "") or "").strip()
            picks = [rescued_pick]
            selection_outcome.selected_rows = list(picks)
            selection_outcome.selected_verdicts = {rescue_id: "downgrade"} if rescue_id else {}
            selection_outcome.selected_downgrade_reasons = (
                {rescue_id: ["shortage_fallback_after_max_phase"]} if rescue_id else {}
            )
            selection_outcome.used_downgrade = True
            selection_outcome.all_selected_downgrade = True

            if "shortage_fallback_after_max_phase" not in selected_quality_reasons:
                selected_quality_reasons.append("shortage_fallback_after_max_phase")

            quality_snapshot = self._attempt_quality_snapshot(
                ranked=ranked,
                picks=picks,
                topic_profile=topic_profile,
                audit_records=audit_records,
                selection_outcome=selection_outcome,
            )
            selected["quality_snapshot"] = dict(quality_snapshot)
            selected["quality_reasons"] = list(selected_quality_reasons)
            selected["selection_outcome"] = selection_outcome

            attempt_idx = int(selected.get("attempt", 1) or 1) - 1
            if 0 <= attempt_idx < len(attempts):
                attempt_payload = dict(attempts[attempt_idx] or {})
                attempt_payload["top_picks_count"] = len(picks)
                attempt_payload["top_picks_min_relevance"] = float(quality_snapshot.get("top_picks_min_relevance", 0.0) or 0.0)
                attempt_payload["top_picks_hard_match_count"] = int(quality_snapshot.get("top_picks_hard_match_count", 0) or 0)
                attempt_payload["bucket_coverage"] = int(quality_snapshot.get("bucket_coverage", 0) or 0)
                attempt_payload["source_coverage"] = int(quality_snapshot.get("source_coverage", 0) or 0)
                attempt_payload["top_picks_min_evidence_quality"] = float(
                    quality_snapshot.get("top_picks_min_evidence_quality", 0.0) or 0.0
                )
                attempt_payload["selected_pass_count"] = 0
                attempt_payload["selected_downgrade_count"] = len(picks)
                attempt_payload["selected_all_downgrade"] = True
                attempt_reasons = [str(value) for value in list(attempt_payload.get("quality_trigger_reasons") or []) if str(value).strip()]
                if "shortage_fallback_after_max_phase" not in attempt_reasons:
                    attempt_reasons.append("shortage_fallback_after_max_phase")
                attempt_payload["quality_trigger_reasons"] = attempt_reasons
                attempts[attempt_idx] = attempt_payload

        audit_payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "topic": str(request.topic or "").strip() or None,
            "requested_top_k": int(top_count),
            "selected_phase": getattr(profile, "phase", "base"),
            "final_top_item_ids": [str(getattr(row.item, "id", "") or "") for row in list(picks or []) if getattr(row, "item", None)],
            "records": [record.model_dump() for record in list(audit_records or [])],
        }
        audit_path = self._write_json(out_dir / "evidence_audit.json", audit_payload)
        self._record_artifact(
            run_id,
            Artifact(
                type=ArtifactType.DIAGNOSIS,
                path=audit_path,
                metadata={"kind": "evidence_audit", "selected_phase": getattr(profile, "phase", "base")},
            ),
        )

        def _drop_reason(row: Any) -> str:
            item_id = str((getattr(row, "item", None).id if getattr(row, "item", None) else "") or "").strip()
            record = records_by_id.get(item_id)
            if record:
                verdict = str(record.verdict or "").strip().lower()
                machine_action = dict(record.machine_action or {})
                reason_code = str(machine_action.get("reason_code") or "").strip().lower()
                if verdict == "reject":
                    return f"rejected:{reason_code or 'audit'}"
                if verdict == "downgrade":
                    return f"downgraded:{reason_code or 'audit'}"

            reasons = " ".join(str(value) for value in list(getattr(row, "reasons", []) or []))
            lowered = reasons.lower()
            if "relevance.relaxed_pick" in lowered:
                return "relaxed_fill"
            if "penalty.relevance" in lowered:
                return "low_relevance"
            if "penalty.body_len" in lowered or "quality.signal.density=0." in lowered:
                return "low_density"
            if "penalty.no_evidence_links" in lowered:
                return "no_evidence_links"
            if "penalty.too_old" in lowered:
                return "too_old"
            if "denylist" in lowered:
                return "denylisted"
            return "deprioritized"

        def _why_ranked(row: Any) -> str:
            item = row.item
            signals = dict((item.metadata or {}).get("quality_signals") or {})
            metadata = dict(item.metadata or {})
            chunks = [f"rel={float(getattr(row, 'relevance_score', 0.0)):.2f}"]
            hard_terms = [str(value).strip() for value in list(metadata.get("topic_hard_match_terms") or []) if str(value).strip()]
            if hard_terms:
                chunks.append(f"hard={','.join(hard_terms[:2])}")
            evidence_links = int(float(signals.get("evidence_links_quality", 0) or 0))
            if evidence_links > 0:
                chunks.append(f"evidence={evidence_links}")
            if bool(signals.get("has_quickstart")):
                chunks.append("quickstart")
            return " | ".join(chunks[:4])

        picked_ids = {pick.item.id for pick in picks}
        candidate_rows = []
        for row in ranked[:30]:
            item = row.item
            metadata = dict(item.metadata or {})
            record = records_by_id.get(str(item.id))
            machine_action = dict((record.machine_action if record else {}) or {})
            candidate_rows.append(
                {
                    "item_id": item.id,
                    "title": item.title,
                    "url": item.url,
                    "source": item.source,
                    "update_time": (item.metadata or {}).get("publish_or_update_time"),
                    "relevance_score": round(float(getattr(row, "relevance_score", 0.0)), 4),
                    "total_score": round(float(getattr(row, "total_score", 0.0)), 4),
                    "hard_match_pass": bool(metadata.get("topic_hard_match_pass")),
                    "hard_match_terms": [str(value) for value in list(metadata.get("topic_hard_match_terms") or []) if str(value).strip()],
                    "bucket_hits": [str(value) for value in list(metadata.get("bucket_hits") or []) if str(value).strip()],
                    "audit_verdict": str(record.verdict if record else "unknown"),
                    "audit_reason_code": str(machine_action.get("reason_code") or ""),
                    "drop_reason": None if item.id in picked_ids else _drop_reason(row),
                }
            )
        top_dropped = [row for row in candidate_rows if row.get("drop_reason")][:10]

        selected_attempt_payload = (attempts[int(selected.get("attempt", 1)) - 1] or {}) if attempts else {}
        hard_match_terms_used = [str(value) for value in list(selected_attempt_payload.get("hard_match_terms_used") or []) if str(value).strip()]
        hard_match_pass_count = int(selected_attempt_payload.get("hard_match_pass_count", 0) or 0)
        top_picks_min_relevance = float(selected_attempt_payload.get("top_picks_min_relevance", 0.0) or 0.0)
        top_picks_hard_match_count = int(selected_attempt_payload.get("top_picks_hard_match_count", 0) or 0)
        selected_bucket_coverage = int(selected_attempt_payload.get("bucket_coverage", 0) or 0)
        top_picks_min_evidence_quality = float(
            selected_attempt_payload.get("top_picks_min_evidence_quality", quality_snapshot.get("top_picks_min_evidence_quality", 0.0))
            or 0.0
        )

        selected_verdicts = dict(selection_outcome.selected_verdicts or {})
        selected_downgrade_reasons = dict(selection_outcome.selected_downgrade_reasons or {})
        selected_pass_count = int(sum(1 for verdict in list(selected_verdicts.values()) if str(verdict).strip().lower() == "pass"))
        selected_downgrade_count = int(
            sum(1 for verdict in list(selected_verdicts.values()) if str(verdict).strip().lower() == "downgrade")
        )
        why_not_more: List[str] = []
        if len(picks) < top_count:
            why_not_more.append(f"top_picks_lt_{top_count}")
        if selected_pass_count < top_count:
            why_not_more.append(f"pass_count_lt_{top_count}")
        if selected_downgrade_count > 0:
            why_not_more.append("used_downgrade_fallback_after_expansion")
        if bool(selection_outcome.all_selected_downgrade):
            why_not_more.append("all_top_picks_downgrade_after_max_phase")
        if len({str(entry.item.source or "").strip().lower() for entry in list(picks or []) if getattr(entry, "item", None)}) < 2:
            why_not_more.append("source_diversity_lt_2")
        if selected_bucket_coverage < 2 and (topic_profile and topic_profile.minimum_bucket_coverage >= 2):
            why_not_more.append("bucket_coverage_lt_2")

        retrieval_diagnosis = {
            "run_id": run_id,
            "topic": str(request.topic or "").strip() or None,
            "time_window": str(request.time_window or "").strip() or None,
            "requested_top_k": top_count,
            "plan": retrieval_plan.model_dump() if retrieval_plan else None,
            "topic_profile": {
                "key": str(topic_profile.key if topic_profile else ""),
                "requires_hard_gate": bool(topic_profile.requires_hard_gate) if topic_profile else False,
                "minimum_bucket_coverage": int(topic_profile.minimum_bucket_coverage) if topic_profile else 1,
            }
            if topic_profile
            else None,
            "attempts": attempts,
            "selected_attempt": int(selected.get("attempt", 1)),
            "selected_phase": getattr(profile, "phase", "base"),
            "selected_queries": list(selected_attempt_payload.get("queries") or []),
            "hard_match_terms_used": hard_match_terms_used,
            "hard_match_pass_count": hard_match_pass_count,
            "top_picks_min_relevance": top_picks_min_relevance,
            "top_picks_hard_match_count": top_picks_hard_match_count,
            "bucket_coverage": selected_bucket_coverage,
            "source_coverage": int(selected_attempt_payload.get("source_coverage", quality_snapshot.get("source_coverage", 0)) or 0),
            "top_picks_min_evidence_quality": top_picks_min_evidence_quality,
            "selected_pass_count": selected_pass_count,
            "selected_downgrade_count": selected_downgrade_count,
            "selected_all_downgrade": bool(selection_outcome.all_selected_downgrade),
            "quality_triggered_expansion": bool(quality_triggered_any),
            "quality_trigger_reasons": list(selected_quality_reasons or []),
            "why_not_more": why_not_more,
            "candidate_rows": candidate_rows,
            "top_dropped_items": top_dropped,
            "evidence_audit_path": audit_path,
        }
        diagnosis_path = self._write_json(out_dir / "retrieval_diagnosis.json", retrieval_diagnosis)
        self._record_artifact(
            run_id,
            Artifact(type=ArtifactType.DIAGNOSIS, path=diagnosis_path, metadata={"selected_phase": getattr(profile, "phase", "base")}),
        )

        run_context["retrieval"] = {
            "diagnosis_path": diagnosis_path,
            "evidence_audit_path": audit_path,
            "selected_phase": getattr(profile, "phase", "base"),
            "selected_attempt": int(selected.get("attempt", 1)),
            "plan": retrieval_plan.model_dump() if retrieval_plan else None,
            "hard_match_terms_used": hard_match_terms_used,
            "hard_match_pass_count": hard_match_pass_count,
            "top_picks_min_relevance": top_picks_min_relevance,
            "top_picks_hard_match_count": top_picks_hard_match_count,
            "bucket_coverage": selected_bucket_coverage,
            "source_coverage": int(selected_attempt_payload.get("source_coverage", quality_snapshot.get("source_coverage", 0)) or 0),
            "top_picks_min_evidence_quality": top_picks_min_evidence_quality,
            "selected_pass_count": selected_pass_count,
            "selected_downgrade_count": selected_downgrade_count,
            "quality_triggered_expansion": bool(quality_triggered_any),
            "quality_trigger_reasons": list(selected_quality_reasons or []),
            "why_not_more": why_not_more,
            "expansion_steps": [
                {
                    "attempt": int(item.get("attempt", 0) or 0),
                    "phase": item.get("phase"),
                    "profile": item.get("profile"),
                    "queries": list(item.get("queries") or []),
                    "bucket_queries": dict(item.get("bucket_queries") or {}),
                    "bucket_coverage": int(item.get("bucket_coverage", 0) or 0),
                    "hard_match_pass_count": int(item.get("hard_match_pass_count", 0) or 0),
                    "top_picks_min_relevance": float(item.get("top_picks_min_relevance", 0.0) or 0.0),
                    "top_picks_hard_match_count": int(item.get("top_picks_hard_match_count", 0) or 0),
                    "source_coverage": int(item.get("source_coverage", 0) or 0),
                    "quality_triggered_expansion": bool(item.get("quality_triggered_expansion")),
                    "quality_trigger_reasons": list(item.get("quality_trigger_reasons") or []),
                    "deep_fetch_applied": bool(item.get("deep_fetch_applied")),
                    "deep_fetch_count": int(item.get("deep_fetch_count", 0) or 0),
                    "deep_fetch_details": list(item.get("deep_fetch_details") or []),
                }
                for item in attempts
                if bool(item.get("expansion_applied"))
            ],
            "attempt_count": len(attempts),
        }

        self._orchestrator.mark_run_progress(run_id, 0.18)
        self._orchestrator.append_event(
            run_id,
            "ingest_done",
            (
                f"raw_items={len(raw_items)} data_mode={data_mode} "
                f"connector_sources={len(run_context.get('connector_stats', {}))} "
                f"selected_phase={getattr(profile, 'phase', 'base')}"
            ),
        )
        self._ensure_not_canceled(run_id)

        picked_ids = {pick.item.id for pick in picks}
        audit_by_item = dict(records_by_id or {})
        verdict_counts = {"pass": 0, "downgrade": 0, "reject": 0}
        for record in list(audit_records or []):
            key = str(record.verdict or "").strip().lower()
            if key in verdict_counts:
                verdict_counts[key] = int(verdict_counts[key]) + 1
        top_verdicts = {}
        top_reason_codes = {}
        top_reasons = {}
        for entry in picks:
            item_id = str(entry.item.id)
            record = audit_by_item.get(item_id)
            top_verdicts[item_id] = str(record.verdict if record else "unknown")
            machine_action = dict((record.machine_action if record else {}) or {})
            top_reason_codes[item_id] = str(machine_action.get("reason_code") or "")
            top_reasons[item_id] = list(record.reasons or []) if record else []
        run_context["ranking_stats"] = {
            "topic": str(request.topic or "").strip() or None,
            "candidate_count": len(ranked),
            "top_picks_count": len(picks),
            "filtered_by_relevance": max(0, len(ranked) - len(relevance_eligible)),
            "requested_top_k": top_count,
            "top_item_ids": [entry.item.id for entry in picks],
            "top_relevance_scores": [round(float(entry.relevance_score), 4) for entry in picks],
            "relevance_threshold": relevance_threshold,
            "relevance_threshold_base": base_relevance_threshold,
            "topic_relevance_threshold_used": relevance_threshold,
            "relaxation_steps": relaxation_steps,
            "candidate_shortage": len(picks) < top_count,
            "diversity_sources": sorted({str(entry.item.source or "").strip().lower() for entry in picks if str(entry.item.source or "").strip()}),
            "selected_recall_phase": getattr(profile, "phase", "base"),
            "recall_attempt_count": len(attempts),
            "evidence_audit_path": audit_path,
            "evidence_audit_verdict_counts": verdict_counts,
            "top_evidence_audit_verdicts": top_verdicts,
            "top_evidence_audit_reason_codes": top_reason_codes,
            "top_evidence_audit_reasons": top_reasons,
            "min_body_len_for_top_picks": 300,
            "hard_match_terms_used": hard_match_terms_used,
            "hard_match_pass_count": hard_match_pass_count,
            "top_picks_min_relevance": top_picks_min_relevance,
            "top_picks_hard_match_count": top_picks_hard_match_count,
            "top_picks_min_evidence_quality": top_picks_min_evidence_quality,
            "selected_pass_count": selected_pass_count,
            "selected_downgrade_count": selected_downgrade_count,
            "selected_all_downgrade": bool(selection_outcome.all_selected_downgrade),
            "bucket_coverage": selected_bucket_coverage,
            "selected_source_coverage": int(selected_attempt_payload.get("source_coverage", quality_snapshot.get("source_coverage", 0)) or 0),
            "top_bucket_hits": sorted(
                {
                    str(bucket).strip()
                    for entry in picks
                    for bucket in list((entry.item.metadata or {}).get("bucket_hits") or [])
                    if str(bucket).strip()
                }
            ),
            "quality_triggered_expansion": bool(quality_triggered_any),
            "quality_trigger_reasons": selected_quality_reasons,
            "why_not_more": why_not_more,
            "top_why_ranked": [_why_ranked(entry) for entry in picks],
            "top_quality_signals": [dict((entry.item.metadata or {}).get("quality_signals") or {}) for entry in picks],
            "drop_reason_samples": [
                {
                    "item_id": entry.item.id,
                    "title": entry.item.title,
                    "reason": _drop_reason(entry),
                }
                for entry in [row for row in ranked if row.item.id not in picked_ids][:3]
            ],
        }
        run_context["diagnosis_path"] = diagnosis_path
        run_context["evidence_audit_path"] = audit_path
        run_context_path = self._write_json(out_dir / "run_context.json", run_context)

        self._orchestrator.mark_run_progress(run_id, 0.42)
        self._orchestrator.append_event(
            run_id,
            "ranking_done",
            (
                f"picked={len(picks)} threshold_used={relevance_threshold:.2f} "
                f"relaxation_steps={relaxation_steps} phase={getattr(profile, 'phase', 'base')}"
            ),
        )
        if len(picks) < top_count:
            self._orchestrator.append_event(
                run_id,
                "ranking_shortage",
                f"requested={top_count} actual={len(picks)} threshold_used={relevance_threshold:.2f}",
            )

        primary = picks[0].item
        duration = self._duration_sec(request)
        facts = build_facts(primary, topic=request.topic)
        primary.metadata["facts"] = facts
        facts_path = self._write_json(out_dir / "facts.json", facts)
        self._record_artifact(run_id, Artifact(type=ArtifactType.FACTS, path=facts_path, metadata={"item_id": primary.id}))
        script = generate_script(
            primary,
            duration_sec=duration,
            platform=self._platform(request),
            tone=self._tone(request),
            facts=facts,
            topic=request.topic,
        )
        script_path = self._write_json(out_dir / "script.json", script)
        self._record_artifact(
            run_id,
            Artifact(type=ArtifactType.SCRIPT, path=script_path, metadata={"item_id": primary.id, "facts_path": facts_path}),
        )
        self._ensure_not_canceled(run_id)

        board = script_to_storyboard(
            script,
            {
                "run_id": run_id,
                "item_id": primary.id,
                "aspect": "9:16",
                "min_shots": 5,
                "max_shots": 8,
            },
        )

        materials_manifest = self._build_materials_manifest(run_id=run_id, primary=primary, picks=picks, run_context=run_context)
        materials_manifest = self._materialize_assets(out_dir=out_dir, manifest=materials_manifest)
        board = self._attach_reference_assets(board=board, materials=materials_manifest)

        valid, errors = validate_storyboard(board)
        if not valid:
            board, fixes = auto_fix_storyboard(board)
            valid2, errors2 = validate_storyboard(board)
            if not valid2:
                raise RuntimeError(f"storyboard validation failed: {errors2}")
            self._orchestrator.append_event(run_id, "storyboard_autofix", ",".join(fixes))
            if errors:
                self._orchestrator.append_event(run_id, "storyboard_errors", " | ".join(errors))

        board_path = self._write_json(out_dir / "storyboard.json", board.model_dump(mode="json"))
        prompt_specs = compile_storyboard(board, style_profile=self._style_profile(request))
        prompt_bundle_path = self._write_json(
            out_dir / "prompt_bundle.json",
            [spec.model_dump(mode="json") for spec in prompt_specs],
        )
        materials_path = self._write_json(
            out_dir / "materials.json",
            materials_manifest,
        )
        self._record_artifact(
            run_id,
            Artifact(
                type=ArtifactType.STORYBOARD,
                path=board_path,
                metadata={
                    "prompt_bundle_path": prompt_bundle_path,
                    "materials_path": materials_path,
                    "shots": len(board.shots),
                    "facts_path": facts_path,
                },
            ),
        )
        self._orchestrator.mark_run_progress(run_id, 0.63)
        self._orchestrator.append_event(run_id, "script_stage_done", f"duration={duration}s shots={len(board.shots)}")
        self._ensure_not_canceled(run_id)

        onepager_path = generate_onepager(picks, primary.citations, out_dir=out_dir, run_context=run_context)
        thumbnail_path = generate_thumbnail(
            primary.title,
            [entry.item.title for entry in picks[1:4]],
            {"fg": "#0b1320", "bg": "#e6eef8", "accent": "#0f766e"},
            out_dir=out_dir,
        )
        self._record_artifact(
            run_id,
            Artifact(
                type=ArtifactType.ONEPAGER,
                path=onepager_path,
                metadata={
                    "top_count": len(picks),
                    "prompt_bundle_path": prompt_bundle_path,
                    "materials_path": materials_path,
                    "run_context_path": run_context_path,
                    "data_mode": data_mode,
                    "facts_path": facts_path,
                },
            ),
        )
        self._record_artifact(run_id, Artifact(type=ArtifactType.THUMBNAIL, path=thumbnail_path, metadata={}))

        audio_path = tts_generate(script, {"voice": self._voice(request), "out_dir": str(out_dir)})
        srt_path = align_subtitles({**script, "output_dir": str(out_dir)}, audio_path)
        mixed_audio = mix_bgm(audio_path, {"out_dir": str(out_dir)})
        self._record_artifact(run_id, Artifact(type=ArtifactType.AUDIO, path=mixed_audio, metadata={"raw_audio_path": audio_path}))
        self._record_artifact(run_id, Artifact(type=ArtifactType.SRT, path=srt_path, metadata={}))

        self._orchestrator.mark_run_progress(run_id, 0.82)
        self._orchestrator.append_event(run_id, "postprocess_done", "audio + subtitles generated")
        self._ensure_not_canceled(run_id)

        render_job_id: Optional[str] = None
        if self._should_render(request):
            render_job_id = self._render_manager.enqueue_render(
                run_id,
                prompt_specs,
                budget=self._render_budget(request),
            )
            self._orchestrator.set_render_job(run_id, render_job_id)
            self._orchestrator.append_event(run_id, "render_enqueued", f"render_job_id={render_job_id}")
        else:
            self._orchestrator.append_event(run_id, "render_skipped", "render disabled by request")

        artifacts = self._orchestrator.list_artifacts(run_id)
        package_path = export_package(out_dir, artifacts, package_name=f"{run_id}_package")
        self._record_artifact(
            run_id,
            Artifact(type=ArtifactType.ZIP, path=package_path, metadata={"artifact_count": len(artifacts)}),
        )

        payload = {
            "run_id": run_id,
            "status": "completed",
            "top_item": primary.title,
            "render_job_id": render_job_id,
        }
        notify_user(request.user_id, f"Run {run_id} completed", out_dir=out_dir)
        if "web" in [str(item).lower() for item in list(request.output_targets or [])]:
            post_to_web(run_id, payload, out_dir=out_dir)
        email_to = str((request.budget or {}).get("email") or "").strip()
        if email_to:
            send_email(email_to, f"Run {run_id} completed", json.dumps(payload, ensure_ascii=False), out_dir=out_dir)

        self._orchestrator.mark_run_progress(run_id, 0.99)
        return RunExecutionResult(
            run_id=run_id,
            output_dir=str(out_dir),
            top_item_ids=[entry.item.id for entry in picks],
            render_job_id=render_job_id,
            data_mode=data_mode,
        )

    async def _collect_raw_items(
        self,
        request: RunRequest,
        *,
        data_mode: str,
        profile: Optional[RecallProfile] = None,
        retrieval_plan: Optional[RetrievalPlan] = None,
        topic_profile: Optional[TopicProfile] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> List[RawItem]:
        phase = profile or RecallProfile(window=str(request.time_window or "today"))
        plan = retrieval_plan
        phase_window = plan.window_for_phase(phase.phase, fallback=phase.window) if plan else phase.window
        phase_queries = plan.queries_for_phase(phase.phase) if plan else []
        bucket_queries = plan.bucket_queries_for_phase(phase.phase) if plan else {}
        topic = str(request.topic or "").strip()
        limit_multiplier = max(1, int(phase.limit_multiplier))
        base_limit = 12 * limit_multiplier
        raw_items: List[RawItem] = []
        connector_calls: List[Dict[str, Any]] = []
        queries_by_source: Dict[str, List[str]] = {}
        include_terms_by_source: Dict[str, List[str]] = {}
        exclude_terms_by_source: Dict[str, List[str]] = {}
        budget = dict(request.budget or {})

        def _resolve_int_budget(name: str, *, env_name: str, default: int, minimum: int = 1) -> int:
            raw = budget.get(name, None)
            if raw in (None, ""):
                raw = os.getenv(env_name)
            try:
                value = int(float(raw))
            except Exception:
                value = int(default)
            return max(minimum, value)

        def _resolve_float_budget(name: str, *, env_name: str, default: float, minimum: float = 1.0) -> float:
            raw = budget.get(name, None)
            if raw in (None, ""):
                raw = os.getenv(env_name)
            try:
                value = float(raw)
            except Exception:
                value = float(default)
            return max(float(minimum), float(value))

        base_query_cap = _resolve_int_budget(
            "query_cap_base",
            env_name="ARA_V2_QUERY_CAP_BASE",
            default=8,
            minimum=1,
        )
        expanded_query_cap = _resolve_int_budget(
            "query_cap_expanded",
            env_name="ARA_V2_QUERY_CAP_EXPANDED",
            default=12,
            minimum=1,
        )
        query_cap = expanded_query_cap if bool(phase.expanded_queries) else base_query_cap
        connector_timeout_sec = _resolve_float_budget(
            "connector_timeout_sec",
            env_name="ARA_V2_CONNECTOR_TIMEOUT_SEC",
            default=20.0,
            minimum=5.0,
        )

        def _trim_queries(values: Sequence[str]) -> List[str]:
            return self._dedupe_tokens(list(values or []))[: max(1, int(query_cap))]

        def _source_constraints(source: str) -> tuple[List[str], List[str]]:
            include_terms = list(plan.must_include_terms) if plan else []
            exclude_terms = list(plan.must_exclude_terms) if plan else []
            source_key = str(source or "").strip().lower()
            plan_filters = dict((plan.source_filters or {}).get(source_key) or {}) if plan else {}
            profile_filters = dict((topic_profile.source_filters or {}).get(source_key) or {}) if topic_profile else {}
            include_terms.extend(list(plan_filters.get("must_include_any") or []))
            include_terms.extend(list(profile_filters.get("must_include_any") or []))
            exclude_terms.extend(list(plan_filters.get("must_exclude_any") or []))
            exclude_terms.extend(list(profile_filters.get("must_exclude_any") or []))
            return self._dedupe_tokens(include_terms), self._dedupe_tokens(exclude_terms)

        async def _call(name: str, **kwargs: Any) -> List[RawItem]:
            started = perf_counter()
            if not self._connector_exists(name):
                connector_calls.append(
                    {
                        "name": name,
                        "count": 0,
                        "duration_ms": 0,
                        "status": "missing",
                        "kwargs": {key: _to_jsonable(value) for key, value in kwargs.items()},
                    }
                )
                return []
            try:
                payload = await asyncio.wait_for(self._invoke_connector(name, **kwargs), timeout=connector_timeout_sec)
                duration_ms = int((perf_counter() - started) * 1000)
                connector_calls.append(
                    {
                        "name": name,
                        "count": len(payload),
                        "duration_ms": duration_ms,
                        "status": "ok",
                        "kwargs": {key: _to_jsonable(value) for key, value in kwargs.items()},
                    }
                )
                return payload
            except asyncio.TimeoutError:
                duration_ms = int((perf_counter() - started) * 1000)
                connector_calls.append(
                    {
                        "name": name,
                        "count": 0,
                        "duration_ms": duration_ms,
                        "status": "timeout",
                        "error": f"timeout>{connector_timeout_sec:.1f}s",
                        "kwargs": {key: _to_jsonable(value) for key, value in kwargs.items()},
                    }
                )
                return []
            except Exception as exc:
                duration_ms = int((perf_counter() - started) * 1000)
                connector_calls.append(
                    {
                        "name": name,
                        "count": 0,
                        "duration_ms": duration_ms,
                        "status": "error",
                        "error": str(exc),
                        "kwargs": {key: _to_jsonable(value) for key, value in kwargs.items()},
                    }
                )
                return []

        if data_mode == "live" and topic:
            github_queries = _trim_queries(plan.queries_for_phase(phase.phase, source="github") if plan else phase_queries)
            github_include_terms, github_exclude_terms = _source_constraints("github")
            queries_by_source["github"] = list(github_queries or [])
            include_terms_by_source["github"] = list(github_include_terms or [])
            exclude_terms_by_source["github"] = list(github_exclude_terms or [])
            hf_queries = _trim_queries(plan.queries_for_phase(phase.phase, source="huggingface") if plan else phase_queries)
            hf_include_terms, hf_exclude_terms = _source_constraints("huggingface")
            queries_by_source["huggingface"] = list(hf_queries or [])
            include_terms_by_source["huggingface"] = list(hf_include_terms or [])
            exclude_terms_by_source["huggingface"] = list(hf_exclude_terms or [])
            hn_queries = _trim_queries(plan.queries_for_phase(phase.phase, source="hackernews") if plan else phase_queries)
            hn_include_terms, hn_exclude_terms = _source_constraints("hackernews")
            queries_by_source["hackernews"] = list(hn_queries or [])
            include_terms_by_source["hackernews"] = list(hn_include_terms or [])
            exclude_terms_by_source["hackernews"] = list(hn_exclude_terms or [])
            topic_tasks: List[Awaitable[List[RawItem]]] = [
                _call(
                    "fetch_github_topic_search",
                    topic=topic,
                    time_window=phase_window,
                    limit=(
                        plan.source_limit_for_phase(source="github", phase=phase.phase, fallback=max(8, base_limit))
                        if plan
                        else max(8, base_limit)
                    ),
                    expanded=bool(phase.expanded_queries),
                    queries=github_queries,
                    must_include_terms=github_include_terms,
                    must_exclude_terms=github_exclude_terms,
                ),
                _call(
                    "fetch_huggingface_search",
                    topic=topic,
                    time_window=phase_window,
                    limit=(
                        plan.source_limit_for_phase(source="huggingface", phase=phase.phase, fallback=max(6, base_limit // 2))
                        if plan
                        else max(6, base_limit // 2)
                    ),
                    expanded=bool(phase.expanded_queries),
                    queries=hf_queries,
                    must_include_terms=hf_include_terms,
                    must_exclude_terms=hf_exclude_terms,
                ),
                _call(
                    "fetch_hackernews_search",
                    topic=topic,
                    time_window=phase_window,
                    limit=(
                        plan.source_limit_for_phase(source="hackernews", phase=phase.phase, fallback=max(6, base_limit // 2))
                        if plan
                        else max(6, base_limit // 2)
                    ),
                    expanded=bool(phase.expanded_queries),
                    queries=hn_queries,
                    must_include_terms=hn_include_terms,
                    must_exclude_terms=hn_exclude_terms,
                ),
            ]
            topic_results = await asyncio.gather(*topic_tasks, return_exceptions=True)
            for result in topic_results:
                if isinstance(result, Exception):
                    continue
                raw_items.extend(list(result or []))

        tasks: List[Awaitable[List[RawItem]]] = [
            _call("fetch_github_trending", max_results=base_limit),
            _call("fetch_huggingface_trending", max_results=base_limit),
            _call("fetch_hackernews_top", max_results=base_limit),
        ]
        tier_a_results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in tier_a_results:
            if isinstance(result, Exception):
                continue
            raw_items.extend(list(result or []))

        repo_names = [
            str(item.title).strip()
            for item in raw_items
            if item.source == "github" and "/" in str(item.title)
        ][:3]
        if repo_names:
            raw_items.extend(
                await _call(
                    "fetch_github_releases",
                    repo_full_names=repo_names,
                    max_results_per_repo=1,
                )
            )

        include_tier_b = bool((request.budget or {}).get("include_tier_b", True))
        if include_tier_b:
            feed_urls = list((request.budget or {}).get("rss_feeds") or [])
            for feed_url in feed_urls[:2]:
                raw_items.extend(await _call("fetch_rss_feed", feed_url=feed_url, max_results=6))

            seed_url = str((request.budget or {}).get("seed_url") or "").strip()
            if seed_url:
                raw_items.extend(await _call("fetch_web_article", url=seed_url))

        deep_fetch_applied = False
        deep_fetch_count = 0
        deep_fetch_details: List[Dict[str, Any]] = []
        if data_mode == "live" and raw_items:
            raw_items, deep_fetch_applied, deep_fetch_count, deep_fetch_details = await self._apply_deep_extraction(
                raw_items,
                max_items=2,
            )
            raw_items = self._annotate_raw_item_topic_context(
                raw_items,
                retrieval_plan=plan,
                topic_profile=topic_profile,
            )

        if diagnostics is not None:
            diagnostics["phase"] = phase.phase
            diagnostics["profile"] = {
                "limit_multiplier": limit_multiplier,
                "window": phase_window,
                "expanded_queries": bool(phase.expanded_queries),
            }
            diagnostics["queries"] = phase_queries
            diagnostics["query_cap"] = int(query_cap)
            diagnostics["connector_timeout_sec"] = float(connector_timeout_sec)
            diagnostics["queries_by_source"] = queries_by_source
            diagnostics["bucket_queries"] = bucket_queries
            diagnostics["must_include_terms"] = self._dedupe_tokens(
                [value for values in include_terms_by_source.values() for value in list(values or [])]
            )
            diagnostics["must_exclude_terms"] = self._dedupe_tokens(
                [value for values in exclude_terms_by_source.values() for value in list(values or [])]
            )
            diagnostics["must_include_terms_by_source"] = include_terms_by_source
            diagnostics["must_exclude_terms_by_source"] = exclude_terms_by_source
            diagnostics["deep_fetch_applied"] = deep_fetch_applied
            diagnostics["deep_fetch_count"] = deep_fetch_count
            diagnostics["deep_fetch_details"] = deep_fetch_details
            diagnostics["connector_calls"] = connector_calls

        if not raw_items and data_mode == "live":
            raise RuntimeError("no live items fetched from connectors")

        if not raw_items:
            raw_items.append(self._synthetic_item(request))
        return raw_items

    async def _apply_deep_extraction(
        self,
        raw_items: Sequence[RawItem],
        *,
        max_items: int = 2,
    ) -> tuple[List[RawItem], bool, int, List[Dict[str, Any]]]:
        refreshed: List[RawItem] = [item if isinstance(item, RawItem) else RawItem(**item) for item in list(raw_items or [])]
        candidates: List[tuple[int, str]] = []

        for idx, item in enumerate(refreshed):
            source = str(item.source or "").strip().lower()
            body_len = len(re.sub(r"\s+", " ", str(item.body or "").strip()))
            if source == "huggingface":
                metadata = dict(item.metadata or {})
                extraction_method = str(metadata.get("extraction_method") or "").strip().lower()
                metadata_only = "metadata" in extraction_method
                if body_len <= 0 or (metadata_only and body_len < 180):
                    candidates.append((idx, "hf_raw_card"))
            elif source in {"hackernews", "web_article"}:
                if body_len >= 600:
                    continue
                url = str(item.url or "").strip()
                if not url.startswith(("http://", "https://")):
                    continue
                host = str(urlparse(url).netloc or "").strip().lower()
                if host in {"news.ycombinator.com", "github.com", "huggingface.co"}:
                    continue
                candidates.append((idx, "web_article"))
            else:
                continue
            if len(candidates) >= max(1, int(max_items)):
                break

        applied = False
        deep_fetch_count = 0
        details: List[Dict[str, Any]] = []
        for idx, mode in candidates:
            item = refreshed[idx]
            old_len = len(re.sub(r"\s+", " ", str(item.body or "").strip()))
            deep_body = ""
            deep_method = ""
            deep_error = ""

            if mode == "hf_raw_card":
                deep_body, deep_method, deep_error = await self._fetch_huggingface_deep_body(item)
            elif mode == "web_article":
                try:
                    payload = await self._invoke_connector("fetch_web_article", url=str(item.url))
                except Exception as exc:
                    payload = []
                    deep_error = str(exc)
                if payload:
                    deep_item = payload[0]
                    deep_body = str(deep_item.body or "")
                    deep_meta = dict(deep_item.metadata or {})
                    deep_method = str(deep_meta.get("extraction_method") or "web_article")
                    if not deep_error:
                        deep_error = str(deep_meta.get("extraction_error") or "")

            new_len = len(re.sub(r"\s+", " ", str(deep_body or "").strip()))
            accepted = False
            if new_len > 0 and (
                old_len <= 0
                or new_len >= old_len + 80
                or (mode == "hf_raw_card" and old_len < 200 and new_len >= 160)
            ):
                accepted = True
            if accepted:
                merged_body = str(deep_body or "").strip()
                if len(merged_body) > 9000:
                    merged_body = merged_body[:8997].rstrip() + "..."

                metadata = dict(item.metadata or {})
                metadata["deep_fetch_applied"] = True
                metadata["deep_fetch_method"] = deep_method or mode
                metadata["deep_fetch_error"] = deep_error or ""
                metadata["deep_fetch_before_body_len"] = int(old_len)
                metadata["deep_fetch_after_body_len"] = int(new_len)
                item.body = merged_body
                item.metadata = metadata
                refreshed[idx] = item
                applied = True
                deep_fetch_count += 1

            details.append(
                {
                    "item_id": str(item.id or ""),
                    "source": str(item.source or ""),
                    "url": str(item.url or ""),
                    "mode": mode,
                    "method": deep_method or mode,
                    "before_body_len": int(old_len),
                    "after_body_len": int(new_len),
                    "accepted": bool(accepted),
                    "error": deep_error or "",
                }
            )

        return refreshed, applied, deep_fetch_count, details

    async def _fetch_huggingface_deep_body(self, item: RawItem) -> tuple[str, str, str]:
        metadata = dict(item.metadata or {})
        item_url = str(item.url or "").strip()
        repo_id = str(metadata.get("repo_id") or "").strip()
        repo_type = str(metadata.get("item_type") or "").strip().lower()

        if not repo_id and item_url:
            parsed = urlparse(item_url)
            path_parts = [part for part in str(parsed.path or "").split("/") if part]
            if path_parts:
                if path_parts[0] == "datasets" and len(path_parts) >= 3:
                    repo_type = "dataset"
                    repo_id = "/".join(path_parts[1:3])
                elif len(path_parts) >= 2:
                    repo_id = "/".join(path_parts[:2])

        if not repo_id:
            return "", "hf_raw_card", "repo_id_missing"

        prefix = "datasets/" if repo_type == "dataset" else ""
        raw_urls = [
            f"https://huggingface.co/{prefix}{repo_id}/raw/main/README.md",
            f"https://huggingface.co/{prefix}{repo_id}/resolve/main/README.md?download=true",
            f"https://huggingface.co/{prefix}{repo_id}/raw/main/README.md?download=true",
        ]

        headers = {
            "User-Agent": "AcademicResearchAgent/2.0",
            "Accept": "text/plain, text/markdown, */*",
        }
        timeout = httpx.Timeout(10.0)
        errors: List[str] = []
        for raw_url in raw_urls:
            try:
                async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                    response = await client.get(raw_url, headers=headers)
                    response.raise_for_status()
                    text = str(response.text or "").strip()
                if len(re.sub(r"\s+", " ", text)) >= 120:
                    return text, "hf_raw_readme", ""
            except Exception as exc:
                errors.append(str(exc))

        # Fallback: try extracting the model card page as generic web article.
        if item_url.startswith(("http://", "https://")):
            try:
                payload = await self._invoke_connector("fetch_web_article", url=item_url)
                if payload:
                    deep_item = payload[0]
                    deep_text = str(deep_item.body or "").strip()
                    if len(re.sub(r"\s+", " ", deep_text)) >= 120:
                        deep_meta = dict(deep_item.metadata or {})
                        return deep_text, str(deep_meta.get("extraction_method") or "hf_web_article"), ""
            except Exception as exc:
                errors.append(str(exc))

        return "", "hf_raw_card", " | ".join([token for token in errors if token][:3])

    @staticmethod
    def _dedupe_tokens(values: Sequence[str]) -> List[str]:
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

    def _resolve_research_planner(self, request: RunRequest) -> ResearchPlanner:
        budget = dict(request.budget or {})
        if bool(budget.get("use_llm_planner")):
            return LLMPlanner(enabled=True)
        return ResearchPlanner()

    def _resolve_evidence_auditor(
        self,
        request: RunRequest,
        *,
        topic_profile: Optional[TopicProfile] = None,
    ) -> EvidenceAuditorProtocol:
        budget = dict(request.budget or {})
        if bool(budget.get("use_llm_auditor")):
            return LLMEvidenceAuditor(
                enabled=True,
                topic=str(request.topic or "").strip() or None,
                topic_profile=topic_profile,
            )
        return EvidenceAuditor(
            topic=str(request.topic or "").strip() or None,
            topic_profile=topic_profile,
        )

    def _build_selection_controller(self, request: RunRequest, *, top_count: int) -> SelectionController:
        budget = dict(request.budget or {})

        def _float(key: str, env_name: str, default: float) -> float:
            if key in budget:
                try:
                    return float(budget.get(key))
                except Exception:
                    return float(default)
            raw_env = os.getenv(env_name)
            if raw_env is None:
                return float(default)
            try:
                return float(raw_env)
            except Exception:
                return float(default)

        def _int(key: str, env_name: str, default: int) -> int:
            if key in budget:
                try:
                    return int(float(budget.get(key)))
                except Exception:
                    return int(default)
            raw_env = os.getenv(env_name)
            if raw_env is None:
                return int(default)
            try:
                return int(float(raw_env))
            except Exception:
                return int(default)

        min_bucket = _int(
            "min_bucket_coverage",
            "ARA_V2_MIN_BUCKET_COVERAGE",
            2 if int(top_count) >= 2 else 1,
        )
        return SelectionController(
            requested_top_k=int(max(1, top_count)),
            min_pass_ratio=_float("min_pass_ratio", "ARA_V2_MIN_PASS_RATIO", 0.30),
            min_evidence_quality=_float("min_evidence_quality", "ARA_V2_MIN_EVIDENCE_QUALITY", 2.0),
            min_bucket_coverage=max(1, min_bucket),
            min_source_coverage=max(1, _int("min_source_coverage", "ARA_V2_MIN_SOURCE_COVERAGE", 2)),
        )

    @staticmethod
    def _row_hard_selection_eligible(row: Any) -> bool:
        item = getattr(row, "item", None)
        if item is None:
            return False
        metadata = dict(getattr(item, "metadata", None) or {})
        try:
            relevance = float(getattr(row, "relevance_score", 0.0) or 0.0)
        except Exception:
            relevance = 0.0
        if relevance <= 0.0:
            return False
        if not bool(metadata.get("topic_hard_match_pass", True)):
            return False
        try:
            body_len = int(float(metadata.get("body_len", len(str(getattr(item, "body_md", "") or ""))) or 0))
        except Exception:
            body_len = len(str(getattr(item, "body_md", "") or "").strip())
        return body_len > 0

    def _shortage_rescue_pick(
        self,
        *,
        ranked: Sequence[Any],
        relevance_eligible: Sequence[Any],
        audit_records: Sequence[Any],
    ) -> Optional[Any]:
        records_by_id = {
            str(getattr(record, "item_id", "") or "").strip(): record
            for record in list(audit_records or [])
            if str(getattr(record, "item_id", "") or "").strip()
        }
        seen = set()
        pool: List[Any] = []
        for row in list(relevance_eligible or []) + list(ranked or []):
            item_id = str((getattr(row, "item", None).id if getattr(row, "item", None) else "") or "").strip()
            if not item_id or item_id in seen:
                continue
            seen.add(item_id)
            pool.append(row)

        for row in pool:
            if not self._row_hard_selection_eligible(row):
                continue
            item_id = str((getattr(row, "item", None).id if getattr(row, "item", None) else "") or "").strip()
            record = records_by_id.get(item_id)
            if record is not None:
                verdict = str(getattr(record, "verdict", "") or "").strip().lower()
                if verdict == "reject":
                    reasons = [str(value) for value in list(getattr(record, "reasons", []) or []) if str(value).strip()]
                    if "shortage_fallback_after_max_phase" not in reasons:
                        reasons.append("shortage_fallback_after_max_phase")
                    record.reasons = reasons
                    record.verdict = "downgrade"
                    machine_action = dict(getattr(record, "machine_action", {}) or {})
                    machine_action["action"] = "downgrade"
                    machine_action["reason_code"] = "shortage_fallback_after_max_phase"
                    machine_action["human_reason"] = (
                        "no pass/downgrade candidates after max phase; selected best hard-relevant candidate"
                    )
                    record.machine_action = machine_action
            return row
        return None

    def _annotate_raw_item_topic_context(
        self,
        raw_items: Sequence[RawItem],
        *,
        retrieval_plan: Optional[RetrievalPlan],
        topic_profile: Optional[TopicProfile],
    ) -> List[RawItem]:
        output: List[RawItem] = []
        for entry in list(raw_items or []):
            item = entry if isinstance(entry, RawItem) else RawItem(**entry)
            metadata = dict(item.metadata or {})
            source_key = str(item.source or "").strip().lower()
            source_query = str(metadata.get("source_query") or "").strip()
            bucket_hits = {
                str(value).strip()
                for value in list(metadata.get("bucket_hits") or [])
                if str(value).strip()
            }

            query_bucket = None
            if retrieval_plan and source_query:
                query_bucket = retrieval_plan.bucket_for_query(source_query, source=source_key)
                if query_bucket:
                    bucket_hits.add(str(query_bucket))

            if topic_profile:
                topic_text = " ".join(
                    [
                        str(item.title or ""),
                        str(item.body or ""),
                    ]
                )
                bucket_hits.update(topic_profile.bucket_hits(topic_text))
                hard_terms = topic_profile.matched_hard_terms(topic_text)
                metadata["topic_hard_match_terms"] = [str(value) for value in hard_terms]
                metadata["topic_hard_match_pass"] = bool(topic_profile.hard_match_pass(topic_text))

            metadata["bucket_hits"] = sorted(bucket_hits)
            if query_bucket:
                metadata["retrieval_bucket"] = str(query_bucket)
            item.metadata = metadata
            output.append(item)
        return output

    @staticmethod
    def _attempt_quality_snapshot(
        *,
        ranked: Sequence[Any],
        picks: Sequence[Any],
        topic_profile: Optional[TopicProfile],
        audit_records: Optional[Sequence[Any]] = None,
        selection_outcome: Optional[Any] = None,
    ) -> Dict[str, Any]:
        hard_match_terms = set()
        hard_match_pass_count = 0
        bucket_hits_summary: Dict[str, int] = {}
        records_by_id = {
            str(getattr(record, "item_id", "") or "").strip(): record
            for record in list(audit_records or [])
            if str(getattr(record, "item_id", "") or "").strip()
        }
        for row in list(ranked or []):
            metadata = dict((getattr(row, "item", None).metadata or {}) if getattr(row, "item", None) else {})
            if bool(metadata.get("topic_hard_match_pass", True)):
                hard_match_pass_count += 1
            for term in list(metadata.get("topic_hard_match_terms") or []):
                token = str(term).strip()
                if token:
                    hard_match_terms.add(token)
            for bucket in list(metadata.get("bucket_hits") or []):
                token = str(bucket).strip()
                if not token:
                    continue
                bucket_hits_summary[token] = int(bucket_hits_summary.get(token, 0)) + 1

        pick_relevance = [float(getattr(row, "relevance_score", 0.0) or 0.0) for row in list(picks or [])]
        top_picks_min_relevance = min(pick_relevance) if pick_relevance else 0.0
        top_picks_hard_match_count = 0
        pick_bucket_hits = set()
        selected_sources = set()
        evidence_quality_values: List[float] = []
        for row in list(picks or []):
            item_id = str((getattr(row, "item", None).id if getattr(row, "item", None) else "") or "").strip()
            metadata = dict((getattr(row, "item", None).metadata or {}) if getattr(row, "item", None) else {})
            if bool(metadata.get("topic_hard_match_pass", True)):
                top_picks_hard_match_count += 1
            for bucket in list(metadata.get("bucket_hits") or []):
                token = str(bucket).strip()
                if token:
                    pick_bucket_hits.add(token)
            source = str((getattr(row, "item", None).source if getattr(row, "item", None) else "") or "").strip().lower()
            if source:
                selected_sources.add(source)
            record = records_by_id.get(item_id)
            if record is not None:
                evidence_quality_values.append(float(getattr(record, "evidence_links_quality", 0) or 0))

        min_evidence_quality = min(evidence_quality_values) if evidence_quality_values else None
        if min_evidence_quality is None and selection_outcome is not None:
            min_evidence_quality = float(getattr(selection_outcome, "top_picks_min_evidence_quality", 0.0) or 0.0)
        if min_evidence_quality is None:
            min_evidence_quality = 0.0

        if topic_profile and topic_profile.hard_include_any:
            terms_used = [str(value) for value in list(topic_profile.hard_include_any) if str(value).strip()]
        else:
            terms_used = sorted(hard_match_terms)
        return {
            "hard_match_terms_used": terms_used,
            "hard_match_pass_count": int(hard_match_pass_count),
            "top_picks_min_relevance": float(top_picks_min_relevance),
            "top_picks_hard_match_count": int(top_picks_hard_match_count),
            "bucket_coverage": int(len(pick_bucket_hits)),
            "bucket_names": sorted(pick_bucket_hits),
            "bucket_hits_summary": bucket_hits_summary,
            "top_picks_min_evidence_quality": float(min_evidence_quality),
            "source_coverage": int(len(selected_sources)),
        }

    @staticmethod
    def _quality_trigger_reasons(
        *,
        snapshot: Mapping[str, Any],
        requested_top_k: int,
        topic_profile: Optional[TopicProfile],
    ) -> List[str]:
        reasons: List[str] = []
        min_relevance = float(snapshot.get("top_picks_min_relevance", 0.0) or 0.0)
        if min_relevance < 0.75:
            reasons.append("min_top_pick_relevance_lt_0.75")

        if topic_profile and topic_profile.requires_hard_gate:
            hard_match_count = int(snapshot.get("top_picks_hard_match_count", 0) or 0)
            if hard_match_count < int(max(1, requested_top_k)):
                reasons.append(f"top_picks_hard_match_count_lt_{int(max(1, requested_top_k))}")

        bucket_coverage = int(snapshot.get("bucket_coverage", 0) or 0)
        required_bucket_coverage = int(topic_profile.minimum_bucket_coverage) if topic_profile else 1
        if bucket_coverage < required_bucket_coverage:
            reasons.append(f"bucket_coverage_lt_{required_bucket_coverage}")
        return reasons

    @staticmethod
    def _selection_priority(*, picks: Sequence[Any], quality_snapshot: Mapping[str, Any]) -> float:
        return (
            float(len(list(picks or []))) * 1000.0
            + float(int(quality_snapshot.get("top_picks_hard_match_count", 0) or 0)) * 50.0
            + float(int(quality_snapshot.get("bucket_coverage", 0) or 0)) * 30.0
            + float(quality_snapshot.get("top_picks_min_relevance", 0.0) or 0.0)
        )

    def _connector_exists(self, name: str) -> bool:
        if self._connector_overrides:
            return name in self._connector_overrides
        return hasattr(connectors, name)

    async def _invoke_connector(self, name: str, /, **kwargs: Any) -> List[RawItem]:
        fn = self._connector_overrides.get(name) if self._connector_overrides else None
        if fn is None:
            fn = getattr(connectors, name)
        filtered_kwargs = dict(kwargs)
        try:
            signature = inspect.signature(fn)
            accepts_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
            if not accepts_var_kwargs:
                allowed = {key for key in signature.parameters.keys()}
                filtered_kwargs = {key: value for key, value in dict(kwargs).items() if key in allowed}
        except Exception:
            filtered_kwargs = dict(kwargs)
        value = fn(**filtered_kwargs)
        if inspect.isawaitable(value):
            value = await value
        return [item if isinstance(item, RawItem) else RawItem(**item) for item in list(value or [])]

    def _synthetic_item(self, request: RunRequest) -> RawItem:
        topic = str(request.topic or "research update").strip()
        seed = hashlib.sha1(topic.encode("utf-8")).hexdigest()[:12]
        return RawItem(
            id=f"synthetic_{seed}",
            source="web_article",
            title=f"{topic} synthetic fallback",
            url="https://example.local/synthetic",
            body=f"Synthetic content for {topic}. Includes architecture, benchmark, and deployment notes.",
            author="system",
            published_at=_utcnow(),
            tier="B",
            metadata={"item_type": "synthetic_fallback", "credibility": "low"},
        )

    @staticmethod
    def _write_json(path: Path, payload: Any) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    @staticmethod
    def _build_materials_manifest(
        *,
        run_id: str,
        primary: Any,
        picks: Sequence[Any],
        run_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        top_items = [entry.item for entry in list(picks or [])]
        facts = dict(primary.metadata.get("facts") or {})
        screenshot_plan = []
        icon_keywords = set()
        source_urls = set()
        for idx, item in enumerate(top_items[:8], start=1):
            item_url = normalize_url(str(item.url or "").strip())
            if item_url and is_allowed_citation_url(item_url):
                source_urls.add(item_url)
                screenshot_plan.append(
                    {
                        "idx": idx,
                        "url": item_url,
                        "purpose": "hero" if idx == 1 else "supporting evidence",
                        "shot_hint": f"shot_{idx:02d}",
                        "title": item.title,
                    }
                )
            for token in [item.source, item.tier, *str(item.title).lower().replace("/", " ").split()[:6]]:
                text = str(token or "").strip().lower()
                if len(text) >= 3 and text.isascii():
                    icon_keywords.add(text)

        citation_sources = []
        for citation in list(primary.citations or [])[:12]:
            url = normalize_url(str(citation.url or "").strip())
            if url and is_allowed_citation_url(url):
                source_urls.add(url)
            citation_sources.append(
                {
                    "title": citation.title,
                    "url": url,
                    "source": citation.source,
                    "snippet": citation.snippet,
                }
            )

        broll_categories = [
            "engineering_team_collaboration",
            "code_editor_terminal_closeup",
            "dashboard_metrics_animation",
            "cloud_infrastructure_datacenter",
            "product_demo_ui_scroll",
        ]

        return {
            "run_id": run_id,
            "item_id": primary.id,
            "data_mode": str(run_context.get("data_mode") or "live"),
            "connector_stats": dict(run_context.get("connector_stats") or {}),
            "extraction_stats": dict(run_context.get("extraction_stats") or {}),
            "top_item_ids": [item.id for item in top_items],
            "source_urls": sorted(source_urls),
            "screenshot_plan": screenshot_plan,
            "local_assets": [],
            "reference_asset_map": {},
            "icon_keyword_suggestions": sorted(icon_keywords)[:16],
            "broll_categories": broll_categories,
            "citations": citation_sources,
            "quality_metrics": {
                "body_len": int(float(primary.metadata.get("body_len", len(str(primary.body_md or ""))) or 0)),
                "citation_count": int(float(primary.metadata.get("citation_count", len(primary.citations)) or 0)),
                "published_recency": primary.metadata.get("published_recency"),
                "link_count": int(float(primary.metadata.get("link_count", 0) or 0)),
            },
            "facts": facts,
        }

    @staticmethod
    def _slug(value: str, *, max_len: int = 48) -> str:
        text = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value or "").strip().lower())
        text = re.sub(r"-{2,}", "-", text).strip("-")
        if not text:
            text = "asset"
        return text[:max_len].strip("-") or "asset"

    def _write_asset_card(self, *, path: Path, title: str, subtitle: str, accent: str = "#0f766e") -> str:
        svg = "\n".join(
            [
                "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1080\" height=\"1920\" viewBox=\"0 0 1080 1920\">",
                "  <rect width=\"1080\" height=\"1920\" fill=\"#0b1320\"/>",
                f"  <rect x=\"56\" y=\"56\" width=\"968\" height=\"1808\" rx=\"36\" fill=\"{accent}\" fill-opacity=\"0.16\"/>",
                f"  <text x=\"88\" y=\"220\" font-size=\"54\" font-family=\"Arial, sans-serif\" fill=\"#e2e8f0\">{self._slug(title, max_len=30)}</text>",
                f"  <text x=\"88\" y=\"330\" font-size=\"38\" font-family=\"Arial, sans-serif\" fill=\"#cbd5e1\">{self._slug(subtitle, max_len=50)}</text>",
                "  <text x=\"88\" y=\"1780\" font-size=\"26\" font-family=\"Arial, sans-serif\" fill=\"#94a3b8\">generated local asset</text>",
                "</svg>",
            ]
        )
        path.write_text(svg, encoding="utf-8")
        return str(path)

    def _materialize_assets(self, *, out_dir: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
        assets_dir = out_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        plan = list(manifest.get("screenshot_plan") or [])
        local_assets: List[str] = []
        url_to_asset: Dict[str, str] = {}
        fetch_logs: List[Dict[str, Any]] = []

        for idx, entry in enumerate(plan, start=1):
            url = normalize_url(str((entry or {}).get("url") or ""))
            title = str((entry or {}).get("title") or f"asset-{idx}")
            shot_hint = str((entry or {}).get("shot_hint") or f"shot_{idx:02d}")
            base = f"{shot_hint}_{self._slug(title, max_len=28)}"
            target = assets_dir / f"{base}.svg"
            error = None

            if url and url.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
                img_target = assets_dir / f"{base}{Path(urlparse(url).path).suffix or '.png'}"
                try:
                    with httpx.Client(timeout=httpx.Timeout(6.0), follow_redirects=True) as client:
                        resp = client.get(url)
                        resp.raise_for_status()
                        img_target.write_bytes(resp.content)
                    target_path = str(img_target)
                except Exception as exc:
                    error = f"download_failed:{exc}"
                    target_path = self._write_asset_card(path=target, title=title, subtitle=url or "local fallback")
            else:
                target_path = self._write_asset_card(path=target, title=title, subtitle=url or "local fallback")

            local_assets.append(target_path)
            if url:
                url_to_asset[url] = target_path
            fetch_logs.append(
                {
                    "idx": idx,
                    "url": url,
                    "local_path": target_path,
                    "status": "ok" if not error else "fallback",
                    "error": error,
                }
            )

        manifest["local_assets"] = local_assets
        manifest["reference_asset_map"] = url_to_asset
        manifest["asset_fetch_results"] = fetch_logs
        return manifest

    @staticmethod
    def _attach_reference_assets(*, board: Any, materials: Dict[str, Any]) -> Any:
        ref_map = {normalize_url(str(key)): str(value) for key, value in dict(materials.get("reference_asset_map") or {}).items()}
        local_assets = [str(path) for path in list(materials.get("local_assets") or []) if str(path).strip()]
        for idx, shot in enumerate(list(board.shots or []), start=1):
            local_pref = local_assets[(idx - 1) % len(local_assets)] if local_assets else ""
            refs: List[str] = []
            if local_pref and idx <= 3:
                refs.append(local_pref)
            for raw_ref in list(shot.reference_assets or []):
                normalized = normalize_url(str(raw_ref or ""))
                mapped = ref_map.get(normalized)
                if mapped:
                    if mapped not in refs:
                        refs.append(mapped)
                elif normalized and is_allowed_citation_url(normalized):
                    if normalized not in refs:
                        refs.append(normalized)
            if not refs and local_assets:
                refs = [local_assets[(idx - 1) % len(local_assets)]]
            shot.reference_assets = refs
        return board

    @staticmethod
    def _build_run_context(
        *,
        data_mode: str,
        raw_items: Sequence[RawItem],
        run_id: str,
        topic: Optional[str],
    ) -> Dict[str, Any]:
        now = _utcnow().isoformat(timespec="seconds")
        connector_stats: Dict[str, Dict[str, Any]] = {}
        extraction_methods: Dict[str, int] = {}
        extraction_failures = 0
        total_body_len = 0
        count = 0

        for item in list(raw_items or []):
            source = str(item.source or "unknown")
            if source not in connector_stats:
                connector_stats[source] = {"count": 0, "fetched_at": now}
            connector_stats[source]["count"] = int(connector_stats[source]["count"]) + 1

            metadata = dict(item.metadata or {})
            method = str(metadata.get("extraction_method") or "fallback").strip().lower()
            extraction_methods[method] = int(extraction_methods.get(method, 0)) + 1
            if bool(metadata.get("extraction_failed")):
                extraction_failures += 1

            body_len = len(re.sub(r"\s+", " ", str(item.body or "").strip()))
            total_body_len += body_len
            count += 1

        avg_body_len = round(float(total_body_len) / float(max(1, count)), 2)
        return {
            "run_id": run_id,
            "data_mode": str(data_mode or "live"),
            "topic": str(topic or "").strip() or None,
            "connector_stats": connector_stats,
            "extraction_stats": {
                "methods": extraction_methods,
                "avg_body_len": avg_body_len,
                "failures": int(extraction_failures),
            },
            "fetched_at": now,
        }

    def _record_artifact(self, run_id: str, artifact: Artifact) -> None:
        self._orchestrator.add_artifact(run_id, artifact)

    def _ensure_not_canceled(self, run_id: str) -> None:
        status = self._orchestrator.get_run_status(run_id)
        if status and status.cancellation_requested:
            raise RunCancelledError("cancellation requested")

    @staticmethod
    def _recall_profiles(
        request: RunRequest,
        *,
        data_mode: str,
        retrieval_plan: Optional[RetrievalPlan] = None,
    ) -> List[RecallProfile]:
        if retrieval_plan and data_mode == "live":
            profiles_from_plan = [
                RecallProfile(
                    limit_multiplier=max(1, int(rule.limit_multiplier)),
                    window=str(rule.window or "today"),
                    expanded_queries=bool(rule.expanded_queries),
                    phase=str(rule.phase or "base"),
                )
                for rule in list(retrieval_plan.time_window_policy or [])
            ]
            if profiles_from_plan:
                return profiles_from_plan

        base_window = str(request.time_window or "today").strip().lower() or "today"
        profiles = [RecallProfile(limit_multiplier=1, window=base_window, expanded_queries=False, phase="base")]
        if data_mode != "live" or not str(request.topic or "").strip():
            return profiles

        profiles.append(RecallProfile(limit_multiplier=2, window=base_window, expanded_queries=False, phase="limit_x2"))
        if base_window in {"today", "24h", "1d"}:
            profiles.append(RecallProfile(limit_multiplier=2, window="3d", expanded_queries=False, phase="window_3d"))
        profiles.append(RecallProfile(limit_multiplier=2, window="7d", expanded_queries=False, phase="window_7d"))
        profiles.append(RecallProfile(limit_multiplier=2, window="7d", expanded_queries=True, phase="query_expanded"))

        deduped: List[RecallProfile] = []
        seen = set()
        for profile in profiles:
            key = (profile.limit_multiplier, profile.window, profile.expanded_queries)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(profile)
        return deduped

    def _rank_with_relevance_gate(
        self,
        *,
        merged: Sequence[Any],
        topic: Optional[str],
        topic_profile: Optional[TopicProfile],
        data_mode: str,
        top_count: int,
    ) -> Dict[str, Any]:
        base_relevance_threshold = 0.55 if data_mode == "live" else 0.0
        min_relevance_threshold = 0.45 if float(base_relevance_threshold) > 0.0 else 0.0
        relevance_threshold = float(base_relevance_threshold)
        relaxation_steps = 0
        ranked = []
        relevance_eligible = []
        picks = []

        while True:
            ranked = rank_items(
                merged,
                topic=topic,
                topic_profile=topic_profile,
                relevance_threshold=relevance_threshold,
            )
            relevance_eligible = (
                [entry for entry in ranked if float(entry.relevance_score) >= float(relevance_threshold)]
                if float(relevance_threshold) > 0.0
                else list(ranked)
            )
            picks = self._select_diverse_top(
                relevance_eligible,
                top_count=top_count,
                min_bucket_coverage=(topic_profile.minimum_bucket_coverage if topic_profile else 1),
            )
            if len(picks) >= top_count:
                break
            if relevance_threshold <= min_relevance_threshold + 1e-9:
                break
            next_threshold = max(min_relevance_threshold, round(relevance_threshold - 0.05, 2))
            if next_threshold >= relevance_threshold:
                break
            relevance_threshold = float(next_threshold)
            relaxation_steps += 1

        if float(relevance_threshold) < float(base_relevance_threshold):
            for pick in picks:
                if float(pick.relevance_score) < float(base_relevance_threshold):
                    reasons = list(pick.reasons or [])
                    if "relevance.relaxed_pick" not in reasons:
                        reasons.append("relevance.relaxed_pick")
                    pick.reasons = reasons

        return {
            "ranked": ranked,
            "relevance_eligible": relevance_eligible,
            "picks": picks,
            "base_relevance_threshold": base_relevance_threshold,
            "relevance_threshold": relevance_threshold,
            "relaxation_steps": relaxation_steps,
        }

    @staticmethod
    def _select_diverse_top(
        rows: Sequence[Any],
        *,
        top_count: int,
        min_bucket_coverage: int = 1,
    ) -> List[Any]:
        if top_count <= 0:
            return []
        selected: List[Any] = []
        selected_ids = set()
        used_sources = set()
        used_buckets = set()
        target_bucket_coverage = max(1, int(min_bucket_coverage))

        def _row_source(row: Any) -> str:
            return str((getattr(row, "item", None).source if getattr(row, "item", None) else "") or "").strip().lower()

        def _row_id(row: Any) -> str:
            return str((getattr(row, "item", None).id if getattr(row, "item", None) else "") or "").strip()

        def _row_buckets(row: Any) -> List[str]:
            item = getattr(row, "item", None)
            metadata = dict((getattr(item, "metadata", None) or {}) if item else {})
            return [str(value).strip() for value in list(metadata.get("bucket_hits") or []) if str(value).strip()]

        def _add(row: Any) -> None:
            item_id = _row_id(row)
            if not item_id or item_id in selected_ids:
                return
            selected.append(row)
            selected_ids.add(item_id)
            source = _row_source(row)
            if source:
                used_sources.add(source)
            used_buckets.update(_row_buckets(row))

        if target_bucket_coverage > 1:
            for row in list(rows or []):
                if len(selected) >= top_count:
                    break
                if _row_id(row) in selected_ids:
                    continue
                buckets = _row_buckets(row)
                if not buckets:
                    continue
                if not any(bucket not in used_buckets for bucket in buckets):
                    continue
                _add(row)
                if len(used_buckets) >= target_bucket_coverage and len(selected) >= 1:
                    break

        for row in list(rows or []):
            if len(selected) >= top_count:
                break
            item_id = _row_id(row)
            if not item_id or item_id in selected_ids:
                continue
            source = _row_source(row)
            if source and source in used_sources:
                continue
            _add(row)

        for row in list(rows or []):
            if len(selected) >= top_count:
                break
            item_id = _row_id(row)
            if not item_id or item_id in selected_ids:
                continue
            _add(row)
        return selected

    @staticmethod
    def _top_n(request: RunRequest) -> int:
        budget = dict(request.budget or {})
        override = int(budget.get("top_k") or 0)
        if override > 0:
            return max(1, min(10, override))
        return 3 if request.mode == RunMode.DAILY else 5

    @staticmethod
    def _duration_sec(request: RunRequest) -> int:
        value = int((request.budget or {}).get("duration_sec") or 35)
        return max(30, min(45, value))

    @staticmethod
    def _platform(request: RunRequest) -> str:
        value = str((request.budget or {}).get("platform") or "tiktok").strip().lower()
        return value or "tiktok"

    @staticmethod
    def _tone(request: RunRequest) -> str:
        value = str((request.budget or {}).get("tone") or "professional").strip()
        return value or "professional"

    @staticmethod
    def _voice(request: RunRequest) -> str:
        value = str((request.budget or {}).get("voice") or "neutral").strip()
        return value or "neutral"

    @staticmethod
    def _style_profile(request: RunRequest) -> Dict[str, str]:
        profile = dict((request.budget or {}).get("style_profile") or {})
        profile.setdefault("aspect", "9:16")
        profile.setdefault("style", "cinematic technical explainer")
        profile.setdefault("mood", "clear and concise engineering narrative")
        profile.setdefault("character_id", "host_01")
        profile.setdefault("style_id", "tech_brief_v1")
        return {str(key): str(value) for key, value in profile.items()}

    @staticmethod
    def _should_render(request: RunRequest) -> bool:
        budget = dict(request.budget or {})
        if "render_enabled" in budget:
            return bool(budget.get("render_enabled"))
        targets = [str(item).strip().lower() for item in list(request.output_targets or []) if str(item).strip()]
        if not targets:
            return True
        return "mp4" in targets

    @staticmethod
    def _render_budget(request: RunRequest) -> Dict[str, Any]:
        budget = dict(request.budget or {})
        max_retries = int(budget.get("max_retries", 1) or 1)
        return {
            "preview": bool(budget.get("preview", True)),
            "confirm_required": bool(budget.get("confirm_required", False)),
            "max_total_cost": float(budget.get("max_total_cost", 100.0) or 100.0),
            "max_retries": max(0, min(2, max_retries)),
            "quality": str(budget.get("quality") or "preview"),
        }
