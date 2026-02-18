"""End-to-end v2 pipeline runtime from RunRequest to artifacts."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
import logging
from pathlib import Path
import re
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence
from urllib.parse import urlparse

from core import Artifact, ArtifactType, RawItem, RenderStatus, RunMode, RunRequest
from orchestrator import RunOrchestrator
from pipeline_v2.data_mode import resolve_data_mode, should_allow_smoke
from pipeline_v2.dedup_cluster import cluster, dedup_exact, embed, merge_cluster
from pipeline_v2.normalize import normalize
from pipeline_v2.notification import notify_user, post_to_web, send_email
from pipeline_v2.prompt_compiler import compile_storyboard
from pipeline_v2.report_export import export_package, generate_onepager, generate_thumbnail
from pipeline_v2.scoring import rank_items
from pipeline_v2.script_generator import build_facts, generate_script
from pipeline_v2.sanitize import is_allowed_citation_url, normalize_url
from pipeline_v2.storyboard_generator import auto_fix_storyboard, script_to_storyboard, validate_storyboard
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

        raw_items = await self._collect_raw_items(request, data_mode=data_mode)
        run_context = self._build_run_context(
            data_mode=data_mode,
            raw_items=raw_items,
            run_id=run_id,
            topic=request.topic,
        )
        run_context_path = self._write_json(out_dir / "run_context.json", run_context)
        self._orchestrator.mark_run_progress(run_id, 0.18)
        self._orchestrator.append_event(
            run_id,
            "ingest_done",
            f"raw_items={len(raw_items)} data_mode={data_mode} connector_sources={len(run_context.get('connector_stats', {}))}",
        )
        self._ensure_not_canceled(run_id)

        normalized = [normalize(item) for item in raw_items]
        unique_items = dedup_exact(normalized)
        embeddings = embed(unique_items)
        grouped = cluster(unique_items, embeddings)
        merged = [merge_cluster(group) for group in grouped]
        top_count = self._top_n(request)
        base_relevance_threshold = 0.55 if data_mode == "live" else 0.0
        min_relevance_threshold = 0.35 if float(base_relevance_threshold) > 0.0 else 0.0
        relevance_threshold = float(base_relevance_threshold)
        relaxation_steps = 0
        ranked = []
        relevance_eligible = []
        picks = []

        while True:
            ranked = rank_items(
                merged,
                topic=request.topic,
                relevance_threshold=relevance_threshold,
            )
            relevance_eligible = (
                [entry for entry in ranked if float(entry.relevance_score) >= float(relevance_threshold)]
                if float(relevance_threshold) > 0.0
                else list(ranked)
            )
            picks = self._select_diverse_top(relevance_eligible, top_count=top_count)
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

        if not picks:
            raise RuntimeError("no relevant ranked items available")

        def _drop_reason(row: Any) -> str:
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
            chunks = [f"rel={float(getattr(row, 'relevance_score', 0.0)):.2f}"]
            recency = signals.get("update_recency_days")
            if recency not in (None, "", "unknown"):
                chunks.append(f"更新{float(recency):.1f}d")
            evidence_links = int(float(signals.get("evidence_links_quality", 0) or 0))
            if evidence_links > 0:
                chunks.append(f"证据链{evidence_links}")
            if bool(signals.get("has_quickstart")):
                chunks.append("含Quickstart")
            points = int(float((item.metadata or {}).get("points", 0) or 0))
            comments = int(float((item.metadata or {}).get("comment_count", 0) or 0))
            if points > 0 or comments > 0:
                chunks.append(f"HN {points}/{comments}")
            if any(str(reason).strip().lower() == "relevance.relaxed_pick" for reason in list(getattr(row, "reasons", []) or [])):
                chunks.append("阈值放宽补位")
            return " · ".join(chunks[:3])

        picked_ids = {pick.item.id for pick in picks}
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
            "min_body_len_for_top_picks": 300,
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
        run_context_path = self._write_json(out_dir / "run_context.json", run_context)

        self._orchestrator.mark_run_progress(run_id, 0.42)
        self._orchestrator.append_event(
            run_id,
            "ranking_done",
            f"picked={len(picks)} threshold_used={relevance_threshold:.2f} relaxation_steps={relaxation_steps}",
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

    async def _collect_raw_items(self, request: RunRequest, *, data_mode: str) -> List[RawItem]:
        tasks: List[Awaitable[List[RawItem]]] = [
            self._invoke_connector("fetch_github_trending", max_results=12),
            self._invoke_connector("fetch_huggingface_trending", max_results=12),
            self._invoke_connector("fetch_hackernews_top", max_results=12),
        ]
        tier_a_results = await asyncio.gather(*tasks, return_exceptions=True)

        raw_items: List[RawItem] = []
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
            try:
                releases = await self._invoke_connector(
                    "fetch_github_releases",
                    repo_full_names=repo_names,
                    max_results_per_repo=1,
                )
                raw_items.extend(releases)
            except Exception:
                pass

        include_tier_b = bool((request.budget or {}).get("include_tier_b", True))
        if include_tier_b:
            feed_urls = list((request.budget or {}).get("rss_feeds") or [])
            for feed_url in feed_urls[:2]:
                try:
                    raw_items.extend(await self._invoke_connector("fetch_rss_feed", feed_url=feed_url, max_results=6))
                except Exception:
                    continue

            seed_url = str((request.budget or {}).get("seed_url") or "").strip()
            if seed_url:
                try:
                    raw_items.extend(await self._invoke_connector("fetch_web_article", url=seed_url))
                except Exception:
                    pass

        if not raw_items and data_mode == "live":
            raise RuntimeError("no live items fetched from connectors")

        if not raw_items:
            raw_items.append(self._synthetic_item(request))
        return raw_items

    async def _invoke_connector(self, name: str, /, **kwargs: Any) -> List[RawItem]:
        fn = self._connector_overrides.get(name) or getattr(connectors, name)
        value = fn(**kwargs)
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
    def _select_diverse_top(rows: Sequence[Any], *, top_count: int) -> List[Any]:
        if top_count <= 0:
            return []
        selected: List[Any] = []
        selected_ids = set()
        used_sources = set()
        for row in list(rows or []):
            source = str((getattr(row, "item", None).source if getattr(row, "item", None) else "") or "").strip().lower()
            item_id = str((getattr(row, "item", None).id if getattr(row, "item", None) else "") or "").strip()
            if not item_id or item_id in selected_ids:
                continue
            if source and source in used_sources:
                continue
            selected.append(row)
            selected_ids.add(item_id)
            if source:
                used_sources.add(source)
            if len(selected) >= top_count:
                return selected
        for row in list(rows or []):
            item_id = str((getattr(row, "item", None).id if getattr(row, "item", None) else "") or "").strip()
            if not item_id or item_id in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(item_id)
            if len(selected) >= top_count:
                break
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
