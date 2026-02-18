"""Asynchronous-style render manager with retry and fallback controls."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4

from core import PromptSpec, RenderStatus, Shot, StatusTimestamps, Storyboard

from .adapters import BaseRendererAdapter, SeedanceAdapter, ShotRenderResult


logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_render_job_id() -> str:
    return f"render_{_utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"


@dataclass
class RenderJob:
    render_job_id: str
    run_id: str
    prompt_specs: List[PromptSpec]
    budget: Dict[str, Any]
    output_dir: Path
    status: RenderStatus
    shot_outputs: Dict[int, str] = field(default_factory=dict)
    failed: Dict[int, str] = field(default_factory=dict)
    total_cost: float = 0.0
    preview_completed: bool = False
    confirmed: bool = False
    preview_output_path: Optional[str] = None


class RenderManager:
    """Queue-backed render manager with retries and fallback MP4 output."""

    def __init__(
        self,
        *,
        renderer_adapter: Optional[BaseRendererAdapter] = None,
        work_dir: Optional[Path] = None,
    ) -> None:
        self._adapter = renderer_adapter or SeedanceAdapter()
        self._work_dir = Path(work_dir or (Path("data") / "outputs" / "render_jobs"))
        self._work_dir.mkdir(parents=True, exist_ok=True)

        self._jobs: Dict[str, RenderJob] = {}
        self._queue: List[str] = []

    def enqueue_render(self, run_id: str, prompt_specs: Sequence[PromptSpec], budget: Optional[Dict[str, Any]] = None) -> str:
        """Queue render job and return render_job_id."""
        render_job_id = _new_render_job_id()
        out_dir = self._work_dir / render_job_id
        out_dir.mkdir(parents=True, exist_ok=True)

        budget_map = dict(budget or {})
        max_retries = int(budget_map.get("max_retries", 1) or 1)

        status = RenderStatus(
            render_job_id=render_job_id,
            run_id=str(run_id),
            state="queued",
            progress=0.0,
            timestamps=StatusTimestamps(),
            max_retries=max(0, max_retries),
        )

        job = RenderJob(
            render_job_id=render_job_id,
            run_id=str(run_id),
            prompt_specs=[PromptSpec(**item.model_dump()) for item in list(prompt_specs or [])],
            budget=budget_map,
            output_dir=out_dir,
            status=status,
        )
        self._jobs[render_job_id] = job
        self._queue.append(render_job_id)
        logger.info("enqueue_render run_id=%s render_job_id=%s shots=%s", run_id, render_job_id, len(job.prompt_specs))
        return render_job_id

    def poll_render(self, render_job_id: str) -> Optional[RenderStatus]:
        """Get current render job status snapshot."""
        job = self._jobs.get(render_job_id)
        if not job:
            return None
        return RenderStatus(**job.status.model_dump())

    def confirm_render(self, render_job_id: str, *, approved: bool = True) -> Optional[RenderStatus]:
        """Confirm preview and queue final render, or cancel when rejected."""
        job = self._jobs.get(render_job_id)
        if not job:
            return None

        confirm_required = bool(job.budget.get("confirm_required", False))
        if not confirm_required:
            return RenderStatus(**job.status.model_dump())

        if not approved:
            self._set_canceled(job, reason="preview rejected by user")
            self._remove_from_queue(render_job_id)
            return RenderStatus(**job.status.model_dump())

        if not job.preview_completed:
            return RenderStatus(**job.status.model_dump())

        job.confirmed = True
        job.shot_outputs.clear()
        job.failed.clear()
        job.status.output_path = None
        job.status.state = "queued"
        job.status.progress = max(job.status.progress, 0.65)
        job.status.timestamps.updated_at = _utcnow()
        if render_job_id not in self._queue:
            self._queue.append(render_job_id)
        return RenderStatus(**job.status.model_dump())

    def cancel_render(self, render_job_id: str) -> bool:
        """Best-effort render cancellation for queued/preview/running jobs."""
        job = self._jobs.get(render_job_id)
        if not job:
            return False
        job.status.cancellation_requested = True
        self._remove_from_queue(render_job_id)
        if job.status.state in {"queued", "retrying", "awaiting_confirmation"}:
            self._set_canceled(job, reason="cancellation requested")
        elif job.status.state == "running":
            job.status.state = "cancel_requested"
            job.status.timestamps.updated_at = _utcnow()
        return True

    def process_next(self) -> Optional[RenderStatus]:
        """Process next queued render job synchronously (worker-style)."""
        if not self._queue:
            return None
        render_job_id = self._queue.pop(0)
        job = self._jobs.get(render_job_id)
        if job is None:
            return None

        if job.status.cancellation_requested:
            self._set_canceled(job, reason="cancellation requested")
            return RenderStatus(**job.status.model_dump())

        confirm_required = bool(job.budget.get("confirm_required", False))
        if confirm_required and not job.preview_completed:
            self._set_running(job, phase="preview")
            self._render_all_shots(job, mode="preview")
            if job.failed:
                self.retry_failed_shots(render_job_id, mode="preview")

            if job.failed:
                fallback_board = self._board_from_prompts(job)
                preview_path = self.fallback_render(fallback_board, out_dir=job.output_dir)
                job.status.errors.append("seedance_failed_using_fallback")
            else:
                preview_path = self.stitch_shots(list(job.shot_outputs.values()), out_dir=job.output_dir)

            job.preview_completed = True
            job.preview_output_path = preview_path
            job.status.output_path = preview_path
            job.status.state = "awaiting_confirmation"
            job.status.progress = max(job.status.progress, 0.6)
            job.status.timestamps.updated_at = _utcnow()
            return RenderStatus(**job.status.model_dump())

        if confirm_required and not job.confirmed:
            job.status.state = "awaiting_confirmation"
            job.status.timestamps.updated_at = _utcnow()
            return RenderStatus(**job.status.model_dump())

        job.shot_outputs.clear()
        job.failed.clear()
        self._set_running(job, phase="final")
        self._render_all_shots(job, mode="final")

        if job.failed:
            self.retry_failed_shots(render_job_id, mode="final")

        if job.failed:
            fallback_board = self._board_from_prompts(job)
            fallback_path = self.fallback_render(fallback_board, out_dir=job.output_dir)
            job.status.output_path = fallback_path
            job.status.errors.append("seedance_failed_using_fallback")

        if not job.status.output_path:
            stitched = self.stitch_shots(list(job.shot_outputs.values()), out_dir=job.output_dir)
            job.status.output_path = stitched

        self._set_completed(job)
        return RenderStatus(**job.status.model_dump())

    def retry_failed_shots(
        self,
        render_job_id: str,
        max_retries: Optional[int] = None,
        *,
        mode: str = "final",
    ) -> Optional[RenderStatus]:
        """Retry failed shots for a render job within retry budget."""
        job = self._jobs.get(render_job_id)
        if not job:
            return None

        allowed = job.status.max_retries if max_retries is None else max(0, int(max_retries))
        if job.status.retry_count >= allowed:
            return RenderStatus(**job.status.model_dump())

        pending = sorted(list(job.failed.keys()))
        if not pending:
            return RenderStatus(**job.status.model_dump())

        job.status.retry_count += 1
        job.status.state = "retrying"
        job.status.timestamps.updated_at = _utcnow()

        current_failed = dict(job.failed)
        job.failed.clear()
        for idx in pending:
            spec = self._prompt_by_idx(job.prompt_specs, idx)
            if spec is None:
                continue
            result = self._render_one_shot(job, spec, mode=mode)
            if result.success and result.output_path:
                job.shot_outputs[idx] = result.output_path
            else:
                job.failed[idx] = result.error or current_failed.get(idx, "unknown render error")

        job.status.failed_shot_indices = sorted(list(job.failed.keys()))
        job.status.timestamps.updated_at = _utcnow()
        return RenderStatus(**job.status.model_dump())

    def fallback_render(self, board: Storyboard, out_dir: Optional[Path] = None) -> str:
        """Fallback render path that must still emit an MP4 artifact."""
        target_dir = Path(out_dir or self._work_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / "fallback_render.mp4"

        payload = [
            b"FALLBACK_MP4_PLACEHOLDER\n",
            f"run_id={board.run_id}\n".encode("utf-8"),
            f"item_id={board.item_id}\n".encode("utf-8"),
            f"duration_sec={board.duration_sec}\n".encode("utf-8"),
            f"shots={len(board.shots)}\n".encode("utf-8"),
        ]
        out_path.write_bytes(b"".join(payload))
        return str(out_path)

    def stitch_shots(self, shot_paths: Sequence[str], out_dir: Optional[Path] = None) -> str:
        """Stitch rendered shots into final MP4 path (placeholder concatenation)."""
        target_dir = Path(out_dir or self._work_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / "rendered_final.mp4"

        with out_path.open("wb") as fh:
            fh.write(b"STITCHED_MP4_PLACEHOLDER\n")
            for shot_path in shot_paths:
                path = Path(str(shot_path))
                if not path.exists() or not path.is_file():
                    continue
                digest = hashlib.sha1(path.read_bytes()).hexdigest()[:12]
                fh.write(f"{path.name}:{digest}\n".encode("utf-8"))

        return str(out_path)

    def _remove_from_queue(self, render_job_id: str) -> None:
        while render_job_id in self._queue:
            self._queue.remove(render_job_id)

    def _set_running(self, job: RenderJob, *, phase: str) -> None:
        now = _utcnow()
        job.status.state = "running"
        job.status.timestamps.started_at = job.status.timestamps.started_at or now
        job.status.timestamps.updated_at = now
        logger.info("render_start run_id=%s render_job_id=%s phase=%s", job.run_id, job.render_job_id, phase)

    def _set_completed(self, job: RenderJob) -> None:
        now = _utcnow()
        job.status.state = "completed"
        job.status.progress = 1.0
        job.status.timestamps.completed_at = now
        job.status.timestamps.updated_at = now
        job.status.failed_shot_indices = sorted(list(job.failed.keys()))
        logger.info(
            "render_done run_id=%s render_job_id=%s output=%s retries=%s",
            job.run_id,
            job.render_job_id,
            job.status.output_path,
            job.status.retry_count,
        )

    def _set_canceled(self, job: RenderJob, *, reason: str) -> None:
        now = _utcnow()
        job.status.state = "canceled"
        job.status.cancellation_requested = True
        job.status.timestamps.cancelled_at = job.status.timestamps.cancelled_at or now
        job.status.timestamps.completed_at = job.status.timestamps.completed_at or now
        job.status.timestamps.updated_at = now
        if reason:
            job.status.errors.append(reason)

    def _render_all_shots(self, job: RenderJob, *, mode: str) -> None:
        total = max(1, len(job.prompt_specs))
        for idx, spec in enumerate(job.prompt_specs, start=1):
            result = self._render_one_shot(job, spec, mode=mode)
            if result.success and result.output_path:
                job.shot_outputs[spec.shot_idx] = result.output_path
            else:
                job.failed[spec.shot_idx] = result.error or "unknown render error"
            job.status.progress = min(0.9, idx / float(total))
            job.status.timestamps.updated_at = _utcnow()
            job.status.failed_shot_indices = sorted(list(job.failed.keys()))

    def _render_one_shot(self, job: RenderJob, spec: PromptSpec, *, mode: str) -> ShotRenderResult:
        budget_limit = float(job.budget.get("max_total_cost", 1e18) or 1e18)
        if job.total_cost > budget_limit:
            return ShotRenderResult(
                shot_idx=spec.shot_idx,
                success=False,
                error="budget exceeded",
                cost=0.0,
            )
        result = self._adapter.render_shot(
            prompt_spec=spec,
            output_dir=job.output_dir,
            mode=mode,
            budget=job.budget,
            run_id=job.run_id,
            render_job_id=job.render_job_id,
        )
        job.total_cost += max(0.0, float(result.cost or 0.0))
        return result

    @staticmethod
    def _prompt_by_idx(prompt_specs: Sequence[PromptSpec], shot_idx: int) -> Optional[PromptSpec]:
        for spec in prompt_specs:
            if int(spec.shot_idx) == int(shot_idx):
                return spec
        return None

    @staticmethod
    def _board_from_prompts(job: RenderJob) -> Storyboard:
        shots: List[Shot] = []
        for idx, spec in enumerate(job.prompt_specs, start=1):
            duration = float(spec.seedance_params.get("duration_sec", 4.0) or 4.0)
            shots.append(
                Shot(
                    idx=idx,
                    duration=max(1.0, duration),
                    camera=str(spec.seedance_params.get("camera", "medium")),
                    scene="fallback scene",
                    subject_id=str(spec.seedance_params.get("character_id", "host_01")),
                    action=spec.prompt_text,
                    overlay_text=None,
                    reference_assets=list(spec.references or []),
                )
            )

        total = int(round(sum(shot.duration for shot in shots))) or 30
        return Storyboard(
            run_id=job.run_id,
            item_id=f"item_{job.render_job_id}",
            duration_sec=total,
            aspect="9:16",
            shots=shots,
        )
