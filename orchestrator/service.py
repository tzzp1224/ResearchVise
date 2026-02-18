"""Orchestrator service layer for daily and on-demand runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Dict, List, Optional, Tuple
from uuid import uuid4
from zoneinfo import ZoneInfo

from core import Artifact, RunMode, RunRequest, RunStatus
from .queue import InMemoryRunQueue
from .store import InMemoryRunStore


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class DailyDigestSubscription:
    """Daily digest schedule registration."""

    subscription_id: str
    user_id: str
    run_at: str
    tz: str
    top_k: int = 3
    created_at: str = field(default_factory=_utc_iso)
    last_triggered_on: Optional[str] = None


class RunOrchestrator:
    """Central orchestrator for run queue lifecycle and status APIs."""

    def __init__(
        self,
        *,
        store: Optional[InMemoryRunStore] = None,
        queue: Optional[InMemoryRunQueue] = None,
    ) -> None:
        self._store = store or InMemoryRunStore()
        self._queue = queue or InMemoryRunQueue()
        self._subs: Dict[str, DailyDigestSubscription] = {}
        self._lock = Lock()

    def schedule_daily_digest(self, user_id: str, run_at: str = "08:00", tz: str = "UTC", top_k: int = 3) -> str:
        """Register daily digest schedule at user-local time."""
        sub_id = f"sub_{uuid4().hex[:10]}"
        with self._lock:
            self._subs[sub_id] = DailyDigestSubscription(
                subscription_id=sub_id,
                user_id=str(user_id).strip(),
                run_at=str(run_at).strip() or "08:00",
                tz=str(tz).strip() or "UTC",
                top_k=max(1, min(5, int(top_k or 3))),
            )
        return sub_id

    def list_subscriptions(self) -> List[DailyDigestSubscription]:
        with self._lock:
            return [DailyDigestSubscription(**item.__dict__) for item in self._subs.values()]

    def trigger_due_daily_runs(self, *, now_utc: Optional[datetime] = None) -> List[str]:
        """Enqueue due daily runs for subscriptions whose local time passed run_at."""
        now = now_utc or datetime.now(timezone.utc)
        created: List[str] = []

        with self._lock:
            subs = [DailyDigestSubscription(**item.__dict__) for item in self._subs.values()]

        for sub in subs:
            try:
                tz = ZoneInfo(sub.tz)
            except Exception:
                tz = ZoneInfo("UTC")
            local_now = now.astimezone(tz)
            run_hour, run_minute = _parse_run_at(sub.run_at)
            target_minute = run_hour * 60 + run_minute
            current_minute = local_now.hour * 60 + local_now.minute
            local_date = local_now.date().isoformat()

            if current_minute < target_minute:
                continue
            if sub.last_triggered_on == local_date:
                continue

            run_request = RunRequest(
                user_id=sub.user_id,
                mode=RunMode.DAILY,
                topic=None,
                time_window="24h",
                tz=sub.tz,
                budget={"top_k": sub.top_k},
                output_targets=["script", "storyboard", "onepager", "mp4", "zip"],
            )
            idempotency_key = f"daily:{sub.user_id}:{local_date}:{sub.run_at}"
            run_id = self.enqueue_run(run_request, idempotency_key=idempotency_key)
            created.append(run_id)

            with self._lock:
                current = self._subs.get(sub.subscription_id)
                if current:
                    current.last_triggered_on = local_date

        return created

    def enqueue_run(self, run_request: RunRequest, *, idempotency_key: Optional[str] = None) -> str:
        """Create and enqueue a run request. Returns stable run_id for same idempotency key."""
        run_id = self._store.create_or_get(run_request, idempotency_key=idempotency_key)
        self._queue.enqueue(run_id)
        return run_id

    def get_run_status(self, run_id: str) -> Optional[RunStatus]:
        """Fetch current run status snapshot."""
        return self._store.get_status(run_id)

    def cancel_run(self, run_id: str) -> bool:
        """Best-effort cancel: remove from queue and set cancel flags/status."""
        _ = self._queue.remove(run_id)
        status = self._store.request_cancel(run_id)
        return status is not None

    def dequeue_next_run(self) -> Optional[Tuple[str, RunRequest]]:
        """Worker-facing method to pick next queued run."""
        run_id = self._queue.dequeue()
        if not run_id:
            return None
        req = self._store.get_request(run_id)
        if req is None:
            return None
        self._store.update_running(run_id)
        return run_id, req

    def mark_run_progress(self, run_id: str, progress: float) -> Optional[RunStatus]:
        return self._store.update_progress(run_id, progress)

    def mark_run_completed(self, run_id: str) -> Optional[RunStatus]:
        return self._store.update_completed(run_id)

    def mark_run_failed(self, run_id: str, error: str) -> Optional[RunStatus]:
        return self._store.update_failed(run_id, error)

    def mark_run_canceled(self, run_id: str, error: Optional[str] = None) -> Optional[RunStatus]:
        return self._store.update_canceled(run_id, error=error)

    def retry_run(self, run_id: str) -> bool:
        """Requeue failed run if retry budget allows."""
        status = self._store.mark_retrying(run_id)
        if status is None:
            return False
        if status.retry_count > status.max_retries:
            return False
        return self._queue.enqueue(run_id)

    def add_artifact(self, run_id: str, artifact: Artifact) -> bool:
        return self._store.add_artifact(run_id, artifact)

    def list_artifacts(self, run_id: str) -> List[Artifact]:
        return self._store.list_artifacts(run_id)

    def set_render_job(self, run_id: str, render_job_id: str) -> bool:
        return self._store.set_render_job(run_id, render_job_id)

    def get_render_job(self, run_id: str) -> Optional[str]:
        return self._store.get_render_job(run_id)

    def append_event(self, run_id: str, event: str, message: str) -> bool:
        return self._store.append_event(run_id, event, message)

    def list_events(self, run_id: str) -> List[Dict[str, str]]:
        return self._store.list_events(run_id)


_DEFAULT_ORCHESTRATOR = RunOrchestrator()


def get_default_orchestrator() -> RunOrchestrator:
    return _DEFAULT_ORCHESTRATOR


def schedule_daily_digest(user_id: str, run_at: str = "08:00", tz: str = "UTC", top_k: int = 3) -> str:
    """Top-level API required by PRD module A."""
    return _DEFAULT_ORCHESTRATOR.schedule_daily_digest(user_id=user_id, run_at=run_at, tz=tz, top_k=top_k)


def enqueue_run(run_request: RunRequest, *, idempotency_key: Optional[str] = None) -> str:
    """Top-level API required by PRD module A."""
    return _DEFAULT_ORCHESTRATOR.enqueue_run(run_request, idempotency_key=idempotency_key)


def get_run_status(run_id: str) -> Optional[RunStatus]:
    """Top-level API required by PRD module A."""
    return _DEFAULT_ORCHESTRATOR.get_run_status(run_id)


def cancel_run(run_id: str) -> bool:
    """Top-level API required by PRD module A."""
    return _DEFAULT_ORCHESTRATOR.cancel_run(run_id)


def list_artifacts(run_id: str) -> List[Artifact]:
    return _DEFAULT_ORCHESTRATOR.list_artifacts(run_id)


def get_render_job(run_id: str) -> Optional[str]:
    return _DEFAULT_ORCHESTRATOR.get_render_job(run_id)


def trigger_due_daily_runs(*, now_utc: Optional[datetime] = None) -> List[str]:
    return _DEFAULT_ORCHESTRATOR.trigger_due_daily_runs(now_utc=now_utc)


def _parse_run_at(run_at: str) -> Tuple[int, int]:
    text = str(run_at or "").strip()
    parts = text.split(":")
    if len(parts) != 2:
        return 8, 0
    try:
        hour = max(0, min(23, int(parts[0])))
        minute = max(0, min(59, int(parts[1])))
        return hour, minute
    except Exception:
        return 8, 0
