"""Orchestrator service layer for daily and on-demand runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Dict, Optional, Tuple
from uuid import uuid4

from core import RunRequest, RunStatus
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
    created_at: str = field(default_factory=_utc_iso)


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

    def schedule_daily_digest(self, user_id: str, run_at: str = "08:00", tz: str = "UTC") -> str:
        """Register daily digest schedule at user-local time."""
        sub_id = f"sub_{uuid4().hex[:10]}"
        with self._lock:
            self._subs[sub_id] = DailyDigestSubscription(
                subscription_id=sub_id,
                user_id=str(user_id).strip(),
                run_at=str(run_at).strip() or "08:00",
                tz=str(tz).strip() or "UTC",
            )
        return sub_id

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

    def retry_run(self, run_id: str) -> bool:
        """Requeue failed run if retry budget allows."""
        status = self._store.mark_retrying(run_id)
        if status is None:
            return False
        if status.retry_count > status.max_retries:
            return False
        return self._queue.enqueue(run_id)


_DEFAULT_ORCHESTRATOR = RunOrchestrator()


def get_default_orchestrator() -> RunOrchestrator:
    return _DEFAULT_ORCHESTRATOR


def schedule_daily_digest(user_id: str, run_at: str = "08:00", tz: str = "UTC") -> str:
    """Top-level API required by PRD module A."""
    return _DEFAULT_ORCHESTRATOR.schedule_daily_digest(user_id=user_id, run_at=run_at, tz=tz)


def enqueue_run(run_request: RunRequest, *, idempotency_key: Optional[str] = None) -> str:
    """Top-level API required by PRD module A."""
    return _DEFAULT_ORCHESTRATOR.enqueue_run(run_request, idempotency_key=idempotency_key)


def get_run_status(run_id: str) -> Optional[RunStatus]:
    """Top-level API required by PRD module A."""
    return _DEFAULT_ORCHESTRATOR.get_run_status(run_id)


def cancel_run(run_id: str) -> bool:
    """Top-level API required by PRD module A."""
    return _DEFAULT_ORCHESTRATOR.cancel_run(run_id)
