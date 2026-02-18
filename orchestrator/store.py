"""In-memory run store for orchestration status tracking."""

from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from typing import Dict, Optional
from uuid import uuid4

from core import RunRequest, RunStatus


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_run_id() -> str:
    return f"run_{_utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"


class InMemoryRunStore:
    """Thread-safe store for run requests and statuses."""

    def __init__(self) -> None:
        self._requests: Dict[str, RunRequest] = {}
        self._statuses: Dict[str, RunStatus] = {}
        self._idempotency_keys: Dict[str, str] = {}
        self._lock = Lock()

    def create_or_get(self, request: RunRequest, *, idempotency_key: Optional[str] = None) -> str:
        """Create run record or return existing run_id for same idempotency key."""
        with self._lock:
            if idempotency_key:
                existing = self._idempotency_keys.get(idempotency_key)
                if existing:
                    return existing

            run_id = _new_run_id()
            self._requests[run_id] = request
            self._statuses[run_id] = RunStatus(run_id=run_id, state="queued", progress=0.0)

            if idempotency_key:
                self._idempotency_keys[idempotency_key] = run_id

            return run_id

    def get_request(self, run_id: str) -> Optional[RunRequest]:
        with self._lock:
            return self._requests.get(run_id)

    def get_status(self, run_id: str) -> Optional[RunStatus]:
        with self._lock:
            status = self._statuses.get(run_id)
            return status.model_copy(deep=True) if status else None

    def update_running(self, run_id: str) -> Optional[RunStatus]:
        with self._lock:
            status = self._statuses.get(run_id)
            if not status:
                return None
            now = _utcnow()
            status.state = "running"
            status.timestamps.started_at = status.timestamps.started_at or now
            status.timestamps.updated_at = now
            return status.model_copy(deep=True)

    def update_progress(self, run_id: str, progress: float) -> Optional[RunStatus]:
        with self._lock:
            status = self._statuses.get(run_id)
            if not status:
                return None
            status.progress = max(0.0, min(1.0, float(progress)))
            status.timestamps.updated_at = _utcnow()
            return status.model_copy(deep=True)

    def update_completed(self, run_id: str) -> Optional[RunStatus]:
        with self._lock:
            status = self._statuses.get(run_id)
            if not status:
                return None
            now = _utcnow()
            status.state = "completed"
            status.progress = 1.0
            status.timestamps.completed_at = now
            status.timestamps.updated_at = now
            return status.model_copy(deep=True)

    def update_failed(self, run_id: str, error: str) -> Optional[RunStatus]:
        with self._lock:
            status = self._statuses.get(run_id)
            if not status:
                return None
            status.state = "failed"
            if error:
                status.errors.append(str(error))
            status.timestamps.updated_at = _utcnow()
            return status.model_copy(deep=True)

    def request_cancel(self, run_id: str) -> Optional[RunStatus]:
        with self._lock:
            status = self._statuses.get(run_id)
            if not status:
                return None
            now = _utcnow()
            status.cancellation_requested = True
            if status.state in {"queued", "retrying"}:
                status.state = "canceled"
                status.timestamps.cancelled_at = now
                status.timestamps.completed_at = status.timestamps.completed_at or now
            elif status.state in {"running"}:
                status.state = "cancel_requested"
            status.timestamps.updated_at = now
            return status.model_copy(deep=True)

    def mark_retrying(self, run_id: str) -> Optional[RunStatus]:
        with self._lock:
            status = self._statuses.get(run_id)
            if not status:
                return None
            if status.retry_count >= status.max_retries:
                return status.model_copy(deep=True)
            status.retry_count += 1
            status.state = "retrying"
            status.timestamps.updated_at = _utcnow()
            return status.model_copy(deep=True)
