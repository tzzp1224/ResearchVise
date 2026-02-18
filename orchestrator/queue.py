"""In-memory FIFO queue with dedup support for run IDs."""

from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Deque, Optional


class InMemoryRunQueue:
    """Best-effort in-memory queue for run orchestration."""

    def __init__(self) -> None:
        self._queue: Deque[str] = deque()
        self._enqueued = set()
        self._lock = Lock()

    def enqueue(self, run_id: str) -> bool:
        """Queue run ID once. Returns True when newly enqueued."""
        with self._lock:
            if run_id in self._enqueued:
                return False
            self._queue.append(run_id)
            self._enqueued.add(run_id)
            return True

    def dequeue(self) -> Optional[str]:
        """Pop next run ID, or None when empty."""
        with self._lock:
            if not self._queue:
                return None
            run_id = self._queue.popleft()
            self._enqueued.discard(run_id)
            return run_id

    def remove(self, run_id: str) -> bool:
        """Best-effort removal for cancellation before start."""
        with self._lock:
            if run_id not in self._enqueued:
                return False
            filtered = deque(item for item in self._queue if item != run_id)
            self._queue = filtered
            self._enqueued.discard(run_id)
            return True

    def size(self) -> int:
        with self._lock:
            return len(self._queue)
