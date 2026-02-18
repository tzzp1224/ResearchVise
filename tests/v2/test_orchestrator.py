from __future__ import annotations

from datetime import datetime, timezone

from core import RunMode, RunRequest
from orchestrator.queue import InMemoryRunQueue
from orchestrator.service import RunOrchestrator
from orchestrator.store import InMemoryRunStore


def _build_req() -> RunRequest:
    return RunRequest(
        user_id="user_1",
        mode=RunMode.ONDEMAND,
        topic="mcp deployment",
        time_window="24h",
        tz="America/Los_Angeles",
        output_targets=["script", "storyboard", "mp4"],
    )


def test_enqueue_is_idempotent_with_same_key() -> None:
    svc = RunOrchestrator(store=InMemoryRunStore(), queue=InMemoryRunQueue())
    req = _build_req()

    run_id_1 = svc.enqueue_run(req, idempotency_key="user_1:mcp:24h")
    run_id_2 = svc.enqueue_run(req, idempotency_key="user_1:mcp:24h")

    assert run_id_1 == run_id_2
    assert svc.get_run_status(run_id_1) is not None


def test_dequeue_updates_running_and_progress_then_complete() -> None:
    svc = RunOrchestrator(store=InMemoryRunStore(), queue=InMemoryRunQueue())
    req = _build_req()
    run_id = svc.enqueue_run(req)

    picked = svc.dequeue_next_run()
    assert picked is not None
    picked_run_id, picked_req = picked
    assert picked_run_id == run_id
    assert picked_req.topic == req.topic

    running = svc.get_run_status(run_id)
    assert running is not None
    assert running.state == "running"

    progressed = svc.mark_run_progress(run_id, 0.55)
    assert progressed is not None
    assert progressed.progress == 0.55

    completed = svc.mark_run_completed(run_id)
    assert completed is not None
    assert completed.state == "completed"
    assert completed.progress == 1.0


def test_cancel_before_start_marks_canceled() -> None:
    svc = RunOrchestrator(store=InMemoryRunStore(), queue=InMemoryRunQueue())
    run_id = svc.enqueue_run(_build_req())

    ok = svc.cancel_run(run_id)
    assert ok is True

    status = svc.get_run_status(run_id)
    assert status is not None
    assert status.cancellation_requested is True
    assert status.state == "canceled"


def test_retry_failed_run_requeues_when_budget_available() -> None:
    svc = RunOrchestrator(store=InMemoryRunStore(), queue=InMemoryRunQueue())
    run_id = svc.enqueue_run(_build_req())
    _ = svc.dequeue_next_run()
    _ = svc.mark_run_failed(run_id, "seedance timeout")

    requeued = svc.retry_run(run_id)
    assert requeued is True

    picked = svc.dequeue_next_run()
    assert picked is not None
    assert picked[0] == run_id


def test_daily_digest_triggers_once_per_local_day() -> None:
    svc = RunOrchestrator(store=InMemoryRunStore(), queue=InMemoryRunQueue())
    sub_id = svc.schedule_daily_digest(user_id="u_daily", run_at="08:00", tz="America/Los_Angeles", top_k=5)
    assert sub_id.startswith("sub_")

    due_time = datetime(2026, 2, 18, 16, 5, tzinfo=timezone.utc)  # 08:05 PST
    created_first = svc.trigger_due_daily_runs(now_utc=due_time)
    assert len(created_first) == 1

    created_second = svc.trigger_due_daily_runs(now_utc=due_time)
    assert created_second == []

    picked = svc.dequeue_next_run()
    assert picked is not None
    run_id, run_request = picked
    assert run_id == created_first[0]
    assert run_request.mode == RunMode.DAILY
    assert int((run_request.budget or {}).get("top_k") or 0) == 5
