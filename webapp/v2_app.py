"""v2 FastAPI app for on-demand and scheduled digest runs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from core import RunMode, RunRequest
from webapp.runtime import get_orchestrator, get_runtime


app = FastAPI(title="AcademicResearchAgent v2 API")


class OnDemandRunPayload(BaseModel):
    user_id: str
    topic: str
    time_window: str = "24h"
    tz: str = "UTC"
    budget: Dict[str, Any] = Field(default_factory=dict)
    output_targets: List[str] = Field(default_factory=lambda: ["web", "mp4"])

    @field_validator("user_id", "topic", "tz")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("value is required")
        return text


class DailySchedulePayload(BaseModel):
    user_id: str
    run_at: str = "08:00"
    tz: str = "UTC"
    top_k: int = 3

    @field_validator("user_id", "run_at", "tz")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("value is required")
        return text


@app.get("/api/v2/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}


@app.post("/api/v2/runs/ondemand")
def create_ondemand_run(payload: OnDemandRunPayload) -> Dict[str, Any]:
    orchestrator = get_orchestrator()
    request = RunRequest(
        user_id=payload.user_id,
        mode=RunMode.ONDEMAND,
        topic=payload.topic,
        time_window=payload.time_window,
        tz=payload.tz,
        budget=payload.budget,
        output_targets=payload.output_targets,
    )
    idem = f"ondemand:{payload.user_id}:{payload.topic}:{payload.time_window}:{payload.tz}"
    run_id = orchestrator.enqueue_run(request, idempotency_key=idem)
    status = orchestrator.get_run_status(run_id)
    return {"run_id": run_id, "status": status.model_dump(mode="json") if status else None}


@app.post("/api/v2/runs/daily/schedule")
def create_daily_schedule(payload: DailySchedulePayload) -> Dict[str, Any]:
    orchestrator = get_orchestrator()
    sub_id = orchestrator.schedule_daily_digest(
        user_id=payload.user_id,
        run_at=payload.run_at,
        tz=payload.tz,
        top_k=payload.top_k,
    )
    return {"subscription_id": sub_id, "run_at": payload.run_at, "tz": payload.tz, "top_k": payload.top_k}


@app.post("/api/v2/runs/daily/tick")
def tick_daily_scheduler(now_utc_iso: Optional[str] = None) -> Dict[str, Any]:
    orchestrator = get_orchestrator()
    now_utc = None
    text = str(now_utc_iso or "").strip()
    if text:
        now_utc = datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    created = orchestrator.trigger_due_daily_runs(now_utc=now_utc)
    return {"created_run_ids": created, "count": len(created)}


@app.get("/api/v2/runs/{run_id}")
def get_run(run_id: str) -> Dict[str, Any]:
    runtime = get_runtime()
    bundle = runtime.get_run_bundle(run_id)
    if bundle["status"] is None:
        raise HTTPException(status_code=404, detail="run not found")
    return bundle


@app.post("/api/v2/runs/{run_id}/cancel")
def cancel_run(run_id: str) -> Dict[str, Any]:
    orchestrator = get_orchestrator()
    ok = orchestrator.cancel_run(run_id)
    status = orchestrator.get_run_status(run_id)
    if not ok and status is None:
        raise HTTPException(status_code=404, detail="run not found")
    return {"run_id": run_id, "canceled": bool(ok), "status": status.model_dump(mode="json") if status else None}


@app.post("/api/v2/workers/runs/next")
def run_worker_next() -> Dict[str, Any]:
    runtime = get_runtime()
    result = runtime.run_next()
    if not result:
        return {"processed": False}
    return {
        "processed": True,
        "run_id": result.run_id,
        "output_dir": result.output_dir,
        "top_item_ids": result.top_item_ids,
        "render_job_id": result.render_job_id,
    }


@app.post("/api/v2/workers/render/next")
def render_worker_next() -> Dict[str, Any]:
    runtime = get_runtime()
    status = runtime.process_next_render()
    if not status:
        return {"processed": False}
    return {
        "processed": True,
        "render_job_id": status.render_job_id,
        "run_id": status.run_id,
        "state": status.state,
        "progress": status.progress,
        "output_path": status.output_path,
        "valid_mp4": status.valid_mp4,
        "probe_error": status.probe_error,
        "retry_count": status.retry_count,
        "errors": list(status.errors or []),
    }


@app.post("/api/v2/renders/{render_job_id}/confirm")
def confirm_render(render_job_id: str, approved: bool = True) -> Dict[str, Any]:
    runtime = get_runtime()
    status = runtime.confirm_render(render_job_id, approved=approved)
    if not status:
        raise HTTPException(status_code=404, detail="render job not found")
    return {
        "render_job_id": status.render_job_id,
        "run_id": status.run_id,
        "state": status.state,
        "progress": status.progress,
        "output_path": status.output_path,
        "valid_mp4": status.valid_mp4,
        "probe_error": status.probe_error,
        "errors": list(status.errors or []),
    }


@app.post("/api/v2/renders/{render_job_id}/cancel")
def cancel_render(render_job_id: str) -> Dict[str, Any]:
    runtime = get_runtime()
    canceled = runtime.cancel_render(render_job_id)
    if not canceled:
        raise HTTPException(status_code=404, detail="render job not found")
    status = runtime.get_render_status(render_job_id)
    return {
        "render_job_id": render_job_id,
        "canceled": True,
        "state": status.state if status else "canceled",
    }
