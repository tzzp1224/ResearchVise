from __future__ import annotations

import importlib
from pathlib import Path

from fastapi.testclient import TestClient

from core import PromptSpec, RawItem
from orchestrator.queue import InMemoryRunQueue
from orchestrator.service import RunOrchestrator
from orchestrator.store import InMemoryRunStore
from pipeline_v2.runtime import RunPipelineRuntime
from render.adapters import BaseRendererAdapter, ShotRenderResult
from render.manager import RenderManager


class ApiAdapter(BaseRendererAdapter):
    provider = "api-mock"

    def render_shot(self, *, prompt_spec: PromptSpec, output_dir: Path, mode: str, budget: dict, run_id: str, render_job_id: str) -> ShotRenderResult:
        _ = mode, budget, run_id, render_job_id
        target = output_dir / f"shot_{prompt_spec.shot_idx:03d}.mp4"
        target.write_bytes(b"api-mock-shot")
        return ShotRenderResult(shot_idx=prompt_spec.shot_idx, success=True, output_path=str(target), cost=0.1)


def _connectors() -> dict:
    async def _gh(max_results: int = 12):
        _ = max_results
        return [
            RawItem(
                id="gh_1",
                source="github",
                title="org/repo-api",
                url="https://github.com/org/repo-api",
                body="API test content with architecture notes.",
                author="org",
                tier="A",
                metadata={"stars": 1000, "item_type": "repo"},
            )
        ]

    async def _gh_release(repo_full_names, max_results_per_repo: int = 1):
        _ = repo_full_names, max_results_per_repo
        return []

    async def _hf(max_results: int = 12):
        _ = max_results
        return []

    async def _hn(max_results: int = 12):
        _ = max_results
        return []

    async def _rss(feed_url: str, max_results: int = 6):
        _ = feed_url, max_results
        return []

    async def _web(url: str):
        _ = url
        return []

    return {
        "fetch_github_trending": _gh,
        "fetch_github_releases": _gh_release,
        "fetch_huggingface_trending": _hf,
        "fetch_hackernews_top": _hn,
        "fetch_rss_feed": _rss,
        "fetch_web_article": _web,
    }


def _client(tmp_path: Path) -> TestClient:
    module = importlib.import_module("webapp.v2_app")
    orchestrator = RunOrchestrator(store=InMemoryRunStore(), queue=InMemoryRunQueue())
    runtime = RunPipelineRuntime(
        orchestrator=orchestrator,
        render_manager=RenderManager(renderer_adapter=ApiAdapter(), work_dir=tmp_path / "render"),
        output_root=tmp_path / "runs",
        connector_overrides=_connectors(),
    )
    module.get_orchestrator = lambda: orchestrator
    module.get_runtime = lambda: runtime
    return TestClient(module.app)


def test_v2_ondemand_run_and_workers(tmp_path: Path) -> None:
    client = _client(tmp_path)
    create = client.post(
        "/api/v2/runs/ondemand",
        json={
            "user_id": "u1",
            "topic": "mcp",
            "time_window": "24h",
            "tz": "UTC",
            "budget": {"duration_sec": 33, "max_total_cost": 5.0},
            "output_targets": ["web", "mp4"],
        },
    )
    assert create.status_code == 200
    run_id = create.json()["run_id"]

    worker = client.post("/api/v2/workers/runs/next")
    assert worker.status_code == 200
    assert worker.json()["processed"] is True
    assert worker.json()["run_id"] == run_id

    status_before = client.get(f"/api/v2/runs/{run_id}")
    assert status_before.status_code == 200
    assert status_before.json()["status"]["state"] == "completed"

    render = client.post("/api/v2/workers/render/next")
    assert render.status_code == 200
    assert render.json()["processed"] is True
    assert render.json()["run_id"] == run_id
    assert render.json()["valid_mp4"] is True
    assert render.json()["probe_error"] is None

    status_after = client.get(f"/api/v2/runs/{run_id}")
    types = {item["type"] for item in status_after.json()["artifacts"]}
    assert "mp4" in types


def test_v2_daily_schedule_tick(tmp_path: Path) -> None:
    client = _client(tmp_path)
    scheduled = client.post(
        "/api/v2/runs/daily/schedule",
        json={"user_id": "u2", "run_at": "08:00", "tz": "America/Los_Angeles", "top_k": 5},
    )
    assert scheduled.status_code == 200
    assert scheduled.json()["top_k"] == 5

    tick = client.post("/api/v2/runs/daily/tick", params={"now_utc_iso": "2026-02-18T16:10:00+00:00"})
    assert tick.status_code == 200
    assert tick.json()["count"] == 1


def test_v2_preview_confirm_final_render_flow(tmp_path: Path) -> None:
    client = _client(tmp_path)
    create = client.post(
        "/api/v2/runs/ondemand",
        json={
            "user_id": "u3",
            "topic": "mcp preview",
            "time_window": "24h",
            "tz": "UTC",
            "budget": {"duration_sec": 35, "max_total_cost": 5.0, "confirm_required": True},
            "output_targets": ["mp4"],
        },
    )
    run_id = create.json()["run_id"]

    worker = client.post("/api/v2/workers/runs/next")
    assert worker.json()["processed"] is True
    render_job_id = worker.json()["render_job_id"]

    preview = client.post("/api/v2/workers/render/next")
    assert preview.status_code == 200
    assert preview.json()["state"] == "awaiting_confirmation"

    confirm = client.post(f"/api/v2/renders/{render_job_id}/confirm", params={"approved": "true"})
    assert confirm.status_code == 200
    assert confirm.json()["state"] == "queued"

    final = client.post("/api/v2/workers/render/next")
    assert final.status_code == 200
    assert final.json()["state"] == "completed"
    assert final.json()["run_id"] == run_id
    assert final.json()["valid_mp4"] is True
