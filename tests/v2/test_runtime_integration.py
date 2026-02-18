from __future__ import annotations

import json
from pathlib import Path

from core import PromptSpec, RawItem, RunMode, RunRequest
from orchestrator import RunOrchestrator
from orchestrator.queue import InMemoryRunQueue
from orchestrator.store import InMemoryRunStore
from pipeline_v2.runtime import RunPipelineRuntime
from render.adapters import BaseRendererAdapter, ShotRenderResult
from render.manager import RenderManager


class AlwaysSuccessAdapter(BaseRendererAdapter):
    provider = "mock-success"

    def render_shot(self, *, prompt_spec: PromptSpec, output_dir: Path, mode: str, budget: dict, run_id: str, render_job_id: str) -> ShotRenderResult:
        _ = mode, budget, run_id, render_job_id
        path = output_dir / f"shot_{prompt_spec.shot_idx:03d}.mp4"
        path.write_bytes(f"shot-{prompt_spec.shot_idx}".encode("utf-8"))
        return ShotRenderResult(shot_idx=prompt_spec.shot_idx, success=True, output_path=str(path), cost=0.3)


def _connectors() -> dict:
    async def _github_trending(max_results: int = 12):
        _ = max_results
        return [
            RawItem(
                id="gh_1",
                source="github",
                title="org/repo-mcp",
                url="https://github.com/org/repo-mcp",
                body="MCP deployment benchmark architecture diagram",
                author="org",
                tier="A",
                metadata={"stars": 1200, "item_type": "repo", "has_diagram": True},
            )
        ]

    async def _github_releases(repo_full_names, max_results_per_repo: int = 1):
        _ = repo_full_names, max_results_per_repo
        return [
            RawItem(
                id="gh_rel_1",
                source="github",
                title="org/repo-mcp v1.2.0",
                url="https://github.com/org/repo-mcp/releases/tag/v1.2.0",
                body="Release with rollback-safe migration and metrics.",
                author="org",
                tier="A",
                metadata={"item_type": "release", "stars": 1300},
            )
        ]

    async def _hf_trending(max_results: int = 12):
        _ = max_results
        return [
            RawItem(
                id="hf_1",
                source="huggingface",
                title="org/mcp-encoder",
                url="https://huggingface.co/org/mcp-encoder",
                body="Model release for protocol embedding.",
                author="org",
                tier="A",
                metadata={"downloads": 22000, "likes": 230, "item_type": "model"},
            )
        ]

    async def _hn_top(max_results: int = 12):
        _ = max_results
        return [
            RawItem(
                id="hn_1",
                source="hackernews",
                title="MCP in production? lessons learned",
                url="https://news.ycombinator.com/item?id=1",
                body="Community discussion with deployment anecdotes.",
                author="hn_user",
                tier="A",
                metadata={"points": 420, "comment_count": 120, "item_type": "story"},
            )
        ]

    async def _rss(feed_url: str, max_results: int = 6):
        _ = feed_url, max_results
        return [
            RawItem(
                id="rss_1",
                source="rss",
                title="Industry blog on MCP adoption",
                url="https://blog.example/mcp",
                body="Blog evidence and summary with references [ref](https://blog.example/ref).",
                author="editor",
                tier="B",
                metadata={"item_type": "rss_entry"},
            )
        ]

    async def _web(url: str):
        _ = url
        return [
            RawItem(
                id="web_1",
                source="web_article",
                title="Deep dive article",
                url="https://article.example/mcp",
                body="A long article about architecture and benchmark details.",
                author="author",
                tier="B",
                metadata={"item_type": "web_article"},
            )
        ]

    return {
        "fetch_github_trending": _github_trending,
        "fetch_github_releases": _github_releases,
        "fetch_huggingface_trending": _hf_trending,
        "fetch_hackernews_top": _hn_top,
        "fetch_rss_feed": _rss,
        "fetch_web_article": _web,
    }


def test_runrequest_to_sync_artifacts_and_async_render(tmp_path: Path) -> None:
    orchestrator = RunOrchestrator(store=InMemoryRunStore(), queue=InMemoryRunQueue())
    render_manager = RenderManager(renderer_adapter=AlwaysSuccessAdapter(), work_dir=tmp_path / "render")
    runtime = RunPipelineRuntime(
        orchestrator=orchestrator,
        render_manager=render_manager,
        output_root=tmp_path / "runs",
        connector_overrides=_connectors(),
    )

    run_id = orchestrator.enqueue_run(
        RunRequest(
            user_id="u_1",
            mode=RunMode.ONDEMAND,
            topic="mcp deployment",
            time_window="24h",
            tz="America/Los_Angeles",
            budget={
                "duration_sec": 36,
                "max_retries": 1,
                "max_total_cost": 10.0,
                "include_tier_b": True,
                "rss_feeds": ["https://rss.example"],
                "seed_url": "https://article.example/mcp",
            },
            output_targets=["web", "mp4"],
        ),
        idempotency_key="u_1:mcp:24h",
    )

    result = runtime.run_next()
    assert result is not None
    assert result.run_id == run_id
    assert Path(result.output_dir).exists()
    assert result.render_job_id is not None

    bundle_before_render = runtime.get_run_bundle(run_id)
    status_before_render = bundle_before_render["status"]
    assert status_before_render["state"] == "completed"

    artifact_types = {item["type"] for item in bundle_before_render["artifacts"]}
    assert "facts" in artifact_types
    assert "script" in artifact_types
    assert "storyboard" in artifact_types
    assert "onepager" in artifact_types
    assert "thumbnail" in artifact_types
    assert "audio" in artifact_types
    assert "srt" in artifact_types
    assert "zip" in artifact_types
    assert "mp4" not in artifact_types

    run_dir = Path(result.output_dir)
    script_payload = json.loads((run_dir / "script.json").read_text(encoding="utf-8"))
    assert "structure" in script_payload
    assert len(script_payload["structure"]["key_points"]) == 3
    assert "facts" in script_payload
    assert (run_dir / "facts.json").exists()
    run_context_payload = json.loads((run_dir / "run_context.json").read_text(encoding="utf-8"))
    assert run_context_payload["data_mode"] == "live"
    assert run_context_payload["topic"] == "mcp deployment"
    assert "ranking_stats" in run_context_payload
    assert "drop_reason_samples" in run_context_payload["ranking_stats"]
    assert "top_quality_signals" in run_context_payload["ranking_stats"]
    assert "top_why_ranked" in run_context_payload["ranking_stats"]
    assert "connector_stats" in run_context_payload
    assert "extraction_stats" in run_context_payload

    materials_payload = json.loads((run_dir / "materials.json").read_text(encoding="utf-8"))
    assert materials_payload["screenshot_plan"]
    assert materials_payload["icon_keyword_suggestions"]
    assert materials_payload["broll_categories"]
    assert "quality_metrics" in materials_payload
    assert materials_payload["data_mode"] == "live"

    onepager_text = (run_dir / "onepager.md").read_text(encoding="utf-8")
    assert "DataMode: `live`" in onepager_text

    render_status = runtime.process_next_render()
    assert render_status is not None
    assert render_status.state == "completed"
    assert render_status.output_path and Path(render_status.output_path).exists()
    assert render_status.valid_mp4 is True
    assert render_status.probe_error is None

    bundle_after_render = runtime.get_run_bundle(run_id)
    artifact_types_after = {item["type"] for item in bundle_after_render["artifacts"]}
    assert "mp4" in artifact_types_after
    assert bundle_after_render["render_status"]["state"] == "completed"
    mp4_artifact = next(item for item in bundle_after_render["artifacts"] if item["type"] == "mp4")
    assert mp4_artifact["metadata"]["valid_mp4"] is True
