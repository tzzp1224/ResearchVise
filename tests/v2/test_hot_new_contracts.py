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


class _NoopRenderer(BaseRendererAdapter):
    provider = "noop"

    def render_shot(self, *, prompt_spec: PromptSpec, output_dir: Path, mode: str, budget: dict, run_id: str, render_job_id: str) -> ShotRenderResult:
        _ = prompt_spec, output_dir, mode, budget, run_id, render_job_id
        path = output_dir / "noop.mp4"
        path.write_bytes(b"noop")
        return ShotRenderResult(shot_idx=1, success=True, output_path=str(path), cost=0.0)


def _runtime(tmp_path: Path, connector_overrides: dict) -> RunPipelineRuntime:
    orchestrator = RunOrchestrator(store=InMemoryRunStore(), queue=InMemoryRunQueue())
    render_manager = RenderManager(renderer_adapter=_NoopRenderer(), work_dir=tmp_path / "render")
    return RunPipelineRuntime(
        orchestrator=orchestrator,
        render_manager=render_manager,
        output_root=tmp_path / "runs",
        connector_overrides=connector_overrides,
    )


def _enqueue(runtime: RunPipelineRuntime, *, user_id: str = "u_hot", topic: str = "AI agent", time_window: str = "7d", top_k: int = 3) -> str:
    return runtime._orchestrator.enqueue_run(  # type: ignore[attr-defined]
        RunRequest(
            user_id=user_id,
            mode=RunMode.ONDEMAND,
            topic=topic,
            time_window=time_window,
            tz="UTC",
            budget={"top_k": top_k, "include_tier_b": False, "render_enabled": False, "min_source_coverage": 1, "min_pass_ratio": 0.0},
            output_targets=["web"],
        ),
        idempotency_key=f"{user_id}:{topic}:{time_window}:{top_k}",
    )


def test_infra_framework_is_partitioned_to_watchlist(tmp_path: Path) -> None:
    async def _github_topic_search(*args, **kwargs):
        _ = args, kwargs
        return [
            RawItem(
                id="infra_langchain_contract",
                source="github",
                title="langchain-ai/langchain",
                url="https://github.com/langchain-ai/langchain",
                body=("Agent framework sdk runtime updates and migration notes. " * 20).strip(),
                tier="A",
                metadata={"stars": 150000, "forks": 40000, "item_type": "repo", "updated_at": "2026-02-19T09:00:00Z"},
            ),
            RawItem(
                id="hot_vertical_contract",
                source="github",
                title="acme/vertical-agent-app",
                url="https://github.com/acme/vertical-agent-app",
                body=("Vertical agent app with MCP demo, tool calling workflow, and quickstart. " * 20).strip(),
                tier="A",
                metadata={"stars": 520, "forks": 55, "item_type": "repo", "created_at": "2026-02-17T09:00:00Z", "updated_at": "2026-02-19T11:00:00Z"},
            ),
        ]

    async def _none(*args, **kwargs):
        _ = args, kwargs
        return []

    runtime = _runtime(
        tmp_path,
        {
            "fetch_github_topic_search": _github_topic_search,
            "fetch_huggingface_search": _none,
            "fetch_hackernews_search": _none,
            "fetch_github_releases": _none,
            "fetch_github_trending": _none,
            "fetch_huggingface_trending": _none,
            "fetch_hackernews_top": _none,
            "fetch_rss_feed": _none,
            "fetch_web_article": _none,
        },
    )
    run_id = _enqueue(runtime, user_id="u_infra")
    result = runtime.run_next()
    assert result is not None and result.run_id == run_id
    run_dir = Path(result.output_dir)
    context = json.loads((run_dir / "run_context.json").read_text(encoding="utf-8"))
    ranking = dict(context.get("ranking_stats") or {})
    top_ids = [str(value) for value in list(ranking.get("top_item_ids") or [])]
    assert "infra_langchain_contract" not in top_ids
    watchlist_ids = {str(item.get("item_id") or "") for item in list(ranking.get("infra_watchlist") or [])}
    assert "infra_langchain_contract" in watchlist_ids


def test_single_source_github_new_repo_can_enter_top_picks(tmp_path: Path) -> None:
    async def _github_topic_search(*args, **kwargs):
        _ = args, kwargs
        return [
            RawItem(
                id="single_source_new_repo",
                source="github",
                title="acme/new-agent-tool",
                url="https://github.com/acme/new-agent-tool",
                body=(
                    "New agent tool with MCP server support, tool-calling runtime, quickstart, "
                    "and benchmark report. "
                    * 25
                ).strip(),
                tier="A",
                metadata={
                    "stars": 780,
                    "forks": 96,
                    "item_type": "repo",
                    "created_at": "2026-02-18T03:00:00Z",
                    "updated_at": "2026-02-19T15:30:00Z",
                    "search_rank": 1,
                    "search_pool_size": 15,
                },
            )
        ]

    async def _none(*args, **kwargs):
        _ = args, kwargs
        return []

    runtime = _runtime(
        tmp_path,
        {
            "fetch_github_topic_search": _github_topic_search,
            "fetch_huggingface_search": _none,
            "fetch_hackernews_search": _none,
            "fetch_github_releases": _none,
            "fetch_github_trending": _none,
            "fetch_huggingface_trending": _none,
            "fetch_hackernews_top": _none,
            "fetch_rss_feed": _none,
            "fetch_web_article": _none,
        },
    )
    _ = _enqueue(runtime, user_id="u_single_source")
    result = runtime.run_next()
    assert result is not None
    run_dir = Path(result.output_dir)
    diagnosis = json.loads((run_dir / "retrieval_diagnosis.json").read_text(encoding="utf-8"))
    context = json.loads((run_dir / "run_context.json").read_text(encoding="utf-8"))
    ranking = dict(context.get("ranking_stats") or {})
    top_ids = [str(value) for value in list(ranking.get("top_item_ids") or [])]
    assert "single_source_new_repo" in top_ids
    candidate_map = {
        str((row or {}).get("item_id") or ""): dict(row or {})
        for row in list(diagnosis.get("candidate_rows") or [])
    }
    assert bool(candidate_map.get("single_source_new_repo", {}).get("single_source_hot"))


def test_handbook_is_background_and_not_top_pick(tmp_path: Path) -> None:
    async def _github_topic_search(*args, **kwargs):
        _ = args, kwargs
        return [
            RawItem(
                id="background_handbook",
                source="github",
                title="acme/awesome-agent-handbook",
                url="https://github.com/acme/awesome-agent-handbook",
                body=("Awesome list and handbook resources roadmap learning path. " * 24).strip(),
                tier="A",
                metadata={"stars": 800, "forks": 120, "item_type": "repo", "updated_at": "2026-02-19T09:00:00Z"},
            ),
            RawItem(
                id="hot_runtime",
                source="github",
                title="acme/agent-runtime-app",
                url="https://github.com/acme/agent-runtime-app",
                body=("Agent runtime app with MCP tool-calling demo, usage guide, and quickstart. " * 24).strip(),
                tier="A",
                metadata={"stars": 430, "forks": 62, "item_type": "repo", "created_at": "2026-02-18T06:00:00Z", "updated_at": "2026-02-19T10:00:00Z"},
            ),
        ]

    async def _none(*args, **kwargs):
        _ = args, kwargs
        return []

    runtime = _runtime(
        tmp_path,
        {
            "fetch_github_topic_search": _github_topic_search,
            "fetch_huggingface_search": _none,
            "fetch_hackernews_search": _none,
            "fetch_github_releases": _none,
            "fetch_github_trending": _none,
            "fetch_huggingface_trending": _none,
            "fetch_hackernews_top": _none,
            "fetch_rss_feed": _none,
            "fetch_web_article": _none,
        },
    )
    _ = _enqueue(runtime, user_id="u_background")
    result = runtime.run_next()
    assert result is not None
    run_dir = Path(result.output_dir)
    context = json.loads((run_dir / "run_context.json").read_text(encoding="utf-8"))
    ranking = dict(context.get("ranking_stats") or {})
    top_ids = [str(value) for value in list(ranking.get("top_item_ids") or [])]
    assert "background_handbook" not in top_ids
    background_ids = {str(item.get("item_id") or "") for item in list(ranking.get("background_reading") or [])}
    assert "background_handbook" in background_ids
