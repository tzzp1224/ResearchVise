from __future__ import annotations

import json
from pathlib import Path

from core import RawItem, RunMode, RunRequest
from orchestrator import RunOrchestrator
from orchestrator.queue import InMemoryRunQueue
from orchestrator.store import InMemoryRunStore
from pipeline_v2.runtime import RunPipelineRuntime
from render.adapters import BaseRendererAdapter, ShotRenderResult
from render.manager import RenderManager


class _NoopRenderer(BaseRendererAdapter):
    provider = "mock-noop"

    def render_shot(self, *, prompt_spec, output_dir: Path, mode: str, budget: dict, run_id: str, render_job_id: str) -> ShotRenderResult:
        _ = prompt_spec, output_dir, mode, budget, run_id, render_job_id
        return ShotRenderResult(shot_idx=1, success=True, output_path=str(output_dir / "noop.mp4"), cost=0.0)


def _connectors() -> dict:
    low_body_github = " ".join(
        [
            f"AI powered runtime for autonomous agent using CLIP ViT image classification adapters github step {idx}."
            for idx in range(1, 42)
        ]
    )
    low_body_hf = " ".join(
        [
            f"AI powered toolkit on huggingface for autonomous agent CLIP ViT checkpoints hf step {idx}."
            for idx in range(1, 42)
        ]
    )
    low_body_hn = " ".join(
        [
            f"AI powered discussion thread about autonomous agent CLIP ViT workflows hn step {idx}."
            for idx in range(1, 42)
        ]
    )
    high_body_github = " ".join(
        [f"AI agent orchestration runtime with MCP tool calling github stage {idx}." for idx in range(1, 42)]
    )
    high_body_hf = " ".join(
        [f"AI agent eval toolkit with function calling and MCP datasets hf stage {idx}." for idx in range(1, 42)]
    )
    high_body_hn = " ".join(
        [f"AI agent production incident report on tool routing and orchestration hn stage {idx}." for idx in range(1, 42)]
    )

    async def _github_topic_search(topic: str, time_window: str = "today", limit: int = 20, expanded: bool = False, **kwargs):
        _ = topic, limit, expanded, kwargs
        if time_window == "today":
            return [
                RawItem(
                    id="gh_low",
                    source="github",
                    title="acme/agent-clip-runtime",
                    url="https://github.com/acme/agent-clip-runtime",
                    body=low_body_github,
                    tier="A",
                    metadata={"stars": 1200, "item_type": "repo", "updated_at": "2026-02-18T09:00:00Z"},
                )
            ]
        return [
            RawItem(
                id="gh_high",
                source="github",
                title="acme/agent-orchestrator",
                url="https://github.com/acme/agent-orchestrator",
                body=high_body_github,
                tier="A",
                metadata={"stars": 2200, "item_type": "repo", "updated_at": "2026-02-17T09:00:00Z"},
            )
        ]

    async def _hf_search(topic: str, time_window: str = "today", limit: int = 20, expanded: bool = False, **kwargs):
        _ = topic, limit, expanded, kwargs
        if time_window == "today":
            return [
                RawItem(
                    id="hf_low",
                    source="huggingface",
                    title="acme/agent-clip-toolkit",
                    url="https://huggingface.co/acme/agent-clip-toolkit",
                    body=low_body_hf,
                    tier="A",
                    metadata={"downloads": 4300, "likes": 210, "item_type": "model", "last_modified": "2026-02-18T07:00:00Z"},
                )
            ]
        return [
            RawItem(
                id="hf_high",
                source="huggingface",
                title="acme/agent-tool-runtime",
                url="https://huggingface.co/acme/agent-tool-runtime",
                body=high_body_hf,
                tier="A",
                metadata={"downloads": 9800, "likes": 510, "item_type": "model", "last_modified": "2026-02-17T05:00:00Z"},
            )
        ]

    async def _hn_search(topic: str, time_window: str = "today", limit: int = 20, expanded: bool = False, **kwargs):
        _ = topic, limit, expanded, kwargs
        if time_window == "today":
            return [
                RawItem(
                    id="hn_low",
                    source="hackernews",
                    title="Agent clip runtime notes",
                    url="https://news.ycombinator.com/item?id=9001",
                    body=low_body_hn,
                    tier="A",
                    metadata={"points": 155, "comment_count": 44, "item_type": "story"},
                )
            ]
        return [
            RawItem(
                id="hn_high",
                source="hackernews",
                title="MCP tool calling orchestration in production",
                url="https://news.ycombinator.com/item?id=9002",
                body=high_body_hn,
                tier="A",
                metadata={"points": 240, "comment_count": 66, "item_type": "story"},
            )
        ]

    async def _none(*args, **kwargs):
        _ = args, kwargs
        return []

    return {
        "fetch_github_topic_search": _github_topic_search,
        "fetch_huggingface_search": _hf_search,
        "fetch_hackernews_search": _hn_search,
        "fetch_github_trending": _none,
        "fetch_huggingface_trending": _none,
        "fetch_hackernews_top": _none,
        "fetch_github_releases": _none,
        "fetch_rss_feed": _none,
        "fetch_web_article": _none,
    }


def test_quality_trigger_forces_expansion_even_when_topk_is_filled(tmp_path: Path) -> None:
    orchestrator = RunOrchestrator(store=InMemoryRunStore(), queue=InMemoryRunQueue())
    render_manager = RenderManager(renderer_adapter=_NoopRenderer(), work_dir=tmp_path / "render")
    runtime = RunPipelineRuntime(
        orchestrator=orchestrator,
        render_manager=render_manager,
        output_root=tmp_path / "runs",
        connector_overrides=_connectors(),
    )

    run_id = orchestrator.enqueue_run(
        RunRequest(
            user_id="u_quality",
            mode=RunMode.ONDEMAND,
            topic="AI agent",
            time_window="today",
            tz="UTC",
            budget={"top_k": 3, "include_tier_b": False, "render_enabled": False},
            output_targets=["web"],
        ),
        idempotency_key="u_quality:ai-agent",
    )

    result = runtime.run_next()
    assert result is not None

    run_dir = Path(result.output_dir)
    diagnosis = json.loads((run_dir / "retrieval_diagnosis.json").read_text(encoding="utf-8"))
    attempts = list(diagnosis.get("attempts") or [])
    assert attempts

    base = attempts[0]
    assert int(base.get("top_picks_count", 0)) == 3
    assert float(base.get("top_picks_min_relevance", 1.0)) < 0.75
    assert bool(base.get("quality_triggered_expansion")) is True

    assert bool(diagnosis.get("quality_triggered_expansion")) is True
    assert str(diagnosis.get("selected_phase") or "") != "base"
