"""Local/CI smoke test for v2 pipeline with mocked external services."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core import RawItem, RunMode, RunRequest
from orchestrator.queue import InMemoryRunQueue
from orchestrator.service import RunOrchestrator
from orchestrator.store import InMemoryRunStore
from pipeline_v2.runtime import RunPipelineRuntime
from render.manager import RenderManager


def _connector_overrides() -> dict:
    async def _github_trending(max_results: int = 12):
        _ = max_results
        return [
            RawItem(
                id="gh_1",
                source="github",
                title="org/repo-smoke",
                url="https://github.com/org/repo-smoke",
                body="Architecture benchmark deployment walkthrough.",
                author="org",
                tier="A",
                metadata={"stars": 999, "item_type": "repo", "has_diagram": True},
            )
        ]

    async def _github_releases(repo_full_names, max_results_per_repo: int = 1):
        _ = repo_full_names, max_results_per_repo
        return [
            RawItem(
                id="rel_1",
                source="github",
                title="org/repo-smoke v0.1.0",
                url="https://github.com/org/repo-smoke/releases/tag/v0.1.0",
                body="Release notes.",
                author="org",
                tier="A",
                metadata={"item_type": "release", "stars": 1000},
            )
        ]

    async def _hf_trending(max_results: int = 12):
        _ = max_results
        return [
            RawItem(
                id="hf_1",
                source="huggingface",
                title="org/mcp-encoder-smoke",
                url="https://huggingface.co/org/mcp-encoder-smoke",
                body="Model card highlights retrieval accuracy and deployment tips.",
                author="org",
                tier="A",
                metadata={"downloads": 8800, "likes": 140, "item_type": "model"},
            )
        ]

    async def _hn_top(max_results: int = 12):
        _ = max_results
        return [
            RawItem(
                id="hn_1",
                source="hackernews",
                title="Show HN: MCP smoke deployment notes",
                url="https://news.ycombinator.com/item?id=1000001",
                body="Community thread on rollout strategy and observability wins.",
                author="hn_user",
                tier="A",
                metadata={"points": 120, "comment_count": 48, "item_type": "story"},
            )
        ]

    async def _rss(feed_url: str, max_results: int = 6):
        _ = feed_url, max_results
        return []

    async def _web(url: str):
        _ = url
        return []

    return {
        "fetch_github_trending": _github_trending,
        "fetch_github_releases": _github_releases,
        "fetch_huggingface_trending": _hf_trending,
        "fetch_hackernews_top": _hn_top,
        "fetch_rss_feed": _rss,
        "fetch_web_article": _web,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v2 smoke flow and emit artifacts")
    parser.add_argument("--out-dir", default="data/outputs/v2_smoke")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    orchestrator = RunOrchestrator(store=InMemoryRunStore(), queue=InMemoryRunQueue())
    # Intentionally uses default renderer path (Seedance unavailable -> fallback motion render).
    render_manager = RenderManager(work_dir=out_dir / "render_jobs")
    runtime = RunPipelineRuntime(
        orchestrator=orchestrator,
        render_manager=render_manager,
        output_root=out_dir / "runs",
        connector_overrides=_connector_overrides(),
    )

    run_id = orchestrator.enqueue_run(
        RunRequest(
            user_id="smoke_user",
            mode=RunMode.ONDEMAND,
            topic="smoke test topic",
            time_window="24h",
            tz="UTC",
            budget={"duration_sec": 34, "max_total_cost": 5.0, "max_retries": 1},
            output_targets=["web", "mp4"],
        )
    )

    result = runtime.run_next()
    if not result:
        raise SystemExit("run worker did not process job")

    render_status = runtime.process_next_render()
    if not render_status:
        raise SystemExit("render worker did not process job")

    bundle = runtime.get_run_bundle(run_id)
    print(json.dumps(bundle, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
