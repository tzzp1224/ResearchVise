from __future__ import annotations

import asyncio
import json
from pathlib import Path

from core import NormalizedItem, PromptSpec, RawItem, RunMode, RunRequest, Shot, Storyboard
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


def _relaxed_connectors() -> dict:
    github_body = ("copilot agent control-plane runtime with mcp sessions, rollback playbook, and deployment metrics. " * 28).strip()
    hf_body = ("agent workflow toolkit for dataset grounding, eval reports, and model card usage examples. " * 28).strip()
    hn_body = ("operator postmortem on agent orchestration failures, fixes, and production incident lessons. " * 28).strip()

    async def _github_trending(max_results: int = 12):
        _ = max_results
        return [
            RawItem(
                id="rel_github",
                source="github",
                title="Copilot agent control plane runtime",
                url="https://github.com/org/agent-control-plane",
                body=github_body,
                tier="A",
                metadata={"stars": 900, "item_type": "repo"},
            )
        ]

    async def _github_releases(repo_full_names, max_results_per_repo: int = 1):
        _ = repo_full_names, max_results_per_repo
        return []

    async def _hf_trending(max_results: int = 12):
        _ = max_results
        return [
            RawItem(
                id="rel_hf",
                source="huggingface",
                title="Agent workflow toolkit",
                url="https://huggingface.co/org/agent-toolkit",
                body=hf_body,
                tier="A",
                metadata={"downloads": 5000, "item_type": "model"},
            )
        ]

    async def _hn_top(max_results: int = 12):
        _ = max_results
        return [
            RawItem(
                id="rel_hn",
                source="hackernews",
                title="Agent orchestration lessons",
                url="https://news.ycombinator.com/item?id=42",
                body=hn_body,
                tier="A",
                metadata={"points": 180, "comment_count": 44, "item_type": "story"},
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


def _expansion_connectors() -> dict:
    low_body = ("frontend dashboard widgets and css theme updates for infra console. " * 24).strip()
    agent_body_github = (
        "ai agent orchestration workflow with tool calling and mcp session planning. "
        "quickstart includes cli usage and benchmark results. "
        "docs https://docs.example/agent runtime notes https://news.ycombinator.com/item?id=77 "
    ) * 18
    agent_body_hf = (
        "agent training toolkit with eval dashboard and reproducible workflow examples. "
        "usage card has install command and inference benchmark table. "
        "card https://huggingface.co/acme/agent-toolkit discussion https://news.ycombinator.com/item?id=88 "
    ) * 18
    agent_body_hn = (
        "production incident review for autonomous agent tool routing and rollback strategy. "
        "thread includes score deltas and failure pattern comparison. "
        "context https://news.ycombinator.com/item?id=77 docs https://acme.dev/agent-case-study "
    ) * 18

    async def _github_topic_search(topic: str, time_window: str = "today", limit: int = 20, expanded: bool = False):
        _ = topic, limit, expanded
        if time_window == "today":
            return [
                RawItem(
                    id="exp_low_gh",
                    source="github",
                    title="infra dashboard theme pack",
                    url="https://github.com/acme/dashboard-theme",
                    body=low_body,
                    tier="A",
                    metadata={"stars": 320, "item_type": "repo"},
                )
            ]
        return [
            RawItem(
                    id="exp_gh",
                    source="github",
                    title="agent orchestrator runtime",
                    url="https://github.com/acme/agent-orchestrator",
                    body=agent_body_github,
                    tier="A",
                    metadata={"stars": 1800, "item_type": "repo", "updated_at": "2026-02-17T10:00:00Z"},
                )
            ]

    async def _hf_search(topic: str, time_window: str = "today", limit: int = 20, expanded: bool = False):
        _ = topic, limit, expanded
        if time_window == "today":
            return []
        return [
            RawItem(
                id="exp_hf",
                source="huggingface",
                title="agent workflow toolkit",
                url="https://huggingface.co/acme/agent-toolkit",
                body=agent_body_hf,
                tier="A",
                metadata={"downloads": 4200, "likes": 190, "item_type": "model", "last_modified": "2026-02-16T08:00:00Z"},
            )
        ]

    async def _hn_search(topic: str, time_window: str = "today", limit: int = 20, expanded: bool = False):
        _ = topic, limit, expanded
        if time_window == "today":
            return []
        return [
            RawItem(
                id="exp_hn",
                source="hackernews",
                title="Agent tool-calling lessons from production",
                url="https://news.ycombinator.com/item?id=77",
                body=agent_body_hn,
                tier="A",
                metadata={"points": 210, "comment_count": 58, "item_type": "story"},
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


def _single_source_only_connectors() -> dict:
    github_body = (
        "AI agent orchestration runtime with MCP tool calling. "
        "Quickstart: pip install acme-agent && acme-agent run. "
        "Includes deterministic evidence audit and deployment workflow checkpoints. "
    ) * 20

    async def _github_topic_search(topic: str, time_window: str = "today", limit: int = 20, expanded: bool = False, **kwargs):
        _ = topic, time_window, limit, expanded, kwargs
        return [
            RawItem(
                id="ss_gh_1",
                source="github",
                title="acme/agent-runtime",
                url="https://github.com/acme/agent-runtime",
                body=github_body,
                tier="A",
                metadata={
                    "stars": 1800,
                    "forks": 200,
                    "item_type": "repo",
                    "updated_at": "2026-02-18T09:00:00Z",
                    "citations": [
                        {
                            "title": "Runtime docs",
                            "url": "https://docs.acme.dev/agent-runtime",
                            "snippet": "Docs include runtime topology and MCP tool-routing examples.",
                            "source": "docs",
                        },
                        {
                            "title": "Benchmark report",
                            "url": "https://acme.dev/agent-runtime-benchmark",
                            "snippet": "Benchmark report compares orchestration latency and rollback outcomes.",
                            "source": "web",
                        },
                    ],
                },
            ),
            RawItem(
                id="ss_gh_2",
                source="github",
                title="acme/agent-orchestration-kit",
                url="https://github.com/acme/agent-orchestration-kit",
                body=github_body,
                tier="A",
                metadata={
                    "stars": 1200,
                    "forks": 120,
                    "item_type": "repo",
                    "updated_at": "2026-02-17T09:00:00Z",
                    "citations": [
                        {
                            "title": "Ops docs",
                            "url": "https://docs.acme.dev/agent-ops",
                            "snippet": "Ops docs show deployment stages and workflow orchestration controls.",
                            "source": "docs",
                        },
                        {
                            "title": "Postmortem",
                            "url": "https://acme.dev/agent-runtime-postmortem",
                            "snippet": "Postmortem summarizes failure handling and tool-calling recovery paths.",
                            "source": "web",
                        },
                    ],
                },
            ),
        ]

    async def _none(*args, **kwargs):
        _ = args, kwargs
        return []

    return {
        "fetch_github_topic_search": _github_topic_search,
        "fetch_huggingface_search": _none,
        "fetch_hackernews_search": _none,
        "fetch_github_trending": _none,
        "fetch_huggingface_trending": _none,
        "fetch_hackernews_top": _none,
        "fetch_github_releases": _none,
        "fetch_rss_feed": _none,
        "fetch_web_article": _none,
    }


def _all_reject_connectors() -> dict:
    body = ("Agent orchestration runtime with tool calling workflow and MCP session router. " * 24).strip()

    async def _hn_search(topic: str, time_window: str = "today", limit: int = 20, expanded: bool = False, **kwargs):
        _ = topic, time_window, limit, expanded, kwargs
        return [
            RawItem(
                id="reject_hn_1",
                source="hackernews",
                title="Agent runtime experiment thread",
                url="https://news.ycombinator.com/item?id=999331",
                body=body,
                tier="A",
                metadata={"points": 1, "comment_count": 0, "item_type": "story"},
            )
        ]

    async def _none(*args, **kwargs):
        _ = args, kwargs
        return []

    return {
        "fetch_github_topic_search": _none,
        "fetch_huggingface_search": _none,
        "fetch_hackernews_search": _hn_search,
        "fetch_github_trending": _none,
        "fetch_huggingface_trending": _none,
        "fetch_hackernews_top": _none,
        "fetch_github_releases": _none,
        "fetch_rss_feed": _none,
        "fetch_web_article": _none,
    }


def _best_attempt_regression_connectors() -> dict:
    strong_body = (
        "AI agent orchestration runtime with MCP tool calling and workflow checkpoints.\n"
        "Quickstart: pip install acme-agent && acme-agent run demo.\n"
        "Docs: https://docs.acme.dev/agent-runtime\n"
        "Release: https://github.com/acme/agent-runtime/releases/tag/v1.0.0\n"
    ) * 18

    async def _github_topic_search(topic: str, time_window: str = "today", limit: int = 20, expanded: bool = False, **kwargs):
        _ = topic, limit, kwargs
        if time_window == "today" and not expanded:
            return [
                RawItem(
                    id="best_gh_pass",
                    source="github",
                    title="acme/agent-runtime",
                    url="https://github.com/acme/agent-runtime",
                    body=strong_body,
                    tier="A",
                    metadata={
                        "stars": 2100,
                        "forks": 280,
                        "item_type": "repo",
                        "updated_at": "2026-02-18T09:00:00Z",
                    },
                )
            ]
        return []

    async def _hn_search(topic: str, time_window: str = "today", limit: int = 20, expanded: bool = False, **kwargs):
        _ = topic, time_window, limit, expanded, kwargs
        return [
            RawItem(
                id=f"best_hn_noise_{time_window}_{'expanded' if expanded else 'base'}",
                source="hackernews",
                title="Frontend dashboard css theme changelog",
                url="https://news.ycombinator.com/item?id=700001",
                body=("frontend css theme tweaks and icon set updates. " * 24).strip(),
                tier="A",
                metadata={"points": 2, "comment_count": 0, "item_type": "story"},
            )
        ]

    async def _none(*args, **kwargs):
        _ = args, kwargs
        return []

    return {
        "fetch_github_topic_search": _github_topic_search,
        "fetch_huggingface_search": _none,
        "fetch_hackernews_search": _hn_search,
        "fetch_github_trending": _none,
        "fetch_huggingface_trending": _none,
        "fetch_hackernews_top": _none,
        "fetch_github_releases": _none,
        "fetch_rss_feed": _none,
        "fetch_web_article": _none,
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
    assert "retrieval" in run_context_payload
    assert Path(str(run_context_payload["retrieval"].get("diagnosis_path") or "")).exists()
    assert Path(str(run_context_payload["retrieval"].get("evidence_audit_path") or "")).exists()
    assert (run_dir / "retrieval_diagnosis.json").exists()
    assert (run_dir / "evidence_audit.json").exists()

    materials_payload = json.loads((run_dir / "materials.json").read_text(encoding="utf-8"))
    assert materials_payload["screenshot_plan"]
    assert materials_payload["local_assets"]
    assert all(Path(path).exists() for path in materials_payload["local_assets"])
    assert materials_payload["icon_keyword_suggestions"]
    assert materials_payload["broll_categories"]
    assert "quality_metrics" in materials_payload
    assert materials_payload["data_mode"] == "live"
    board_payload = json.loads((run_dir / "storyboard.json").read_text(encoding="utf-8"))
    local_ref_shots = 0
    for shot in list(board_payload.get("shots") or []):
        refs = [str(value) for value in list((shot or {}).get("reference_assets") or []) if str(value).strip()]
        if any(ref.startswith(str(run_dir / "assets")) for ref in refs):
            local_ref_shots += 1
    assert local_ref_shots >= 2

    onepager_text = (run_dir / "onepager.md").read_text(encoding="utf-8")
    assert "DataMode: `live`" in onepager_text
    assert "CandidateCount: `" in onepager_text
    assert "FilteredByRelevance: `" in onepager_text
    assert "EvidenceAuditPath" in onepager_text
    assert "HardMatchPassCount" in onepager_text
    assert "TopPicksMinRelevance" in onepager_text
    assert "QualityTriggeredExpansion" in onepager_text

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


def test_runtime_adaptive_relevance_relaxation_and_diversity(tmp_path: Path) -> None:
    orchestrator = RunOrchestrator(store=InMemoryRunStore(), queue=InMemoryRunQueue())
    render_manager = RenderManager(renderer_adapter=AlwaysSuccessAdapter(), work_dir=tmp_path / "render")
    runtime = RunPipelineRuntime(
        orchestrator=orchestrator,
        render_manager=render_manager,
        output_root=tmp_path / "runs",
        connector_overrides=_relaxed_connectors(),
    )

    run_id = orchestrator.enqueue_run(
        RunRequest(
            user_id="u_relaxed",
            mode=RunMode.ONDEMAND,
            topic="copilot agent",
            time_window="24h",
            tz="UTC",
            budget={"top_k": 3},
            output_targets=["web"],
        ),
        idempotency_key="u_relaxed:copilot-agent",
    )
    result = runtime.run_next()
    assert result is not None
    run_dir = Path(result.output_dir)
    run_context = json.loads((run_dir / "run_context.json").read_text(encoding="utf-8"))
    ranking = dict(run_context.get("ranking_stats") or {})

    assert int(ranking.get("top_picks_count", 0)) >= 1
    assert float(ranking.get("topic_relevance_threshold_used", 0.55)) <= 0.55
    assert int(ranking.get("relaxation_steps", 0)) >= 1
    assert int(ranking.get("requested_top_k", 0)) == 3
    if int(ranking.get("top_picks_count", 0)) < int(ranking.get("requested_top_k", 0)):
        why_not_more = [str(value) for value in list(ranking.get("why_not_more") or [])]
        assert any("top_picks_lt_3" in reason for reason in why_not_more)

    onepager = (run_dir / "onepager.md").read_text(encoding="utf-8")
    assert "TopicRelevanceThresholdUsed" in onepager
    assert "RelevanceRelaxationSteps" in onepager


def test_runtime_recall_expansion_selects_window_3d_and_records_diagnosis(tmp_path: Path) -> None:
    orchestrator = RunOrchestrator(store=InMemoryRunStore(), queue=InMemoryRunQueue())
    render_manager = RenderManager(renderer_adapter=AlwaysSuccessAdapter(), work_dir=tmp_path / "render")
    runtime = RunPipelineRuntime(
        orchestrator=orchestrator,
        render_manager=render_manager,
        output_root=tmp_path / "runs",
        connector_overrides=_expansion_connectors(),
    )

    run_id = orchestrator.enqueue_run(
        RunRequest(
            user_id="u_expand",
            mode=RunMode.ONDEMAND,
            topic="AI agent",
            time_window="today",
            tz="UTC",
            budget={"top_k": 3, "include_tier_b": False},
            output_targets=["web"],
        ),
        idempotency_key="u_expand:ai-agent",
    )
    result = runtime.run_next()
    assert result is not None
    run_dir = Path(result.output_dir)
    run_context = json.loads((run_dir / "run_context.json").read_text(encoding="utf-8"))
    ranking = dict(run_context.get("ranking_stats") or {})
    retrieval = dict(run_context.get("retrieval") or {})
    diagnosis = json.loads((run_dir / "retrieval_diagnosis.json").read_text(encoding="utf-8"))
    evidence_audit = json.loads((run_dir / "evidence_audit.json").read_text(encoding="utf-8"))

    assert int(ranking.get("requested_top_k", 0)) == 3
    assert int(ranking.get("top_picks_count", 0)) >= 2
    assert str(ranking.get("selected_recall_phase") or "") in {"window_3d", "window_7d", "query_expanded"}
    assert int(ranking.get("recall_attempt_count", 0)) >= 3
    assert len(list(retrieval.get("expansion_steps") or [])) >= 1
    assert int(retrieval.get("attempt_count", 0) or 0) >= 3
    assert str(diagnosis.get("selected_phase") or "") == str(retrieval.get("selected_phase") or "")
    assert len(list(diagnosis.get("attempts") or [])) >= 3
    assert any(bool(item.get("expansion_applied")) for item in list(diagnosis.get("attempts") or []))
    assert isinstance(diagnosis.get("plan"), dict)
    assert "quality_triggered_expansion" in diagnosis
    assert "hard_match_terms_used" in diagnosis
    assert "top_picks_min_relevance" in diagnosis
    assert "top_picks_hard_match_count" in diagnosis
    attempts = list(diagnosis.get("attempts") or [])
    required_attempt_fields = {
        "hard_match_terms_used",
        "hard_match_pass_count",
        "top_picks_min_relevance",
        "top_picks_hard_match_count",
        "quality_triggered_expansion",
    }
    assert all(required_attempt_fields.issubset(set(dict(item).keys())) for item in attempts)
    assert any(bool(item.get("quality_triggered_expansion")) for item in attempts)
    base_attempt = next((item for item in attempts if str(item.get("phase")) == "base"), {})
    expanded_attempt = next((item for item in attempts if str(item.get("phase")) == "query_expanded"), {})
    base_queries = {str(q).strip().lower() for q in list(base_attempt.get("queries") or []) if str(q).strip()}
    expanded_queries = {str(q).strip().lower() for q in list(expanded_attempt.get("queries") or []) if str(q).strip()}
    if expanded_queries:
        assert expanded_queries != base_queries
    assert list(evidence_audit.get("records") or [])


def test_runtime_returns_shortage_with_reason_when_source_coverage_cannot_reach_two(tmp_path: Path) -> None:
    orchestrator = RunOrchestrator(store=InMemoryRunStore(), queue=InMemoryRunQueue())
    render_manager = RenderManager(renderer_adapter=AlwaysSuccessAdapter(), work_dir=tmp_path / "render")
    runtime = RunPipelineRuntime(
        orchestrator=orchestrator,
        render_manager=render_manager,
        output_root=tmp_path / "runs",
        connector_overrides=_single_source_only_connectors(),
    )

    run_id = orchestrator.enqueue_run(
        RunRequest(
            user_id="u_single_source",
            mode=RunMode.ONDEMAND,
            topic="AI agent",
            time_window="today",
            tz="UTC",
            budget={"top_k": 3, "include_tier_b": False, "render_enabled": False},
            output_targets=["web"],
        ),
        idempotency_key="u_single_source:ai-agent",
    )
    result = runtime.run_next()
    assert result is not None
    run_dir = Path(result.output_dir)

    run_context = json.loads((run_dir / "run_context.json").read_text(encoding="utf-8"))
    ranking = dict(run_context.get("ranking_stats") or {})
    diagnosis = json.loads((run_dir / "retrieval_diagnosis.json").read_text(encoding="utf-8"))
    onepager = (run_dir / "onepager.md").read_text(encoding="utf-8")

    assert int(ranking.get("top_picks_count", 0)) < 3
    assert int(ranking.get("selected_source_coverage", 0) or 0) < 2
    assert int(ranking.get("recall_attempt_count", 0) or 0) >= 2
    reasons = [str(value) for value in list(ranking.get("why_not_more") or [])]
    assert any("source_diversity_lt_2" in reason for reason in reasons)
    assert "## Why not more?" in onepager
    assert len(list(diagnosis.get("attempts") or [])) >= 2
    assert any("source_coverage_lt_2" in str(reason) for item in list(diagnosis.get("attempts") or []) for reason in list(item.get("quality_trigger_reasons") or []))


def test_hf_metadata_only_item_triggers_deep_fetch_and_updates_body(tmp_path: Path) -> None:
    orchestrator = RunOrchestrator(store=InMemoryRunStore(), queue=InMemoryRunQueue())
    render_manager = RenderManager(renderer_adapter=AlwaysSuccessAdapter(), work_dir=tmp_path / "render")
    runtime = RunPipelineRuntime(
        orchestrator=orchestrator,
        render_manager=render_manager,
        output_root=tmp_path / "runs",
        connector_overrides={},
    )

    async def _fake_hf_fetch(_item):
        return (
            "## Overview\nAgent runtime supports tool calling and orchestration workflows.\nQuickstart: pip install acme-agent",
            "hf_raw_readme",
            "",
        )

    runtime._fetch_huggingface_deep_body = _fake_hf_fetch  # type: ignore[attr-defined]

    raw = RawItem(
        id="hf_meta_only",
        source="huggingface",
        title="acme/agent-runtime",
        url="https://huggingface.co/acme/agent-runtime",
        body="",
        tier="A",
        metadata={"repo_id": "acme/agent-runtime", "item_type": "model", "extraction_method": "hf_metadata"},
    )

    refreshed, applied, deep_count, details = asyncio.run(runtime._apply_deep_extraction([raw], max_items=1))
    assert applied is True
    assert deep_count == 1
    assert len(refreshed) == 1
    assert len(str(refreshed[0].body or "").strip()) > 0
    assert bool((refreshed[0].metadata or {}).get("deep_fetch_applied")) is True
    assert details and bool(details[0].get("accepted")) is True


def test_runtime_cross_source_corroboration_annotation_marks_shared_entity(tmp_path: Path) -> None:
    orchestrator = RunOrchestrator(store=InMemoryRunStore(), queue=InMemoryRunQueue())
    render_manager = RenderManager(renderer_adapter=AlwaysSuccessAdapter(), work_dir=tmp_path / "render")
    runtime = RunPipelineRuntime(
        orchestrator=orchestrator,
        render_manager=render_manager,
        output_root=tmp_path / "runs",
        connector_overrides={},
    )

    github_item = NormalizedItem(
        id="gh_item",
        source="github",
        title="acme/agent-runtime",
        url="https://github.com/acme/agent-runtime",
        body_md="Agent runtime orchestration quickstart.",
        tier="A",
        hash="hash-gh",
        metadata={},
    )
    hn_item = NormalizedItem(
        id="hn_item",
        source="hackernews",
        title="HN discussion: acme/agent-runtime",
        url="https://github.com/acme/agent-runtime",
        body_md="Discussion on production rollout.",
        tier="A",
        hash="hash-hn",
        metadata={"hn_url": "https://news.ycombinator.com/item?id=123"},
    )

    annotated = runtime._annotate_cross_source_corroboration([github_item, hn_item])  # type: ignore[attr-defined]
    assert bool((annotated[0].metadata or {}).get("cross_source_corroborated")) is True
    assert bool((annotated[1].metadata or {}).get("cross_source_corroborated")) is True
    assert int((annotated[0].metadata or {}).get("cross_source_corroboration_count", 0) or 0) >= 2


def test_runtime_shortage_rescue_prevents_crash_when_all_audit_verdicts_reject(tmp_path: Path) -> None:
    orchestrator = RunOrchestrator(store=InMemoryRunStore(), queue=InMemoryRunQueue())
    render_manager = RenderManager(renderer_adapter=AlwaysSuccessAdapter(), work_dir=tmp_path / "render")
    runtime = RunPipelineRuntime(
        orchestrator=orchestrator,
        render_manager=render_manager,
        output_root=tmp_path / "runs",
        connector_overrides=_all_reject_connectors(),
    )

    run_id = orchestrator.enqueue_run(
        RunRequest(
            user_id="u_reject_rescue",
            mode=RunMode.ONDEMAND,
            topic="AI agent",
            time_window="today",
            tz="UTC",
            budget={"top_k": 3, "include_tier_b": False, "render_enabled": False},
            output_targets=["web"],
        ),
        idempotency_key="u_reject_rescue:ai-agent",
    )
    result = runtime.run_next()
    assert result is not None
    run_dir = Path(result.output_dir)
    run_context = json.loads((run_dir / "run_context.json").read_text(encoding="utf-8"))
    ranking = dict(run_context.get("ranking_stats") or {})
    assert int(ranking.get("top_picks_count", 0) or 0) >= 1
    assert int(ranking.get("selected_downgrade_count", 0) or 0) >= 1
    assert any(
        "shortage_fallback_after_max_phase" in str(reason)
        for reason in list(ranking.get("quality_trigger_reasons") or []) + list(ranking.get("why_not_more") or [])
    )


def test_runtime_prefers_best_attempt_when_late_phase_degrades(tmp_path: Path) -> None:
    orchestrator = RunOrchestrator(store=InMemoryRunStore(), queue=InMemoryRunQueue())
    render_manager = RenderManager(renderer_adapter=AlwaysSuccessAdapter(), work_dir=tmp_path / "render")
    runtime = RunPipelineRuntime(
        orchestrator=orchestrator,
        render_manager=render_manager,
        output_root=tmp_path / "runs",
        connector_overrides=_best_attempt_regression_connectors(),
    )

    run_id = orchestrator.enqueue_run(
        RunRequest(
            user_id="u_best_attempt",
            mode=RunMode.ONDEMAND,
            topic="AI agent",
            time_window="today",
            tz="UTC",
            budget={"top_k": 3, "include_tier_b": False, "render_enabled": False},
            output_targets=["web"],
        ),
        idempotency_key="u_best_attempt:ai-agent",
    )
    result = runtime.run_next()
    assert result is not None
    run_dir = Path(result.output_dir)
    run_context = json.loads((run_dir / "run_context.json").read_text(encoding="utf-8"))
    ranking = dict(run_context.get("ranking_stats") or {})
    diagnosis = json.loads((run_dir / "retrieval_diagnosis.json").read_text(encoding="utf-8"))

    assert int(ranking.get("selected_pass_count", 0) or 0) >= 1
    assert int(ranking.get("top_picks_count", 0) or 0) >= 1
    assert any("selected_from_best_attempt" in str(reason) for reason in list(ranking.get("quality_trigger_reasons") or []))
    assert int(diagnosis.get("selected_attempt", 0) or 0) < len(list(diagnosis.get("attempts") or []))


def test_attach_reference_assets_keeps_shot_asset_semantic_alignment() -> None:
    board = Storyboard(
        run_id="run_test",
        item_id="item_test",
        duration_sec=30,
        aspect="9:16",
        shots=[
            Shot(idx=1, duration=3.0, camera="wide", scene="technical studio", action="agent update", reference_assets=["https://x.com/acme/status/1"]),
            Shot(idx=2, duration=5.0, camera="medium", scene="technical studio", action="repo evidence", reference_assets=["https://github.com/acme/agent-runtime"]),
            Shot(idx=3, duration=5.0, camera="medium", scene="technical studio", action="hf evidence", reference_assets=["https://huggingface.co/acme/agent-runtime"]),
        ],
    )
    materials = {
        "local_assets": ["/tmp/asset_primary.svg", "/tmp/asset_secondary.svg"],
        "reference_asset_map": {
            "https://github.com/acme/agent-runtime": "/tmp/asset_primary.svg",
            "https://huggingface.co/acme/agent-runtime": "/tmp/asset_secondary.svg",
        },
    }
    updated = RunPipelineRuntime._attach_reference_assets(board=board, materials=materials)
    assert updated.shots[0].reference_assets[0] == "/tmp/asset_primary.svg"
    assert updated.shots[1].reference_assets[0] == "/tmp/asset_primary.svg"
    assert updated.shots[2].reference_assets[0] == "/tmp/asset_secondary.svg"
