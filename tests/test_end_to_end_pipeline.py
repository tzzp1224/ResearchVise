"""
Integration tests for keyword-driven end-to-end research pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from intelligence.llm.base import BaseLLM, LLMResponse, Message
from intelligence.pipeline import run_research_from_search_results
from outputs.video_generator import BaseVideoGenerator


class FakeLLM(BaseLLM):
    """Deterministic LLM for offline integration tests."""

    def __init__(self):
        super().__init__(model="fake-llm")

    @property
    def provider(self) -> str:
        return "fake"

    async def acomplete(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        prompt = "\n".join([m.content for m in messages])

        if "提取 8-20 个关键事实" in prompt or "请分析以下关于" in prompt:
            content = {
                "facts": [
                    {
                        "claim": "Model uses hybrid sparse + dense attention with block-local routing.",
                        "evidence": ["arxiv_a1", "github_repo_1"],
                        "confidence": 0.91,
                        "source_type": "paper",
                        "category": "architecture",
                    },
                    {
                        "claim": "On long-context benchmark, model improves exact-match by 7.8 points over baseline.",
                        "evidence": ["arxiv_a1", "hackernews_1"],
                        "confidence": 0.88,
                        "source_type": "paper",
                        "category": "performance",
                    },
                    {
                        "claim": "Training recipe uses mixed curriculum with 30% synthetic reasoning traces.",
                        "evidence": ["arxiv_a2"],
                        "confidence": 0.84,
                        "source_type": "paper",
                        "category": "training",
                    },
                    {
                        "claim": "Compared with dense Transformer at equal FLOPs, latency reduces by 23%.",
                        "evidence": ["github_repo_1", "stackoverflow_q1"],
                        "confidence": 0.81,
                        "source_type": "code",
                        "category": "comparison",
                    },
                    {
                        "claim": "Failure mode appears on compositional arithmetic beyond 20-step chains.",
                        "evidence": ["reddit_1", "arxiv_a2"],
                        "confidence": 0.77,
                        "source_type": "paper",
                        "category": "limitation",
                    },
                    {
                        "claim": "Deployment requires KV-cache sharding to keep p95 latency under 300ms.",
                        "evidence": ["github_repo_1"],
                        "confidence": 0.79,
                        "source_type": "code",
                        "category": "deployment",
                    },
                    {
                        "claim": "Community reports memory pressure when context exceeds 128K tokens.",
                        "evidence": ["reddit_1", "hackernews_1"],
                        "confidence": 0.72,
                        "source_type": "social",
                        "category": "community",
                    },
                    {
                        "claim": "Ablation shows sparse router temperature is key to stability.",
                        "evidence": ["arxiv_a2"],
                        "confidence": 0.83,
                        "source_type": "paper",
                        "category": "training",
                    },
                ],
                "knowledge_gaps": ["Need public cost breakdown across GPU classes."],
            }
            return LLMResponse(content=json.dumps(content), model=self.model, usage={})

        if "发展时间轴" in prompt:
            content = {
                "events": [
                    {
                        "date": "2023-09",
                        "title": "Sparse routing prototype",
                        "description": "First public implementation of sparse router for long-context inference.",
                        "importance": 4,
                        "source_refs": ["fact_1"],
                    },
                    {
                        "date": "2024-03",
                        "title": "Hybrid attention release",
                        "description": "Hybrid sparse+dense attention demonstrates stable training at larger context.",
                        "importance": 5,
                        "source_refs": ["fact_2", "fact_3"],
                    },
                    {
                        "date": "2024-11",
                        "title": "Production deployment optimization",
                        "description": "KV-sharding and quantization reduce serving latency and memory overhead.",
                        "importance": 5,
                        "source_refs": ["fact_6"],
                    },
                ]
            }
            return LLMResponse(content=json.dumps(content), model=self.model, usage={})

        if "技术一页纸摘要" in prompt or "一页纸摘要" in prompt:
            content = {
                "title": "Long-Context Hybrid Attention One-Pager",
                "executive_summary": "Hybrid sparse+dense attention improves long-context quality while reducing serving latency.",
                "key_findings": [
                    "Sparse router + local dense attention keeps quality at long context windows.",
                    "Latency improves by 23% at equal FLOPs versus dense baseline.",
                    "Synthetic reasoning traces improve step-wise reasoning robustness.",
                    "KV-cache sharding is required to meet p95 latency SLA.",
                    "Compositional arithmetic remains a known failure region.",
                ],
                "metrics": {
                    "Long-context EM gain": "+7.8",
                    "Latency reduction": "23%",
                    "Target p95 latency": "<300ms",
                    "Context size": "128K+",
                },
                "strengths": ["Efficient long-context inference", "Good scaling with routing"],
                "weaknesses": ["Router temperature sensitivity", "Arithmetic chain failures"],
                "technical_deep_dive": [
                    "Router entropy regularization prevents expert collapse in late-stage training.",
                    "Layer-wise mixed attention schedule avoids instability in early warmup.",
                ],
                "implementation_notes": [
                    "Use paged KV-cache with per-shard admission control.",
                    "Monitor expert load-balance and router entropy as first-class metrics.",
                ],
                "risks_and_mitigations": [
                    "Risk: expert collapse -> Mitigation: entropy penalty + temperature floor.",
                    "Risk: OOM under long context -> Mitigation: KV offload + chunked prefilling.",
                ],
                "resources": [
                    {"title": "Technical report", "url": "https://example.com/report"},
                    {"title": "Reference implementation", "url": "https://example.com/repo"},
                ],
            }
            return LLMResponse(content=json.dumps(content), model=self.model, usage={})

        if "视频简报脚本" in prompt:
            content = {
                "title": "Hybrid Attention Deep Dive",
                "duration_estimate": "4-6 minutes",
                "hook": "How do you scale context length without destroying latency?",
                "target_audience": "ML infra engineers",
                "visual_style": "cinematic technical explainer with diagrams and benchmark overlays",
                "segments": [
                    {
                        "title": "Architecture",
                        "content": "Explain hybrid sparse+dense attention and router dynamics.",
                        "talking_points": ["router entropy", "expert balance", "attention locality"],
                        "duration_sec": 60,
                        "visual_prompt": "close-up GPU datacenter, animated sparse attention graph, technical overlay",
                    },
                    {
                        "title": "Benchmarks",
                        "content": "Show long-context quality and latency trade-off curves.",
                        "talking_points": ["+7.8 EM", "-23% latency", "p95 under 300ms"],
                        "duration_sec": 75,
                        "visual_prompt": "benchmark charts, animated line plots, modern technical dashboard",
                    },
                    {
                        "title": "Deployment Playbook",
                        "content": "Cover KV-sharding and memory control strategies in production.",
                        "talking_points": ["paged KV cache", "offload policy", "SLA instrumentation"],
                        "duration_sec": 70,
                        "visual_prompt": "production control room, latency heatmaps, infra diagrams",
                    },
                ],
                "conclusion": "Hybrid attention is practical when routing stability and memory plans are engineered upfront.",
                "call_to_action": "Validate with your own workload traces before full rollout.",
            }
            return LLMResponse(content=json.dumps(content), model=self.model, usage={})

        return LLMResponse(content="{}", model=self.model, usage={})

    async def astream(self, messages: List[Message], **kwargs: Any):
        yield ""


class SparseOutputLLM(FakeLLM):
    """Returns intentionally incomplete content to test pipeline normalization."""

    async def acomplete(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        prompt = "\n".join([m.content for m in messages])

        if "技术一页纸摘要" in prompt or "一页纸摘要" in prompt:
            content = {
                "title": "Sparse One-Pager",
                "executive_summary": "Summary",
                "key_findings": ["A", "B", "C", "D", "E"],
                "metrics": {"m1": "v1", "m2": "v2", "m3": "v3"},
                "technical_deep_dive": ["d1", "d2"],
                "implementation_notes": ["i1", "i2"],
                "risks_and_mitigations": ["r1", "r2"],
                "resources": [
                    {"title": "placeholder", "url": "（请自行搜索）"},
                    {"title": "empty", "url": ""},
                    {"title": "fake", "url": "https://github.com/your-repo/demo"},
                ],
            }
            return LLMResponse(content=json.dumps(content), model=self.model, usage={})

        if "视频简报脚本" in prompt:
            content = {
                "title": "Sparse Video Brief",
                "duration_estimate": "3-5 minutes",
                "hook": "hook",
                "segments": [
                    {"title": "S1", "content": "C1", "talking_points": ["p1"]},
                    {"title": "S2", "content": "C2", "talking_points": ["p2"], "duration_sec": 50},
                ],
                "conclusion": "conclusion",
                "call_to_action": "cta",
            }
            return LLMResponse(content=json.dumps(content), model=self.model, usage={})

        return await super().acomplete(messages, tools=tools, **kwargs)


def _build_search_results() -> List[Dict[str, Any]]:
    return [
        {
            "id": "arxiv_a1",
            "source": "arxiv",
            "title": "Hybrid Attention for Long Context",
            "content": "Reports +7.8 EM and 23% lower latency with sparse+dense routing.",
            "url": "https://example.com/arxiv1",
            "metadata": {"year": 2024, "benchmarks": ["LongBench"]},
        },
        {
            "id": "arxiv_a2",
            "source": "arxiv",
            "title": "Router Stability in Sparse Models",
            "content": "Ablation on entropy regularization and temperature scheduling.",
            "url": "https://example.com/arxiv2",
            "metadata": {"year": 2024},
        },
        {
            "id": "github_repo_1",
            "source": "github",
            "title": "org/hybrid-attention",
            "content": "Production inference stack with paged KV-cache sharding.",
            "url": "https://example.com/repo",
            "metadata": {"stars": 1234, "language": "python"},
        },
        {
            "id": "reddit_1",
            "source": "reddit",
            "title": "Long context memory issue discussion",
            "content": "Users report OOM above 128K unless KV offload is enabled.",
            "url": "https://example.com/reddit",
            "metadata": {"score": 321},
        },
        {
            "id": "hackernews_1",
            "source": "hackernews",
            "title": "Latency trade-offs thread",
            "content": "Deployment post reports p95 SLA wins with cache sharding.",
            "url": "https://example.com/hn",
            "metadata": {"points": 201},
        },
    ]


def _build_manus_search_results() -> List[Dict[str, Any]]:
    return [
        {
            "id": "arxiv_a1",
            "source": "arxiv",
            "title": "Manus-style Agentic Planning with Tool Execution",
            "content": "Describes a planner-executor architecture with explicit verification loops and cost controls.",
            "url": "https://example.com/manus-arxiv-1",
            "metadata": {"year": 2025, "benchmarks": ["GAIA", "ToolBench"]},
        },
        {
            "id": "arxiv_a2",
            "source": "arxiv",
            "title": "Failure Modes in Autonomous Web Agents",
            "content": "Reports prompt injection, stale context, and tool misuse as dominant failure sources.",
            "url": "https://example.com/manus-arxiv-2",
            "metadata": {"year": 2025},
        },
        {
            "id": "github_repo_1",
            "source": "github",
            "title": "open-manus/agent-runtime",
            "content": "Open-source runtime for multi-tool orchestration with retries and state snapshots.",
            "url": "https://example.com/manus-github",
            "metadata": {"stars": 4821, "language": "python"},
        },
        {
            "id": "stackoverflow_q1",
            "source": "stackoverflow",
            "title": "How to avoid context drift in long-running autonomous agents?",
            "content": "Practitioners discuss memory compaction, checkpointing, and deterministic tool wrappers.",
            "url": "https://example.com/manus-so",
            "metadata": {"score": 142, "answer_count": 9},
        },
        {
            "id": "reddit_1",
            "source": "reddit",
            "title": "Manus workflow stability discussion",
            "content": "Users share breakpoints in browser automation and compare fallback recovery policies.",
            "url": "https://example.com/manus-reddit",
            "metadata": {"score": 420},
        },
        {
            "id": "hackernews_1",
            "source": "hackernews",
            "title": "Show HN: Agentic research assistant inspired by Manus",
            "content": "Thread compares latency, quality and operational cost versus manual workflows.",
            "url": "https://example.com/manus-hn",
            "metadata": {"points": 336},
        },
    ]


def test_run_research_from_search_results_e2e(tmp_path: Path):
    llm = FakeLLM()
    search_results = _build_search_results()

    import asyncio

    result = asyncio.run(
        run_research_from_search_results(
            topic="Hybrid Attention",
            search_results=search_results,
            llm=llm,
            out_dir=tmp_path,
            generate_video=False,
            enable_knowledge_indexing=False,
        )
    )

    assert result["search_results_count"] == len(search_results)
    assert len(result["facts"]) >= 8

    depth = result["depth_assessment"]
    assert depth["pass"] is True
    assert set(["architecture", "performance", "training"]).issubset(set(depth["fact_categories"]))

    one_pager = result["one_pager"]
    assert len(one_pager["key_findings"]) >= 5
    assert len(one_pager["metrics"]) >= 3
    assert len(one_pager["technical_deep_dive"]) >= 2
    assert len(one_pager["implementation_notes"]) >= 2
    assert len(one_pager["risks_and_mitigations"]) >= 2

    video_brief = result["video_brief"]
    assert len(video_brief["segments"]) >= 3
    assert all(seg.get("visual_prompt") for seg in video_brief["segments"])
    assert all(seg.get("duration_sec") for seg in video_brief["segments"])

    for path in result["written_files"].values():
        assert Path(path).exists()

    assert result["video_artifact"] is None


def test_run_research_from_search_results_manus_case(tmp_path: Path):
    llm = FakeLLM()
    search_results = _build_manus_search_results()

    import asyncio

    result = asyncio.run(
        run_research_from_search_results(
            topic="manus",
            search_results=search_results,
            llm=llm,
            out_dir=tmp_path,
            generate_video=False,
            enable_knowledge_indexing=False,
        )
    )

    assert result["search_results_count"] == len(search_results)
    assert result["depth_assessment"]["pass"] is True
    assert len(result["facts"]) >= 8
    assert len(result["one_pager"]["technical_deep_dive"]) >= 2
    assert len(result["video_brief"]["segments"]) >= 3
    assert Path(result["written_files"]["report_md"]).exists()


def test_pipeline_normalizes_resources_and_video_fields(tmp_path: Path):
    llm = SparseOutputLLM()
    search_results = _build_search_results()

    import asyncio

    result = asyncio.run(
        run_research_from_search_results(
            topic="Hybrid Attention",
            search_results=search_results,
            llm=llm,
            out_dir=tmp_path,
            generate_video=False,
            enable_knowledge_indexing=False,
        )
    )

    resources = result["one_pager"]["resources"]
    assert resources
    assert all(str(item.get("url", "")).startswith("http") for item in resources)
    assert all("your-repo" not in str(item.get("url", "")) for item in resources)

    segments = result["video_brief"]["segments"]
    assert segments
    assert all(seg.get("duration_sec") for seg in segments)
    assert all(seg.get("visual_prompt") for seg in segments)


def test_pipeline_keeps_documents_when_video_generation_fails(tmp_path: Path):
    class FailingVideoGenerator(BaseVideoGenerator):
        provider = "failing"

        async def generate(self, **kwargs):
            raise RuntimeError("simulated video failure")

    llm = FakeLLM()
    search_results = _build_search_results()

    import asyncio

    result = asyncio.run(
        run_research_from_search_results(
            topic="Hybrid Attention",
            search_results=search_results,
            llm=llm,
            out_dir=tmp_path,
            generate_video=True,
            video_generator=FailingVideoGenerator(),
            enable_knowledge_indexing=False,
        )
    )

    assert result["video_artifact"] is None
    assert "simulated video failure" in (result.get("video_error") or "")
    assert Path(result["written_files"]["report_md"]).exists()
