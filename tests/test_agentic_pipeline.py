"""Tests for planner/react/critic/chat orchestration."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

import intelligence.agents.chat_agent as chat_agent_module
import intelligence.pipeline as pipeline_module
from intelligence.agents import ChatAgent, CriticAgent, PlannerAgent, SearchAgent
from intelligence.llm.base import BaseLLM, LLMResponse, Message, ToolCall


class _NoopLLM(BaseLLM):
    def __init__(self):
        super().__init__(model="noop")

    @property
    def provider(self) -> str:
        return "noop"

    async def acomplete(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(content="{}", model=self.model, usage={})

    async def astream(self, messages: List[Message], **kwargs: Any):
        yield ""


class _PlannerOffTopicLLM(BaseLLM):
    def __init__(self):
        super().__init__(model="planner-off-topic")

    @property
    def provider(self) -> str:
        return "fake"

    async def acomplete(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        payload = {
            "is_technical": True,
            "reason": "technical",
            "normalized_topic": "OpenClaw 技术研究",
            "query_rewrites": ["OpenClaw 技术研究", "OpenClaw architecture"],
            "research_questions": ["q1"],
            "search_plan": [{"dimension": "architecture", "query": "OpenClaw architecture"}],
        }
        return LLMResponse(content=json.dumps(payload, ensure_ascii=False), model=self.model, usage={})

    async def astream(self, messages: List[Message], **kwargs: Any):
        yield ""


class _SearchLLM(BaseLLM):
    def __init__(self):
        super().__init__(model="search-llm")
        self.calls = 0

    @property
    def provider(self) -> str:
        return "fake"

    async def acomplete(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content="开始检索",
                model=self.model,
                usage={},
                tool_calls=[
                    ToolCall(
                        id="tc_1",
                        name="github_search",
                        arguments={
                            "query": "MCP production deployment architecture benchmark comparison limitation deployment",
                            "max_results": 3,
                        },
                    )
                ],
            )
        return LLMResponse(content="搜索完成", model=self.model, usage={})

    async def astream(self, messages: List[Message], **kwargs: Any):
        yield ""


class _ChatLLM(BaseLLM):
    def __init__(self):
        super().__init__(model="chat-llm")

    @property
    def provider(self) -> str:
        return "fake"

    async def acomplete(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        payload = {
            "answer": "MCP 在生产环境建议优先做会话隔离与可观测性。",
            "highlights": ["会话隔离", "可观测性", "故障回滚"],
            "limitations": ["缺少统一 benchmark"],
            "follow_up_queries": ["MCP latency benchmark 2025"],
        }
        return LLMResponse(content=json.dumps(payload, ensure_ascii=False), model=self.model, usage={})

    async def astream(self, messages: List[Message], **kwargs: Any):
        yield ""


class _CacheCfg:
    enabled = False
    similarity_threshold = 0.82
    top_k = 3
    min_quality_score = 0.0
    require_video_for_video_request = True
    collection_name = "research_artifacts"


def test_planner_filters_non_technical_request():
    planner = PlannerAgent(llm=_NoopLLM())
    result = asyncio.run(planner.plan(topic="今天晚饭吃什么", user_query="给我推荐几个家常菜"))
    assert result["is_technical"] is False


def test_planner_rejects_off_topic_normalized_topic():
    planner = PlannerAgent(llm=_PlannerOffTopicLLM())
    result = asyncio.run(planner.plan(topic="MCP production deployment", user_query=None))
    assert result["normalized_topic"] == "MCP production deployment"
    assert all("openclaw" not in q.lower() for q in result["query_rewrites"])


@pytest.mark.asyncio
async def test_search_agent_react_executes_tool_and_returns_trace():
    async def _tool_executor(name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        assert name == "github_search"
        return [
            {
                "id": "github_1",
                "source": "github",
                "title": "MCP gateway runtime",
                "content": "architecture benchmark comparison limitation deployment",
                "url": "https://example.com/repo1",
                "metadata": {},
            },
            {
                "id": "github_2",
                "source": "github",
                "title": "MCP production guide",
                "content": "production deployment rollback monitoring and limitation analysis",
                "url": "https://example.com/repo2",
                "metadata": {},
            },
        ]

    agent = SearchAgent(
        llm=_SearchLLM(),
        max_iterations=2,
        min_total_results=2,
        allowed_sources=["github"],
        tool_executor=_tool_executor,
    )

    result = await agent.run(
        {
            "topic": "MCP production deployment",
            "query_rewrites": ["MCP production deployment"],
            "search_plan": [{"dimension": "architecture", "query": "MCP architecture"}],
            "max_results_per_source": 4,
        }
    )

    assert len(result["search_results"]) == 2
    assert result["search_trace"]
    assert result["search_trace"][0]["tool"] == "github_search"


@pytest.mark.asyncio
async def test_critic_agent_returns_quality_metrics():
    critic = CriticAgent(quality_threshold=0.5)
    evaluation = await critic.evaluate(
        facts=[
            {
                "id": "fact1",
                "category": "architecture",
                "claim": "A vs B trade-off with mitigation plan.",
                "evidence": ["src1", "src2"],
            },
            {
                "id": "fact2",
                "category": "performance",
                "claim": "Latency reduced by 20%.",
                "evidence": ["src1", "src3"],
            },
            {
                "id": "fact3",
                "category": "comparison",
                "claim": "Compared with baseline however cost rises slightly.",
                "evidence": ["src2", "src3"],
            },
            {
                "id": "fact4",
                "category": "limitation",
                "claim": "Main risk is memory pressure but mitigated by chunking.",
                "evidence": ["src2", "src4"],
            },
            {
                "id": "fact5",
                "category": "training",
                "claim": "Curriculum schedule stabilizes convergence.",
                "evidence": ["src1"],
            },
        ],
        search_results=[
            {"id": "src1", "source": "arxiv"},
            {"id": "src2", "source": "github"},
            {"id": "src3", "source": "stackoverflow"},
            {"id": "src4", "source": "hackernews"},
        ],
        one_pager={
            "metrics": {"latency": "120ms", "throughput": "80rps", "cost": "$0.12/1k"},
            "implementation_notes": ["Deploy with canary and monitor p95.", "Prepare rollback switch."],
            "risks_and_mitigations": ["risk->mitigation", "risk->mitigation"],
            "key_findings": ["f1", "f2", "f3", "f4", "f5"],
        },
        video_brief={"segments": [{"talking_points": ["a"]}, {"talking_points": ["b"]}, {"talking_points": ["c"]}]},
        knowledge_gaps=[],
    )
    assert "quality_metrics" in evaluation
    assert "overall_score" in evaluation["quality_metrics"]
    assert isinstance(evaluation["pass"], bool)


@pytest.mark.asyncio
async def test_chat_agent_over_kb(monkeypatch):
    async def _fake_hybrid_search(
        query: str,
        sources: List[str] | None = None,
        year_filter: int | None = None,
        top_k: int = 10,
        score_threshold: float = 0.15,
        namespace: str | None = None,
        topic_hash: str | None = None,
    ) -> List[Dict[str, Any]]:
        return [
            {
                "id": "doc_1",
                "content": "MCP deployment guide with observability and rollback strategy.",
                "metadata": {"source": "github", "url": "https://example.com/mcp-guide"},
                "score": 0.92,
            }
        ]

    monkeypatch.setattr(chat_agent_module, "hybrid_search", _fake_hybrid_search)

    agent = ChatAgent(llm=_ChatLLM())
    result = await agent.ask(question="MCP 生产部署要注意什么？", use_hybrid=True, top_k=4)

    assert result["retrieved_count"] == 1
    assert result["citations"]
    assert "会话隔离" in result["answer"]


@pytest.mark.asyncio
async def test_chat_agent_deduplicates_citations(monkeypatch):
    async def _fake_hybrid_search(
        query: str,
        sources: List[str] | None = None,
        year_filter: int | None = None,
        top_k: int = 10,
        score_threshold: float = 0.15,
        namespace: str | None = None,
        topic_hash: str | None = None,
    ) -> List[Dict[str, Any]]:
        return [
            {
                "id": "doc_1",
                "content": "snippet A",
                "metadata": {"source": "github", "url": "https://example.com/a"},
                "score": 0.91,
            },
            {
                "id": "doc_1",
                "content": "snippet B",
                "metadata": {"source": "github", "url": "https://example.com/a"},
                "score": 0.90,
            },
        ]

    monkeypatch.setattr(chat_agent_module, "hybrid_search", _fake_hybrid_search)

    agent = ChatAgent(llm=_ChatLLM())
    result = await agent.ask(question="MCP 风险?", use_hybrid=True, top_k=4)

    assert result["retrieved_count"] == 1
    assert len(result["citations"]) == 1


@pytest.mark.asyncio
async def test_pipeline_end_to_end_uses_planner_and_agentic_search(monkeypatch):
    async def _fake_plan(self, *, topic: str, user_query: str | None = None):
        return {
            "is_technical": True,
            "reason": "technical",
            "normalized_topic": "Model Context Protocol production deployment",
            "query_rewrites": ["MCP production deployment", "Model Context Protocol deployment"],
            "research_questions": ["q1"],
            "search_plan": [{"dimension": "architecture", "query": "MCP architecture"}],
        }

    async def _fake_search_run(self, state: Dict[str, Any]):
        return {
            "search_results": [
                {
                    "id": "github_1",
                    "source": "github",
                    "title": "repo",
                    "content": "architecture performance comparison limitation deployment",
                    "url": "https://example.com/repo",
                    "metadata": {},
                }
            ],
            "search_trace": [{"iteration": 1, "tool": "github_search", "result_count": 1}],
            "coverage": {"result_count": 1, "source_coverage": ["github"]},
            "strategy": "react_agent",
        }

    async def _fake_research_from_search_results(**kwargs: Any):
        return {
            "topic": kwargs["topic"],
            "search_results_count": len(kwargs["search_results"]),
            "facts": [{"id": "f1"}],
            "knowledge_gaps": [],
            "timeline": [],
            "one_pager": {"title": "x"},
            "video_brief": {"title": "v"},
            "depth_assessment": {"pass": True, "score": 10, "max_score": 13},
            "quality_metrics": {"overall_score": 0.8},
            "quality_gate_pass": True,
            "quality_recommendations": [],
            "output_dir": None,
            "written_files": {},
            "video_artifact": None,
            "video_error": None,
        }

    class _UnexpectedAggregator:
        def __init__(self, *args: Any, **kwargs: Any):
            pass

        async def __aenter__(self):
            raise AssertionError("DataAggregator should not be called when agentic search returns results")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(pipeline_module.PlannerAgent, "plan", _fake_plan)
    monkeypatch.setattr(pipeline_module.SearchAgent, "run", _fake_search_run)
    monkeypatch.setattr(pipeline_module, "get_research_cache_settings", lambda: _CacheCfg())
    monkeypatch.setattr(
        pipeline_module,
        "run_research_from_search_results",
        _fake_research_from_search_results,
    )
    monkeypatch.setattr(pipeline_module, "DataAggregator", _UnexpectedAggregator)

    result = await pipeline_module.run_research_end_to_end(
        topic="MCP production deployment",
        llm=_NoopLLM(),
        use_agentic_search=True,
        max_results_per_source=4,
    )

    assert result["search_results_count"] == 1
    assert result["planner"]["is_technical"] is True
    assert result["search_strategy"] == "react_agent"
    assert result["aggregated_summary"]["total"] == 1


@pytest.mark.asyncio
async def test_pipeline_blocks_non_technical_request(monkeypatch):
    async def _fake_non_tech_plan(self, *, topic: str, user_query: str | None = None):
        return {
            "is_technical": False,
            "reason": "non technical intent",
            "normalized_topic": topic,
            "query_rewrites": [],
            "research_questions": [],
            "search_plan": [],
        }

    class _UnexpectedAggregator:
        def __init__(self, *args: Any, **kwargs: Any):
            pass

        async def __aenter__(self):
            raise AssertionError("DataAggregator should not be called for blocked request")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(pipeline_module.PlannerAgent, "plan", _fake_non_tech_plan)
    monkeypatch.setattr(pipeline_module, "get_research_cache_settings", lambda: _CacheCfg())
    monkeypatch.setattr(pipeline_module, "DataAggregator", _UnexpectedAggregator)

    result = await pipeline_module.run_research_end_to_end(
        topic="周末去哪玩",
        llm=_NoopLLM(),
    )

    assert result["blocked"] is True
    assert result["search_results_count"] == 0
    assert result["quality_gate_pass"] is False


@pytest.mark.asyncio
async def test_pipeline_returns_cached_result_when_similarity_hit(monkeypatch, tmp_path: Path):
    class _CacheEnabledCfg(_CacheCfg):
        enabled = True
        similarity_threshold = 0.7
        top_k = 3

    snapshot = {
        "topic": "Model Context Protocol production deployment",
        "search_results_count": 23,
        "facts": [{"id": "f1"}],
        "knowledge_gaps": [],
        "timeline": [],
        "one_pager": {"title": "cached"},
        "video_brief": {"title": "cached"},
        "depth_assessment": {"pass": True, "score": 11, "max_score": 13},
        "quality_metrics": {"overall_score": 0.72},
        "quality_gate_pass": True,
        "quality_recommendations": [],
        "output_dir": str(tmp_path / "cached_output"),
        "written_files": {},
        "video_artifact": {
            "provider": "slidev",
            "output_path": str(tmp_path / "cached_output" / "video_brief.mp4"),
        },
        "video_error": None,
    }
    snapshot_path = tmp_path / "cached_snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
    video_path = Path(snapshot["video_artifact"]["output_path"])
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"video")

    class _FakeArtifactStore:
        def __init__(self, collection_name: str = "research_artifacts"):
            self.collection_name = collection_name

        def find_similar(self, *, query: str, score_threshold: float = 0.82, top_k: int = 3):
            return [
                    {
                        "topic": "Model Context Protocol production deployment",
                        "score": 0.97,
                        "snapshot_path": str(snapshot_path),
                        "quality_score": 0.72,
                        "artifact_schema_version": "2026-02-08-slidev-v2",
                    }
            ]

        def close(self):
            return None

    class _UnexpectedSearchAgent:
        def __init__(self, *args: Any, **kwargs: Any):
            pass

        async def run(self, state: Dict[str, Any]):
            raise AssertionError("Search should be skipped on cache hit")

    class _UnexpectedAggregator:
        def __init__(self, *args: Any, **kwargs: Any):
            pass

        async def __aenter__(self):
            raise AssertionError("Aggregator should be skipped on cache hit")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(pipeline_module, "get_research_cache_settings", lambda: _CacheEnabledCfg())
    monkeypatch.setattr(pipeline_module, "ResearchArtifactStore", _FakeArtifactStore)
    monkeypatch.setattr(pipeline_module, "SearchAgent", _UnexpectedSearchAgent)
    monkeypatch.setattr(pipeline_module, "DataAggregator", _UnexpectedAggregator)

    result = await pipeline_module.run_research_end_to_end(
        topic="MCP production deployment",
        llm=_NoopLLM(),
        generate_video=True,
    )

    assert result["cache_hit"] is True
    assert result["search_strategy"] == "cache_reuse"
    assert result["search_results_count"] == 23


@pytest.mark.asyncio
async def test_chat_over_kb_propagates_session_id(monkeypatch):
    async def _fake_ask(self, **kwargs: Any):
        return {
            "answer": "ok",
            "citations": [],
            "retrieved_count": 0,
            "session_id": kwargs.get("namespace"),
        }

    monkeypatch.setattr(pipeline_module.ChatAgent, "ask", _fake_ask)

    result = await pipeline_module.chat_over_kb(question="test", topic="MCP production deployment", llm=_NoopLLM())
    assert result["answer"] == "ok"
    assert str(result.get("session_id", "")).strip()
