from __future__ import annotations

from intelligence.pipeline_helpers import (
    normalize_one_pager_content,
    normalize_timeline_dates,
    normalize_video_brief,
)


def test_normalize_one_pager_content_recovers_sparse_sections():
    one_pager = {
        "title": "Sparse",
        "executive_summary": "",
        "key_findings": [],
        "metrics": {},
        "resources": [{"title": "doc", "url": "https://example.com/doc"}],
    }
    facts = [
        {
            "id": "fact_1",
            "claim": "Latency reduction: 23%",
            "category": "performance",
            "confidence": 0.9,
            "evidence": ["arxiv_1"],
        },
        {
            "id": "fact_2",
            "claim": "Deployment requires KV cache sharding and rollback plan.",
            "category": "deployment",
            "confidence": 0.82,
            "evidence": ["github_1"],
        },
        {
            "id": "fact_3",
            "claim": "Known limitation: compositional arithmetic failure at long chains.",
            "category": "limitation",
            "confidence": 0.78,
            "evidence": ["arxiv_2"],
        },
    ]
    search_results = [
        {"id": "arxiv_1", "source": "arxiv", "title": "Paper A", "content": "content", "metadata": {"year": 2024}},
        {"id": "github_1", "source": "github", "title": "Repo", "content": "content", "metadata": {"stars": 1200}},
    ]

    normalized = normalize_one_pager_content(
        topic="Hybrid Attention",
        one_pager=one_pager,
        facts=facts,
        search_results=search_results,
    )

    assert normalized["executive_summary"]
    assert len(normalized["key_findings"]) >= 3
    assert normalized["metrics"]
    assert "SOTA_Metric" in normalized["metrics"]
    assert "Hardware_Requirement" in normalized["metrics"]
    assert "Core_Formula" in normalized["metrics"]
    assert "Key_Optimization" in normalized["metrics"]
    assert len(normalized["technical_deep_dive"]) >= 1
    assert len(normalized["implementation_notes"]) >= 2
    assert len(normalized["risks_and_mitigations"]) >= 1


def test_normalize_timeline_dates_prefers_verified_result_dates():
    timeline = [
        {
            "date": "2099-01",
            "title": "Unverified event",
            "description": "desc",
            "importance": 4,
            "source_refs": ["fact_1"],
        }
    ]
    facts = [
        {"id": "fact_1", "claim": "claim", "evidence": ["arxiv_1"], "category": "architecture"},
    ]
    search_results = [
        {
            "id": "arxiv_1",
            "source": "arxiv",
            "title": "Paper",
            "content": "content",
            "metadata": {"published_date": "2024-05-12"},
        }
    ]

    normalized = normalize_timeline_dates(
        topic="Hybrid Attention",
        timeline=timeline,
        facts=facts,
        search_results=search_results,
    )

    assert isinstance(normalized, list)
    assert normalized[0]["date"] == "2024-05-12"


def test_normalize_timeline_dates_matches_claim_when_refs_are_hallucinated():
    timeline = [
        {
            "date": "2024-02",
            "title": "Kimi K2.5 引入 Agent Swarm 并行编排",
            "description": "通过并行智能体降低复杂任务延迟。",
            "importance": 5,
            "source_refs": ["fact_10"],
        }
    ]
    facts = [
        {
            "id": "fact_real",
            "claim": "Kimi K2.5 introduces Agent Swarm for parallel task decomposition and latency reduction.",
            "evidence": ["arxiv_1"],
            "category": "deployment",
        }
    ]
    search_results = [
        {
            "id": "arxiv_1",
            "source": "arxiv",
            "title": "Kimi K2.5: Visual Agentic Intelligence",
            "content": "Agent Swarm reduces latency by up to 4.5x.",
            "metadata": {"published_date": "2026-02-02"},
        }
    ]

    normalized = normalize_timeline_dates(
        topic="Kimi 2.5",
        timeline=timeline,
        facts=facts,
        search_results=search_results,
    )

    assert isinstance(normalized, list)
    assert normalized[0]["date"] == "2026-02-02"


def test_normalize_video_brief_expands_sparse_segments():
    normalized = normalize_video_brief(
        topic="Kimi 2.5",
        video_brief={
            "title": "brief",
            "segments": [
                {
                    "title": "Overview",
                    "content": "Core summary",
                    "talking_points": ["architecture"],
                    "duration_sec": 30,
                }
            ],
        },
        facts=[
            {
                "id": "fact_1",
                "claim": "Kimi 2.5 introduces hybrid routing and reports +7.8 benchmark gain.",
                "category": "performance",
                "confidence": 0.9,
                "evidence": ["arxiv_1"],
            }
        ],
        search_results=[
            {
                "id": "arxiv_1",
                "source": "arxiv",
                "title": "Kimi 2.5 Technical Report",
                "content": "Benchmark gain +7.8 and latency reduction 23%.",
                "metadata": {"published_date": "2026-02-02"},
            }
        ],
    )

    assert normalized is not None
    assert len(normalized["segments"]) >= 1


def test_normalize_timeline_dates_uses_fact_evidence_date_over_unrelated_fallback():
    timeline = [
        {
            "date": "2026-01",
            "title": "Agent Swarm architecture release",
            "description": "new orchestration runtime",
            "importance": 5,
            "source_refs": ["fact_1"],
        }
    ]
    facts = [
        {
            "id": "fact_1",
            "claim": "Agent Swarm runtime shipped with Kimi 2.5.",
            "evidence": ["arxiv_target"],
            "category": "deployment",
        }
    ]
    search_results = [
        {
            "id": "arxiv_old",
            "source": "arxiv",
            "title": "Older unrelated item",
            "content": "legacy baseline",
            "metadata": {"published_date": "2024-01-01"},
        },
        {
            "id": "arxiv_target",
            "source": "arxiv",
            "title": "Kimi 2.5 Agent Swarm",
            "content": "parallel orchestration runtime",
            "metadata": {"published_date": "2026-02-02"},
        },
    ]

    normalized = normalize_timeline_dates(
        topic="Kimi 2.5",
        timeline=timeline,
        facts=facts,
        search_results=search_results,
    )

    assert isinstance(normalized, list)
    assert normalized[0]["date"] == "2026-02-02"
