"""Unit tests for intelligence.tools.rag_tools."""

from __future__ import annotations

import pytest

from intelligence.tools import rag_tools


@pytest.mark.asyncio
async def test_hybrid_search_uses_in_filter_for_multi_sources(monkeypatch):
    calls = []

    async def _fake_vector_search(
        query,
        top_k=5,
        filter=None,
        score_threshold=0.3,
        namespace=None,
        topic_hash=None,
    ):
        calls.append(
            {
                "query": query,
                "top_k": top_k,
                "filter": filter,
                "score_threshold": score_threshold,
                "namespace": namespace,
                "topic_hash": topic_hash,
            }
        )
        return [{"id": "doc-1", "content": "x", "metadata": {"source": "arxiv"}, "score": 0.9}]

    monkeypatch.setattr(rag_tools, "vector_search", _fake_vector_search)

    results = await rag_tools.hybrid_search(
        query="multi-source retrieval",
        sources=["arxiv", "github"],
        year_filter=2024,
        top_k=6,
    )

    assert len(results) == 1
    assert len(calls) == 1
    assert calls[0]["filter"] == {
        "source": {"$in": ["arxiv", "github"]},
        "year": {"$gte": 2024},
    }


@pytest.mark.asyncio
async def test_hybrid_search_falls_back_to_fanout_and_dedupes(monkeypatch):
    calls = []

    async def _fake_vector_search(
        query,
        top_k=5,
        filter=None,
        score_threshold=0.3,
        namespace=None,
        topic_hash=None,
    ):
        calls.append(
            {
                "query": query,
                "top_k": top_k,
                "filter": filter,
                "namespace": namespace,
                "topic_hash": topic_hash,
            }
        )
        source_filter = (filter or {}).get("source")
        if isinstance(source_filter, dict):
            raise RuntimeError("IN filter unsupported")
        if source_filter == "arxiv":
            return [
                {"id": "shared", "content": "same", "metadata": {"source": "arxiv"}, "score": 0.61},
                {"id": "a-only", "content": "arxiv only", "metadata": {"source": "arxiv"}, "score": 0.55},
            ]
        if source_filter == "github":
            return [
                {"id": "shared", "content": "same newer", "metadata": {"source": "github"}, "score": 0.93},
                {"id": "g-only", "content": "github only", "metadata": {"source": "github"}, "score": 0.74},
            ]
        return []

    monkeypatch.setattr(rag_tools, "vector_search", _fake_vector_search)

    results = await rag_tools.hybrid_search(
        query="mcp runtime",
        sources=["arxiv", "github"],
        top_k=3,
    )

    assert len(calls) == 3
    assert calls[0]["filter"] == {"source": {"$in": ["arxiv", "github"]}}
    assert {call["filter"]["source"] for call in calls[1:]} == {"arxiv", "github"}
    assert [item["id"] for item in results] == ["shared", "g-only", "a-only"]
    assert results[0]["score"] == 0.93
