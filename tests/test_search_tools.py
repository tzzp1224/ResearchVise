"""Unit tests for intelligence.tools.search_tools."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from intelligence.tools.search_tools import (
    arxiv_rss_search,
    create_search_tools,
    execute_search_tool,
    huggingface_search,
    openreview_search,
    reddit_search,
    twitter_search,
)


class _FakeHF:
    def __init__(self):
        self.called_models = None
        self.called_datasets = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def search_models(self, query: str, max_results=None):
        self.called_models = (query, max_results)
        return [
            SimpleNamespace(
                id="m1",
                name="Model 1",
                description="desc",
                url="https://hf.co/m1",
                downloads=100,
                likes=10,
                tags=["tag"],
                pipeline_tag="text-generation",
            )
        ]

    async def search_datasets(self, query: str, max_results=None):
        self.called_datasets = (query, max_results)
        return [
            SimpleNamespace(
                id="d1",
                name="Dataset 1",
                description="desc",
                url="https://hf.co/d1",
                downloads=50,
                likes=5,
                tags=["tag"],
            )
        ]


class _FakeSocialScraper:
    def __init__(self, posts):
        self._posts = posts

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def search(self, *args, **kwargs):
        return self._posts


@pytest.mark.asyncio
async def test_huggingface_search_passes_max_results_keyword():
    fake = _FakeHF()
    with patch("intelligence.tools.search_tools.HuggingFaceScraper", return_value=fake):
        results = await huggingface_search("manus", search_type="all", max_results=7)

    assert fake.called_models == ("manus", 7)
    assert fake.called_datasets == ("manus", 7)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_execute_search_tool_unknown_name():
    with pytest.raises(ValueError):
        await execute_search_tool("unknown_tool", {"query": "manus"})


def test_create_search_tools_includes_new_fresh_sources():
    tools = create_search_tools(allowed_sources=["openreview", "arxiv_rss"])
    names = [tool["function"]["name"] for tool in tools]
    assert "openreview_search" in names
    assert "arxiv_rss_search" in names


@pytest.mark.asyncio
async def test_openreview_search_parses_notes(monkeypatch):
    class _FakeResponse:
        status_code = 200

        def json(self):
            return {
                "notes": [
                    {
                        "id": "abc123",
                        "cdate": 1739059200000,
                        "content": {
                            "title": {"value": "Kimi 2.5 Technical Report"},
                            "abstract": {"value": "Benchmark score reaches 76.5 with 32B active params."},
                            "venue": {"value": "CoRR 2026"},
                        },
                    }
                ]
            }

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, *args, **kwargs):
            return _FakeResponse()

    async def _fake_run_with_timeout(coro, *, context, timeout_sec=15):
        return await coro

    monkeypatch.setattr("intelligence.tools.search_tools.httpx.AsyncClient", lambda *args, **kwargs: _FakeClient())
    monkeypatch.setattr("intelligence.tools.search_tools._run_with_timeout", _fake_run_with_timeout)

    results = await openreview_search("Kimi 2.5", max_results=3)
    assert results
    assert results[0]["source"] == "openreview"
    assert "Technical Report" in results[0]["title"]
    assert results[0]["url"].startswith("https://openreview.net/forum?id=")


@pytest.mark.asyncio
async def test_arxiv_rss_search_parses_feed(monkeypatch):
    rss_xml = """<?xml version="1.0"?>
    <rss version="2.0">
      <channel>
        <item>
          <title>Kimi 2.5 architecture update</title>
          <description>Latency improves by 20% on benchmark.</description>
          <link>https://arxiv.org/abs/2602.02276</link>
          <pubDate>Mon, 09 Feb 2026 08:00:00 GMT</pubDate>
        </item>
      </channel>
    </rss>
    """

    class _FakeResponse:
        status_code = 200
        text = rss_xml

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, *args, **kwargs):
            return _FakeResponse()

    monkeypatch.setattr("intelligence.tools.search_tools.httpx.AsyncClient", lambda *args, **kwargs: _FakeClient())
    results = await arxiv_rss_search("Kimi 2.5", max_results=5)
    assert results
    assert results[0]["source"] == "arxiv_rss"
    assert "arxiv.org/abs/2602.02276" in (results[0]["url"] or "")


@pytest.mark.asyncio
async def test_twitter_search_defensive_mapping_and_item_isolation(monkeypatch, caplog):
    fake = _FakeSocialScraper(
        [
            SimpleNamespace(
                id="t1",
                author="alice",
                content="tweet content",
                url="https://x.com/alice/status/1",
                likes=7,
                created_at=None,
                extra={"retweets": 9, "replies": 4},
            ),
            SimpleNamespace(
                author="broken",
                content="missing id should be skipped",
                url="https://x.com/broken/status/2",
                extra={},
            ),
        ]
    )
    monkeypatch.setattr("intelligence.tools.search_tools.TwitterScraper", lambda: fake)
    caplog.set_level(logging.ERROR, logger="intelligence.tools.search_tools")

    results = await twitter_search("agent platform", max_results=5)
    assert len(results) == 1
    metadata = results[0]["metadata"]
    assert metadata["likes"] == 7
    assert metadata["reposts"] == 9
    assert metadata["comments"] == 4
    assert metadata["retweets"] == 9
    assert metadata["replies"] == 4
    assert any("Twitter post parse failed" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_twitter_search_timeout_is_logged_as_warning(monkeypatch, caplog):
    fake = _FakeSocialScraper([])
    monkeypatch.setattr("intelligence.tools.search_tools.TwitterScraper", lambda: fake)

    async def _timeout_wait_for(coro, timeout):
        coro.close()
        raise asyncio.TimeoutError()

    monkeypatch.setattr("intelligence.tools.search_tools.asyncio.wait_for", _timeout_wait_for)
    caplog.set_level(logging.WARNING, logger="intelligence.tools.search_tools")

    results = await twitter_search("timeout case", max_results=3)
    assert results == []
    assert any("timed out" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_reddit_search_defensive_mapping_and_item_isolation(monkeypatch, caplog):
    fake = _FakeSocialScraper(
        [
            SimpleNamespace(
                id="r1",
                content="Title line\n\ndetails",
                author="u/alice",
                url="https://reddit.com/r/MachineLearning/comments/r1",
                likes=42,
                comments=11,
                created_at=None,
                extra={"subreddit": "MachineLearning"},
            ),
            SimpleNamespace(
                id="r2",
                content="bad extra payload",
                author="u/bob",
                url="https://reddit.com/r/LocalLLaMA/comments/r2",
                extra="invalid",
            ),
        ]
    )
    monkeypatch.setattr("intelligence.tools.search_tools.RedditScraper", lambda: fake)
    caplog.set_level(logging.ERROR, logger="intelligence.tools.search_tools")

    results = await reddit_search("agent platform", max_results=5)
    assert len(results) == 1
    assert results[0]["title"] == "Title line"
    metadata = results[0]["metadata"]
    assert metadata["subreddit"] == "MachineLearning"
    assert metadata["score"] == 42
    assert metadata["likes"] == 42
    assert metadata["num_comments"] == 11
    assert metadata["comments"] == 11
    assert any("Reddit post parse failed" in record.message for record in caplog.records)
