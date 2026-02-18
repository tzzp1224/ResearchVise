from __future__ import annotations

import pytest

from sources import connectors


@pytest.mark.asyncio
async def test_fetch_rss_feed_parses_items(monkeypatch) -> None:
    rss_xml = """
    <rss><channel>
      <item>
        <title>AI Infra Weekly</title>
        <link>https://example.com/post-1</link>
        <description><![CDATA[Latency tuning tips]]></description>
        <pubDate>2026-02-17T10:00:00+00:00</pubDate>
      </item>
    </channel></rss>
    """.strip()

    article_html = """
    <html>
      <head><title>AI Infra Weekly Deep Dive</title></head>
      <body>
        <article>
          <p>Latency tuning moved p95 from 210ms to 130ms in staging.</p>
          <p>Canary rollout covered 5% traffic before full release.</p>
        </article>
      </body>
    </html>
    """.strip()

    async def _fake_get_text(url: str, *, headers=None) -> str:
        if url == "https://example.com/feed.xml":
            return rss_xml
        if url == "https://example.com/post-1":
            return article_html
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(connectors, "_http_get_text", _fake_get_text)

    items = await connectors.fetch_rss_feed("https://example.com/feed.xml", max_results=5)
    assert len(items) == 1
    assert items[0].tier == "B"
    assert items[0].source == "rss"
    assert items[0].title == "AI Infra Weekly"
    assert "extraction_method" in items[0].metadata


@pytest.mark.asyncio
async def test_fetch_web_article_extracts_title_and_body(monkeypatch) -> None:
    html = """
    <html>
      <head><title>Edge Deployment Playbook</title></head>
      <body>
        <article>
          <p>Step 1: collect p95 latency and GPU memory metrics.</p>
          <p>Step 2: deploy canary and monitor rollback signals.</p>
        </article>
      </body>
    </html>
    """.strip()

    async def _fake_get_text(url: str, *, headers=None) -> str:
        assert url == "https://example.com/article"
        return html

    monkeypatch.setattr(connectors, "_http_get_text", _fake_get_text)

    items = await connectors.fetch_web_article("https://example.com/article")
    assert len(items) == 1
    assert items[0].tier == "B"
    assert "Edge Deployment Playbook" in items[0].title
    assert "Step 1" in items[0].body
    assert items[0].metadata.get("extraction_method")


@pytest.mark.asyncio
async def test_fetch_github_releases_maps_response(monkeypatch) -> None:
    payload = [
        {
            "id": 123,
            "name": "v2.1.0",
            "tag_name": "v2.1.0",
            "html_url": "https://github.com/acme/agent/releases/tag/v2.1.0",
            "body": "Adds fallback renderer",
            "published_at": "2026-02-16T11:30:00Z",
            "author": {"login": "acme-bot"},
        }
    ]

    async def _fake_get_json(url: str, *, headers=None):
        assert "api.github.com/repos/acme/agent/releases" in url
        return payload

    monkeypatch.setattr(connectors, "_http_get_json", _fake_get_json)

    items = await connectors.fetch_github_releases(["acme/agent"], max_results_per_repo=1)
    assert len(items) == 1
    assert items[0].source == "github"
    assert items[0].tier == "A"
    assert items[0].metadata.get("item_type") == "release"
