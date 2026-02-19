from __future__ import annotations

from core import RawItem
from pipeline_v2.normalize import content_hash, extract_citations, normalize


def test_extract_citations_from_markdown_and_url() -> None:
    raw = RawItem(
        id="raw_1",
        source="web_article",
        title="MCP rollout notes",
        url="https://example.com/main",
        body="See [official report](https://example.com/report) and https://example.com/bench",
        metadata={},
    )

    citations = extract_citations(raw)
    urls = {item.url for item in citations}

    assert "https://example.com/report" in urls
    assert "https://example.com/bench" in urls
    assert "https://example.com/main" in urls


def test_extract_citations_filters_badge_and_image_urls() -> None:
    raw = RawItem(
        id="raw_bad",
        source="github",
        title="Repo",
        url="https://github.com/acme/repo",
        body=(
            "[badge](https://img.shields.io/badge/ci-pass)\n"
            "See https://buymeacoffee.com/demo\n"
            "image https://example.com/logo.svg\n"
            "docs https://example.com/docs"
        ),
        metadata={},
    )

    citations = extract_citations(raw)
    urls = {item.url for item in citations}
    assert "https://example.com/docs" in urls
    assert "https://img.shields.io/badge/ci-pass" not in urls
    assert "https://buymeacoffee.com/demo" not in urls
    assert "https://example.com/logo.svg" not in urls


def test_extract_citations_uses_contextual_snippets_per_url() -> None:
    raw = RawItem(
        id="raw_ctx",
        source="github",
        title="acme/agent",
        url="https://github.com/acme/agent",
        body=(
            "Quickstart guide [install docs](https://docs.example.com/agent/install) for first boot.\n"
            "Benchmark notes [results](https://docs.example.com/agent/results) include latency and throughput.\n"
            "Community discussion https://news.ycombinator.com/item?id=42 with production feedback.\n"
        ),
        metadata={},
    )
    citations = extract_citations(raw)
    snippets = [str(item.snippet or "").strip() for item in citations if str(item.url or "").strip().startswith("https://docs.example.com")]
    assert len(snippets) >= 2
    assert len(set(snippets)) >= 2


def test_extract_citations_strips_html_anchor_snippets() -> None:
    raw = RawItem(
        id="raw_html_ctx",
        source="github",
        title="acme/agent-html",
        url="https://github.com/acme/agent-html",
        body=(
            '<a href="https://platform.example.com/docs/faq/contact-us" target="_blank" style="font-size:17px;">'
            "Contact docs</a>\n"
            "Learn more about MCP runtime [guide](https://platform.example.com/docs/agent-runtime)\n"
        ),
        metadata={},
    )
    citations = extract_citations(raw)
    by_url = {str(item.url or ""): str(item.snippet or "") for item in citations}
    faq_snippet = by_url.get("https://platform.example.com/docs/faq/contact-us", "")
    assert "href=" not in faq_snippet.lower()
    assert "<a" not in faq_snippet.lower()
    assert faq_snippet.strip() != ""


def test_extract_citations_excludes_tooling_links_and_canonicalizes_noise() -> None:
    raw = RawItem(
        id="raw_tool",
        source="github",
        title="agent repo",
        url='https://github.com/acme/agent"]}',
        body=(
            "API endpoint https://api.openai.com/v1/chat/completions should not be citation.\n"
            "Install notes https://bun.sh/docs/runtime should be tooling.\n"
            "Local test http://myapp.localhost:1355 should be dropped.\n"
            "Primary source [release](https://github.com/acme/agent/releases/tag/v1.2.0?utm_source=x).\n"
        ),
        metadata={},
    )
    citations = extract_citations(raw)
    urls = {item.url for item in citations}
    assert "https://api.openai.com/v1/chat/completions" not in urls
    assert "https://bun.sh/docs/runtime" not in urls
    assert "https://myapp.localhost:1355" not in urls
    assert "https://github.com/acme/agent/releases/tag/v1.2.0" in urls
    assert "https://github.com/acme/agent" in urls


def test_normalize_assigns_tier_b_credibility_and_hash() -> None:
    raw = RawItem(
        id="raw_2",
        source="rss",
        title="News digest",
        url="https://example.com/news",
        body="Community write-up",
        author=None,
        metadata={},
    )

    item = normalize(raw)
    assert item.tier == "B"
    assert item.metadata.get("credibility") in {"low", "medium"}
    assert item.metadata.get("body_len", 0) > 0
    assert item.metadata.get("citation_count", 0) >= 1
    assert "published_recency" in item.metadata
    assert item.metadata.get("link_count", 0) >= 1
    assert isinstance(item.metadata.get("quality_metrics"), dict)
    assert item.hash


def test_content_hash_is_stable() -> None:
    raw = RawItem(
        id="raw_3",
        source="github",
        title="acme/agent",
        url="https://github.com/acme/agent",
        body="Fast path for inference",
        metadata={},
    )

    h1 = content_hash(raw)
    h2 = content_hash(raw)
    assert h1 == h2


def test_quality_signals_extracted_github_like_item() -> None:
    raw = RawItem(
        id="gh_quality",
        source="github",
        title="acme/agent-runner",
        url="https://github.com/acme/agent-runner",
        body=(
            "# Agent Runner\n"
            "Quickstart: install via pip and run `agent-run --demo`.\n"
            "## Results\n"
            "Benchmark shows 28% lower latency with multi-agent routing.\n"
            "![demo](https://raw.githubusercontent.com/acme/agent-runner/main/assets/demo.png)\n"
        ),
        author="acme",
        published_at="2026-02-16T10:00:00Z",
        metadata={"last_push": "2026-02-17T11:30:00Z"},
    )

    item = normalize(raw)
    signals = dict(item.metadata.get("quality_signals") or {})
    assert float(signals.get("content_density", 0.0)) > 0.05
    assert bool(signals.get("has_quickstart")) is True
    assert bool(signals.get("has_results_or_bench")) is True
    assert bool(signals.get("has_images_non_badge")) is True
    assert str(signals.get("publish_or_update_time") or "").startswith("2026-02-17")
    assert signals.get("update_recency_days") is not None
