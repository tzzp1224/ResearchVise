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
