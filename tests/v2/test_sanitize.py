from __future__ import annotations

from pipeline_v2.sanitize import canonicalize_url, classify_link, is_allowed_citation_url, is_valid_http_url, normalize_url, sanitize_markdown


def test_sanitize_markdown_removes_html_badges_and_promotions() -> None:
    raw = """
<p align=\"center\"><img src=\"https://img.shields.io/badge/test-green\"></p>
# Awesome Agent
[![badge](https://img.shields.io/badge/ci-pass-brightgreen)](https://img.shields.io)
Support this project: https://buymeacoffee.com/demo
<div>Agent orchestration for production workflows.</div>
See docs at https://example.com/docs
"""
    clean_md, clean_text, stats = sanitize_markdown(raw)

    lowered = clean_text.lower()
    assert "img.shields.io" not in lowered
    assert "buymeacoffee" not in lowered
    assert "<p" not in lowered
    assert "orchestration" in lowered
    assert "https://example.com/docs" in clean_md
    assert stats["clean_len"] > 20


def test_is_allowed_citation_url_applies_denylist() -> None:
    assert is_allowed_citation_url("https://example.com/report") is True
    assert is_allowed_citation_url("https://img.shields.io/badge/x") is False
    assert is_allowed_citation_url("https://github.com/sponsors/foo") is False
    assert is_allowed_citation_url("https://buymeacoffee.com/foo") is False
    assert is_allowed_citation_url("https://example.com/chart.svg") is False
    assert is_allowed_citation_url("https://[invalid") is False


def test_normalize_url_strips_trailing_quotes_and_validates() -> None:
    raw = 'https://example.com/path?x=1"'
    normalized = normalize_url(raw)
    assert normalized == "https://example.com/path"
    assert is_valid_http_url(normalized) is True


def test_canonicalize_url_removes_json_tail_and_normalizes_scheme() -> None:
    raw = "http://example.com/a/b?utm_source=x&id=123\"]}"
    assert canonicalize_url(raw) == "https://example.com/a/b?id=123"


def test_classify_link_distinguishes_tooling_and_evidence() -> None:
    assert classify_link("https://bun.sh/docs/runtime") == "tooling"
    assert classify_link("https://github.com/org/repo/releases/tag/v1.0.0") == "evidence"
