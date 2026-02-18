from __future__ import annotations

from pipeline_v2.sanitize import is_allowed_citation_url, sanitize_markdown


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
