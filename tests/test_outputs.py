"""
Tests for Phase 4 Outputs module
"""

from __future__ import annotations


def test_render_timeline_markdown_empty():
    from outputs import render_timeline_markdown

    md = render_timeline_markdown("Test Topic", None)
    assert "# Test Topic Timeline" in md
    assert "No timeline events" in md


def test_render_timeline_markdown_sorted_by_date():
    from outputs import render_timeline_markdown

    timeline = [
        {"date": "2018-10", "title": "B", "description": "b", "importance": 3, "source_refs": []},
        {"date": "2017-06", "title": "A", "description": "a", "importance": 5, "source_refs": []},
    ]

    md = render_timeline_markdown("X", timeline, include_mermaid=False)

    # 2017 的事件应该在 2018 之前出现
    assert md.index("2017-06") < md.index("2018-10")


def test_render_one_pager_markdown_sections():
    from outputs import render_one_pager_markdown

    one_pager = {
        "title": "T",
        "executive_summary": "S",
        "key_findings": ["k1", "k2"],
        "metrics": {"m1": "v1"},
        "strengths": ["s1"],
        "weaknesses": ["w1"],
        "technical_deep_dive": ["d1"],
        "implementation_notes": ["i1"],
        "risks_and_mitigations": ["r1 -> m1"],
        "resources": [{"title": "r1", "url": "https://example.com"}],
    }

    md = render_one_pager_markdown(one_pager)
    assert "# T" in md
    assert "> S" in md
    assert "## Key Findings" in md
    assert "## Metrics" in md
    assert "| m1 | v1 |" in md
    assert "## Strengths" in md
    assert "## Weaknesses" in md
    assert "## Technical Deep Dive" in md
    assert "## Implementation Notes" in md
    assert "## Risks and Mitigations" in md
    assert "## Resources" in md
    assert "[r1](https://example.com)" in md


def test_export_research_outputs_writes_files(tmp_path):
    from outputs import export_research_outputs

    written = export_research_outputs(
        tmp_path,
        topic="Topic",
        timeline=[{"date": "2020", "title": "E", "description": "d", "importance": 3, "source_refs": []}],
        one_pager={"title": "OP", "executive_summary": "ES"},
        video_brief={"title": "VB", "duration_estimate": "5m", "hook": "H"},
        write_report=True,
    )

    assert (tmp_path / "timeline.md").exists()
    assert (tmp_path / "timeline.json").exists()
    assert (tmp_path / "one_pager.md").exists()
    assert (tmp_path / "one_pager.json").exists()
    assert (tmp_path / "video_brief.md").exists()
    assert (tmp_path / "video_brief.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "manifest.json").exists()

    assert "manifest_json" in written


def test_render_video_brief_markdown_segment_metadata():
    from outputs import render_video_brief_markdown

    md = render_video_brief_markdown(
        {
            "title": "VB",
            "duration_estimate": "5m",
            "hook": "Hook",
            "segments": [
                {
                    "title": "S1",
                    "content": "C1",
                    "duration_sec": 42,
                    "visual_prompt": "cinematic benchmark overlay",
                }
            ],
        }
    )

    assert "- Duration: 42s" in md
    assert "- Visual prompt: cinematic benchmark overlay" in md
