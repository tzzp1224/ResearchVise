"""
Tests for video generator implementations.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from outputs.video_generator import (
    OnePager,
    SlidevVideoGenerator,
    VideoBrief,
    VideoGenerationError,
    build_video_prompt,
    create_video_generator,
)


def _sample_video_brief() -> dict:
    return {
        "title": "Hybrid Attention Deep Dive",
        "duration_estimate": "7-8 min",
        "hook": "Long-context quality and latency often conflict in production.",
        "segments": [
            {
                "title": "Architecture",
                "content": "Sparse+dense hybrid routing with KV cache partitioning.",
                "talking_points": ["Router", "Cache", "Memory"],
                "duration_sec": 80,
            },
            {
                "title": "Benchmarks",
                "content": "Latency and quality trade-offs across workloads.",
                "talking_points": ["p95 latency", "LongBench", "OOM rate"],
                "duration_sec": 70,
            },
        ],
        "conclusion": "Use hybrid attention with explicit guardrails.",
        "call_to_action": "Run controlled A/B and monitor cost-quality frontier.",
    }


def _sample_one_pager() -> dict:
    return {
        "title": "Hybrid Attention Report",
        "executive_summary": "Hybrid routing improves throughput while preserving quality under long contexts.",
        "key_findings": [
            "Sparse routing lowers p95 latency by 23% in production traces.",
            "KV cache sharding reduces OOM for 128K prompts.",
            "Entropy regularization improves router stability.",
        ],
        "metrics": {
            "p95_latency": "-23%",
            "longbench_em": "+7.8",
            "oom_rate": "-41%",
        },
        "technical_deep_dive": [
            "Two-stage route scoring combines lexical salience and learned policy.",
            "Paged KV cache with shard-aware eviction keeps memory bounded.",
            "Router temperature scheduling avoids early collapse.",
        ],
        "implementation_notes": [
            "Warmup with dense-only decoding for first N steps.",
            "Enable per-shard admission control before enabling sparse mode.",
            "Expose fallback path for tail latency spikes.",
        ],
        "risks_and_mitigations": [
            "Risk: routing collapse under distribution shift; mitigation: entropy floor + drift alarms.",
            "Risk: cache fragmentation; mitigation: periodic compaction + page size tuning.",
        ],
    }


def _sample_facts() -> list[dict]:
    return [
        {
            "claim": "Hybrid attention reduced p95 latency by 23% on 8xA100 service.",
            "category": "performance",
            "confidence": 0.92,
        },
        {
            "claim": "LongBench EM improved by +7.8 with sparse+dense routing.",
            "category": "benchmark",
            "confidence": 0.88,
        },
        {
            "claim": "KV cache sharding lowered OOM incidents for 128K prompts.",
            "category": "deployment",
            "confidence": 0.86,
        },
    ]


def test_build_video_prompt_contains_fact_and_segments():
    prompt = build_video_prompt(
        topic="Transformer",
        facts=_sample_facts(),
        video_brief=_sample_video_brief(),
        one_pager=_sample_one_pager(),
    )
    assert "Transformer" in prompt
    assert "Segment:" in prompt
    assert "confidence=0.92" in prompt


def test_create_video_generator_slides_provider_only():
    generator = create_video_generator("slidev")
    assert isinstance(generator, SlidevVideoGenerator)

    with pytest.raises(ValueError):
        create_video_generator("veo")

    with pytest.raises(ValueError):
        create_video_generator("storyboard")


def test_slides_build_specs_produces_chart_formula_and_diagram():
    generator = SlidevVideoGenerator(target_duration_sec=420)
    slides = generator._build_slide_specs(
        topic="Hybrid Attention",
        video_brief=VideoBrief.from_dict(_sample_video_brief(), default_title="Hybrid Attention"),
        one_pager=OnePager.from_dict(_sample_one_pager(), default_title="Hybrid Attention"),
        facts=_sample_facts(),
    )

    assert len(slides) >= 8
    kinds = {slide.visual_kind for slide in slides}
    assert "chart" in kinds
    assert "formula" in kinds
    assert "diagram" in kinds

    total_duration = sum(slide.duration_sec for slide in slides)
    assert 360 <= total_duration <= 600


def test_slides_build_specs_sanitizes_placeholder_bullets():
    generator = SlidevVideoGenerator(target_duration_sec=420)

    dirty_one_pager = _sample_one_pager()
    dirty_one_pager["key_findings"] = ["{{placeholder}}", "  ", "N/A", "Valid metric: p95 latency -12%"]
    dirty_one_pager["technical_deep_dive"] = ["[[todo]]", "（待补充）", "Deep detail: API context retrieval pipeline"]

    dirty_brief = _sample_video_brief()
    dirty_brief["segments"][0]["content"] = "<placeholder>"
    dirty_brief["segments"][0]["talking_points"] = ["  ", "N/A", "Real point: API retrieval chain"]

    slides = generator._build_slide_specs(
        topic="Hybrid Attention",
        video_brief=VideoBrief.from_dict(dirty_brief, default_title="Hybrid Attention"),
        one_pager=OnePager.from_dict(dirty_one_pager, default_title="Hybrid Attention"),
        facts=_sample_facts(),
    )

    bullet_text = "\n".join(["\n".join(slide.bullets) for slide in slides])
    assert "placeholder" not in bullet_text.lower()
    assert "{{" not in bullet_text
    assert "[[" not in bullet_text
    assert "Valid metric" in bullet_text
    assert "Real point" in bullet_text


def test_slides_generator_generate_with_mocked_sync_flow(tmp_path: Path):
    class DummySlides(SlidevVideoGenerator):
        def _generate_video_sync(self, *, topic, out_dir, video_brief, one_pager, facts, output_path):
            output_path.write_bytes(b"FAKE_MP4")
            (out_dir / "video_slides_plan.md").write_text("# plan\n", encoding="utf-8")
            (out_dir / "video_slides_plan.json").write_text("{}\n", encoding="utf-8")
            return {"slide_count": 12, "estimated_duration_sec": 420}

    generator = DummySlides()
    artifact = asyncio.run(
        generator.generate(
            topic="Hybrid Attention",
            out_dir=tmp_path,
            facts=_sample_facts(),
            video_brief=_sample_video_brief(),
            one_pager=_sample_one_pager(),
        )
    )

    assert artifact.provider == "slidev"
    assert artifact.output_path.exists()
    assert artifact.output_path.read_bytes() == b"FAKE_MP4"

    metadata = json.loads(artifact.metadata_path.read_text(encoding="utf-8"))
    assert metadata["provider"] == "slidev"
    assert metadata["flow"]["slide_count"] == 12


def test_slides_generator_raises_when_output_missing(tmp_path: Path):
    class BrokenSlides(SlidevVideoGenerator):
        def _generate_video_sync(self, *, topic, out_dir, video_brief, one_pager, facts, output_path):
            return {"slide_count": 3}

    generator = BrokenSlides()
    with pytest.raises(VideoGenerationError):
        asyncio.run(
            generator.generate(
                topic="Hybrid Attention",
                out_dir=tmp_path,
                facts=_sample_facts(),
                video_brief=_sample_video_brief(),
                one_pager=_sample_one_pager(),
            )
        )
