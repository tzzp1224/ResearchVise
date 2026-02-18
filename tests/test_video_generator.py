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
    SlideSpec,
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


def test_slides_build_specs_produces_chart_and_diagram():
    generator = SlidevVideoGenerator(target_duration_sec=420)
    slides = generator._build_slide_specs(
        topic="Hybrid Attention",
        video_brief=VideoBrief.from_dict(_sample_video_brief(), default_title="Hybrid Attention"),
        one_pager=OnePager.from_dict(_sample_one_pager(), default_title="Hybrid Attention"),
        facts=_sample_facts(),
    )

    assert len(slides) >= 5
    assert len(slides) <= 10
    kinds = {slide.visual_kind for slide in slides}
    assert "chart" in kinds
    assert "diagram" in kinds

    total_duration = sum(slide.duration_sec for slide in slides)
    assert 360 <= total_duration <= 600


def test_pick_section_title_keeps_decimal_version():
    generator = SlidevVideoGenerator(target_duration_sec=180)
    title = generator._pick_section_title(
        topic="Kimi Chat 2.5 模型技术分析",
        theme="overview",
        hints=["Kimi Chat 2.5 模型技术分析 technical overview"],
    )
    assert "2.5" in title


def test_slides_build_specs_uses_segment_title_as_dynamic_section_hint():
    generator = SlidevVideoGenerator(target_duration_sec=420)
    brief = _sample_video_brief()
    brief["segments"][0]["title"] = "Router Pathology and Cache Policy"

    slides = generator._build_slide_specs(
        topic="Hybrid Attention",
        video_brief=VideoBrief.from_dict(brief, default_title="Hybrid Attention"),
        one_pager=OnePager.from_dict(_sample_one_pager(), default_title="Hybrid Attention"),
        facts=_sample_facts(),
    )

    titles = [slide.title.lower() for slide in slides]
    assert any("router" in title for title in titles)


def test_slides_build_specs_sparse_inputs_create_evidence_gap_sections():
    generator = SlidevVideoGenerator(target_duration_sec=180)
    slides = generator._build_slide_specs(
        topic="MCP",
        video_brief=VideoBrief.from_dict(
            {"title": "MCP brief", "duration_estimate": "3 min", "segments": []},
            default_title="MCP",
        ),
        one_pager=OnePager.from_dict(
            {"title": "MCP", "executive_summary": "", "key_findings": [], "metrics": {}},
            default_title="MCP",
        ),
        facts=[],
    )

    assert len(slides) >= 1
    all_bullets = "\n".join("\n".join(slide.bullets) for slide in slides)
    assert ("证据不足" in all_bullets) or ("补充可验证证据" in all_bullets)


def test_slide_specs_dedupe_cross_slide_duplicate_bullets():
    generator = SlidevVideoGenerator(target_duration_sec=420)
    repeated = "统一目标函数用于平衡质量、覆盖、延迟与成本。"
    one_pager = _sample_one_pager()
    one_pager["key_findings"] = [repeated] * 8
    one_pager["technical_deep_dive"] = [repeated] * 6
    one_pager["implementation_notes"] = [repeated] * 6
    one_pager["risks_and_mitigations"] = [repeated] * 6
    brief = _sample_video_brief()
    for seg in brief["segments"]:
        seg["content"] = repeated
        seg["talking_points"] = [repeated, repeated]

    slides = generator._build_slide_specs(
        topic="Hybrid Attention",
        video_brief=VideoBrief.from_dict(brief, default_title="Hybrid Attention"),
        one_pager=OnePager.from_dict(one_pager, default_title="Hybrid Attention"),
        facts=_sample_facts(),
    )

    assert len(slides) >= 4
    keys = [
        generator._bullet_uniqueness_key(bullet)
        for slide in slides
        for bullet in slide.bullets
        if generator._bullet_uniqueness_key(bullet)
    ]
    assert keys
    assert len(keys) == len(set(keys))


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


def test_narration_specs_are_built_from_slides(tmp_path: Path):
    generator = SlidevVideoGenerator(target_duration_sec=420)
    slides = generator._build_slide_specs(
        topic="Hybrid Attention",
        video_brief=VideoBrief.from_dict(_sample_video_brief(), default_title="Hybrid Attention"),
        one_pager=OnePager.from_dict(_sample_one_pager(), default_title="Hybrid Attention"),
        facts=_sample_facts(),
    )
    specs = generator._narration_pipeline.build_narration_specs(slides[:2])
    assert len(specs) == 2
    assert all(item.text for item in specs)
    assert all(item.text.count(".") + item.text.count("。") >= 2 for item in specs)
    assert specs[0].duration_sec == slides[0].duration_sec

    script_path = generator._narration_pipeline.write_narration_script(specs, tmp_path)
    content = script_path.read_text(encoding="utf-8")
    assert "Video Narration Script" in content
    assert "Slide 1" in content


def test_narration_fit_script_expands_short_seed_for_long_target():
    generator = SlidevVideoGenerator(target_duration_sec=180, tts_speed=1.25)
    seed = "这一页介绍 Kimi Delta Attention 的核心门控机制。"
    fitted = generator._narration_pipeline._fit_script_to_duration(seed, duration_sec=40)
    assert len(fitted) > len(seed)
    assert "门控机制" in fitted


def test_narration_retime_keeps_natural_pace_and_copies_audio(tmp_path: Path):
    generator = SlidevVideoGenerator(target_duration_sec=180)
    pipeline = generator._narration_pipeline
    source = tmp_path / "raw.m4a"
    output = tmp_path / "retimed.m4a"
    source.write_bytes(b"raw-audio")

    def _fake_probe(path: Path):
        if path == source:
            return 12.0
        if path == output and output.exists():
            return 12.0
        return None

    def _fake_run_subprocess(cmd, *, context):
        raise AssertionError("Retime should not run ffmpeg atempo path")

    pipeline._probe_media_duration = _fake_probe  # type: ignore[method-assign]
    pipeline._run_subprocess = _fake_run_subprocess  # type: ignore[method-assign]

    retimed = pipeline._retime_audio_to_target(
        source_path=source,
        output_path=output,
        target_duration_sec=30,
        ffmpeg_bin="ffmpeg",
    )

    assert output.exists()
    assert output.read_bytes() == source.read_bytes()
    assert retimed == 12.0


def test_concat_segments_with_audio_runs_mux_step(tmp_path: Path):
    class ProbeSlides(SlidevVideoGenerator):
        def __init__(self):
            super().__init__()
            self.calls = []

        def _run_subprocess(self, cmd, *, context, cwd=None, env=None):
            self.calls.append((context, cmd))
            output_path = Path(cmd[-1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"FAKE")

    generator = ProbeSlides()
    segment_1 = tmp_path / "segment_001.mp4"
    segment_2 = tmp_path / "segment_002.mp4"
    segment_1.write_bytes(b"segment")
    segment_2.write_bytes(b"segment")
    narration_audio = tmp_path / "video_narration.m4a"
    narration_audio.write_bytes(b"audio")

    output_path = tmp_path / "video_brief.mp4"
    generator._concat_segments(
        segment_paths=[segment_1, segment_2],
        output_path=output_path,
        ffmpeg_bin="ffmpeg",
        narration_audio_path=narration_audio,
    )

    assert output_path.exists()
    contexts = [item[0] for item in generator.calls]
    assert "ffmpeg concat" in contexts
    assert "ffmpeg mux narration" in contexts
    mux_cmd = next(cmd for context, cmd in generator.calls if context == "ffmpeg mux narration")
    assert "-af" in mux_cmd
    assert "apad" in mux_cmd

    mux_cmd = [cmd for context, cmd in generator.calls if context == "ffmpeg mux narration"][0]
    assert "-map" in mux_cmd
    assert "1:a:0" in mux_cmd


def test_generate_sync_uses_narration_duration_for_slide_timing(tmp_path: Path, monkeypatch):
    class DummySlides(SlidevVideoGenerator):
        def __init__(self):
            super().__init__(enable_narration=True)
            self.rendered_slide_durations = []

        def _build_slide_specs(self, *, topic, video_brief, one_pager, facts):
            return [
                SlideSpec(title="S1", bullets=["A", "B"], duration_sec=30, notes="n1"),
                SlideSpec(title="S2", bullets=["C", "D"], duration_sec=30, notes="n2"),
            ]

        def _write_slide_plan_markdown(self, slides, out_dir):
            path = out_dir / "video_slides_plan.md"
            path.write_text("plan\n", encoding="utf-8")
            return path

        def _write_slide_plan_json(self, slides, out_dir):
            path = out_dir / "video_slides_plan.json"
            path.write_text("{}\n", encoding="utf-8")
            return path

        def _write_script_contract(self, out_dir):
            path = out_dir / "video_script_contract.md"
            path.write_text("contract\n", encoding="utf-8")
            return path

        def _ensure_runtime_environment(self):
            return tmp_path / "runtime"

        def _write_slidev_source(self, *, topic, slides, build_dir):
            build_dir.mkdir(parents=True, exist_ok=True)
            entry = build_dir / "slides.md"
            entry.write_text("# slides\n", encoding="utf-8")
            return entry

        def _export_slides_with_slidev(self, *, runtime_dir, entry_path, slides_dir):
            slides_dir.mkdir(parents=True, exist_ok=True)
            for i in range(1, 3):
                (slides_dir / f"{i}.png").write_bytes(b"png")

        def _sorted_exported_images(self, slides_dir):
            return [slides_dir / "1.png", slides_dir / "2.png"]

        def _render_narration_segments(self, *, slides, out_dir, ffmpeg_bin):
            script = out_dir / "video_narration_script.md"
            script.write_text("script\n", encoding="utf-8")
            seg_dir = out_dir / "video_audio_segments"
            seg_dir.mkdir(parents=True, exist_ok=True)
            seg1 = seg_dir / "segment_001.m4a"
            seg2 = seg_dir / "segment_002.m4a"
            seg1.write_bytes(b"a")
            seg2.write_bytes(b"a")
            return {
                "script_path": script,
                "segment_paths": [seg1, seg2],
                "segment_durations": [4.2, 6.8],
                "providers_used": ["edge_tts"],
                "narration_dir": seg_dir,
                "narration_specs": [],
            }

        def _concat_audio_segments(self, audio_paths, output_path, ffmpeg_bin):
            output_path.write_bytes(b"audio")

        def _render_video_segments(self, image_paths, slides, segments_dir, ffmpeg_bin):
            self.rendered_slide_durations = [slide.duration_sec for slide in slides]
            segments_dir.mkdir(parents=True, exist_ok=True)
            paths = []
            for idx in range(1, 3):
                path = segments_dir / f"segment_{idx:03d}.mp4"
                path.write_bytes(b"video")
                paths.append(path)
            return paths

        def _concat_segments(self, segment_paths, output_path, ffmpeg_bin, narration_audio_path=None):
            output_path.write_bytes(b"final")

    generator = DummySlides()
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None)
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_meta = generator._generate_video_sync(
        topic="Hybrid",
        out_dir=out_dir,
        video_brief=VideoBrief.from_dict(_sample_video_brief(), default_title="Hybrid"),
        one_pager=OnePager.from_dict(_sample_one_pager(), default_title="Hybrid"),
        facts=_sample_facts(),
        output_path=out_dir / "video_brief.mp4",
    )

    # slide时长会参考真实旁白，但存在最小时长保护避免视频过短
    assert generator.rendered_slide_durations == [20, 20]
    assert artifact_meta["narration_audio_path"].endswith("video_narration.m4a")


def test_generate_sync_rewrites_plan_json_after_narration(tmp_path: Path, monkeypatch):
    class DummySlides(SlidevVideoGenerator):
        def __init__(self):
            super().__init__(enable_narration=True)
            self.plan_durations_snapshots = []

        def _build_slide_specs(self, *, topic, video_brief, one_pager, facts):
            return [
                SlideSpec(title="S1", bullets=["A"], duration_sec=30, notes="n1"),
                SlideSpec(title="S2", bullets=["B"], duration_sec=30, notes="n2"),
            ]

        def _write_slide_plan_markdown(self, slides, out_dir):
            path = out_dir / "video_slides_plan.md"
            path.write_text("plan\n", encoding="utf-8")
            return path

        def _write_slide_plan_json(self, slides, out_dir):
            self.plan_durations_snapshots.append([slide.duration_sec for slide in slides])
            path = out_dir / "video_slides_plan.json"
            payload = {"slides": [{"duration_sec": slide.duration_sec} for slide in slides]}
            path.write_text(json.dumps(payload), encoding="utf-8")
            return path

        def _write_script_contract(self, out_dir):
            path = out_dir / "video_script_contract.md"
            path.write_text("contract\n", encoding="utf-8")
            return path

        def _ensure_runtime_environment(self):
            return tmp_path / "runtime"

        def _write_slidev_source(self, *, topic, slides, build_dir):
            build_dir.mkdir(parents=True, exist_ok=True)
            entry = build_dir / "slides.md"
            entry.write_text("# slides\n", encoding="utf-8")
            return entry

        def _export_slides_with_slidev(self, *, runtime_dir, entry_path, slides_dir):
            slides_dir.mkdir(parents=True, exist_ok=True)
            for i in range(1, 3):
                (slides_dir / f"{i}.png").write_bytes(b"png")

        def _sorted_exported_images(self, slides_dir):
            return [slides_dir / "1.png", slides_dir / "2.png"]

        def _render_narration_segments(self, *, slides, out_dir, ffmpeg_bin):
            script = out_dir / "video_narration_script.md"
            script.write_text("script\n", encoding="utf-8")
            seg_dir = out_dir / "video_audio_segments"
            seg_dir.mkdir(parents=True, exist_ok=True)
            seg1 = seg_dir / "segment_001.m4a"
            seg2 = seg_dir / "segment_002.m4a"
            seg1.write_bytes(b"a")
            seg2.write_bytes(b"a")
            return {
                "script_path": script,
                "segment_paths": [seg1, seg2],
                "segment_durations": [4.2, 6.8],
                "providers_used": ["edge_tts"],
                "narration_dir": seg_dir,
                "narration_specs": [],
            }

        def _concat_audio_segments(self, audio_paths, output_path, ffmpeg_bin):
            output_path.write_bytes(b"audio")

        def _render_video_segments(self, image_paths, slides, segments_dir, ffmpeg_bin):
            segments_dir.mkdir(parents=True, exist_ok=True)
            out = []
            for i in range(1, 3):
                path = segments_dir / f"segment_{i:03d}.mp4"
                path.write_bytes(b"video")
                out.append(path)
            return out

        def _concat_segments(self, segment_paths, output_path, ffmpeg_bin, narration_audio_path=None):
            output_path.write_bytes(b"final")

    generator = DummySlides()
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None)
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    generator._generate_video_sync(
        topic="Hybrid",
        out_dir=out_dir,
        video_brief=VideoBrief.from_dict(_sample_video_brief(), default_title="Hybrid"),
        one_pager=OnePager.from_dict(_sample_one_pager(), default_title="Hybrid"),
        facts=_sample_facts(),
        output_path=out_dir / "video_brief.mp4",
    )

    assert len(generator.plan_durations_snapshots) >= 2
    assert generator.plan_durations_snapshots[0] == [30, 30]
    assert generator.plan_durations_snapshots[-1] == [20, 20]


def test_tts_provider_candidates_prefers_free_first(monkeypatch):
    generator = SlidevVideoGenerator(tts_provider="auto")
    monkeypatch.setattr(generator._narration_pipeline, "_is_edge_tts_available", lambda: True)
    monkeypatch.setattr(
        "shutil.which",
        lambda name: (
            "/usr/bin/say"
            if name == "say"
            else ("/usr/bin/espeak" if name == "espeak" else None)
        ),
    )

    providers = generator._narration_pipeline.tts_provider_candidates("test")
    assert providers == ["edge_tts", "say", "espeak"]


def test_tts_provider_alias_normalized():
    generator = SlidevVideoGenerator(tts_provider="edge-tts")
    assert generator._narration_pipeline.tts_provider == "edge_tts"


def test_tts_provider_edge_falls_back_to_system_voices(monkeypatch):
    generator = SlidevVideoGenerator(tts_provider="edge-tts")
    monkeypatch.setattr(generator._narration_pipeline, "_is_edge_tts_available", lambda: False)
    monkeypatch.setattr(
        "shutil.which",
        lambda name: "/usr/bin/say" if name == "say" else ("/usr/bin/espeak" if name == "espeak" else None),
    )
    providers = generator._narration_pipeline.tts_provider_candidates("test")
    assert providers == ["say", "espeak"]
