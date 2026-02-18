from __future__ import annotations

from pathlib import Path

from render.audio_subtitles import align_subtitles, mix_bgm, tts_generate


def _script(tmp_path: Path) -> dict:
    return {
        "title": "MCP explainer",
        "duration_sec": 8,
        "output_dir": str(tmp_path),
        "lines": [
            {"idx": 1, "start_sec": 0.0, "end_sec": 3.2, "text": "What is MCP and why now?"},
            {"idx": 2, "start_sec": 3.2, "end_sec": 8.0, "text": "Deployment path with rollback strategy."},
        ],
    }


def test_tts_generate_align_subtitles_and_mix_bgm(tmp_path: Path) -> None:
    script = _script(tmp_path)

    audio_path = tts_generate(script, {"voice": "neutral", "out_dir": str(tmp_path)})
    assert audio_path.endswith(".wav")
    assert Path(audio_path).exists()

    srt_path = align_subtitles(script, audio_path)
    assert srt_path.endswith(".srt")
    srt_text = Path(srt_path).read_text(encoding="utf-8")
    assert "What is MCP and why now?" in srt_text
    assert "00:00:00,000 --> 00:00:03," in srt_text

    mixed = mix_bgm(audio_path, {"track": "calm", "out_dir": str(tmp_path)})
    assert mixed.endswith(".wav")
    assert Path(mixed).exists()


def test_align_subtitles_monotonic_when_timeline_is_invalid(tmp_path: Path) -> None:
    script = {
        "title": "test",
        "duration_sec": 6,
        "output_dir": str(tmp_path),
        "lines": [
            {"idx": 1, "start_sec": 1.5, "end_sec": 1.6, "text": "first"},
            {"idx": 2, "start_sec": 1.0, "end_sec": 1.1, "text": "second"},
        ],
    }
    audio_path = tts_generate(script, {"out_dir": str(tmp_path)})
    srt_path = align_subtitles(script, audio_path)
    text = Path(srt_path).read_text(encoding="utf-8")
    assert "first" in text
    assert "second" in text


def test_mix_bgm_returns_source_when_bgm_missing(tmp_path: Path) -> None:
    script = _script(tmp_path)
    audio_path = tts_generate(script, {"out_dir": str(tmp_path)})
    mixed = mix_bgm(audio_path, {"bgm_path": str(tmp_path / "missing.wav"), "out_dir": str(tmp_path)})
    assert mixed == audio_path
