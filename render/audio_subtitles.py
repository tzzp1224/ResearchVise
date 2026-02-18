"""Audio, subtitles, and BGM post-processing utilities."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Dict, List
import wave


def _script_lines(script: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [dict(item) for item in list(script.get("lines") or []) if isinstance(item, dict)]


def _format_srt_timestamp(seconds: float) -> str:
    value = max(0.0, float(seconds))
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    secs = int(value % 60)
    millis = int(round((value - int(value)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def tts_generate(script: Dict[str, Any], voice_profile: Dict[str, Any]) -> str:
    """Generate a deterministic local WAV placeholder for narration."""
    out_dir = Path(str(voice_profile.get("out_dir") or "/tmp"))
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_path = out_dir / "tts_narration.wav"

    lines = _script_lines(script)
    duration_sec = float(script.get("duration_sec") or 0.0)
    if duration_sec <= 0:
        duration_sec = sum(float(line.get("duration_sec") or 0.0) for line in lines)
    duration_sec = max(1.0, min(duration_sec, 300.0))

    sample_rate = 16000
    frame_count = int(sample_rate * duration_sec)

    with wave.open(str(audio_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frame_count)

    return str(audio_path)


def align_subtitles(script: Dict[str, Any], audio_path: str) -> str:
    """Generate subtitle SRT aligned to time-coded script lines."""
    _ = audio_path
    lines = _script_lines(script)
    if not lines:
        lines = [
            {
                "idx": 1,
                "start_sec": 0.0,
                "end_sec": float(script.get("duration_sec") or 3.0),
                "text": str(script.get("title") or "Generated narration"),
            }
        ]

    out_dir = Path(str(script.get("output_dir") or "/tmp"))
    out_dir.mkdir(parents=True, exist_ok=True)
    srt_path = out_dir / "captions.srt"

    chunks: List[str] = []
    for idx, line in enumerate(lines, start=1):
        start_sec = float(line.get("start_sec") or 0.0)
        end_sec = float(line.get("end_sec") or (start_sec + float(line.get("duration_sec") or 2.0)))
        text = str(line.get("text") or "").strip() or "..."
        chunks.append(
            "\n".join(
                [
                    str(idx),
                    f"{_format_srt_timestamp(start_sec)} --> {_format_srt_timestamp(end_sec)}",
                    text,
                    "",
                ]
            )
        )

    srt_path.write_text("\n".join(chunks), encoding="utf-8")
    return str(srt_path)


def mix_bgm(audio: str, bgm_profile: Dict[str, Any]) -> str:
    """Best-effort BGM mixing placeholder by copying source audio."""
    src = Path(str(audio))
    if not src.exists():
        raise FileNotFoundError(f"audio not found: {src}")

    out_dir = Path(str(bgm_profile.get("out_dir") or src.parent))
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "tts_narration_with_bgm.wav"
    shutil.copyfile(src, target)
    return str(target)
