"""Audio, subtitles, and BGM post-processing utilities."""

from __future__ import annotations

from array import array
import math
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Dict, List, Tuple
import wave
from uuid import uuid4


def _script_lines(script: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [dict(item) for item in list(script.get("lines") or []) if isinstance(item, dict)]


def _clean_text(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"\b(placeholder|dummy|lorem|todo|testsrc|colorbars)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def _format_srt_timestamp(seconds: float) -> str:
    value = max(0.0, float(seconds))
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    secs = int(value % 60)
    millis = int(round((value - int(value)) * 1000))
    if millis >= 1000:
        millis -= 1000
        secs += 1
        if secs >= 60:
            secs = 0
            minutes += 1
            if minutes >= 60:
                minutes = 0
                hours += 1
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _timeline(script: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], float]:
    raw_lines = _script_lines(script)
    total = float(script.get("duration_sec") or 0.0)

    if not raw_lines:
        structure = dict(script.get("structure") or {})
        fallback_lines = [
            structure.get("hook"),
            structure.get("main_thesis"),
            *list(structure.get("key_points") or []),
            structure.get("cta"),
            script.get("title"),
        ]
        raw_lines = [{"idx": idx + 1, "text": text} for idx, text in enumerate(fallback_lines) if str(text or "").strip()]

    if not raw_lines:
        raw_lines = [{"idx": 1, "text": "Generated narration"}]

    ordered = sorted(raw_lines, key=lambda item: float(item.get("start_sec") or item.get("idx") or 0.0))
    normalized: List[Dict[str, Any]] = []
    cursor = 0.0

    for idx, line in enumerate(ordered, start=1):
        text = _clean_text(str(line.get("text") or "")) or "Update"
        requested_start = float(line.get("start_sec") or cursor)
        start = max(cursor, requested_start)

        requested_end = line.get("end_sec")
        if requested_end is not None:
            end = max(start + 0.35, float(requested_end))
        else:
            end = max(start + 0.35, start + float(line.get("duration_sec") or 2.2))

        normalized.append({"idx": idx, "start_sec": start, "end_sec": end, "text": text})
        cursor = end

    total = max(total, cursor)
    if total <= 0:
        total = 1.0

    # Clamp all lines into [0, total] while preserving monotonicity.
    fixed: List[Dict[str, Any]] = []
    cursor = 0.0
    for line in normalized:
        start = max(cursor, float(line["start_sec"]))
        end = min(total, max(start + 0.35, float(line["end_sec"])))
        fixed.append({"idx": int(line["idx"]), "start_sec": start, "end_sec": end, "text": str(line["text"])})
        cursor = end

    if fixed and fixed[-1]["end_sec"] < total:
        fixed[-1]["end_sec"] = total

    return fixed, total


def tts_generate(script: Dict[str, Any], voice_profile: Dict[str, Any]) -> str:
    """Generate deterministic local narration WAV without external dependencies."""
    out_dir = Path(str(voice_profile.get("out_dir") or "/tmp"))
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_path = out_dir / "tts_narration.wav"

    lines, duration_sec = _timeline(script)
    duration_sec = max(1.0, min(duration_sec, 300.0))
    sample_rate = int(voice_profile.get("sample_rate") or 22050)
    sample_rate = max(8000, min(sample_rate, 48000))

    pcm = array("h")
    cursor_samples = 0
    total_samples = int(duration_sec * sample_rate)

    for idx, line in enumerate(lines, start=1):
        start_sample = int(float(line["start_sec"]) * sample_rate)
        end_sample = int(float(line["end_sec"]) * sample_rate)
        if end_sample <= start_sample:
            end_sample = start_sample + int(0.35 * sample_rate)

        if start_sample > cursor_samples:
            pcm.extend([0] * (start_sample - cursor_samples))
            cursor_samples = start_sample

        segment_samples = max(1, end_sample - start_sample)
        text = str(line["text"])
        token_len = max(1, len(re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]", text)))
        base_freq = 160.0 + (idx % 5) * 28.0 + min(80.0, token_len * 0.9)

        for n in range(segment_samples):
            t = n / float(sample_rate)
            fade_in = min(1.0, n / max(1.0, sample_rate * 0.03))
            fade_out = min(1.0, (segment_samples - n) / max(1.0, sample_rate * 0.05))
            env = max(0.0, min(fade_in, fade_out))
            sample = int(6500.0 * env * math.sin(2.0 * math.pi * base_freq * t))
            pcm.append(sample)

        cursor_samples = end_sample

    if cursor_samples < total_samples:
        pcm.extend([0] * (total_samples - cursor_samples))

    with wave.open(str(audio_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())

    return str(audio_path)


def align_subtitles(script: Dict[str, Any], audio_path: str) -> str:
    """Generate subtitle SRT with monotonic timestamps from script timeline."""
    _ = audio_path
    lines, _duration = _timeline(script)

    out_dir = Path(str(script.get("output_dir") or "/tmp"))
    out_dir.mkdir(parents=True, exist_ok=True)
    srt_path = out_dir / "captions.srt"

    chunks: List[str] = []
    for idx, line in enumerate(lines, start=1):
        start_sec = float(line.get("start_sec") or 0.0)
        end_sec = float(line.get("end_sec") or (start_sec + 1.0))
        if end_sec <= start_sec:
            end_sec = start_sec + 0.35
        text = _clean_text(str(line.get("text") or "")) or "Update"
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
    """Mix optional BGM into narration; return original audio when no BGM is configured."""
    src = Path(str(audio))
    if not src.exists():
        raise FileNotFoundError(f"audio not found: {src}")

    bgm_path = Path(str(bgm_profile.get("bgm_path") or "")).expanduser()
    ffmpeg_bin = shutil.which("ffmpeg")
    if not bgm_path.exists() or not ffmpeg_bin:
        return str(src)

    out_dir = Path(str(bgm_profile.get("out_dir") or src.parent))
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "tts_narration_with_bgm.wav"
    tmp_target = out_dir / f".{target.name}.{uuid4().hex}.tmp"

    bgm_volume = float(bgm_profile.get("bgm_volume") or 0.12)
    narration_volume = float(bgm_profile.get("narration_volume") or 1.0)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-stream_loop",
        "-1",
        "-i",
        str(bgm_path),
        "-filter_complex",
        (
            f"[0:a]volume={narration_volume}[main];"
            f"[1:a]volume={bgm_volume}[bg];"
            "[main][bg]amix=inputs=2:duration=first:dropout_transition=2"
        ),
        "-c:a",
        "pcm_s16le",
        str(tmp_target),
    ]

    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0 or not tmp_target.exists() or tmp_target.stat().st_size <= 0:
        if tmp_target.exists():
            tmp_target.unlink()
        return str(src)

    tmp_target.replace(target)
    return str(target)
