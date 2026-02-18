"""Script generation for short-form video runs."""

from __future__ import annotations

from datetime import datetime, timezone
import math
import re
from typing import Dict, List, Sequence

from core import NormalizedItem


def _compact_text(value: str, max_len: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _split_sentences(text: str) -> List[str]:
    normalized = str(text or "")
    # Keep decimal numbers intact while splitting sentence-ending dots.
    normalized = re.sub(r"(?<!\d)\.(?!\d)", "。", normalized)
    chunks = re.split(r"[。！？!?;；\n]+", normalized)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _estimate_sentence_duration(sentence: str) -> float:
    zh_chars = len(re.findall(r"[\u4e00-\u9fff]", sentence))
    en_words = len(re.findall(r"[a-zA-Z0-9_]+", sentence))
    est = zh_chars / 4.0 + en_words / 2.2
    return max(1.6, min(8.5, est))


def _timeline_segments(total_duration: int, count: int) -> List[float]:
    if count <= 0:
        return []
    base = total_duration / float(count)
    return [base for _ in range(count)]


def _platform_hint(platform: str) -> Dict[str, str]:
    key = str(platform or "").strip().lower()
    if key in {"tiktok", "douyin", "reels", "shorts"}:
        return {
            "cadence": "fast",
            "hook": "first 2 seconds question hook",
        }
    if key in {"youtube", "bilibili"}:
        return {
            "cadence": "balanced",
            "hook": "problem-context-solution hook",
        }
    return {
        "cadence": "balanced",
        "hook": "evidence-first hook",
    }


def generate_script(
    item: NormalizedItem,
    duration_sec: int,
    platform: str,
    tone: str,
) -> Dict[str, object]:
    """Generate a time-coded script with deterministic structure."""
    target = max(20, min(int(duration_sec), 75))
    tone_text = str(tone or "professional").strip() or "professional"
    hints = _platform_hint(platform)

    sentences = _split_sentences(item.body_md)
    if not sentences:
        sentences = [item.title, "Key update with evidence-backed impact and deployment implications."]

    max_lines = max(4, min(10, int(math.ceil(target / 5.0))))
    selected = sentences[:max_lines]
    while len(selected) < min(4, max_lines):
        base = selected[len(selected) % len(selected)] if selected else item.title
        selected.append(f"{base} (evidence-backed detail)")

    # Ensure first line is a clear hook.
    opening = _compact_text(
        f"{item.title}: why this matters now?",
        max_len=140,
    )
    selected[0] = opening

    durations = _timeline_segments(target, len(selected))
    lines: List[Dict[str, object]] = []
    cursor = 0.0
    for idx, (sentence, slot) in enumerate(zip(selected, durations), start=1):
        est = _estimate_sentence_duration(sentence)
        duration = max(1.8, min(float(slot) * 1.25, est + 1.8))
        end = min(float(target), cursor + duration)
        line_text = _compact_text(sentence, max_len=180)
        lines.append(
            {
                "idx": idx,
                "start_sec": round(cursor, 2),
                "end_sec": round(end, 2),
                "duration_sec": round(end - cursor, 2),
                "text": line_text,
            }
        )
        cursor = end

    if lines:
        lines[-1]["end_sec"] = float(target)
        lines[-1]["duration_sec"] = round(float(target) - float(lines[-1]["start_sec"]), 2)

    return {
        "item_id": item.id,
        "title": item.title,
        "platform": str(platform or "").strip() or "generic",
        "tone": tone_text,
        "duration_sec": target,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hook_strategy": hints["hook"],
        "cadence": hints["cadence"],
        "lines": lines,
        "evidence": [
            {
                "title": citation.title,
                "url": citation.url,
                "snippet": citation.snippet,
            }
            for citation in item.citations[:8]
        ],
    }


def generate_variants(script: Dict[str, object], platforms: Sequence[str]) -> Dict[str, Dict[str, object]]:
    """Generate platform-specific variants from base script."""
    base_lines = list(script.get("lines") or [])
    variants: Dict[str, Dict[str, object]] = {}

    for platform in platforms:
        key = str(platform or "").strip().lower()
        if not key:
            continue
        hint = _platform_hint(key)

        lines = [dict(line) for line in base_lines]
        if lines and hint["cadence"] == "fast":
            for line in lines:
                text = str(line.get("text") or "")
                line["text"] = _compact_text(text, max_len=120)

        variants[key] = {
            **dict(script),
            "platform": key,
            "cadence": hint["cadence"],
            "hook_strategy": hint["hook"],
            "lines": lines,
        }

    return variants
