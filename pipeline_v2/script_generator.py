"""Script generation for short-form video runs."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Dict, List, Sequence
from urllib.parse import urlparse

from core import NormalizedItem

_FORBIDDEN_TOKENS = re.compile(r"\b(placeholder|dummy|lorem|todo|testsrc|colorbars)\b", re.IGNORECASE)


def _compact_text(value: str, max_len: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _clean_sentence(text: str, *, max_len: int = 220) -> str:
    value = str(text or "")
    value = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1", value)
    value = re.sub(r"https?://[^\s)\]>]+", "", value)
    value = re.sub(r"\s+", " ", value).strip(" -:;,.")
    value = _FORBIDDEN_TOKENS.sub("", value)
    value = re.sub(r"\s{2,}", " ", value).strip()
    return _compact_text(value, max_len=max_len)


def _split_sentences(text: str) -> List[str]:
    normalized = str(text or "")
    normalized = normalized.replace("\r\n", "\n")
    normalized = re.sub(r"(?<!\d)\.(?!\d)", "。", normalized)
    chunks = re.split(r"[。！？!?;；\n]+", normalized)
    sentences: List[str] = []
    seen = set()
    for chunk in chunks:
        sentence = _clean_sentence(chunk, max_len=260)
        if len(sentence) < 18:
            continue
        token = sentence.lower()
        if token in seen:
            continue
        seen.add(token)
        sentences.append(sentence)
    return sentences


def _domain(url: str) -> str:
    parsed = urlparse(str(url or ""))
    host = str(parsed.netloc or "").strip().lower()
    return host or "source"


def _platform_hint(platform: str) -> Dict[str, str]:
    key = str(platform or "").strip().lower()
    if key in {"tiktok", "douyin", "reels", "shorts"}:
        return {
            "cadence": "fast",
            "hook": "first 3 seconds impact hook",
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


def _derive_thesis(item: NormalizedItem, sentences: Sequence[str]) -> str:
    for sentence in list(sentences or []):
        if len(sentence) >= 24:
            return _compact_text(sentence, max_len=190)
    domain = _domain(item.url)
    fallback = f"{item.title} is gaining momentum because teams are reporting measurable production impact across {domain} discussions."
    return _compact_text(fallback, max_len=190)


def _metric_point(item: NormalizedItem) -> str:
    metadata = dict(item.metadata or {})
    stars = int(float(metadata.get("stars", 0) or 0))
    downloads = int(float(metadata.get("downloads", 0) or 0))
    points = int(float(metadata.get("points", 0) or 0))
    comments = int(float(metadata.get("comment_count", metadata.get("comments", 0)) or 0))
    if stars > 0:
        return f"GitHub traction is visible with about {stars} stars, suggesting strong developer pull."
    if downloads > 0:
        return f"Adoption signals are strong with roughly {downloads} downloads in the recent window."
    if points > 0 or comments > 0:
        return f"Community interest is high with {points} points and {comments} comments driving discussion."
    return "Community signals indicate this topic is moving from experimentation toward repeatable deployment."


def _derive_key_points(item: NormalizedItem, thesis: str, sentences: Sequence[str]) -> List[str]:
    points: List[str] = []
    thesis_token = thesis.lower()
    for sentence in list(sentences or []):
        token = sentence.lower()
        if token == thesis_token:
            continue
        if len(sentence) < 22:
            continue
        points.append(_compact_text(sentence, max_len=180))
        if len(points) >= 2:
            break

    citation_snippets = [
        _clean_sentence(citation.snippet or citation.title, max_len=180)
        for citation in list(item.citations or [])
        if _clean_sentence(citation.snippet or citation.title, max_len=180)
    ]
    for snippet in citation_snippets:
        if snippet.lower() == thesis_token:
            continue
        if snippet not in points:
            points.append(snippet)
        if len(points) >= 3:
            break

    if len(points) < 3:
        points.append(_metric_point(item))

    if len(points) < 3:
        domain = _domain(item.url)
        points.append(f"Primary evidence and references are traceable back to {domain}, reducing ambiguity in the narrative.")

    while len(points) < 3:
        points.append(f"{item.title} provides concrete implementation cues that teams can adapt this week.")

    return [points[0], points[1], points[2]]


def _build_hook(item: NormalizedItem) -> str:
    domain = _domain(item.url)
    return _compact_text(f"What changed in {item.title}, and why are teams shipping it now across {domain}?", max_len=170)


def _build_cta(item: NormalizedItem) -> str:
    domain = _domain(item.url)
    return _compact_text(
        f"Save this brief, review the linked sources, and follow for tomorrow's ranked updates from {domain} and adjacent communities.",
        max_len=170,
    )


def _allocate_sections(total_duration: int) -> List[float]:
    target = float(max(20, min(total_duration, 75)))
    hook = max(2.2, min(3.0, target * 0.09))
    cta = max(2.2, min(3.4, target * 0.1))
    middle = max(12.0, target - hook - cta)
    thesis = middle * 0.25
    points_total = middle - thesis
    point = points_total / 3.0
    return [hook, thesis, point, point, point, cta]


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
    hook = _build_hook(item)
    thesis = _derive_thesis(item, sentences)
    key_points = _derive_key_points(item, thesis, sentences)
    cta_text = _build_cta(item)

    section_plan = [
        ("hook", hook),
        ("thesis", thesis),
        ("point_1", key_points[0]),
        ("point_2", key_points[1]),
        ("point_3", key_points[2]),
        ("cta", cta_text),
    ]
    durations = _allocate_sections(target)

    lines: List[Dict[str, object]] = []
    cursor = 0.0
    for idx, ((section, sentence), slot) in enumerate(zip(section_plan, durations), start=1):
        duration = max(1.6, float(slot))
        end = min(float(target), cursor + duration)
        line_text = _clean_sentence(sentence, max_len=180)
        if not line_text:
            continue
        lines.append(
            {
                "idx": idx,
                "start_sec": round(cursor, 2),
                "end_sec": round(end, 2),
                "duration_sec": round(end - cursor, 2),
                "section": section,
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
        "structure": {
            "hook": hook,
            "main_thesis": thesis,
            "key_points": key_points,
            "cta": cta_text,
        },
        "lines": lines,
        "evidence": [
            {
                "title": _clean_sentence(citation.title, max_len=140),
                "url": citation.url,
                "snippet": _clean_sentence(citation.snippet, max_len=200),
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
