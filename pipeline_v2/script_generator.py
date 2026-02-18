"""Script generation for short-form video runs."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Dict, List, Optional, Sequence
from urllib.parse import urlparse

from core import NormalizedItem
from pipeline_v2.sanitize import is_allowed_citation_url

_FORBIDDEN_TOKENS = re.compile(r"\b(placeholder|dummy|lorem|todo|testsrc|colorbars)\b", re.IGNORECASE)
_HTML_TAGS = re.compile(r"<[^>]+>")


def _compact_text(value: str, max_len: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _clean_sentence(text: str, *, max_len: int = 220) -> str:
    value = str(text or "")
    value = _HTML_TAGS.sub(" ", value)
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
        if len(sentence) < 20:
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


def _body_text(item: NormalizedItem) -> str:
    metadata = dict(item.metadata or {})
    return str(metadata.get("clean_text") or item.body_md or "")


def _links(item: NormalizedItem) -> List[str]:
    urls: List[str] = []
    for citation in list(item.citations or []):
        url = str(citation.url or "").strip()
        if is_allowed_citation_url(url) and url not in urls:
            urls.append(url)
    item_url = str(item.url or "").strip()
    if is_allowed_citation_url(item_url) and item_url not in urls:
        urls.append(item_url)
    return urls[:4]


def _engagement_line(item: NormalizedItem) -> str:
    metadata = dict(item.metadata or {})
    stars = int(float(metadata.get("stars", 0) or 0))
    downloads = int(float(metadata.get("downloads", 0) or 0))
    points = int(float(metadata.get("points", 0) or 0))
    comments = int(float(metadata.get("comment_count", metadata.get("comments", 0)) or 0))
    if stars > 0:
        return f"Community traction is visible with around {stars} stars and active contributor interest."
    if downloads > 0:
        return f"Adoption signal is strong with roughly {downloads} downloads in the recent cycle."
    if points > 0 or comments > 0:
        return f"Discussion signal is high with about {points} points and {comments} comments from practitioners."
    return "Evidence suggests this is moving from isolated demos toward repeatable production workflows."


def _why_now(item: NormalizedItem) -> str:
    metadata = dict(item.metadata or {})
    recency = metadata.get("published_recency")
    if recency not in (None, "", "unknown"):
        try:
            days = float(recency)
            return f"The update is recent (about {days:.1f} days old), so teams can apply these changes immediately."
        except Exception:
            pass
    return _engagement_line(item)


def build_facts(item: NormalizedItem, *, topic: Optional[str] = None) -> Dict[str, object]:
    """Extract a structured fact brief used by script/storyboard generation."""
    text = _body_text(item)
    sentences = _split_sentences(text)
    topic_text = str(topic or "").strip()

    what_it_is = ""
    for sentence in sentences:
        if len(sentence) >= 32:
            what_it_is = sentence
            break
    if not what_it_is:
        what_it_is = f"{item.title} is a practical update that targets production engineering workflows."

    how_it_works: List[str] = []
    for sentence in sentences:
        if sentence.lower() == what_it_is.lower():
            continue
        how_it_works.append(sentence)
        if len(how_it_works) >= 3:
            break

    while len(how_it_works) < 3:
        fallback = _engagement_line(item) if len(how_it_works) == 0 else f"{item.title} provides concrete implementation signals for engineering teams this week."
        how_it_works.append(_clean_sentence(fallback, max_len=180))

    proof: List[str] = [_engagement_line(item)]
    for citation in list(item.citations or [])[:3]:
        snippet = _clean_sentence(citation.snippet or citation.title, max_len=180)
        if snippet and snippet not in proof:
            proof.append(snippet)
        if len(proof) >= 3:
            break

    links = _links(item)
    hook = (
        f"If you're tracking {topic_text}, this is one update worth watching today."
        if topic_text
        else f"{item.title} is surfacing as a practical signal teams are shipping now."
    )
    cta = "Save this brief, review the cited sources, and use the checklist before your next rollout."

    return {
        "topic": topic_text or None,
        "hook": _clean_sentence(hook, max_len=170),
        "what_it_is": _clean_sentence(what_it_is, max_len=190),
        "why_now": _clean_sentence(_why_now(item), max_len=190),
        "how_it_works": [_clean_sentence(point, max_len=180) for point in how_it_works[:3]],
        "proof": [_clean_sentence(point, max_len=180) for point in proof[:3]],
        "links": links,
        "cta": _clean_sentence(cta, max_len=170),
    }


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
    *,
    facts: Optional[Dict[str, object]] = None,
    topic: Optional[str] = None,
) -> Dict[str, object]:
    """Generate a time-coded script from structured facts (no raw snippet stitching)."""
    target = max(20, min(int(duration_sec), 75))
    tone_text = str(tone or "professional").strip() or "professional"
    hints = _platform_hint(platform)

    facts_payload = dict(facts or build_facts(item, topic=topic))
    hook = _clean_sentence(str(facts_payload.get("hook") or ""), max_len=170)
    thesis = _clean_sentence(str(facts_payload.get("what_it_is") or ""), max_len=190)
    key_points_raw = list(facts_payload.get("how_it_works") or [])
    key_points = [_clean_sentence(str(point), max_len=180) for point in key_points_raw if _clean_sentence(str(point), max_len=180)]
    while len(key_points) < 3:
        key_points.append(_clean_sentence(_engagement_line(item), max_len=180))
    key_points = key_points[:3]
    cta_text = _clean_sentence(str(facts_payload.get("cta") or ""), max_len=170)

    links = [str(url).strip() for url in list(facts_payload.get("links") or []) if is_allowed_citation_url(str(url).strip())]
    if not links:
        links = _links(item)

    section_plan = [
        ("hook", hook, links[:1]),
        ("thesis", thesis, links[:1]),
        ("point_1", key_points[0], links[:2]),
        ("point_2", key_points[1], links[1:3] or links[:1]),
        ("point_3", key_points[2], links[2:4] or links[:1]),
        ("cta", cta_text, links[:1]),
    ]
    durations = _allocate_sections(target)

    lines: List[Dict[str, object]] = []
    cursor = 0.0
    for idx, ((section, sentence, refs), slot) in enumerate(zip(section_plan, durations), start=1):
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
                "references": [ref for ref in refs if str(ref).strip()],
            }
        )
        cursor = end

    if lines:
        lines[-1]["end_sec"] = float(target)
        lines[-1]["duration_sec"] = round(float(target) - float(lines[-1]["start_sec"]), 2)

    evidence: List[Dict[str, str]] = []
    for idx, proof in enumerate(list(facts_payload.get("proof") or [])[:3], start=1):
        text = _clean_sentence(str(proof), max_len=180)
        if not text:
            continue
        evidence.append(
            {
                "title": f"fact_{idx}",
                "url": links[min(idx - 1, len(links) - 1)] if links else "",
                "snippet": text,
            }
        )

    for citation in item.citations[:5]:
        snippet = _clean_sentence(citation.snippet or citation.title, max_len=200)
        if not snippet:
            continue
        evidence.append(
            {
                "title": _clean_sentence(citation.title, max_len=140),
                "url": citation.url,
                "snippet": snippet,
            }
        )

    return {
        "item_id": item.id,
        "title": item.title,
        "platform": str(platform or "").strip() or "generic",
        "tone": tone_text,
        "duration_sec": target,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hook_strategy": hints["hook"],
        "cadence": hints["cadence"],
        "facts": facts_payload,
        "structure": {
            "hook": hook,
            "main_thesis": thesis,
            "key_points": key_points,
            "cta": cta_text,
        },
        "lines": lines,
        "evidence": evidence[:8],
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
