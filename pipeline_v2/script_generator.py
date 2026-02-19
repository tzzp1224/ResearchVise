"""Script generation for short-form video runs."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Dict, List, Optional, Sequence
from urllib.parse import urlparse

from core import NormalizedItem
from pipeline_v2.sanitize import canonicalize_url, classify_link, is_allowed_citation_url

_FORBIDDEN_TOKENS = re.compile(r"\b(placeholder|dummy|lorem|todo|testsrc|colorbars)\b", re.IGNORECASE)
_HTML_TAGS = re.compile(r"<[^>]+>")
_MARKDOWN_NOISE = re.compile(r"^\s*[*#>\-|`]+\s*$")
_URL_FRAGMENT = re.compile(r"\b(?:[a-z0-9-]+\.)+[a-z]{2,}/[^\s]+", re.IGNORECASE)
_PATH_FRAGMENT = re.compile(r"^(?:[a-z0-9._-]+/){1,4}[a-z0-9._%-]+$", re.IGNORECASE)
_FACT_NOISE_PATTERNS = (
    re.compile(r"\bstars?\s*:\s*\d+", re.IGNORECASE),
    re.compile(r"\bforks?\s*:\s*\d+", re.IGNORECASE),
    re.compile(r"\blast(push|_push|_modified)?\s*:\s*", re.IGNORECASE),
    re.compile(r"\b(?:readme|license|contributing)\b", re.IGNORECASE),
)
_HN_META_LINE = re.compile(r"^\s*Points:\s*\d+\s*(?:\|\s*Comments:\s*\d+)?\s*$", re.IGNORECASE)
_SECTION_HEADER = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")
_SECTION_PRIORITY = {
    "overview": 1.2,
    "what": 1.1,
    "feature": 1.5,
    "quickstart": 1.8,
    "getting started": 1.8,
    "usage": 1.6,
    "how it works": 1.8,
    "example": 1.5,
}
_VERIFIABLE_TOKENS = (
    "cli",
    "api",
    "sdk",
    "mcp",
    "tool calling",
    "orchestration",
    "workflow",
    "pipeline",
    "quickstart",
    "usage",
    "deployment",
    "benchmark",
    "latency",
    "throughput",
)
_COMMAND_PATTERN = re.compile(r"\b(pip\s+install|npm\s+i|npm\s+install|curl\s+|python\s+\S+|docker\s+run)\b", re.IGNORECASE)
_HYPE_PATTERN = re.compile(
    r"\b(best|most|ultimate|revolutionary|amazing|world[- ]class|ever built|game[- ]changing)\b",
    re.IGNORECASE,
)


def _compact_text(value: str, max_len: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    trimmed = text[: max_len - 3].rstrip()
    if " " in trimmed:
        safe = trimmed.rsplit(" ", 1)[0].rstrip()
        if len(safe) >= max(12, int(max_len * 0.55)):
            trimmed = safe
    return trimmed + "..."


def _clean_sentence(text: str, *, max_len: int = 220) -> str:
    value = str(text or "")
    if _HN_META_LINE.match(value.strip()):
        return ""
    value = re.sub(r"^\s*>+\s*", "", value)
    value = _HTML_TAGS.sub(" ", value)
    value = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1", value)
    value = re.sub(r"https?://[^\s)\]>]+", "", value)
    value = _URL_FRAGMENT.sub(" ", value)
    value = re.sub(r"[*_`]{1,3}", " ", value)
    value = re.sub(r"\s*[-]{2,}\s*", " ", value)
    value = re.sub(r"\s+", " ", value).strip(" -:;,.")
    value = _FORBIDDEN_TOKENS.sub("", value)
    value = re.sub(r"\s{2,}", " ", value).strip()
    return _compact_text(value, max_len=max_len)


def _split_sentences(text: str, *, min_len: int = 20) -> List[str]:
    normalized = str(text or "")
    normalized = normalized.replace("\r\n", "\n")
    normalized = re.sub(r"(?<!\d)\.(?!\d)", "。", normalized)
    chunks = re.split(r"[。！？!?;；\n]+", normalized)
    sentences: List[str] = []
    seen = set()
    for chunk in chunks:
        sentence = _clean_sentence(chunk, max_len=260)
        if len(sentence) < int(min_len):
            continue
        if _MARKDOWN_NOISE.match(sentence):
            continue
        token = sentence.lower()
        if token in seen:
            continue
        seen.add(token)
        sentences.append(sentence)
    return sentences


def _split_sections(text: str) -> Dict[str, str]:
    sections: Dict[str, List[str]] = {"root": []}
    current = "root"
    for line in str(text or "").replace("\r\n", "\n").split("\n"):
        if _HN_META_LINE.match(str(line or "").strip()):
            continue
        header = _SECTION_HEADER.match(str(line or "").strip())
        if header:
            current = str(header.group(1) or "").strip().lower()
            if current not in sections:
                sections[current] = []
            continue
        sections.setdefault(current, []).append(str(line))
    return {title: "\n".join(lines).strip() for title, lines in sections.items()}


def _section_boost(section_name: str) -> float:
    key = str(section_name or "").strip().lower()
    for marker, weight in _SECTION_PRIORITY.items():
        if marker in key:
            return float(weight)
    return 0.0


def _sentence_score(sentence: str, *, section_name: str = "root") -> float:
    text = _clean_sentence(sentence, max_len=220)
    if not text or _is_fact_noise(text) or _looks_like_fragment(text):
        return -10.0

    lowered = text.lower()
    score = 0.0
    score += min(1.5, float(len(lowered)) / 120.0)
    score += _section_boost(section_name)

    if any(token in lowered for token in _VERIFIABLE_TOKENS):
        score += 2.0
    if _COMMAND_PATTERN.search(lowered):
        score += 1.8
    if re.search(r"\b\d+(\.\d+)?(%|ms|s|x)?\b", lowered):
        score += 1.1
    if re.search(r"\b(implements?|supports?|routes?|orchestrates?|deploys?|provides?)\b", lowered):
        score += 0.8

    if _HYPE_PATTERN.search(lowered):
        score -= 2.6
    if not _has_verifiable_signal(text):
        score -= 0.9
    if re.search(r"\b(no|not|never|nothing)\b", lowered) and not _has_verifiable_signal(text):
        score -= 0.8

    return score


def _has_verifiable_signal(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return False
    if any(token in lowered for token in _VERIFIABLE_TOKENS):
        return True
    if _COMMAND_PATTERN.search(lowered):
        return True
    if re.search(r"\b\d+(\.\d+)?(%|ms|s|x)?\b", lowered):
        return True
    return False


def _feature_sentences(text: str) -> List[str]:
    sections = _split_sections(text)
    scored: List[tuple[float, str]] = []
    seen = set()
    for section_name, section_text in sections.items():
        for sentence in _split_sentences(section_text, min_len=18):
            token = sentence.lower()
            if token in seen:
                continue
            seen.add(token)
            scored.append((_sentence_score(sentence, section_name=section_name), sentence))
    scored.sort(key=lambda row: row[0], reverse=True)
    return [sentence for score, sentence in scored if score > -5.0]


def _is_fact_noise(sentence: str) -> bool:
    text = str(sentence or "").strip()
    if not text:
        return True
    if _HN_META_LINE.match(text):
        return True
    lowered = text.lower()
    if lowered in {"---", "*", "-"}:
        return True
    if any(pattern.search(text) for pattern in _FACT_NOISE_PATTERNS):
        return True
    return False


def _looks_like_fragment(text: str) -> bool:
    value = str(text or "").strip().lower()
    if not value:
        return True
    if value.startswith(("http://", "https://", "www.")):
        return True
    if _URL_FRAGMENT.search(value):
        return True
    if _PATH_FRAGMENT.match(value):
        return True
    if value.startswith(("com/", "org/", "net/", "io/")):
        return True
    return False


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
    text = str(metadata.get("clean_text") or item.body_md or "")
    cleaned_lines = []
    for line in text.replace("\r\n", "\n").split("\n"):
        if _HN_META_LINE.match(str(line or "").strip()):
            continue
        cleaned_lines.append(str(line))
    return "\n".join(cleaned_lines).strip()


def _links(item: NormalizedItem) -> List[str]:
    urls: List[str] = []
    metadata = dict(item.metadata or {})
    for raw in list(metadata.get("evidence_links") or []):
        token = canonicalize_url(str(raw or "").strip())
        if classify_link(token) == "evidence" and token not in urls:
            urls.append(token)
    for citation in list(item.citations or []):
        url = canonicalize_url(str(citation.url or "").strip())
        if is_allowed_citation_url(url) and url not in urls:
            urls.append(url)
    item_url = canonicalize_url(str(item.url or "").strip())
    if is_allowed_citation_url(item_url) and item_url not in urls:
        urls.append(item_url)
    return urls[:6]


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
    quality_signals = dict(metadata.get("quality_signals") or {})
    item_type = str(metadata.get("item_type") or "").strip().lower()
    if item_type == "release":
        version = str(metadata.get("version") or "").strip()
        if version:
            return f"Release {version} just landed, giving teams a concrete upgrade window."
    recency = quality_signals.get("update_recency_days") or metadata.get("published_recency")
    if recency not in (None, "", "unknown"):
        try:
            days = float(recency)
            return f"Updated {days:.1f} days ago, so this is still inside the decision window."
        except Exception:
            pass
    points = int(float(metadata.get("points", 0) or 0))
    comments = int(float(metadata.get("comment_count", metadata.get("comments", 0)) or 0))
    if points > 0 or comments > 0:
        return f"HN is active at {points} points/{comments} comments, so the operator interest is real."
    stars = int(float(metadata.get("stars", 0) or 0))
    if stars > 0:
        return f"GitHub traction already crossed {stars}, suggesting immediate replication value."
    return "Signals are emerging; verify primary sources before treating this as a strong trend."


def _clean_fact_point(text: str, *, max_len: int = 160) -> str:
    value = _clean_sentence(text, max_len=max_len)
    value = re.sub(r"^(?:[-*•]+\s*)", "", value).strip()
    value = re.sub(r"\s+", " ", value).strip()
    if value and re.search(r"[a-zA-Z0-9]$", value) and not re.search(r"[.!?。！？]$", value):
        value = value + "."
    return value


def build_facts(item: NormalizedItem, *, topic: Optional[str] = None) -> Dict[str, object]:
    """Extract a structured fact brief with verifiable functional points."""
    text = _body_text(item)
    topic_text = str(topic or "").strip()
    feature_candidates = _feature_sentences(text)
    metadata = dict(item.metadata or {})
    quality_signals = dict(metadata.get("quality_signals") or {})

    what_it_is = ""
    for sentence in feature_candidates:
        cleaned = _clean_fact_point(sentence, max_len=150)
        if not cleaned or cleaned.lower().startswith(("we believe", "our vision", "the most")):
            continue
        if _has_verifiable_signal(cleaned):
            what_it_is = cleaned
            break
    if not what_it_is:
        what_it_is = _clean_fact_point(
            f"{item.title} provides a concrete workflow for engineering teams to run and validate.",
            max_len=150,
        )

    how_it_works: List[str] = []
    seen = {what_it_is.lower()}
    for sentence in feature_candidates:
        cleaned = _clean_fact_point(sentence, max_len=88)
        token = cleaned.lower()
        if not cleaned or token in seen:
            continue
        if not _has_verifiable_signal(cleaned):
            continue
        if _looks_like_fragment(cleaned):
            continue
        seen.add(token)
        how_it_works.append(cleaned)
        if len(how_it_works) >= 3:
            break

    while len(how_it_works) < 3:
        if len(how_it_works) == 0:
            fallback = "Quickstart includes reproducible setup and run commands."
        elif len(how_it_works) == 1:
            fallback = "Evidence audit and citation checks gate weak candidates before final picks."
        else:
            fallback = "Outputs ship as facts, script, storyboard, prompts, and validator-ready artifacts."
        how_it_works.append(_clean_fact_point(fallback, max_len=88))

    links = _links(item)
    link_domains = []
    for raw in links:
        domain = _domain(raw)
        if domain not in link_domains:
            link_domains.append(domain)

    metrics = {
        "stars": int(float(metadata.get("stars", 0) or 0)),
        "forks": int(float(metadata.get("forks", 0) or 0)),
        "downloads": int(float(metadata.get("downloads", 0) or 0)),
        "hn_points": int(float(metadata.get("points", 0) or 0)),
        "hn_comments": int(float(metadata.get("comment_count", metadata.get("comments", 0)) or 0)),
        "publish_or_update_time": str(quality_signals.get("publish_or_update_time") or metadata.get("publish_or_update_time") or ""),
    }

    proof: List[str] = []
    proof_links: List[str] = []
    used_pairs = set()
    for citation in list(item.citations or []):
        citation_url = canonicalize_url(str(citation.url or "").strip())
        if not is_allowed_citation_url(citation_url):
            continue
        pair = (_domain(citation_url), str(urlparse(citation_url).path or ""))
        if pair in used_pairs:
            continue
        snippet = _clean_fact_point(citation.snippet or citation.title, max_len=120)
        if not snippet or _looks_like_fragment(snippet):
            continue
        proof.append(snippet)
        proof_links.append(citation_url)
        used_pairs.add(pair)
        if len(proof) >= 2:
            break

    if len(proof) < 2 and metrics["publish_or_update_time"]:
        proof.append(f"Latest publish/update time: {metrics['publish_or_update_time']}.")
        proof_links.append(links[0] if links else "")
    if len(proof) < 2 and (metrics["hn_points"] > 0 or metrics["hn_comments"] > 0):
        proof.append(f"HN discussion: {metrics['hn_points']} points / {metrics['hn_comments']} comments.")
        proof_links.append(next((url for url in links if "news.ycombinator.com/item" in url), links[0] if links else ""))
    while len(proof) < 2:
        fallback = _clean_fact_point("Evidence is limited; verify with release notes or docs before publishing.", max_len=120)
        proof.append(fallback)
        proof_links.append(links[0] if links else "")

    unique_links: List[str] = []
    for url in links:
        token = canonicalize_url(url)
        if token and token not in unique_links:
            unique_links.append(token)
        if len(unique_links) >= 4:
            break
    for url in list(proof_links or []):
        if len(unique_links) >= 2:
            break
        token = canonicalize_url(url)
        if token and token not in unique_links:
            unique_links.append(token)

    dedup_proof_links: List[str] = []
    for link in proof_links:
        token = canonicalize_url(link)
        if not token or token in dedup_proof_links:
            continue
        dedup_proof_links.append(token)
        if len(dedup_proof_links) >= 2:
            break
    for link in unique_links:
        token = canonicalize_url(link)
        if not token or token in dedup_proof_links:
            continue
        dedup_proof_links.append(token)
        if len(dedup_proof_links) >= 2:
            break

    hook = (
        f"If you're tracking {topic_text}, this update has concrete implementation signals."
        if topic_text
        else f"{item.title} has practical implementation details worth checking now."
    )
    cta = "Review the cited links, run quickstart commands, and validate before rollout."

    return {
        "topic": topic_text or None,
        "hook": _clean_sentence(hook, max_len=170),
        "what_it_is": _clean_sentence(what_it_is, max_len=150),
        "why_now": _clean_sentence(_why_now(item), max_len=190),
        "how_it_works": [_clean_fact_point(point, max_len=88) for point in how_it_works[:3]],
        "proof": [_clean_fact_point(point, max_len=120) for point in proof[:2]],
        "proof_links": dedup_proof_links,
        "metrics": metrics,
        "links": unique_links[:4],
        "link_domains": link_domains[:4],
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

    links = [canonicalize_url(str(url).strip()) for url in list(facts_payload.get("links") or []) if is_allowed_citation_url(str(url).strip())]
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
    proof_links = [canonicalize_url(str(url or "").strip()) for url in list(facts_payload.get("proof_links") or [])]
    for idx, proof in enumerate(list(facts_payload.get("proof") or [])[:3], start=1):
        text = _clean_sentence(str(proof), max_len=180)
        if not text:
            continue
        linked = proof_links[min(idx - 1, len(proof_links) - 1)] if proof_links else ""
        evidence.append(
            {
                "title": f"fact_{idx}",
                "url": linked if is_allowed_citation_url(linked) else (links[min(idx - 1, len(links) - 1)] if links else ""),
                "snippet": text,
            }
        )

    for citation in item.citations[:5]:
        citation_url = canonicalize_url(str(citation.url or "").strip())
        if not is_allowed_citation_url(citation_url):
            continue
        snippet = _clean_sentence(citation.snippet or citation.title, max_len=200)
        if not snippet:
            continue
        evidence.append(
            {
                "title": _clean_sentence(citation.title, max_len=140),
                "url": citation_url,
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
