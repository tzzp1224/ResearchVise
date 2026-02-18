"""Content sanitization and citation URL filtering for v2 pipeline."""

from __future__ import annotations

import html as html_lib
import re
from typing import Dict, List, Tuple
from urllib.parse import urlparse


_CLEAN_BLOCK_PATTERNS = [
    re.compile(r"<script[^>]*>.*?</script>", flags=re.IGNORECASE | re.DOTALL),
    re.compile(r"<style[^>]*>.*?</style>", flags=re.IGNORECASE | re.DOTALL),
    re.compile(r"```[\\s\\S]*?```", flags=re.MULTILINE),
]
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
_MD_IMAGE_ONLY_RE = re.compile(r"^\s*!\[[^\]]*\]\(([^)]+)\)\s*$")
_MD_IMAGE_INLINE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
_URL_RE = re.compile(r"https?://[^\s)\]>]+")

_BADGE_OR_PROMO_MARKERS = (
    "img.shields.io",
    "badge",
    "buymeacoffee.com",
    "github.com/sponsors",
    "patreon.com",
)
_DENYLIST_DOMAINS = {
    "img.shields.io",
    "buymeacoffee.com",
    "www.buymeacoffee.com",
    "github.com/sponsors",
    "www.github.com/sponsors",
    "localhost",
    "127.0.0.1",
}
_IMAGE_SUFFIXES = (".svg", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico")


def normalize_url(url: str) -> str:
    value = str(url or "").strip()
    if not value:
        return ""
    value = value.strip("\"'`")
    value = re.sub(r"[\s]+", "", value)
    value = value.rstrip(".,;:")
    while value and value[-1] in {"\"", "'", "`"}:
        value = value[:-1].rstrip(".,;:")
    if value.endswith(")") and "(" not in value:
        value = value[:-1].rstrip(".,;:")
    return value


def is_valid_http_url(url: str) -> bool:
    normalized = normalize_url(url)
    if not normalized.startswith(("http://", "https://")):
        return False
    if " " in normalized:
        return False
    if len(normalized) > 2048:
        return False
    try:
        parsed = urlparse(normalized)
    except ValueError:
        return False
    if str(parsed.scheme or "").lower() not in {"http", "https"}:
        return False
    host = str(parsed.netloc or "").strip()
    if not host or "." not in host and host not in {"localhost", "127.0.0.1"}:
        return False
    return True


def _hostname(url: str) -> str:
    try:
        parsed = urlparse(normalize_url(url))
    except ValueError:
        return ""
    host = str(parsed.netloc or "").strip().lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def is_allowed_citation_url(url: str) -> bool:
    normalized = normalize_url(url)
    if not is_valid_http_url(normalized):
        return False

    host = _hostname(normalized)
    if not host:
        return False

    if host in _DENYLIST_DOMAINS:
        return False

    lowered = normalized.lower()
    if any(lowered.endswith(ext) for ext in _IMAGE_SUFFIXES):
        return False

    if "github.com/sponsors" in lowered or "buymeacoffee.com" in lowered:
        return False

    return True


def _drop_line(line: str) -> bool:
    lowered = str(line or "").strip().lower()
    if not lowered:
        return True

    if _MD_IMAGE_ONLY_RE.match(lowered):
        return True

    if any(marker in lowered for marker in _BADGE_OR_PROMO_MARKERS):
        return True

    if lowered.startswith("<img") or lowered.startswith("<picture") or lowered.startswith("<svg"):
        return True

    return False


def sanitize_markdown(raw: str, *, max_len: int = 9000) -> Tuple[str, str, Dict[str, int]]:
    """Remove low-signal markup/badges and return clean markdown + clean text."""
    text = html_lib.unescape(str(raw or ""))
    for pattern in _CLEAN_BLOCK_PATTERNS:
        text = pattern.sub("\n", text)

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    kept: List[str] = []
    dropped = 0
    for raw_line in text.split("\n"):
        line = str(raw_line or "").strip()
        if _drop_line(line):
            dropped += 1
            continue

        # strip inline image markdown
        line = _MD_IMAGE_INLINE_RE.sub(" ", line)

        # markdown links: keep anchor text, drop denylist targets
        def _replace_md_link(match: re.Match[str]) -> str:
            label = str(match.group(1) or "").strip()
            url = str(match.group(2) or "").strip()
            return label if is_allowed_citation_url(url) else ""

        line = _MD_LINK_RE.sub(_replace_md_link, line)

        # raw urls: keep only allowed evidence URLs
        line = _URL_RE.sub(lambda m: m.group(0) if is_allowed_citation_url(m.group(0)) else "", line)

        line = _HTML_TAG_RE.sub(" ", line)
        line = re.sub(r"`{1,3}", "", line)
        line = re.sub(r"\s+", " ", line).strip(" -|\t")
        if not line:
            dropped += 1
            continue

        kept.append(line)

    clean_md = "\n".join(kept)
    clean_md = re.sub(r"\n{3,}", "\n\n", clean_md).strip()
    clean_text = re.sub(r"\s+", " ", clean_md).strip()

    if len(clean_md) > max_len:
        clean_md = clean_md[: max_len - 3].rstrip() + "..."
    if len(clean_text) > max_len:
        clean_text = clean_text[: max_len - 3].rstrip() + "..."

    return clean_md, clean_text, {
        "raw_len": len(str(raw or "")),
        "clean_len": len(clean_text),
        "kept_lines": len(kept),
        "dropped_lines": dropped,
    }


def contains_html_like_tokens(text: str) -> bool:
    lowered = str(text or "").lower()
    tokens = ("<p", "<img", "<div", "align=", "<picture", "<svg")
    return any(token in lowered for token in tokens)
