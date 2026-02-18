"""Content sanitization, URL canonicalization, and link classification for v2 pipeline."""

from __future__ import annotations

import html as html_lib
import re
from typing import Dict, List, Literal, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


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
    "0.0.0.0",
}
_IMAGE_SUFFIXES = (".svg", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico")
_ASSET_SUFFIXES = _IMAGE_SUFFIXES + (".mp4", ".webm", ".mov", ".m4v", ".wav", ".mp3")
_QUERY_ALLOWLIST = {
    "id",
    "v",
    "p",
    "q",
    "t",
    "page",
    "tab",
    "model",
    "dataset",
    "paper",
    "release",
    "version",
}
_TOOLING_DOMAINS = {
    "bun.sh",
    "www.bun.sh",
    "aistudio.google.com",
    "api.openai.com",
    "platform.openai.com",
    "api.deepseek.com",
    "openrouter.ai",
    "www.openrouter.ai",
}
LinkCategory = Literal["evidence", "tooling", "asset", "blocked", "invalid"]


def _is_local_host(host: str) -> bool:
    value = str(host or "").strip().lower()
    if not value:
        return False
    if value in {"localhost", "127.0.0.1", "0.0.0.0"}:
        return True
    return value.endswith(".localhost")


def _strip_url_tail_noise(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.strip("\"'`")
    text = text.replace("\\\"", "\"").replace("\\'", "'")
    text = re.sub(r"\s+", "", text)

    # remove json-ish trailing fragments and punctuation noise
    while text and text[-1] in {"\"", "'", "`", ")", "]", "}", ",", ";", ".", ">", ":"}:
        text = text[:-1].rstrip()
    text = re.sub(r"(?:[\]\}\"'`]+)$", "", text)
    text = re.sub(r"(?:\\u[0-9a-fA-F]{4})+$", "", text)
    return text.strip()


def canonicalize_url(url: str) -> str:
    value = _strip_url_tail_noise(url)
    if not value:
        return ""
    if not value.startswith(("http://", "https://")):
        return value

    try:
        parsed = urlparse(value)
    except ValueError:
        return value

    scheme = str(parsed.scheme or "").lower()
    if scheme not in {"http", "https"}:
        return value

    host = str(parsed.netloc or "").strip().lower()
    if host.endswith(":80"):
        host = host[:-3]
    elif host.endswith(":443"):
        host = host[:-4]
    if host.startswith("www."):
        host = host[4:]

    path = re.sub(r"/{2,}", "/", str(parsed.path or ""))
    if path != "/" and path.endswith("/"):
        path = path[:-1]

    query_pairs = []
    for key, val in parse_qsl(str(parsed.query or ""), keep_blank_values=False):
        key_clean = str(key or "").strip().lower()
        if key_clean in _QUERY_ALLOWLIST:
            query_pairs.append((key_clean, str(val or "").strip()))
    query = urlencode(query_pairs, doseq=False)

    normalized = urlunparse(("https", host, path, "", query, ""))
    return _strip_url_tail_noise(normalized)


def normalize_url(url: str) -> str:
    """Backward-compatible alias; prefer `canonicalize_url`."""
    return canonicalize_url(url)


def is_valid_http_url(url: str) -> bool:
    normalized = canonicalize_url(url)
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
        parsed = urlparse(canonicalize_url(url))
    except ValueError:
        return ""
    host = str(parsed.hostname or parsed.netloc or "").strip().lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def classify_link(url: str) -> LinkCategory:
    normalized = canonicalize_url(url)
    if not is_valid_http_url(normalized):
        return "invalid"

    host = _hostname(normalized)
    lowered = normalized.lower()
    if not host:
        return "invalid"

    if host in _DENYLIST_DOMAINS or _is_local_host(host):
        return "blocked"
    if "github.com/sponsors" in lowered or "buymeacoffee.com" in lowered:
        return "blocked"
    if "aistudio.google.com/apikey" in lowered or "/apikey" in lowered:
        return "tooling"
    if "/v1/chat/completions" in lowered or "/compatible-mode/" in lowered:
        return "tooling"
    if host.startswith("api.") and "/docs" not in lowered and "/documentation" not in lowered:
        return "tooling"
    if host in _TOOLING_DOMAINS or host.startswith("api.") and ("openai.com" in host or "deepseek.com" in host):
        return "tooling"
    if any(lowered.endswith(ext) for ext in _ASSET_SUFFIXES):
        return "asset"
    return "evidence"


def is_allowed_citation_url(url: str) -> bool:
    return classify_link(url) == "evidence"


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
            category = classify_link(url)
            return label if category in {"evidence", "asset"} else ""

        line = _MD_LINK_RE.sub(_replace_md_link, line)

        # raw urls: keep only allowed evidence URLs
        line = _URL_RE.sub(
            lambda m: canonicalize_url(m.group(0)) if classify_link(m.group(0)) == "evidence" else "",
            line,
        )

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
