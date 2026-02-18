"""Report, thumbnail, and package export helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Iterable, List, Sequence
from urllib.parse import urlparse
import zipfile

from core import Artifact, ArtifactType, Citation, NormalizedItem, RankedItem
from pipeline_v2.sanitize import is_allowed_citation_url


_MAX_BULLETS_PER_PICK = 6
_MAX_BULLET_BYTES = 90


def _safe_slug(value: str, *, max_len: int = 72) -> str:
    text = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value or "").strip().lower())
    text = re.sub(r"-{2,}", "-", text).strip("-")
    if not text:
        text = "artifact"
    return text[:max_len].strip("-") or "artifact"


def _extract_item(item: object) -> NormalizedItem:
    if isinstance(item, RankedItem):
        return item.item
    if isinstance(item, NormalizedItem):
        return item
    raise TypeError("items must contain RankedItem or NormalizedItem")


def _extract_citations(items: Sequence[object], citations: Sequence[Citation] | None = None) -> List[Citation]:
    merged: List[Citation] = []
    seen = set()
    for item in items:
        normalized = _extract_item(item)
        for citation in normalized.citations:
            key = str(citation.url or "").strip() or f"{citation.title}|{citation.snippet}"
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(citation)
    for citation in list(citations or []):
        key = str(citation.url or "").strip() or f"{citation.title}|{citation.snippet}"
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(citation)
    return merged


def _compact_text(value: str, max_len: int = 220) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _domain(url: str) -> str:
    host = str(urlparse(str(url or "")).netloc or "").strip().lower()
    return host or "unknown"


def _quality_metrics(item: NormalizedItem) -> dict:
    metadata = dict(item.metadata or {})
    return {
        "body_len": int(float(metadata.get("body_len", len(item.body_md or "")) or 0)),
        "citation_count": int(float(metadata.get("citation_count", len(item.citations)) or 0)),
        "published_recency": metadata.get("published_recency"),
        "link_count": int(float(metadata.get("link_count", 0) or 0)),
    }


def _fact_bullets(item: NormalizedItem) -> List[str]:
    metadata = dict(item.metadata or {})
    metrics = _quality_metrics(item)
    facts = dict(metadata.get("facts") or {})
    signals = dict(metadata.get("quality_signals") or {})

    def _trim_bytes(text: str, limit: int = _MAX_BULLET_BYTES) -> str:
        value = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(value.encode("utf-8")) <= limit:
            return value
        shrunk = value
        while shrunk and len((shrunk + "...").encode("utf-8")) > limit:
            shrunk = shrunk[:-1]
        return (shrunk.rstrip() + "...") if shrunk else value[:20]

    what = str(facts.get("what_it_is") or item.title or "")
    why_now = str(facts.get("why_now") or "")
    if not why_now:
        recency = signals.get("update_recency_days")
        if recency not in (None, "", "unknown"):
            why_now = f"更新 {float(recency):.1f} 天内，窗口期明确。"
        else:
            why_now = "热度抬升，讨论集中在落地能力。"

    how_points = [str(value or "") for value in list(facts.get("how_it_works") or []) if str(value or "").strip()]
    while len(how_points) < 2:
        how_points.append("给出可执行流程，减少试错成本。")

    proof = ""
    points = int(float(metadata.get("points", 0) or 0))
    comments = int(float(metadata.get("comment_count", 0) or 0))
    stars = int(float(metadata.get("stars", 0) or 0))
    downloads = int(float(metadata.get("downloads", 0) or 0))
    evidence_links = int(float(signals.get("evidence_links_quality", 0) or 0))
    update_days = signals.get("update_recency_days")
    if points > 0 or comments > 0:
        proof = f"HN {points} 分 / {comments} 评，讨论已成形。"
    elif stars > 0:
        proof = f"GitHub {stars} stars，开发者采用信号明确。"
    elif downloads > 0:
        proof = f"HF 下载 {downloads}，需求侧已验证。"
    elif update_days not in (None, "", "unknown"):
        proof = f"最近更新 {float(update_days):.1f} 天内。"
    else:
        proof = f"证据链 {evidence_links} 条，可回查源链接。"

    cta = "先跑官方示例，再替换你的真实任务。"

    bullets = [
        f"WHAT｜{_trim_bytes(what)}",
        f"WHY NOW｜{_trim_bytes(why_now)}",
        f"HOW｜{_trim_bytes(how_points[0])}",
        f"HOW｜{_trim_bytes(how_points[1])}",
        f"PROOF｜{_trim_bytes(proof)}",
    ]
    if metrics["citation_count"] > 0:
        bullets.append(f"CTA｜{_trim_bytes(cta)}")

    normalized: List[str] = []
    seen = set()
    for bullet in bullets:
        key = bullet.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        normalized.append(_trim_bytes(bullet))
        if len(normalized) >= _MAX_BULLETS_PER_PICK:
            break
    return normalized


def _citation_bullets(item: NormalizedItem) -> List[str]:
    lines: List[str] = []
    for citation in list(item.citations or [])[:2]:
        snippet = _compact_text(citation.snippet or citation.title or "", max_len=160)
        url = str(citation.url or "").strip()
        if url and not is_allowed_citation_url(url):
            continue
        if not snippet and not url:
            continue
        lines.append(f"{snippet or '引用片段缺失'} ({url or 'N/A'})")
    if not lines:
        return ["无引用"]
    return lines


def _relevance(item: object) -> float | None:
    if isinstance(item, RankedItem):
        try:
            return float(item.relevance_score)
        except Exception:
            return None
    return None


def _why_ranked(item: object, normalized: NormalizedItem) -> str:
    relevance = _relevance(item)
    signals = dict((normalized.metadata or {}).get("quality_signals") or {})
    parts: List[str] = []
    if relevance is not None:
        parts.append(f"rel={float(relevance):.2f}")
    recency = signals.get("update_recency_days")
    if recency not in (None, "", "unknown"):
        parts.append(f"更新{float(recency):.1f}d")
    evidence = int(float(signals.get("evidence_links_quality", 0) or 0))
    if evidence > 0:
        parts.append(f"证据链{evidence}")
    if bool(signals.get("has_quickstart")):
        parts.append("含Quickstart")
    return " · ".join(parts[:3]) if parts else "rel=N/A"


def generate_onepager(
    items: Sequence[object],
    citations: Sequence[Citation],
    out_dir: str | Path,
    *,
    run_context: dict | None = None,
) -> str:
    """Generate one-page markdown report with item ranking and evidence."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "onepager.md"

    ranked_items = list(items or [])
    normalized_items = [_extract_item(item) for item in ranked_items]
    citation_list = _extract_citations(items, citations)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    context = dict(run_context or {})
    data_mode = str(context.get("data_mode") or "live").strip().lower() or "live"
    connector_stats = dict(context.get("connector_stats") or {})
    extraction_stats = dict(context.get("extraction_stats") or {})
    connector_stats_line = json.dumps(connector_stats, ensure_ascii=False, sort_keys=True)
    extraction_stats_line = json.dumps(extraction_stats, ensure_ascii=False, sort_keys=True)

    lines: List[str] = [
        "# One Pager",
        "",
        f"- DataMode: `{data_mode}`",
        f"- ConnectorStats: `{connector_stats_line}`",
        f"- ExtractionStats: `{extraction_stats_line}`",
        f"- GeneratedAt(UTC): `{generated_at}`",
        f"- CandidateCount: `{len(normalized_items)}`",
        f"- CitationCount: `{len(citation_list)}`",
        "",
        "## Top Picks",
        "",
    ]

    for idx, item in enumerate(normalized_items, start=1):
        payload = ranked_items[idx - 1]
        credibility = str((item.metadata or {}).get("credibility") or "unknown")
        metrics = _quality_metrics(item)
        fact_bullets = _fact_bullets(item)
        citation_bullets = _citation_bullets(item)
        relevance_value = _relevance(payload)

        lines.extend(
            [
                f"### {idx}. {item.title}",
                f"- Source URL: {item.url or 'N/A'}",
                f"- Source Domain: `{_domain(item.url)}`",
                f"- Tier/Credibility: `{item.tier}` / `{credibility}`",
                f"- Topic Relevance: `{relevance_value:.2f}`" if relevance_value is not None else "- Topic Relevance: `N/A`",
                f"- Why Ranked: `{_why_ranked(payload, item)}`",
                (
                    "- Quality Metrics: "
                    f"body_len={metrics['body_len']}, "
                    f"citation_count={metrics['citation_count']}, "
                    f"published_recency={metrics['published_recency']}, "
                    f"link_count={metrics['link_count']}"
                ),
                "",
                "#### Compact Brief",
            ]
        )
        for bullet in fact_bullets:
            lines.append(f"- {bullet}")
        lines.append("")
        lines.append("#### Citations")
        for bullet in citation_bullets:
            lines.append(f"- {bullet}")
        lines.append("")

    lines.extend(
        [
            "## Evidence",
            "",
        ]
    )

    for idx, citation in enumerate(citation_list, start=1):
        lines.extend(
            [
                f"{idx}. {citation.title or 'Untitled'}",
                f"   - source: `{citation.source or 'unknown'}`",
                f"   - url: {citation.url or 'N/A'}",
                f"   - snippet: {citation.snippet or 'N/A'}",
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def generate_thumbnail(title: str, keywords: Iterable[str], style: dict, out_dir: str | Path) -> str:
    """Generate deterministic SVG thumbnail placeholder for packaging."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fg = str((style or {}).get("fg") or "#111111")
    bg = str((style or {}).get("bg") or "#f0f4f8")
    accent = str((style or {}).get("accent") or "#0f766e")
    text_title = str(title or "Research Digest").strip() or "Research Digest"
    subtitle = ", ".join([str(item).strip() for item in list(keywords or []) if str(item).strip()][:4]) or "top picks"

    slug = _safe_slug(text_title, max_len=48)
    path = output_dir / f"thumbnail_{slug}.svg"
    svg = "\n".join(
        [
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1080\" height=\"1920\" viewBox=\"0 0 1080 1920\">",
            f"  <rect width=\"1080\" height=\"1920\" fill=\"{bg}\"/>",
            f"  <rect x=\"64\" y=\"64\" width=\"952\" height=\"1792\" rx=\"42\" fill=\"{accent}\" fill-opacity=\"0.12\"/>",
            f"  <text x=\"96\" y=\"220\" font-size=\"56\" font-family=\"Arial, sans-serif\" fill=\"{fg}\">Tech Brief</text>",
            f"  <text x=\"96\" y=\"360\" font-size=\"88\" font-family=\"Arial, sans-serif\" fill=\"{fg}\">{text_title[:32]}</text>",
            f"  <text x=\"96\" y=\"480\" font-size=\"42\" font-family=\"Arial, sans-serif\" fill=\"{fg}\">{subtitle[:56]}</text>",
            "</svg>",
            "",
        ]
    )
    path.write_text(svg, encoding="utf-8")
    return str(path)


def export_package(
    out_dir: str | Path,
    artifacts: Sequence[Artifact],
    package_name: str = "run_export",
) -> str:
    """Package artifacts into a zip for export/download."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    package_path = output_dir / f"{_safe_slug(package_name)}.zip"

    manifest = {
        "package_name": package_name,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "artifact_count": len(list(artifacts or [])),
        "artifacts": [
            {
                "type": str(item.type.value if isinstance(item.type, ArtifactType) else item.type),
                "path": item.path,
                "metadata": dict(item.metadata or {}),
            }
            for item in list(artifacts or [])
        ],
    }

    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        for artifact in list(artifacts or []):
            src = Path(str(artifact.path or ""))
            if not src.exists() or not src.is_file():
                continue
            arcname = f"{artifact.type.value}/{src.name}"
            zf.write(src, arcname=arcname)

    return str(package_path)
