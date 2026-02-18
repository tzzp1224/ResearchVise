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


def _summary_paragraphs(body: str) -> List[str]:
    text = re.sub(r"\s+", " ", str(body or "")).strip()
    chunks = [piece.strip() for piece in re.split(r"(?<=[.!?。！？])\s+", text) if piece.strip()]
    if not chunks:
        return ["暂无可用正文摘要。"]
    if len(chunks) == 1:
        return [chunks[0]]
    first = " ".join(chunks[:2]).strip()
    second = " ".join(chunks[2:4]).strip() if len(chunks) > 2 else ""
    return [first] + ([second] if second else [])


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
    metrics = _quality_metrics(item)
    metadata = dict(item.metadata or {})
    bullets: List[str] = []
    facts = dict(metadata.get("facts") or {})
    for entry in list(facts.get("how_it_works") or [])[:3]:
        line = _compact_text(str(entry or ""), max_len=180)
        if line:
            bullets.append(line)
    for entry in list(facts.get("proof") or [])[:2]:
        line = _compact_text(str(entry or ""), max_len=180)
        if line and line not in bullets:
            bullets.append(line)

    stars = int(float(metadata.get("stars", 0) or 0))
    forks = int(float(metadata.get("forks", 0) or 0))
    downloads = int(float(metadata.get("downloads", 0) or 0))
    points = int(float(metadata.get("points", 0) or 0))
    comments = int(float(metadata.get("comment_count", 0) or 0))
    if stars > 0:
        bullets.append(f"GitHub stars `{stars}`，forks `{forks}`，显示社区采用强度。")
    if downloads > 0:
        bullets.append(f"Hugging Face downloads `{downloads}`，表明近期使用热度。")
    if points > 0 or comments > 0:
        bullets.append(f"HN 讨论热度 `points={points}`、`comments={comments}`。")
    if metrics["published_recency"] is not None:
        bullets.append(f"发布时间距今约 `{metrics['published_recency']}` 天。")
    if metrics["citation_count"] > 0:
        bullets.append(f"引用密度：`citation_count={metrics['citation_count']}`，`body_len={metrics['body_len']}`。")

    for paragraph in _summary_paragraphs(item.body_md):
        line = _compact_text(paragraph, max_len=180)
        if line and line not in bullets:
            bullets.append(line)
        if len(bullets) >= 4:
            break

    deduped: List[str] = []
    seen = set()
    for bullet in bullets:
        key = bullet.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(bullet)
    if len(deduped) < 2:
        fallback = _compact_text(item.body_md, max_len=170) or _compact_text(item.title, max_len=170)
        while len(deduped) < 2:
            deduped.append(fallback)
    return deduped[:4]


def _citation_bullets(item: NormalizedItem) -> List[str]:
    lines: List[str] = []
    for citation in list(item.citations or [])[:2]:
        snippet = _compact_text(citation.snippet or citation.title or "", max_len=160)
        url = str(citation.url or "").strip()
        if not snippet and not url:
            continue
        lines.append(f"{snippet or '引用片段缺失'} ({url or 'N/A'})")
    if not lines:
        return ["无引用"]
    return lines


def _rank_reasons(item: object) -> str:
    if isinstance(item, RankedItem):
        reasons = [str(reason) for reason in list(item.reasons or [])[:6] if str(reason).strip()]
        if reasons:
            return " | ".join(reasons)
    return "无"


def _relevance(item: object) -> float | None:
    if isinstance(item, RankedItem):
        try:
            return float(item.relevance_score)
        except Exception:
            return None
    return None


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
                (
                    "- Quality Metrics: "
                    f"body_len={metrics['body_len']}, "
                    f"citation_count={metrics['citation_count']}, "
                    f"published_recency={metrics['published_recency']}, "
                    f"link_count={metrics['link_count']}"
                ),
                f"- Ranking Reasons: `{_rank_reasons(payload)}`",
                "",
                "#### Facts",
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
