"""Report, thumbnail, and package export helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Iterable, List, Sequence
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


def generate_onepager(items: Sequence[object], citations: Sequence[Citation], out_dir: str | Path) -> str:
    """Generate one-page markdown report with item ranking and evidence."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "onepager.md"

    ranked_items = [_extract_item(item) for item in list(items or [])]
    citation_list = _extract_citations(items, citations)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    lines: List[str] = [
        "# One Pager",
        "",
        f"- GeneratedAt(UTC): `{generated_at}`",
        f"- CandidateCount: `{len(ranked_items)}`",
        f"- CitationCount: `{len(citation_list)}`",
        "",
        "## Top Picks",
        "",
    ]

    for idx, item in enumerate(ranked_items, start=1):
        credibility = str((item.metadata or {}).get("credibility") or "unknown")
        lines.extend(
            [
                f"### {idx}. {item.title}",
                f"- Source: `{item.source}`",
                f"- Tier: `{item.tier}`",
                f"- Credibility: `{credibility}`",
                f"- URL: {item.url or 'N/A'}",
                f"- Summary: {str(item.body_md or '').strip()[:280] or 'N/A'}",
                "",
            ]
        )

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
