"""Input ingestion helpers for Phase5: arXiv URL / uploaded PDF."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

import httpx

from scrapers.arxiv_scraper import ArxivScraper


@dataclass
class InputSeedBundle:
    topic: str
    search_results: List[Dict[str, Any]]
    notes: List[str]


def _compact(text: Any, max_len: int = 220) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3] + "..."


def _extract_arxiv_id(url: str) -> Optional[str]:
    value = str(url or "").strip()
    if not value:
        return None
    # Supports:
    # - https://arxiv.org/abs/2401.12345
    # - https://arxiv.org/pdf/2401.12345.pdf
    # - 2401.12345
    patterns = [
        r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})(?:v\d+)?(?:\.pdf)?",
        r"^([0-9]{4}\.[0-9]{4,5})(?:v\d+)?$",
    ]
    for pattern in patterns:
        match = re.search(pattern, value, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _extract_pdf_text_from_bytes(pdf_bytes: bytes, *, max_pages: int, max_chars: int) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        return ""

    from io import BytesIO

    try:
        reader = PdfReader(BytesIO(pdf_bytes))
    except Exception:
        return ""

    chunks: List[str] = []
    remaining = max(2000, int(max_chars))
    for page in reader.pages[: max(1, int(max_pages))]:
        if remaining <= 0:
            break
        try:
            text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
        except Exception:
            text = ""
        if not text:
            continue
        if len(text) > remaining:
            text = text[:remaining]
        chunks.append(text)
        remaining -= len(text)
    return "\n".join(chunks).strip()


def _guess_title_from_text(text: str, *, fallback: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in str(text or "").splitlines()]
    lines = [line for line in lines if 8 <= len(line) <= 180]
    if lines:
        ranked = sorted(lines[:18], key=lambda line: (-(len(line.split())), len(line)))
        title = ranked[0]
        if title:
            return _compact(title, max_len=120)
    return _compact(fallback, max_len=120)


def _build_seed_result(
    *,
    seed_id: str,
    title: str,
    content: str,
    url: Optional[str],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "id": seed_id,
        "source": "user_document",
        "title": title,
        "content": content,
        "url": url or "",
        "metadata": metadata,
    }


async def ingest_arxiv_url(
    *,
    arxiv_url: str,
    max_pdf_pages: int = 12,
    max_chars: int = 24000,
) -> InputSeedBundle:
    arxiv_id = _extract_arxiv_id(arxiv_url)
    if not arxiv_id:
        raise ValueError("Invalid arXiv URL/ID. Expected arxiv.org/abs/<id> or arxiv.org/pdf/<id>.pdf")

    async with ArxivScraper() as scraper:
        paper = await scraper.get_details(arxiv_id)
    if not paper:
        raise ValueError(f"Unable to fetch arXiv metadata for id={arxiv_id}")

    title = _compact(paper.title or f"arXiv {arxiv_id}", max_len=140)
    abstract = _compact(paper.abstract or "", max_len=5000)
    authors = ", ".join([item.name for item in (paper.authors or [])[:8]])
    categories = ", ".join(list(paper.categories or [])[:6])

    content_parts = [
        f"Title: {title}",
        f"Abstract: {abstract}",
        f"Authors: {authors}" if authors else "",
        f"Categories: {categories}" if categories else "",
    ]
    notes: List[str] = [f"arXiv metadata loaded: {arxiv_id}"]

    pdf_text = ""
    pdf_url = str(getattr(paper, "pdf_url", "") or "").strip()
    if pdf_url:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
                response = await client.get(pdf_url, follow_redirects=True)
                response.raise_for_status()
                pdf_bytes = response.content
            pdf_text = await asyncio.to_thread(
                _extract_pdf_text_from_bytes,
                pdf_bytes,
                max_pages=max_pdf_pages,
                max_chars=max_chars,
            )
            if pdf_text:
                notes.append(f"arXiv PDF extracted: {max_pdf_pages} pages budget")
        except Exception as exc:
            notes.append(f"arXiv PDF extraction skipped: {exc}")
    if pdf_text:
        content_parts.append(f"PDF Excerpt:\n{_compact(pdf_text, max_len=max_chars)}")

    topic = _guess_title_from_text(title or abstract, fallback=f"arXiv {arxiv_id}")
    content = "\n\n".join([item for item in content_parts if item]).strip()
    seed = _build_seed_result(
        seed_id=f"user_arxiv_{arxiv_id}",
        title=title,
        content=content,
        url=str(paper.url or arxiv_url),
        metadata={
            "input_mode": "arxiv_url",
            "arxiv_id": arxiv_id,
            "pdf_url": pdf_url,
            "authors": [item.name for item in (paper.authors or [])],
            "categories": list(paper.categories or []),
        },
    )
    return InputSeedBundle(topic=topic, search_results=[seed], notes=notes)


async def ingest_uploaded_pdf(
    *,
    file_path: Path,
    original_name: str,
    max_pdf_pages: int = 14,
    max_chars: int = 26000,
) -> InputSeedBundle:
    raw = await asyncio.to_thread(file_path.read_bytes)
    text = await asyncio.to_thread(
        _extract_pdf_text_from_bytes,
        raw,
        max_pages=max_pdf_pages,
        max_chars=max_chars,
    )
    if not text:
        raise ValueError("PDF text extraction failed. Please provide a text-selectable PDF.")

    title = _guess_title_from_text(text, fallback=original_name)
    topic = _compact(title, max_len=120)
    snippet = _compact(text, max_len=max_chars)
    seed = _build_seed_result(
        seed_id=f"user_pdf_{file_path.stem}",
        title=title,
        content=f"Document: {original_name}\n\nPDF Excerpt:\n{snippet}",
        url=f"file:{file_path}",
        metadata={
            "input_mode": "pdf_upload",
            "file_path": str(file_path),
            "file_name": original_name,
        },
    )
    return InputSeedBundle(
        topic=topic,
        search_results=[seed],
        notes=[f"Uploaded PDF extracted from {original_name}"],
    )

