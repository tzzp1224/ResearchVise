"""
Deep content enrichment for search results.

Goals:
- Pull deeper technical evidence from papers/repos/model cards.
- Keep runtime bounded and resilient (best-effort, skip-on-failure).
"""

from __future__ import annotations

import asyncio
from io import BytesIO
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from config import get_settings


logger = logging.getLogger(__name__)

_TEXT_KEYWORDS = [
    "architecture",
    "algorithm",
    "training",
    "benchmark",
    "latency",
    "throughput",
    "ablation",
    "deployment",
    "evaluation",
    "complexity",
    "memory",
    "cost",
    "架构",
    "训练",
    "基准",
    "延迟",
    "吞吐",
    "部署",
    "复杂度",
    "成本",
    "method",
    "methodology",
    "formula",
    "equation",
    "implementation",
    "ablation",
    "hardware",
    "memory",
    "optimizer",
    "loss",
    "伪代码",
    "公式",
    "实现",
    "消融",
    "硬件",
    "显存",
    "优化器",
    "损失",
]

_CODE_EXTENSIONS = (
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".go",
    ".rs",
    ".java",
    ".cpp",
    ".cc",
    ".c",
    ".cs",
    ".swift",
    ".kt",
)

_CODE_HINTS = ("train", "model", "inference", "serve", "deploy", "agent", "pipeline", "retrieval", "rag")

_DEPENDENCY_FILE_HINTS = (
    "requirements.txt",
    "requirements-dev.txt",
    "pyproject.toml",
    "environment.yml",
    "environment.yaml",
    "poetry.lock",
    "pdm.lock",
)


def _clean_text(value: Any, *, max_chars: int = 5000) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def _split_sentences(value: str) -> List[str]:
    text = _clean_text(value, max_chars=20000)
    if not text:
        return []
    parts = re.split(r"[。！？!?；;\n]+", text)
    return [p.strip() for p in parts if p and p.strip()]


def _snippet_score(sentence: str) -> int:
    lowered = sentence.lower()
    score = 0
    if re.search(r"\d", sentence):
        score += 3
    for token in _TEXT_KEYWORDS:
        if token in lowered:
            score += 2
    if ":" in sentence or "：" in sentence or "->" in sentence or "→" in sentence:
        score += 1
    return score


def _select_snippets(text: str, *, max_items: int = 6, max_chars: int = 1800) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return ""

    ranked = sorted(sentences, key=_snippet_score, reverse=True)
    selected: List[str] = []
    used = set()
    total_chars = 0
    for sent in ranked:
        cleaned = _clean_text(sent, max_chars=260)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in used:
            continue
        projected = total_chars + len(cleaned)
        if selected and projected > max_chars:
            break
        selected.append(cleaned)
        used.add(key)
        total_chars = projected
        if len(selected) >= max_items:
            break

    if not selected:
        selected = [_clean_text(sentences[0], max_chars=min(260, max_chars))]
    return "\n".join([f"- {item}" for item in selected if item])


def _extract_targeted_technical_text(text: str, *, max_chars: int) -> str:
    cleaned = _clean_text(text, max_chars=max_chars * 3)
    if not cleaned:
        return ""
    blocks = [segment.strip() for segment in re.split(r"\n{2,}|(?<=\.)\s{2,}", cleaned) if segment.strip()]
    if not blocks:
        return cleaned[:max_chars]

    ranked: List[tuple[int, str]] = []
    for block in blocks:
        score = _snippet_score(block)
        lowered = block.lower()
        if any(token in lowered for token in ("method", "approach", "implementation", "algorithm", "公式", "方法", "实现")):
            score += 3
        if any(token in lowered for token in ("table", "benchmark", "ablation", "latency", "throughput", "memory", "显存", "吞吐")):
            score += 3
        if any(token in block for token in ("=", "≈", "∑", "∂", "O(", "Θ(", "L=", "loss")):
            score += 2
        if re.search(r"\d", block):
            score += 1
        ranked.append((score, block))

    ranked.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    selected: List[str] = []
    total = 0
    for score, block in ranked:
        if score <= 0:
            continue
        chunk = _clean_text(block, max_chars=720)
        if not chunk:
            continue
        projected = total + len(chunk)
        if selected and projected > max_chars:
            break
        selected.append(chunk)
        total = projected
        if len(selected) >= 18:
            break
    if not selected:
        return cleaned[:max_chars]
    return "\n".join(selected)


def _merge_content(original: str, appendix_title: str, appendix_body: str, *, max_chars: int) -> str:
    base = _clean_text(original, max_chars=max_chars)
    body = _clean_text(appendix_body, max_chars=max_chars)
    if not body:
        return base
    merged = f"{base}\n\n[{appendix_title}]\n{body}".strip()
    return _clean_text(merged, max_chars=max_chars)


def _item_source(item: Dict[str, Any]) -> str:
    return str(item.get("source", "")).strip().lower()


def _item_score(item: Dict[str, Any]) -> float:
    metadata = dict(item.get("metadata", {}) or {})
    source = _item_source(item)
    if source in {"arxiv", "semantic_scholar", "openreview", "arxiv_rss"}:
        return float(metadata.get("citation_count") or 0)
    if source == "github":
        return float(metadata.get("stars") or 0)
    if source == "huggingface":
        return float(metadata.get("downloads") or 0)
    return 0.0


def _arxiv_pdf_url(item: Dict[str, Any]) -> Optional[str]:
    metadata = dict(item.get("metadata", {}) or {})
    pdf_url = str(metadata.get("pdf_url") or "").strip()
    if pdf_url:
        return pdf_url
    url = str(item.get("url") or "").strip()
    if "arxiv.org/abs/" in url:
        arxiv_id = url.rstrip("/").split("/")[-1]
        if arxiv_id:
            return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    return None


def _semantic_pdf_url(item: Dict[str, Any]) -> Optional[str]:
    metadata = dict(item.get("metadata", {}) or {})
    pdf_url = str(metadata.get("pdf_url") or "").strip()
    return pdf_url or None


def _openreview_pdf_url(item: Dict[str, Any]) -> Optional[str]:
    metadata = dict(item.get("metadata", {}) or {})
    pdf_url = str(metadata.get("pdf_url") or "").strip()
    if pdf_url:
        return pdf_url
    url = str(item.get("url") or "").strip()
    match = re.search(r"[?&]id=([A-Za-z0-9._-]+)", url)
    if match:
        return f"https://openreview.net/pdf?id={match.group(1)}"
    return None


def _github_owner_repo(item: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    metadata = dict(item.get("metadata", {}) or {})
    full_name = str(metadata.get("repo_full_name") or "").strip()
    if "/" in full_name:
        owner, repo = full_name.split("/", 1)
        if owner and repo:
            return owner, repo

    url = str(item.get("url") or "").strip()
    try:
        parsed = urlparse(url)
    except Exception:
        return None
    if "github.com" not in parsed.netloc:
        return None
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return None


def _huggingface_repo(item: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    metadata = dict(item.get("metadata", {}) or {})
    repo_id = str(metadata.get("repo_id") or "").strip()
    if repo_id:
        repo_type = str(metadata.get("type") or "model").strip().lower()
        return repo_type, repo_id

    url = str(item.get("url") or "").strip()
    try:
        parsed = urlparse(url)
    except Exception:
        return None
    if "huggingface.co" not in parsed.netloc:
        return None
    parts = [p for p in parsed.path.split("/") if p]
    if not parts:
        return None
    if parts[0] == "datasets" and len(parts) >= 3:
        return "dataset", f"{parts[1]}/{parts[2]}"
    if len(parts) >= 2:
        return "model", f"{parts[0]}/{parts[1]}"
    return None


async def _http_get_text(client: httpx.AsyncClient, url: str) -> str:
    try:
        resp = await client.get(url, follow_redirects=True)
        resp.raise_for_status()
        return resp.text
    except Exception:
        return ""


async def _http_get_bytes(client: httpx.AsyncClient, url: str) -> bytes:
    try:
        resp = await client.get(url, follow_redirects=True)
        resp.raise_for_status()
        return bytes(resp.content or b"")
    except Exception:
        return b""


def _selected_pdf_page_indices(total_pages: int, max_pages: int) -> List[int]:
    total = max(0, int(total_pages))
    limit = max(1, int(max_pages))
    if total <= limit:
        return list(range(total))

    # Bias sampling toward middle and later sections where Methodology/Experiments
    # are commonly located, while still preserving intro/conclusion context.
    anchors: List[int] = []
    for ratio in (0.0, 0.02, 0.05, 0.10, 0.18, 0.28, 0.38, 0.50, 0.62, 0.72, 0.82, 0.90, 0.96, 0.99):
        idx = int(round((total - 1) * ratio))
        anchors.append(idx)

    method_window_start = max(0, int(total * 0.32))
    method_window_end = min(total - 1, int(total * 0.92))
    method_window_width = max(1, method_window_end - method_window_start + 1)
    method_step = max(1, method_window_width // max(3, limit // 2))
    anchors.extend(range(method_window_start, method_window_end + 1, method_step))

    deduped: List[int] = []
    seen = set()
    for idx in anchors:
        if idx < 0 or idx >= total or idx in seen:
            continue
        seen.add(idx)
        deduped.append(idx)
        if len(deduped) >= limit:
            break

    if len(deduped) < limit:
        step = max(1, total // max(1, limit))
        for idx in range(0, total, step):
            if idx in seen:
                continue
            deduped.append(idx)
            seen.add(idx)
            if len(deduped) >= limit:
                break
    return sorted(deduped)[:limit]


def _extract_pdf_text(pdf_bytes: bytes, *, max_pages: int, max_chars: int) -> str:
    if not pdf_bytes:
        return ""
    try:
        from pypdf import PdfReader
    except Exception:
        return ""

    try:
        reader = PdfReader(BytesIO(pdf_bytes))
    except Exception:
        return ""

    pages = []
    page_indices = _selected_pdf_page_indices(len(reader.pages), max_pages=max_pages)
    for idx in page_indices:
        if idx < 0 or idx >= len(reader.pages):
            continue
        page = reader.pages[idx]
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text:
            pages.append(text)
    return _clean_text("\n\n".join(pages), max_chars=max_chars)


def _extract_html_text(html: str, *, max_chars: int) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    text_parts: List[str] = []

    abstract_block = soup.select_one("blockquote.abstract") or soup.select_one("[data-selenium-selector='abstract']")
    if abstract_block:
        text_parts.append(abstract_block.get_text(" ", strip=True))

    for selector in ("section", "article", "p"):
        for node in soup.select(selector):
            snippet = node.get_text(" ", strip=True)
            if len(snippet) < 40:
                continue
            text_parts.append(snippet)
            if len(" ".join(text_parts)) > max_chars * 2:
                break
        if len(" ".join(text_parts)) > max_chars * 2:
            break

    return _clean_text(" ".join(text_parts), max_chars=max_chars)


async def _enrich_paper_item(
    item: Dict[str, Any],
    *,
    source: str,
    client: httpx.AsyncClient,
    max_pdf_pages: int,
    max_chars_per_item: int,
) -> Optional[Dict[str, Any]]:
    new_item = dict(item)
    metadata = dict(item.get("metadata", {}) or {})
    if source in {"arxiv", "arxiv_rss"}:
        pdf_url = _arxiv_pdf_url(item)
    elif source == "semantic_scholar":
        pdf_url = _semantic_pdf_url(item)
    elif source == "openreview":
        pdf_url = _openreview_pdf_url(item)
    else:
        pdf_url = None
    url = str(item.get("url") or "").strip()

    pdf_excerpt = ""
    if pdf_url:
        pdf_bytes = await _http_get_bytes(client, pdf_url)
        pdf_excerpt = _extract_pdf_text(
            pdf_bytes,
            max_pages=max_pdf_pages,
            max_chars=max_chars_per_item * 2,
        )
        if pdf_excerpt:
            metadata["pdf_url"] = pdf_url

    html_excerpt = ""
    if not pdf_excerpt and url:
        html = await _http_get_text(client, url)
        html_excerpt = _extract_html_text(html, max_chars=max_chars_per_item * 2)

    technical_text = _extract_targeted_technical_text(
        pdf_excerpt or html_excerpt,
        max_chars=max_chars_per_item * 2,
    )
    merged_snippets = _select_snippets(
        technical_text or pdf_excerpt or html_excerpt,
        max_items=8,
        max_chars=max_chars_per_item,
    )
    if not merged_snippets:
        return None

    new_item["content"] = _merge_content(
        str(item.get("content") or ""),
        "Deep Paper Evidence",
        merged_snippets,
        max_chars=max_chars_per_item * 2,
    )
    metadata["deep_enriched"] = True
    metadata["deep_enrichment_source"] = "pdf" if pdf_excerpt else "html"
    metadata["deep_enrichment_focus"] = "methods+implementation+benchmarks"
    new_item["metadata"] = metadata
    return new_item


def _is_code_candidate(path: str, size: int) -> bool:
    lowered = path.lower()
    if size <= 0 or size > 180000:
        return False
    if any(lowered.endswith(ext) for ext in _CODE_EXTENSIONS):
        return True
    return any(token in lowered for token in _CODE_HINTS)


def _is_dependency_candidate(path: str, size: int) -> bool:
    lowered = str(path or "").lower()
    if size <= 0 or size > 220000:
        return False
    if any(lowered.endswith(hint) for hint in _DEPENDENCY_FILE_HINTS):
        return True
    return any(hint in lowered for hint in _DEPENDENCY_FILE_HINTS)


def _extract_dependency_lines(text: str, *, max_items: int = 14) -> List[str]:
    rows: List[str] = []
    for line in str(text or "").splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#"):
            continue
        if re.search(r"[A-Za-z0-9_.-]+\s*(==|>=|<=|~=|>|<)\s*[A-Za-z0-9_.-]+", cleaned):
            rows.append(cleaned)
        elif "python" in cleaned.lower() and re.search(r"\d", cleaned):
            rows.append(cleaned)
        if len(rows) >= max_items:
            break
    return rows


async def _enrich_github_item(
    item: Dict[str, Any],
    *,
    client: httpx.AsyncClient,
    max_chars_per_item: int,
    github_token: str,
) -> Optional[Dict[str, Any]]:
    owner_repo = _github_owner_repo(item)
    if not owner_repo:
        return None
    owner, repo = owner_repo

    headers: Dict[str, str] = {"Accept": "application/vnd.github+json"}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    readme_candidates = [
        f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/README.md",
        f"https://raw.githubusercontent.com/{owner}/{repo}/main/README.md",
        f"https://raw.githubusercontent.com/{owner}/{repo}/master/README.md",
    ]

    readme_text = ""
    for url in readme_candidates:
        try:
            resp = await client.get(url, headers=headers, follow_redirects=True)
            if resp.status_code == 200 and resp.text:
                readme_text = resp.text
                break
        except Exception:
            continue

    code_snippets: List[str] = []
    dependency_signals: List[str] = []
    try:
        listing_resp = await client.get(
            f"https://api.github.com/repos/{owner}/{repo}/contents",
            headers=headers,
            follow_redirects=True,
        )
        if listing_resp.status_code == 200:
            entries = listing_resp.json()
            if isinstance(entries, list):
                candidates: List[tuple[int, str, str]] = []
                dependency_candidates: List[tuple[int, str, str]] = []
                directory_candidates: List[str] = []
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    entry_type = str(entry.get("type") or "").strip()
                    path = str(entry.get("path") or "")
                    lowered = path.lower()
                    if entry_type == "dir":
                        if any(token in lowered for token in ("model", "models", "src", "train", "inference", "serve", "pipeline")):
                            dir_api_url = str(entry.get("url") or "").strip()
                            if dir_api_url:
                                directory_candidates.append(dir_api_url)
                        continue
                    if entry_type != "file":
                        continue
                    size = int(entry.get("size") or 0)
                    download_url = str(entry.get("download_url") or "")
                    if not download_url:
                        continue
                    if _is_code_candidate(path, size):
                        score = 1
                        if any(token in lowered for token in _CODE_HINTS):
                            score += 2
                        if lowered.endswith(".py"):
                            score += 1
                        candidates.append((score, path, download_url))
                    if _is_dependency_candidate(path, size):
                        dep_score = 2
                        if "requirements" in lowered:
                            dep_score += 2
                        dependency_candidates.append((dep_score, path, download_url))

                # One-level directory expansion for code hotspots.
                for dir_api_url in directory_candidates[:3]:
                    try:
                        dir_resp = await client.get(dir_api_url, headers=headers, follow_redirects=True)
                    except Exception:
                        continue
                    if dir_resp.status_code != 200:
                        continue
                    dir_entries = dir_resp.json()
                    if not isinstance(dir_entries, list):
                        continue
                    for entry in dir_entries:
                        if not isinstance(entry, dict) or str(entry.get("type") or "") != "file":
                            continue
                        path = str(entry.get("path") or "")
                        size = int(entry.get("size") or 0)
                        download_url = str(entry.get("download_url") or "")
                        if not download_url:
                            continue
                        if _is_code_candidate(path, size):
                            score = 1
                            lowered = path.lower()
                            if any(token in lowered for token in _CODE_HINTS):
                                score += 2
                            if lowered.endswith(".py"):
                                score += 1
                            candidates.append((score, path, download_url))
                        if _is_dependency_candidate(path, size):
                            dependency_candidates.append((2, path, download_url))

                candidates.sort(reverse=True)
                for _, path, download_url in candidates[:3]:
                    try:
                        file_text = await _http_get_text(client, download_url)
                    except Exception:
                        file_text = ""
                    snippet = _select_snippets(file_text, max_items=4, max_chars=700)
                    if snippet:
                        code_snippets.append(f"{path}\n{snippet}")

                dependency_candidates.sort(reverse=True)
                for _, path, download_url in dependency_candidates[:3]:
                    dep_text = await _http_get_text(client, download_url)
                    dep_lines = _extract_dependency_lines(dep_text, max_items=8)
                    if dep_lines:
                        dependency_signals.append(f"{path}: " + "; ".join(dep_lines[:8]))
    except Exception:
        pass

    if not readme_text and not code_snippets and not dependency_signals:
        return None

    blocks: List[str] = []
    if readme_text:
        blocks.append("README Highlights:")
        blocks.append(_select_snippets(readme_text, max_items=8, max_chars=1300))
    if code_snippets:
        blocks.append("Code Evidence:")
        blocks.extend([f"- {snippet}" for snippet in code_snippets])
    if dependency_signals:
        blocks.append("Dependency Signals:")
        blocks.extend([f"- {line}" for line in dependency_signals])
    appendix = "\n".join([item for item in blocks if item]).strip()
    if not appendix:
        return None

    new_item = dict(item)
    metadata = dict(item.get("metadata", {}) or {})
    new_item["content"] = _merge_content(
        str(item.get("content") or ""),
        "Deep Repo Evidence",
        appendix,
        max_chars=max_chars_per_item * 2,
    )
    metadata["deep_enriched"] = True
    metadata["deep_enrichment_source"] = "github_readme_code_dependencies"
    new_item["metadata"] = metadata
    return new_item


async def _enrich_huggingface_item(
    item: Dict[str, Any],
    *,
    client: httpx.AsyncClient,
    max_chars_per_item: int,
) -> Optional[Dict[str, Any]]:
    repo = _huggingface_repo(item)
    if not repo:
        return None
    repo_type, repo_id = repo

    if repo_type == "dataset":
        readme_url = f"https://huggingface.co/datasets/{repo_id}/raw/main/README.md"
        cfg_url = f"https://huggingface.co/datasets/{repo_id}/raw/main/dataset_infos.json"
    else:
        readme_url = f"https://huggingface.co/{repo_id}/raw/main/README.md"
        cfg_url = f"https://huggingface.co/{repo_id}/raw/main/config.json"

    readme_text = await _http_get_text(client, readme_url)
    cfg_text = await _http_get_text(client, cfg_url)

    cfg_summary = ""
    if cfg_text:
        try:
            cfg = json.loads(cfg_text)
            if isinstance(cfg, dict):
                key_fields = []
                for key in (
                    "architectures",
                    "model_type",
                    "hidden_size",
                    "num_hidden_layers",
                    "num_attention_heads",
                    "max_position_embeddings",
                    "vocab_size",
                ):
                    if key in cfg:
                        key_fields.append(f"{key}={cfg[key]}")
                if key_fields:
                    cfg_summary = "; ".join(key_fields)
        except Exception:
            cfg_summary = _clean_text(cfg_text, max_chars=300)

    if not readme_text and not cfg_summary:
        return None

    blocks: List[str] = []
    if readme_text:
        blocks.append("Model Card / Dataset Card Highlights:")
        blocks.append(_select_snippets(readme_text, max_items=8, max_chars=1300))
    if cfg_summary:
        blocks.append(f"Config Signals: {cfg_summary}")
    appendix = "\n".join([item for item in blocks if item]).strip()
    if not appendix:
        return None

    new_item = dict(item)
    metadata = dict(item.get("metadata", {}) or {})
    new_item["content"] = _merge_content(
        str(item.get("content") or ""),
        "Deep HF Evidence",
        appendix,
        max_chars=max_chars_per_item * 2,
    )
    metadata["deep_enriched"] = True
    metadata["deep_enrichment_source"] = "hf_card_config"
    new_item["metadata"] = metadata
    return new_item


async def enrich_search_results_deep(
    search_results: List[Dict[str, Any]],
    *,
    max_items_per_source: int = 2,
    concurrency: int = 4,
    timeout_sec: float = 12.0,
    max_pdf_pages: int = 14,
    max_chars_per_item: int = 12000,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Best-effort deep enrichment over selected high-value results.
    """
    if not search_results:
        return search_results, {"attempted": 0, "enriched": 0, "sources": {}}

    by_source: Dict[str, List[int]] = {}
    for idx, item in enumerate(search_results):
        source = _item_source(item)
        if source in {"arxiv", "arxiv_rss", "semantic_scholar", "openreview", "github", "huggingface"}:
            by_source.setdefault(source, []).append(idx)

    selected_indices: List[int] = []
    for source, indices in by_source.items():
        scored = sorted(
            indices,
            key=lambda i: _item_score(search_results[i]),
            reverse=True,
        )
        selected_indices.extend(scored[: max(0, int(max_items_per_source))])

    if not selected_indices:
        return search_results, {"attempted": 0, "enriched": 0, "sources": {}}

    settings = get_settings()
    github_token = str(settings.github.token or "").strip()
    timeout = httpx.Timeout(max(3.0, float(timeout_sec)))
    limits = httpx.Limits(max_connections=max(4, int(concurrency) * 2), max_keepalive_connections=max(4, int(concurrency)))
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))

    result_map = {idx: dict(search_results[idx]) for idx in selected_indices}
    source_stats: Dict[str, int] = {}
    enriched_count = 0

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        async def _enrich_index(idx: int) -> None:
            nonlocal enriched_count
            source = _item_source(search_results[idx])
            candidate = dict(search_results[idx])
            enriched: Optional[Dict[str, Any]] = None
            async with semaphore:
                try:
                    if source in {"arxiv", "arxiv_rss", "semantic_scholar", "openreview"}:
                        enriched = await _enrich_paper_item(
                            candidate,
                            source=source,
                            client=client,
                            max_pdf_pages=max_pdf_pages,
                            max_chars_per_item=max_chars_per_item,
                        )
                    elif source == "github":
                        enriched = await _enrich_github_item(
                            candidate,
                            client=client,
                            max_chars_per_item=max_chars_per_item,
                            github_token=github_token,
                        )
                    elif source == "huggingface":
                        enriched = await _enrich_huggingface_item(
                            candidate,
                            client=client,
                            max_chars_per_item=max_chars_per_item,
                        )
                except Exception as exc:
                    logger.debug(f"Deep enrichment failed for {source} item={candidate.get('id')}: {exc}")
                    enriched = None

            if enriched:
                result_map[idx] = enriched
                enriched_count += 1
                source_stats[source] = source_stats.get(source, 0) + 1

        await asyncio.gather(*[_enrich_index(idx) for idx in selected_indices], return_exceptions=True)

    output = list(search_results)
    for idx, item in result_map.items():
        output[idx] = item

    summary = {
        "attempted": len(selected_indices),
        "enriched": enriched_count,
        "sources": source_stats,
    }
    return output, summary
