"""Phase 5 Web App: FastAPI + SSE + three-column UI."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Set
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Response
from fastapi import File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator

from config import get_research_cache_settings
from intelligence.pipeline import (
    _load_video_flow,
    _sync_video_brief_with_flow,
    chat_over_kb,
    run_research_end_to_end,
)
from intelligence.pipeline_helpers import (
    normalize_one_pager_content,
    normalize_one_pager_resources,
    normalize_timeline_dates,
    normalize_video_brief,
)
from intelligence.tools.rag_tools import add_to_knowledge_base
from outputs import export_research_outputs
from outputs.video_generator import SlidevVideoGenerator
from storage import ResearchArtifactStore
from webapp.input_ingest import ingest_arxiv_url, ingest_uploaded_pdf


logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT_DIR / "webapp" / "static"
ARTIFACT_ROOT = ROOT_DIR / "data"
UPLOAD_ROOT = ARTIFACT_ROOT / "uploads"

DEFAULT_SOURCES = [
    "arxiv",
    "arxiv_rss",
    "openreview",
    "semantic_scholar",
    "huggingface",
    "github",
    "stackoverflow",
    "hackernews",
]

SEARCH_ONLY_SOURCES = {"arxiv_rss", "openreview"}

SOURCE_FLAG_MAP: Dict[str, str] = {
    "arxiv": "enable_arxiv",
    "huggingface": "enable_huggingface",
    "twitter": "enable_twitter",
    "reddit": "enable_reddit",
    "github": "enable_github",
    "semantic_scholar": "enable_semantic_scholar",
    "stackoverflow": "enable_stackoverflow",
    "hackernews": "enable_hackernews",
}

STREAMABLE_DOC_KEYS = [
    "knowledge_tree_md",
    "one_pager_md",
    "timeline_md",
    "report_md",
    "video_brief_md",
]

PROFESSIONAL_SOURCES = {"arxiv", "arxiv_rss", "openreview", "semantic_scholar", "github", "huggingface", "user_document"}
COMMUNITY_SOURCES = {"stackoverflow", "hackernews", "reddit", "twitter"}


class ResearchRequest(BaseModel):
    topic: str = Field(default="", description="Research topic")
    arxiv_url: Optional[str] = Field(default=None, description="arXiv abs/pdf URL")
    uploaded_pdf_path: Optional[str] = Field(default=None, exclude=True)
    max_results: int = Field(default=8, ge=1, le=30)
    sources: List[str] = Field(default_factory=lambda: list(DEFAULT_SOURCES))
    generate_video: bool = Field(default=True)
    disable_narration: bool = Field(default=False)
    tts_provider: str = Field(default="edge-tts")
    tts_voice: Optional[str] = None
    tts_speed: float = Field(default=1.25, ge=0.8, le=1.5)
    slides_target_duration_sec: int = Field(default=180, ge=60, le=1200)
    slides_fps: int = Field(default=24, ge=10, le=60)
    narration_model: str = Field(default="deepseek-chat")
    search_max_iterations: int = Field(default=2, ge=1, le=16)
    search_tool_timeout_sec: int = Field(default=6, ge=3, le=60)
    react_thought_timeout_sec: int = Field(default=5, ge=4, le=45)
    search_time_budget_sec: int = Field(default=45, ge=20, le=900)
    analysis_timeout_sec: int = Field(default=60, ge=10, le=300)
    content_timeout_sec: int = Field(default=45, ge=12, le=360)
    critic_timeout_sec: int = Field(default=15, ge=8, le=120)
    enable_critic_gate: bool = Field(default=False)
    allow_cache_hit: bool = Field(
        default=False,
        description="Allow automatic cache reuse during fresh research start",
    )

    @field_validator("topic")
    @classmethod
    def _validate_topic(cls, value: str) -> str:
        return str(value or "").strip()

    @field_validator("arxiv_url")
    @classmethod
    def _validate_arxiv_url(cls, value: Optional[str]) -> Optional[str]:
        text = str(value or "").strip()
        return text or None

    @field_validator("uploaded_pdf_path")
    @classmethod
    def _validate_uploaded_pdf_path(cls, value: Optional[str]) -> Optional[str]:
        text = str(value or "").strip()
        return text or None

    @model_validator(mode="after")
    def _validate_input_mode(self):
        topic = str(self.topic or "").strip()
        arxiv_url = str(self.arxiv_url or "").strip()
        uploaded = str(self.uploaded_pdf_path or "").strip()
        modes = int(bool(topic)) + int(bool(arxiv_url)) + int(bool(uploaded))
        if modes != 1:
            raise ValueError("Provide exactly one input mode: topic or arxiv_url or uploaded_pdf_path")
        return self

    @field_validator("sources", mode="before")
    @classmethod
    def _normalize_sources(cls, value: Any) -> List[str]:
        if value is None:
            return list(DEFAULT_SOURCES)

        raw_items: Iterable[str]
        if isinstance(value, str):
            raw_items = [item.strip() for item in value.split(",")]
        elif isinstance(value, list):
            raw_items = [str(item).strip() for item in value]
        else:
            return list(DEFAULT_SOURCES)

        normalized: List[str] = []
        seen = set()
        for item in raw_items:
            key = item.lower().replace("-", "_")
            if not key:
                continue
            if key == "semantic scholar":
                key = "semantic_scholar"
            if key == "stack overflow":
                key = "stackoverflow"
            if key in SOURCE_FLAG_MAP and key not in seen:
                normalized.append(key)
                seen.add(key)
                continue
            if key in SEARCH_ONLY_SOURCES and key not in seen:
                normalized.append(key)
                seen.add(key)

        return normalized or list(DEFAULT_SOURCES)


class ChatRequest(BaseModel):
    question: str = Field(..., description="Question to ask over KB")
    topic: Optional[str] = Field(default=None, description="Optional topic for session-scoped KB retrieval")
    session_id: Optional[str] = Field(default=None, description="Optional session id for namespace-isolated KB retrieval")
    top_k: int = Field(default=6, ge=1, le=20)
    sources: Optional[List[str]] = None
    year_filter: Optional[int] = None
    use_hybrid: bool = True

    @field_validator("question")
    @classmethod
    def _validate_question(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("question is required")
        return text

    @field_validator("topic", "session_id")
    @classmethod
    def _normalize_optional_text(cls, value: Optional[str]) -> Optional[str]:
        text = str(value or "").strip()
        return text or None


class CacheOpenRequest(BaseModel):
    snapshot_path: str = Field(..., description="Path to research_result_snapshot.json")

    @field_validator("snapshot_path")
    @classmethod
    def _validate_snapshot_path(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("snapshot_path is required")
        return text


@dataclass
class ResearchJob:
    run_id: str
    request: ResearchRequest
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "running"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    events: List[Dict[str, Any]] = field(default_factory=list)
    last_event: str = "init"
    done: asyncio.Event = field(default_factory=asyncio.Event)
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    streamed_docs: Set[str] = field(default_factory=set)

    async def emit(self, event: str, payload: Dict[str, Any]) -> None:
        self.last_event = event
        entry = {
            "event": event,
            "data": payload,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        async with self.condition:
            self.events.append(entry)
            self.condition.notify_all()

    async def finalize(self, *, status: str, result: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> None:
        self.status = status
        self.result = result
        self.error = error
        self.done.set()
        async with self.condition:
            self.condition.notify_all()


JOBS: Dict[str, ResearchJob] = {}


def _sse(event: str, data: Dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def _sanitize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "search_results",
            "timeline",
            "one_pager",
            "video_brief",
            "facts",
        }
    }


def _resolve_artifact_path(path_text: str) -> Path:
    text = str(path_text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="path is required")

    raw = Path(text)
    if raw.is_absolute():
        candidate = raw.resolve()
    else:
        candidate = (ROOT_DIR / raw).resolve()

    roots = [ROOT_DIR.resolve(), ARTIFACT_ROOT.resolve()]
    if not any(root == candidate or root in candidate.parents for root in roots):
        raise HTTPException(status_code=403, detail="path is outside allowed roots")

    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")

    return candidate


def _read_snapshot_file(snapshot_path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse snapshot: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid snapshot payload")
    return payload


def _video_artifact_exists(payload: Dict[str, Any]) -> bool:
    artifact = payload.get("video_artifact") or {}
    output_path = str(artifact.get("output_path") or "").strip()
    if not output_path:
        return False
    try:
        resolved = _resolve_artifact_path(output_path)
    except HTTPException:
        return False
    return resolved.exists()


def _summary_from_snapshot(snapshot: Dict[str, Any]) -> str:
    one_pager = dict(snapshot.get("one_pager") or {})
    summary = str(one_pager.get("executive_summary", "")).strip()
    if summary:
        return summary
    key_findings = [str(x).strip() for x in list(one_pager.get("key_findings") or []) if str(x).strip()]
    if key_findings:
        return key_findings[0]
    facts = list(snapshot.get("facts") or [])
    for item in facts:
        claim = str((item or {}).get("claim", "")).strip()
        if claim:
            return claim
    return ""


def _is_temporary_artifact_path(path_text: str) -> bool:
    text = str(path_text or "").strip()
    if not text:
        return False
    try:
        path = Path(text)
    except Exception:
        return False
    name = path.name.lower()
    parent = path.parent.name.lower() if path.parent else ""
    markers = (
        "tmp_",
        "test_",
        "debug_",
        "_debug",
        "verify_",
        "_verify",
        "check_",
        "_check",
    )
    return any(name.startswith(marker) for marker in markers) or any(
        parent.startswith(marker) for marker in markers
    )


def _topic_token_overlap(query: str, candidate_topic: str) -> float:
    q_tokens = {
        token
        for token in re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?|[\u4e00-\u9fff]+", str(query or "").lower())
        if token and token not in {"the", "and", "for", "with", "model", "models", "技术", "分析"}
    }
    c_tokens = {
        token
        for token in re.findall(
            r"[A-Za-z]+|\d+(?:\.\d+)?|[\u4e00-\u9fff]+",
            str(candidate_topic or "").lower(),
        )
        if token and token not in {"the", "and", "for", "with", "model", "models", "技术", "分析"}
    }
    if not q_tokens or not c_tokens:
        return 0.0
    return len(q_tokens.intersection(c_tokens)) / float(len(q_tokens))


def _refresh_cached_outputs(snapshot: Dict[str, Any], *, snapshot_path: Path) -> Dict[str, Any]:
    payload = dict(snapshot)
    topic = str(payload.get("topic") or payload.get("input_topic") or "").strip() or "Cached Research"
    output_dir_text = str(payload.get("output_dir") or "").strip() or str(snapshot_path.parent)
    output_dir = Path(output_dir_text)
    cached_results = list(payload.get("search_results") or [])
    cached_facts = list(payload.get("facts") or [])

    payload["timeline"] = normalize_timeline_dates(
        topic=topic,
        timeline=payload.get("timeline"),
        facts=cached_facts,
        search_results=cached_results,
    )
    payload["one_pager"] = normalize_one_pager_content(
        topic=topic,
        one_pager=payload.get("one_pager"),
        facts=cached_facts,
        search_results=cached_results,
    )
    payload["one_pager"] = normalize_one_pager_resources(
        one_pager=payload.get("one_pager"),
        facts=cached_facts,
        search_results=cached_results,
    )
    payload["video_brief"] = normalize_video_brief(
        topic=topic,
        video_brief=payload.get("video_brief"),
    )

    artifact_meta_path = str((payload.get("video_artifact") or {}).get("metadata_path", "")).strip()
    if artifact_meta_path:
        flow = _load_video_flow(Path(artifact_meta_path))
        payload["video_brief"] = _sync_video_brief_with_flow(
            video_brief=payload.get("video_brief"),
            flow=flow,
        )

    try:
        refreshed_written = export_research_outputs(
            output_dir,
            topic=topic,
            timeline=payload.get("timeline"),
            one_pager=payload.get("one_pager"),
            video_brief=payload.get("video_brief"),
            facts=cached_facts,
            search_results=cached_results,
            write_report=True,
        )
        payload["written_files"] = {k: str(v) for k, v in refreshed_written.items()}
        payload["output_dir"] = str(output_dir)
    except Exception as exc:
        logger.warning("Failed to refresh cached outputs: %s", exc)
        payload["written_files"] = {
            key: str(value) for key, value in dict(payload.get("written_files") or {}).items()
        }
        payload["output_dir"] = str(output_dir)

    payload["topic"] = topic
    payload["search_results_count"] = int(payload.get("search_results_count") or len(cached_results))
    payload["cache_hit"] = True
    return payload


async def _warmup_kb_from_results(search_results: List[Dict[str, Any]]) -> None:
    if not search_results:
        return
    documents: List[Dict[str, Any]] = []
    for item in search_results:
        metadata = dict(item.get("metadata", {}) or {})
        metadata["source"] = str(item.get("source", "")).strip() or metadata.get("source", "unknown")
        url = str(item.get("url", "")).strip()
        if url:
            metadata["url"] = url
        documents.append(
            {
                "id": str(item.get("id", "")).strip(),
                "content": f"{item.get('title', '')}\n\n{item.get('content', '')}",
                "metadata": metadata,
                "type": "search_result",
            }
        )
    try:
        await add_to_knowledge_base(documents)
    except Exception as exc:
        logger.warning("Cache KB warmup skipped: %s", exc)


def _aggregator_kwargs_from_sources(sources: List[str]) -> Dict[str, bool]:
    allowed = set(sources)
    return {flag: source in allowed for source, flag in SOURCE_FLAG_MAP.items()}


def _chunk_text(text: str, chunk_size: int = 1200) -> Iterable[str]:
    cleaned = text or ""
    for index in range(0, len(cleaned), chunk_size):
        yield cleaned[index : index + chunk_size]


def _dedupe_citations(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for item in items:
        source = str(item.get("source", "")).strip()
        url = str(item.get("url", "")).strip()
        key = (source, url, str(item.get("id", "")).strip())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _build_source_groups(search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    professional: Dict[str, int] = {}
    community: Dict[str, int] = {}
    other: Dict[str, int] = {}
    for item in search_results:
        source = str(item.get("source", "")).strip() or "unknown"
        if source in PROFESSIONAL_SOURCES:
            professional[source] = professional.get(source, 0) + 1
        elif source in COMMUNITY_SOURCES:
            community[source] = community.get(source, 0) + 1
        else:
            other[source] = other.get(source, 0) + 1
    return {
        "professional": {"total": int(sum(professional.values())), "breakdown": professional},
        "community": {"total": int(sum(community.values())), "breakdown": community},
        "other": {"total": int(sum(other.values())), "breakdown": other},
    }


async def _stream_documents(job: ResearchJob, written_files: Dict[str, Any]) -> None:
    if not isinstance(written_files, dict):
        return

    for key in STREAMABLE_DOC_KEYS:
        if key in job.streamed_docs:
            continue
        path_text = str(written_files.get(key, "")).strip()
        if not path_text:
            continue

        try:
            path = _resolve_artifact_path(path_text)
            content = path.read_text(encoding="utf-8")
        except Exception as exc:
            await job.emit(
                "document_error",
                {"name": key, "path": path_text, "error": str(exc)},
            )
            job.streamed_docs.add(key)
            continue

        await job.emit(
            "document_start",
            {"name": key, "path": str(path), "length": len(content)},
        )
        for idx, chunk in enumerate(_chunk_text(content), start=1):
            await job.emit(
                "document_chunk",
                {"name": key, "chunk_index": idx, "chunk": chunk},
            )
        await job.emit("document_done", {"name": key, "path": str(path)})
        job.streamed_docs.add(key)


def _make_video_generator(req: ResearchRequest, job: ResearchJob) -> Optional[SlidevVideoGenerator]:
    if not req.generate_video:
        return None

    loop = asyncio.get_running_loop()

    def _video_progress(message: str) -> None:
        loop.call_soon_threadsafe(
            lambda: asyncio.create_task(
                job.emit("video_progress", {"message": str(message)})
            )
        )

    return SlidevVideoGenerator(
        target_duration_sec=req.slides_target_duration_sec,
        fps=req.slides_fps,
        enable_narration=not req.disable_narration,
        tts_provider=req.tts_provider,
        tts_voice=req.tts_voice,
        tts_speed=req.tts_speed,
        narration_model=req.narration_model,
        progress_callback=_video_progress,
    )


def _result_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    written_files = {
        key: str(value) for key, value in dict(result.get("written_files") or {}).items()
    }
    video_artifact = dict(result.get("video_artifact") or {})
    source_groups = _build_source_groups(list(result.get("search_results") or []))
    return {
        "topic": result.get("topic"),
        "input_topic": result.get("input_topic"),
        "output_dir": result.get("output_dir"),
        "search_results_count": int(result.get("search_results_count") or 0),
        "facts_count": len(result.get("facts") or []),
        "cache_hit": bool(result.get("cache_hit", False)),
        "cache_score": result.get("cache_score"),
        "cache_matched_topic": result.get("cache_matched_topic"),
        "quality_metrics": result.get("quality_metrics"),
        "quality_gate_pass": result.get("quality_gate_pass"),
        "quality_recommendations": list(result.get("quality_recommendations") or []),
        "knowledge_gaps": list(result.get("knowledge_gaps") or []),
        "written_files": written_files,
        "video_artifact": video_artifact,
        "video_error": result.get("video_error"),
        "blocked": bool(result.get("blocked", False)),
        "blocked_reason": result.get("blocked_reason"),
        "search_strategy": result.get("search_strategy"),
        "source_groups": source_groups,
    }


async def _run_job(job: ResearchJob) -> None:
    req = job.request

    async def _progress_callback(payload: Dict[str, Any]) -> None:
        event = str(payload.get("event") or "progress")
        await job.emit(event, _sanitize_payload(payload))
        if event in {"documents_exported", "documents_reexported_after_video"}:
            if event == "documents_reexported_after_video":
                job.streamed_docs.clear()
            await _stream_documents(job, payload.get("written_files") or {})

    await job.emit(
        "job_started",
        {
            "run_id": job.run_id,
            "topic": req.topic or req.arxiv_url or req.uploaded_pdf_path,
            "generate_video": req.generate_video,
            "sources": req.sources,
            "input_mode": (
                "arxiv_url"
                if req.arxiv_url
                else ("pdf_upload" if req.uploaded_pdf_path else "topic")
            ),
        },
    )

    try:
        generator = _make_video_generator(req, job)
        effective_topic = req.topic
        seed_results: List[Dict[str, Any]] = []
        seed_user_query: Optional[str] = None

        if req.arxiv_url:
            await job.emit("input_ingestion_started", {"mode": "arxiv_url", "value": req.arxiv_url})
            bundle = await ingest_arxiv_url(arxiv_url=req.arxiv_url)
            effective_topic = bundle.topic
            seed_results = list(bundle.search_results)
            if seed_results:
                seed_user_query = str(seed_results[0].get("content", ""))[:1800]
            for note in bundle.notes:
                await job.emit("input_ingestion_note", {"mode": "arxiv_url", "note": note})
            await job.emit(
                "input_ingestion_completed",
                {"mode": "arxiv_url", "derived_topic": effective_topic, "seed_count": len(seed_results)},
            )

        if req.uploaded_pdf_path:
            await job.emit("input_ingestion_started", {"mode": "pdf_upload", "value": req.uploaded_pdf_path})
            bundle = await ingest_uploaded_pdf(
                file_path=Path(req.uploaded_pdf_path),
                original_name=Path(req.uploaded_pdf_path).name,
            )
            effective_topic = bundle.topic
            seed_results = list(bundle.search_results)
            if seed_results:
                seed_user_query = str(seed_results[0].get("content", ""))[:1800]
            for note in bundle.notes:
                await job.emit("input_ingestion_note", {"mode": "pdf_upload", "note": note})
            await job.emit(
                "input_ingestion_completed",
                {"mode": "pdf_upload", "derived_topic": effective_topic, "seed_count": len(seed_results)},
            )

        result = await run_research_end_to_end(
            topic=effective_topic,
            max_results_per_source=req.max_results,
            show_progress=False,
            generate_video=req.generate_video,
            video_generator=generator,
            aggregator_kwargs=_aggregator_kwargs_from_sources(req.sources),
            allowed_sources=req.sources,
            user_query=seed_user_query,
            search_max_iterations=req.search_max_iterations,
            search_tool_timeout_sec=req.search_tool_timeout_sec,
            react_thought_timeout_sec=req.react_thought_timeout_sec,
            search_time_budget_sec=req.search_time_budget_sec,
            search_stage_timeout_sec=max(20, int(req.search_time_budget_sec) + 30),
            analysis_timeout_sec=req.analysis_timeout_sec,
            content_timeout_sec=req.content_timeout_sec,
            critic_timeout_sec=req.critic_timeout_sec,
            enable_critic_gate=req.enable_critic_gate,
            allow_cache_hit=req.allow_cache_hit,
            seed_search_results=seed_results,
            progress_callback=_progress_callback,
        )

        await _stream_documents(job, result.get("written_files") or {})

        summary = _result_summary(result)
        await job.emit("result", summary)

        video_artifact = summary.get("video_artifact") or {}
        video_path = str(video_artifact.get("output_path", "")).strip()
        if video_path:
            await job.emit(
                "video_ready",
                {
                    "output_path": video_path,
                    "metadata_path": video_artifact.get("metadata_path"),
                    "narration_script_path": video_artifact.get("narration_script_path"),
                },
            )

        await job.emit("job_completed", {"run_id": job.run_id})
        await job.finalize(status="completed", result=summary)
    except Exception as exc:
        logger.exception("Research job failed")
        message = str(exc)
        await job.emit("job_failed", {"error": message})
        await job.finalize(status="failed", error=message)


app = FastAPI(title="Academic Research Agent UI", version="phase5")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    return Response(status_code=204)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/research")
async def start_research(req: ResearchRequest) -> Dict[str, Any]:
    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    job = ResearchJob(run_id=run_id, request=req)
    JOBS[run_id] = job
    asyncio.create_task(_run_job(job))
    return {"run_id": run_id, "status": "running"}


@app.get("/api/cache/suggestions")
async def cache_suggestions(
    topic: str,
    top_k: int = 5,
    score_threshold: Optional[float] = None,
    require_video: bool = False,
) -> Dict[str, Any]:
    query = str(topic or "").strip()
    if not query:
        return {"topic": "", "candidates": []}

    cache_cfg = get_research_cache_settings()
    threshold = (
        float(score_threshold)
        if score_threshold is not None
        else max(0.30, float(getattr(cache_cfg, "similarity_threshold", 0.82)) - 0.45)
    )
    threshold = float(max(0.0, min(1.0, threshold)))
    top_k = max(1, min(int(top_k), 8))
    min_quality_score = float(getattr(cache_cfg, "min_quality_score", 0.0) or 0.0)
    require_quality_gate_pass = bool(getattr(cache_cfg, "require_quality_gate_pass", False))
    min_facts_count = max(0, int(getattr(cache_cfg, "min_facts_count", 0) or 0))

    store = ResearchArtifactStore(collection_name=cache_cfg.collection_name)
    try:
        candidates = store.find_similar(query=query, score_threshold=threshold, top_k=max(top_k, 6))
    finally:
        store.close()

    payload: List[Dict[str, Any]] = []
    for candidate in candidates:
        candidate_quality = float(candidate.get("quality_score", 0.0) or 0.0)
        # Backward compatibility: older artifacts may not have quality_score indexed.
        if candidate_quality > 0 and candidate_quality < min_quality_score:
            continue
        if require_quality_gate_pass and not bool(candidate.get("quality_gate_pass", False)):
            continue

        snapshot_path_text = str(candidate.get("snapshot_path", "")).strip()
        if not snapshot_path_text:
            continue
        if _is_temporary_artifact_path(snapshot_path_text):
            continue
        try:
            snapshot_path = _resolve_artifact_path(snapshot_path_text)
            snapshot = _read_snapshot_file(snapshot_path)
        except HTTPException:
            continue

        output_dir = str(snapshot.get("output_dir") or candidate.get("output_dir") or "").strip()
        if _is_temporary_artifact_path(output_dir):
            continue

        facts_count = len(snapshot.get("facts") or [])
        if min_facts_count and facts_count < min_facts_count:
            continue

        snapshot_quality = float(((snapshot.get("quality_metrics") or {}).get("overall_score") or 0.0))
        effective_quality = candidate_quality if candidate_quality > 0 else snapshot_quality

        candidate_topic = str(snapshot.get("topic") or candidate.get("topic") or "").strip()
        overlap_ratio = _topic_token_overlap(query=query, candidate_topic=candidate_topic)
        if overlap_ratio <= 0.0 and float(candidate.get("score", 0.0) or 0.0) < 0.72:
            continue

        has_video = _video_artifact_exists(snapshot)
        if require_video and not has_video:
            continue

        summary = _summary_from_snapshot(snapshot)
        payload.append(
            {
                "artifact_id": str(candidate.get("artifact_id", "")).strip(),
                "topic": candidate_topic,
                "score": float(candidate.get("score", 0.0)),
                "topic_overlap": round(float(overlap_ratio), 3),
                "created_at": str(candidate.get("created_at") or snapshot.get("created_at") or "").strip(),
                "quality_score": float(effective_quality),
                "quality_gate_pass": bool(candidate.get("quality_gate_pass", False)),
                "search_results_count": int(
                    candidate.get("search_results_count")
                    or snapshot.get("search_results_count")
                    or len(snapshot.get("search_results") or [])
                ),
                "facts_count": facts_count,
                "has_video": has_video,
                "output_dir": output_dir,
                "snapshot_path": str(snapshot_path),
                "summary": summary,
            }
        )
        if len(payload) >= top_k:
            break

    payload.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return {"topic": query, "candidates": payload}


@app.post("/api/research/upload")
async def start_research_with_upload(
    file: UploadFile = File(...),
    config_json: str = Form(default="{}"),
) -> Dict[str, Any]:
    try:
        config = json.loads(config_json or "{}")
        if not isinstance(config, dict):
            raise ValueError("config_json must be an object")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid config_json: {exc}") from exc

    filename = str(file.filename or "").strip() or "document.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF upload is supported")

    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", filename).strip("_") or "document.pdf"
    upload_path = UPLOAD_ROOT / f"{run_stamp}_{uuid4().hex[:8]}_{safe_name}"

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded PDF is empty")
        upload_path.write_bytes(content)
    finally:
        await file.close()

    payload = dict(config)
    payload["topic"] = ""
    payload["arxiv_url"] = None
    payload["uploaded_pdf_path"] = str(upload_path)
    try:
        req = ResearchRequest(**payload)
    except Exception as exc:
        upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid request config: {exc}") from exc

    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    job = ResearchJob(run_id=run_id, request=req)
    JOBS[run_id] = job
    asyncio.create_task(_run_job(job))
    return {"run_id": run_id, "status": "running"}


@app.get("/api/research/{run_id}")
async def get_research_status(run_id: str) -> Dict[str, Any]:
    job = JOBS.get(run_id)
    if not job:
        raise HTTPException(status_code=404, detail="run_id not found")
    return {
        "run_id": run_id,
        "status": job.status,
        "error": job.error,
        "created_at": job.created_at.isoformat(timespec="seconds"),
        "events": len(job.events),
        "result": job.result,
    }


@app.post("/api/cache/open")
async def open_cached_result(req: CacheOpenRequest) -> Dict[str, Any]:
    snapshot_path = _resolve_artifact_path(req.snapshot_path)
    snapshot = _read_snapshot_file(snapshot_path)
    normalized = _refresh_cached_outputs(snapshot, snapshot_path=snapshot_path)
    await _warmup_kb_from_results(list(normalized.get("search_results") or []))
    summary = _result_summary(normalized)
    summary["cache_hit"] = True
    summary["cache_matched_topic"] = summary.get("topic")
    summary["cache_score"] = None
    return {"result": summary}


@app.get("/api/research/{run_id}/events")
async def stream_research_events(run_id: str) -> StreamingResponse:
    job = JOBS.get(run_id)
    if not job:
        raise HTTPException(status_code=404, detail="run_id not found")

    async def _event_stream():
        idx = 0
        yield "retry: 3000\n\n"
        yield _sse("connected", {"run_id": run_id, "status": job.status})

        while True:
            while idx < len(job.events):
                item = job.events[idx]
                idx += 1
                yield _sse(item["event"], item["data"])

            if job.done.is_set() and idx >= len(job.events):
                break

            try:
                async with job.condition:
                    await asyncio.wait_for(job.condition.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                elapsed = (datetime.now(timezone.utc) - job.created_at).total_seconds()
                yield _sse(
                    "heartbeat",
                    {
                        "run_id": run_id,
                        "status": job.status,
                        "events": len(job.events),
                        "last_event": job.last_event,
                        "elapsed_sec": round(float(elapsed), 1),
                    },
                )

        yield _sse(
            "stream_end",
            {"run_id": run_id, "status": job.status, "error": job.error},
        )

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(_event_stream(), media_type="text/event-stream", headers=headers)


@app.get("/api/artifacts/file")
async def get_artifact_file(path: str) -> FileResponse:
    artifact_path = _resolve_artifact_path(path)
    return FileResponse(artifact_path)


@app.post("/api/chat")
async def chat(req: ChatRequest) -> JSONResponse:
    answer = await chat_over_kb(
        question=req.question,
        topic=req.topic,
        session_id=req.session_id,
        top_k=req.top_k,
        sources=req.sources,
        year_filter=req.year_filter,
        use_hybrid=req.use_hybrid,
    )
    citations = _dedupe_citations(list(answer.get("citations") or []))
    payload = dict(answer)
    payload["citations"] = citations
    return JSONResponse(payload)
