"""
Research Pipeline
关键词驱动的端到端研究流水线（Phase 1 -> 4 + 可选视频生成）
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import logging

from aggregator import DataAggregator
from config import get_research_cache_settings
from intelligence.agents import (
    AnalystAgent,
    ChatAgent,
    ContentAgent,
    CriticAgent,
    PlannerAgent,
    SearchAgent,
)
from intelligence.llm import BaseLLM, get_llm
from intelligence.pipeline_helpers import (
    aggregated_result_to_search_results,
    evaluate_output_depth,
    evaluate_research_quality,
    normalize_one_pager_resources,
    normalize_video_brief,
)
from intelligence.tools.rag_tools import add_to_knowledge_base, close_knowledge_base
from outputs import export_research_outputs
from outputs.video_generator import BaseVideoGenerator, create_video_generator
from storage import ResearchArtifactStore


logger = logging.getLogger(__name__)

_RESULT_SNAPSHOT_FILE = "research_result_snapshot.json"
_ARTIFACT_SCHEMA_VERSION = "2026-02-08-slidev-v2"

_AGGREGATOR_FLAG_TO_SOURCE = {
    "enable_arxiv": "arxiv",
    "enable_huggingface": "huggingface",
    "enable_twitter": "twitter",
    "enable_reddit": "reddit",
    "enable_github": "github",
    "enable_semantic_scholar": "semantic_scholar",
    "enable_stackoverflow": "stackoverflow",
    "enable_hackernews": "hackernews",
}

_SOURCE_ALIASES = {
    "semantic-scholar": "semantic_scholar",
    "semantic scholar": "semantic_scholar",
    "stack-overflow": "stackoverflow",
    "stack overflow": "stackoverflow",
    "hn": "hackernews",
}


def _normalize_sources(values: Optional[List[str]]) -> List[str]:
    if not values:
        return []
    normalized: List[str] = []
    seen = set()
    for item in values:
        key = str(item or "").strip().lower().replace("-", "_")
        if not key:
            continue
        key = _SOURCE_ALIASES.get(key, key)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    return normalized


def _sources_from_aggregator_kwargs(kwargs: Dict[str, Any]) -> List[str]:
    sources: List[str] = []
    for flag, source in _AGGREGATOR_FLAG_TO_SOURCE.items():
        if bool(kwargs.get(flag, False)):
            sources.append(source)
    return sorted(sources)


def _aggregator_kwargs_from_sources(sources: List[str]) -> Dict[str, bool]:
    allowed = set(_normalize_sources(sources))
    if not allowed:
        return {}
    return {
        flag: source in allowed for flag, source in _AGGREGATOR_FLAG_TO_SOURCE.items()
    }


def _summarize_search_results(search_results: List[Dict[str, Any]]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for item in search_results:
        source = str(item.get("source", "")).strip() or "unknown"
        summary[source] = summary.get(source, 0) + 1
    summary["total"] = len(search_results)
    return summary


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _result_snapshot_path(output_dir: Path) -> Path:
    return output_dir / _RESULT_SNAPSHOT_FILE


def _write_result_snapshot(output_dir: Path, payload: Dict[str, Any]) -> Path:
    snapshot_path = _result_snapshot_path(output_dir)
    snapshot_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return snapshot_path


def _read_result_snapshot(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _build_artifact_summary_text(topic: str, result: Dict[str, Any]) -> str:
    lines = [str(topic or "").strip()]
    one_pager = result.get("one_pager") or {}
    summary = str(one_pager.get("executive_summary", "")).strip()
    if summary:
        lines.append(summary)

    key_findings = [
        str(item).strip()
        for item in (one_pager.get("key_findings") or [])
        if str(item).strip()
    ]
    if key_findings:
        lines.extend(key_findings[:6])

    facts = result.get("facts") or []
    for fact in facts[:8]:
        claim = str((fact or {}).get("claim", "")).strip()
        if claim:
            lines.append(claim)

    return "\n".join([line for line in lines if line]).strip() or str(topic or "").strip()


def _video_artifact_exists(payload: Dict[str, Any]) -> bool:
    artifact = payload.get("video_artifact") or {}
    output_path = str(artifact.get("output_path") or "").strip()
    if not output_path:
        return False
    return Path(output_path).exists()


def _search_results_to_documents(search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
    return documents


async def _warmup_knowledge_base_from_search_results(search_results: List[Dict[str, Any]]) -> None:
    if not search_results:
        return
    documents = _search_results_to_documents(search_results)
    await add_to_knowledge_base(documents)


def _load_video_flow(metadata_path: Path) -> Dict[str, Any]:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if isinstance(metadata, dict) and isinstance(metadata.get("flow"), dict):
            return metadata["flow"]
    except Exception:
        pass
    return {}


async def _generate_video_artifact(
    *,
    topic: str,
    output_dir: Path,
    generated: Dict[str, Any],
    facts: List[Dict[str, Any]],
    video_generator: Optional[BaseVideoGenerator],
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    generator = video_generator or create_video_generator(provider="slidev")

    try:
        artifact = await generator.generate(
            topic=topic,
            out_dir=output_dir,
            video_brief=generated.get("video_brief"),
            one_pager=generated.get("one_pager"),
            facts=facts,
        )
        flow = _load_video_flow(Path(artifact.metadata_path))
        payload: Dict[str, Any] = {
            "provider": artifact.provider,
            "output_path": str(artifact.output_path),
            "metadata_path": str(artifact.metadata_path),
        }
        for key in ("narration_audio_path", "narration_script_path"):
            value = flow.get(key)
            if value:
                payload[key] = str(value)
        return payload, None
    except Exception as exc:
        message = str(exc)
        logger.warning(f"Video generation failed, keeping document outputs: {message}")
        return None, message


async def run_research_from_search_results(
    *,
    topic: str,
    search_results: List[Dict[str, Any]],
    llm: Optional[BaseLLM] = None,
    out_dir: Optional[Path] = None,
    generate_video: bool = False,
    video_generator: Optional[BaseVideoGenerator] = None,
    enable_knowledge_indexing: bool = True,
    enable_critic_gate: bool = True,
    critic_threshold: float = 0.65,
) -> Dict[str, Any]:
    """
    使用已准备好的搜索结果执行 Phase 3 + Phase 4。
    """
    llm = llm or get_llm()

    try:
        analyst = AnalystAgent(
            llm=llm,
            enable_knowledge_indexing=enable_knowledge_indexing,
        )
        analysis = await analyst.analyze(topic, search_results)
        facts = analysis.get("facts", [])

        content_agent = ContentAgent(llm=llm)
        generated = await content_agent.generate(topic, facts)
        generated["one_pager"] = normalize_one_pager_resources(
            one_pager=generated.get("one_pager"),
            facts=facts,
            search_results=search_results,
        )
        generated["video_brief"] = normalize_video_brief(
            topic=topic,
            video_brief=generated.get("video_brief"),
        )

        output_dir = out_dir or Path("data") / "outputs" / (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{topic.replace(' ', '_')}"
        )
        written_files = export_research_outputs(
            output_dir,
            topic=topic,
            timeline=generated.get("timeline"),
            one_pager=generated.get("one_pager"),
            video_brief=generated.get("video_brief"),
            write_report=True,
        )

        depth = evaluate_output_depth(
            facts=facts,
            one_pager=generated.get("one_pager"),
            video_brief=generated.get("video_brief"),
        )

        quality_metrics = None
        quality_gate_pass = None
        quality_recommendations: List[str] = []
        if enable_critic_gate:
            critic = CriticAgent(quality_threshold=critic_threshold)
            critic_result = await critic.evaluate(
                facts=facts,
                search_results=search_results,
                one_pager=generated.get("one_pager"),
                video_brief=generated.get("video_brief"),
                knowledge_gaps=analysis.get("knowledge_gaps", []),
            )
            quality_metrics = critic_result.get("quality_metrics")
            quality_gate_pass = bool(critic_result.get("pass", False))
            quality_recommendations = list(critic_result.get("recommendations", []))

        video_artifact = None
        video_error = None
        if generate_video:
            video_artifact, video_error = await _generate_video_artifact(
                topic=topic,
                output_dir=output_dir,
                generated=generated,
                facts=facts,
                video_generator=video_generator,
            )

        return {
            "topic": topic,
            "search_results_count": len(search_results),
            "search_results": search_results,
            "facts": facts,
            "knowledge_gaps": analysis.get("knowledge_gaps", []),
            "timeline": generated.get("timeline"),
            "one_pager": generated.get("one_pager"),
            "video_brief": generated.get("video_brief"),
            "depth_assessment": depth,
            "quality_metrics": quality_metrics,
            "quality_gate_pass": quality_gate_pass,
            "quality_recommendations": quality_recommendations,
            "output_dir": str(output_dir),
            "written_files": {k: str(v) for k, v in written_files.items()},
            "video_artifact": video_artifact,
            "video_error": video_error,
            "cache_hit": False,
        }
    finally:
        if enable_knowledge_indexing:
            close_knowledge_base()


async def run_research_end_to_end(
    *,
    topic: str,
    max_results_per_source: int = 10,
    llm: Optional[BaseLLM] = None,
    out_dir: Optional[Path] = None,
    show_progress: bool = True,
    generate_video: bool = False,
    video_generator: Optional[BaseVideoGenerator] = None,
    enable_knowledge_indexing: bool = True,
    aggregator_kwargs: Optional[Dict[str, Any]] = None,
    user_query: Optional[str] = None,
    allowed_sources: Optional[List[str]] = None,
    use_agentic_search: bool = True,
    enable_planner: bool = True,
    enable_critic_gate: bool = True,
    critic_threshold: float = 0.65,
    search_max_iterations: int = 8,
) -> Dict[str, Any]:
    """
    完整端到端流程：
    Planner -> ReAct Search -> AnalystAgent -> ContentAgent -> CriticGate -> export -> (optional) video.
    """
    llm = llm or get_llm()
    aggregator_kwargs = dict(aggregator_kwargs or {})

    resolved_sources = _normalize_sources(allowed_sources)
    if not resolved_sources:
        resolved_sources = _sources_from_aggregator_kwargs(aggregator_kwargs)

    planned_topic = topic
    planner_output: Dict[str, Any] = {
        "is_technical": True,
        "normalized_topic": topic,
        "query_rewrites": [topic],
        "research_questions": [],
        "search_plan": [],
        "reason": "planner skipped",
    }
    if enable_planner:
        planner = PlannerAgent(llm=llm)
        planner_output = await planner.plan(topic=topic, user_query=user_query)
        normalized_topic = str(planner_output.get("normalized_topic") or "").strip()
        if normalized_topic:
            planned_topic = normalized_topic
        if not bool(planner_output.get("is_technical", True)):
            reason = str(planner_output.get("reason") or "non-technical request").strip()
            quality = evaluate_research_quality(
                facts=[],
                search_results=[],
                one_pager=None,
                video_brief=None,
                knowledge_gaps=[reason],
                threshold=critic_threshold,
            )
            return {
                "topic": planned_topic,
                "input_topic": topic,
                "blocked": True,
                "blocked_reason": reason,
                "planner": planner_output,
                "search_results_count": 0,
                "search_results": [],
                "facts": [],
                "knowledge_gaps": [f"Planner blocked request: {reason}"],
                "timeline": None,
                "one_pager": None,
                "video_brief": None,
                "depth_assessment": evaluate_output_depth(
                    facts=[],
                    one_pager=None,
                    video_brief=None,
                ),
                "quality_metrics": quality,
                "quality_gate_pass": False,
                "quality_recommendations": list(quality.get("recommendations", [])),
                "output_dir": None,
                "written_files": {},
                "video_artifact": None,
                "video_error": None,
                "aggregated_summary": {"total": 0},
                "search_trace": [],
                "search_coverage": {},
                "search_strategy": "blocked",
                "cache_hit": False,
            }

    cache_cfg = get_research_cache_settings()
    cache_enabled = bool(cache_cfg.enabled)
    cache_min_quality = _safe_float(cache_cfg.min_quality_score, default=0.0)

    if cache_enabled:
        store = None
        try:
            store = ResearchArtifactStore(collection_name=cache_cfg.collection_name)
            candidates = store.find_similar(
                query=planned_topic,
                score_threshold=_safe_float(cache_cfg.similarity_threshold, 0.82),
                top_k=max(1, int(cache_cfg.top_k)),
            )
            for candidate in candidates:
                if str(candidate.get("artifact_schema_version", "")).strip() != _ARTIFACT_SCHEMA_VERSION:
                    continue
                if _safe_float(candidate.get("quality_score"), 0.0) < cache_min_quality:
                    continue
                snapshot_path = Path(str(candidate.get("snapshot_path", "")).strip())
                if not snapshot_path.exists():
                    continue

                cached = _read_result_snapshot(snapshot_path)
                if not cached:
                    continue
                if generate_video and bool(cache_cfg.require_video_for_video_request):
                    if not _video_artifact_exists(cached):
                        continue

                cached["input_topic"] = topic
                cached["topic"] = cached.get("topic") or planned_topic
                cached["planner"] = planner_output
                cached["cache_hit"] = True
                cached["cache_score"] = _safe_float(candidate.get("score"), 0.0)
                cached["cache_matched_topic"] = str(candidate.get("topic", "")).strip() or cached["topic"]
                cached["search_strategy"] = "cache_reuse"
                cached.setdefault("search_trace", [])
                cached.setdefault("search_coverage", {})
                cached.setdefault("aggregated_summary", {"total": cached.get("search_results_count", 0)})
                if enable_knowledge_indexing:
                    try:
                        cached_results = list(cached.get("search_results") or [])
                        await _warmup_knowledge_base_from_search_results(cached_results)
                    except Exception as exc:
                        logger.warning(f"Cache hit KB warmup skipped: {exc}")
                    finally:
                        close_knowledge_base()
                return cached
        except Exception as exc:
            logger.warning(f"Research artifact cache lookup skipped: {exc}")
        finally:
            if store is not None:
                store.close()

    search_results: List[Dict[str, Any]] = []
    aggregated_summary: Dict[str, Any] = {}
    search_trace: List[Dict[str, Any]] = []
    search_coverage: Dict[str, Any] = {}
    search_strategy = "aggregator"

    if use_agentic_search:
        search_agent = SearchAgent(
            llm=llm,
            max_iterations=max(1, int(search_max_iterations)),
            min_total_results=max(8, max_results_per_source * 2),
            allowed_sources=resolved_sources or None,
        )
        search_output = await search_agent.run(
            {
                "topic": planned_topic,
                "user_query": user_query,
                "query_rewrites": planner_output.get("query_rewrites", []),
                "research_questions": planner_output.get("research_questions", []),
                "search_plan": planner_output.get("search_plan", []),
                "max_results_per_source": max_results_per_source,
            }
        )
        search_results = list(search_output.get("search_results", []))
        search_trace = list(search_output.get("search_trace", []))
        search_coverage = dict(search_output.get("coverage", {}))
        search_strategy = str(search_output.get("strategy", "react_agent"))
        aggregated_summary = _summarize_search_results(search_results)

    if not search_results:
        fallback_kwargs = dict(aggregator_kwargs)
        if resolved_sources:
            fallback_kwargs.update(_aggregator_kwargs_from_sources(resolved_sources))
        async with DataAggregator(**fallback_kwargs) as aggregator:
            aggregated = await aggregator.aggregate(
                topic=planned_topic,
                max_results_per_source=max_results_per_source,
                show_progress=show_progress,
            )
        search_results = aggregated_result_to_search_results(aggregated)
        aggregated_summary = aggregated.summary()
        if search_trace:
            search_strategy = f"{search_strategy}+aggregator_fallback"
        else:
            search_strategy = "aggregator"
            search_coverage = {}

    result = await run_research_from_search_results(
        topic=planned_topic,
        search_results=search_results,
        llm=llm,
        out_dir=out_dir,
        generate_video=generate_video,
        video_generator=video_generator,
        enable_knowledge_indexing=enable_knowledge_indexing,
        enable_critic_gate=enable_critic_gate,
        critic_threshold=critic_threshold,
    )
    result["input_topic"] = topic
    result["search_results"] = search_results
    result["planner"] = planner_output
    result["aggregated_summary"] = aggregated_summary
    result["search_trace"] = search_trace
    result["search_coverage"] = search_coverage
    result["search_strategy"] = search_strategy
    result["cache_hit"] = False

    output_dir_text = str(result.get("output_dir") or "").strip()
    snapshot_path: Optional[Path] = None
    if output_dir_text:
        output_dir_path = Path(output_dir_text)
        try:
            output_dir_path.mkdir(parents=True, exist_ok=True)
            snapshot_path = _write_result_snapshot(output_dir_path, result)
            result["result_snapshot_path"] = str(snapshot_path)
        except Exception as exc:
            logger.warning(f"Failed to write result snapshot: {exc}")

    if cache_enabled and snapshot_path is not None:
        store = None
        try:
            store = ResearchArtifactStore(collection_name=cache_cfg.collection_name)
            manifest_path = str((result.get("written_files") or {}).get("manifest_json", "")).strip() or None
            video_output_path = str((result.get("video_artifact") or {}).get("output_path", "")).strip() or None
            store.index_artifact(
                topic=planned_topic,
                summary_text=_build_artifact_summary_text(planned_topic, result),
                output_dir=output_dir_text,
                snapshot_path=str(snapshot_path),
                manifest_path=manifest_path,
                video_output_path=video_output_path,
                quality_score=_safe_float((result.get("quality_metrics") or {}).get("overall_score"), 0.0),
                quality_gate_pass=bool(result.get("quality_gate_pass", False)),
                search_results_count=int(result.get("search_results_count", 0)),
                artifact_schema_version=_ARTIFACT_SCHEMA_VERSION,
            )
        except Exception as exc:
            logger.warning(f"Research artifact cache index skipped: {exc}")
        finally:
            if store is not None:
                store.close()
    return result


async def chat_over_knowledge_base(
    *,
    question: str,
    llm: Optional[BaseLLM] = None,
    top_k: int = 6,
    sources: Optional[List[str]] = None,
    year_filter: Optional[int] = None,
    use_hybrid: bool = True,
) -> Dict[str, Any]:
    """
    基于向量知识库进行对话问答。
    """
    try:
        chat_agent = ChatAgent(llm=llm or get_llm(), default_top_k=top_k)
        return await chat_agent.ask(
            question=question,
            top_k=top_k,
            sources=sources,
            year_filter=year_filter,
            use_hybrid=use_hybrid,
        )
    finally:
        close_knowledge_base()


async def chat_over_kb(
    *,
    question: str,
    llm: Optional[BaseLLM] = None,
    top_k: int = 6,
    sources: Optional[List[str]] = None,
    year_filter: Optional[int] = None,
    use_hybrid: bool = True,
) -> Dict[str, Any]:
    """chat_over_knowledge_base 的简写别名。"""
    return await chat_over_knowledge_base(
        question=question,
        llm=llm,
        top_k=top_k,
        sources=sources,
        year_filter=year_filter,
        use_hybrid=use_hybrid,
    )
