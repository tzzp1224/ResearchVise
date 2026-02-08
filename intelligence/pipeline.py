"""
Research Pipeline
关键词驱动的端到端研究流水线（Phase 1 -> 4 + 可选视频生成）
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
from urllib.parse import urlparse

from aggregator import DataAggregator
from intelligence.agents import AnalystAgent, ContentAgent
from intelligence.llm import BaseLLM, get_llm
from models import AggregatedResult
from outputs import export_research_outputs
from outputs.video_generator import BaseVideoGenerator, create_video_generator


logger = logging.getLogger(__name__)

_REQUIRED_FACT_CATEGORIES = {
    "architecture",
    "performance",
    "training",
    "comparison",
    "limitation",
}


def _is_valid_http_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    url = value.strip()
    if not url:
        return False
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _looks_placeholder_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    url = value.strip().lower()
    if not url:
        return True
    if "example.com" in url:
        return True
    if "your-repo" in url or "your_org" in url or "your-project" in url:
        return True
    if "placeholder" in url or "todo" in url:
        return True
    return False


def _normalize_one_pager_resources(
    *,
    one_pager: Optional[Dict[str, Any]],
    facts: List[Dict[str, Any]],
    search_results: List[Dict[str, Any]],
    max_resources: int = 8,
) -> Optional[Dict[str, Any]]:
    if not one_pager:
        return one_pager

    normalized = dict(one_pager)
    id_to_result = {
        str(item.get("id", "")).strip(): item
        for item in search_results
        if str(item.get("id", "")).strip()
    }

    resources: List[Dict[str, str]] = []
    seen_urls = set()

    for resource in normalized.get("resources", []) or []:
        if not isinstance(resource, dict):
            continue
        url = str(resource.get("url", "")).strip()
        title = str(resource.get("title", "")).strip() or "Resource"
        if not _is_valid_http_url(url) or _looks_placeholder_url(url):
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)
        resources.append({"title": title, "url": url})
        if len(resources) >= max_resources:
            break

    evidence_ids: List[str] = []
    for fact in facts:
        for evidence in fact.get("evidence", []) or []:
            evidence_id = str(evidence).strip()
            if evidence_id and evidence_id not in evidence_ids:
                evidence_ids.append(evidence_id)

    for evidence_id in evidence_ids:
        item = id_to_result.get(evidence_id)
        if not item:
            continue
        url = str(item.get("url", "")).strip()
        if not _is_valid_http_url(url) or url in seen_urls:
            continue
        title = str(item.get("title", "")).strip() or evidence_id
        resources.append({"title": title, "url": url})
        seen_urls.add(url)
        if len(resources) >= max_resources:
            break

    if len(resources) < 3:
        preferred_sources = ["arxiv", "semantic_scholar", "huggingface", "github", "stackoverflow"]
        source_rank = {source: idx for idx, source in enumerate(preferred_sources)}
        sorted_results = sorted(
            search_results,
            key=lambda item: (
                source_rank.get(str(item.get("source", "")).strip(), len(preferred_sources)),
                str(item.get("title", "")).lower(),
            ),
        )
        for item in sorted_results:
            url = str(item.get("url", "")).strip()
            if not _is_valid_http_url(url) or url in seen_urls:
                continue
            title = str(item.get("title", "")).strip() or str(item.get("id", "Resource")).strip()
            resources.append({"title": title, "url": url})
            seen_urls.add(url)
            if len(resources) >= max_resources:
                break

    normalized["resources"] = resources
    return normalized


def _normalize_video_brief(
    *,
    topic: str,
    video_brief: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not video_brief:
        return video_brief

    normalized = dict(video_brief)
    raw_segments = list(normalized.get("segments") or [])

    if not raw_segments:
        raw_segments = [
            {
                "title": f"{topic} technical overview",
                "content": f"Explain core mechanisms, benchmarks, and deployment trade-offs for {topic}.",
                "talking_points": ["architecture", "performance", "deployment"],
            }
        ]

    segment_count = max(1, min(5, len(raw_segments)))
    fallback_duration = max(35, min(90, int(210 / segment_count)))
    segments: List[Dict[str, Any]] = []

    for idx, seg in enumerate(raw_segments[:5], 1):
        title = str(seg.get("title", "")).strip() or f"Segment {idx}"
        content = str(seg.get("content", "")).strip() or f"Technical segment about {topic}."
        talking_points = list(seg.get("talking_points") or [])

        duration_raw = seg.get("duration_sec")
        try:
            duration_sec = int(duration_raw)
        except Exception:
            duration_sec = fallback_duration
        duration_sec = max(20, min(120, duration_sec))

        visual_prompt = str(seg.get("visual_prompt", "")).strip()
        if not visual_prompt:
            visual_prompt = (
                f"technical explainer for {topic}, segment '{title}', "
                "cinematic macro hardware shots, architecture diagrams, "
                "benchmark overlays, precise engineering visual language"
            )

        segments.append(
            {
                "title": title,
                "content": content,
                "talking_points": [str(p).strip() for p in talking_points if str(p).strip()],
                "duration_sec": duration_sec,
                "visual_prompt": visual_prompt,
            }
        )

    normalized["segments"] = segments
    return normalized


def _safe_iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    text = str(value).strip()
    return text or None


def aggregated_result_to_search_results(result: AggregatedResult) -> List[Dict[str, Any]]:
    """
    将聚合层数据统一转换为 Agent 搜索结果格式。
    """
    items: List[Dict[str, Any]] = []

    for paper in result.papers:
        items.append(
            {
                "id": f"{paper.source.value}_{paper.id}",
                "source": paper.source.value,
                "title": paper.title,
                "content": paper.abstract,
                "url": paper.url,
                "metadata": {
                    "authors": [a.name for a in paper.authors],
                    "published_date": _safe_iso(paper.published_date),
                    "updated_date": _safe_iso(paper.updated_date),
                    "categories": paper.categories,
                    "citation_count": paper.citation_count,
                    **paper.extra,
                },
            }
        )

    for model in result.models:
        items.append(
            {
                "id": f"hf_model_{model.id}",
                "source": "huggingface",
                "title": model.name,
                "content": model.description or "",
                "url": model.url,
                "metadata": {
                    "author": model.author,
                    "downloads": model.downloads,
                    "likes": model.likes,
                    "tags": model.tags,
                    "created_at": _safe_iso(model.created_at),
                    "updated_at": _safe_iso(model.updated_at),
                    **model.extra,
                },
            }
        )

    for dataset in result.datasets:
        items.append(
            {
                "id": f"hf_dataset_{dataset.id}",
                "source": "huggingface",
                "title": dataset.name,
                "content": dataset.description or "",
                "url": dataset.url,
                "metadata": {
                    "author": dataset.author,
                    "downloads": dataset.downloads,
                    "tags": dataset.tags,
                    **dataset.extra,
                },
            }
        )

    for post in result.social_posts:
        items.append(
            {
                "id": f"{post.source.value}_{post.id}",
                "source": post.source.value,
                "title": f"{post.author} @ {post.source.value}",
                "content": post.content,
                "url": post.url,
                "metadata": {
                    "author": post.author,
                    "likes": post.likes,
                    "comments": post.comments,
                    "reposts": post.reposts,
                    "created_at": _safe_iso(post.created_at),
                    **post.extra,
                },
            }
        )

    for repo in result.github_repos:
        items.append(
            {
                "id": f"github_repo_{repo.id}",
                "source": "github",
                "title": repo.full_name,
                "content": repo.description or "",
                "url": repo.url,
                "metadata": {
                    "owner": repo.owner,
                    "language": repo.language,
                    "stars": repo.stars,
                    "forks": repo.forks,
                    "watchers": repo.watchers,
                    "topics": repo.topics,
                    "updated_at": _safe_iso(repo.updated_at),
                    **repo.extra,
                },
            }
        )

    for question in result.stackoverflow_questions:
        items.append(
            {
                "id": f"stackoverflow_{question.id}",
                "source": "stackoverflow",
                "title": question.title,
                "content": question.body or "",
                "url": question.url,
                "metadata": {
                    "author": question.author,
                    "tags": question.tags,
                    "score": question.score,
                    "view_count": question.view_count,
                    "answer_count": question.answer_count,
                    "is_answered": question.is_answered,
                    **question.extra,
                },
            }
        )

    for hn in result.hackernews_items:
        items.append(
            {
                "id": f"hackernews_{hn.id}",
                "source": "hackernews",
                "title": hn.title,
                "content": hn.text or "",
                "url": hn.url or hn.hn_url,
                "metadata": {
                    "author": hn.author,
                    "points": hn.points,
                    "comment_count": hn.comment_count,
                    "created_at": _safe_iso(hn.created_at),
                    **hn.extra,
                },
            }
        )

    return items


def evaluate_output_depth(
    *,
    facts: List[Dict[str, Any]],
    one_pager: Optional[Dict[str, Any]],
    video_brief: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    对输出深度做简单可解释评分，便于判断是否满足“技术细节充足”。
    """
    score = 0
    max_score = 13

    fact_categories = {str(f.get("category", "")).strip().lower() for f in facts}
    covered_required = sorted(list(_REQUIRED_FACT_CATEGORIES.intersection(fact_categories)))
    facts_with_evidence = [f for f in facts if f.get("evidence")]
    high_confidence = [f for f in facts if float(f.get("confidence", 0.0)) >= 0.7]

    if len(facts) >= 8:
        score += 2
    if len(covered_required) >= 4:
        score += 2
    if len(facts_with_evidence) >= max(4, len(facts) // 3):
        score += 1
    if len(high_confidence) >= max(4, len(facts) // 3):
        score += 1

    one_pager = one_pager or {}
    if len(one_pager.get("key_findings", [])) >= 5:
        score += 1
    if len(one_pager.get("metrics", {})) >= 3:
        score += 1
    if len(one_pager.get("technical_deep_dive", [])) >= 2:
        score += 1
    if len(one_pager.get("implementation_notes", [])) >= 2:
        score += 1
    if len(one_pager.get("risks_and_mitigations", [])) >= 2:
        score += 1

    video_brief = video_brief or {}
    segments = list(video_brief.get("segments", []))
    if len(segments) >= 3:
        score += 1
    if segments and all(
        bool(seg.get("visual_prompt")) and bool(seg.get("duration_sec")) for seg in segments
    ):
        score += 1

    return {
        "score": score,
        "max_score": max_score,
        "pass": score >= 8,
        "fact_count": len(facts),
        "fact_categories": sorted(list(fact_categories)),
        "covered_required_categories": covered_required,
        "facts_with_evidence": len(facts_with_evidence),
        "high_confidence_facts": len(high_confidence),
    }


async def run_research_from_search_results(
    *,
    topic: str,
    search_results: List[Dict[str, Any]],
    llm: Optional[BaseLLM] = None,
    out_dir: Optional[Path] = None,
    generate_video: bool = False,
    video_generator: Optional[BaseVideoGenerator] = None,
    enable_knowledge_indexing: bool = True,
) -> Dict[str, Any]:
    """
    使用已准备好的搜索结果执行 Phase 3 + Phase 4。
    """
    llm = llm or get_llm()

    analyst = AnalystAgent(
        llm=llm,
        enable_knowledge_indexing=enable_knowledge_indexing,
    )
    analysis = await analyst.analyze(topic, search_results)
    facts = analysis.get("facts", [])

    content_agent = ContentAgent(llm=llm)
    generated = await content_agent.generate(topic, facts)
    generated["one_pager"] = _normalize_one_pager_resources(
        one_pager=generated.get("one_pager"),
        facts=facts,
        search_results=search_results,
    )
    generated["video_brief"] = _normalize_video_brief(
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

    video_artifact = None
    video_error = None
    if generate_video:
        generator = video_generator or create_video_generator(provider="slidev")
        try:
            artifact = await generator.generate(
                topic=topic,
                out_dir=output_dir,
                video_brief=generated.get("video_brief"),
                one_pager=generated.get("one_pager"),
                facts=facts,
            )
            video_artifact = {
                "provider": artifact.provider,
                "output_path": str(artifact.output_path),
                "metadata_path": str(artifact.metadata_path),
            }
        except Exception as e:
            video_error = str(e)
            logger.warning(f"Video generation failed, keeping document outputs: {e}")

    return {
        "topic": topic,
        "search_results_count": len(search_results),
        "facts": facts,
        "knowledge_gaps": analysis.get("knowledge_gaps", []),
        "timeline": generated.get("timeline"),
        "one_pager": generated.get("one_pager"),
        "video_brief": generated.get("video_brief"),
        "depth_assessment": depth,
        "output_dir": str(output_dir),
        "written_files": {k: str(v) for k, v in written_files.items()},
        "video_artifact": video_artifact,
        "video_error": video_error,
    }


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
) -> Dict[str, Any]:
    """
    完整端到端流程：
    topic -> DataAggregator -> AnalystAgent -> ContentAgent -> export -> (optional) video.
    """
    aggregator_kwargs = aggregator_kwargs or {}

    async with DataAggregator(**aggregator_kwargs) as aggregator:
        aggregated = await aggregator.aggregate(
            topic=topic,
            max_results_per_source=max_results_per_source,
            show_progress=show_progress,
        )

    search_results = aggregated_result_to_search_results(aggregated)
    result = await run_research_from_search_results(
        topic=topic,
        search_results=search_results,
        llm=llm,
        out_dir=out_dir,
        generate_video=generate_video,
        video_generator=video_generator,
        enable_knowledge_indexing=enable_knowledge_indexing,
    )
    result["aggregated_summary"] = aggregated.summary()
    return result
