"""
Helper functions for research pipeline data transformation and normalization.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from models import AggregatedResult

_REQUIRED_FACT_CATEGORIES = {
    "architecture",
    "performance",
    "training",
    "comparison",
    "limitation",
}

_PREFERRED_RESOURCE_SOURCES = (
    "arxiv",
    "semantic_scholar",
    "huggingface",
    "github",
    "stackoverflow",
)

_CONFLICT_HINTS = ("vs", "versus", "trade-off", "however", "but", "对比", "取舍", "风险", "缓解")

_ACTION_HINTS = (
    "deploy",
    "monitor",
    "rollback",
    "config",
    "pipeline",
    "instrument",
    "步骤",
    "监控",
    "回滚",
    "配置",
    "上线",
)


def _safe_iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    text = str(value).strip()
    return text or None


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
    return any(
        marker in url
        for marker in (
            "example.com",
            "your-repo",
            "your_org",
            "your-project",
            "placeholder",
            "todo",
        )
    )


def normalize_one_pager_resources(
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
        if not _is_valid_http_url(url) or _looks_placeholder_url(url) or url in seen_urls:
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
        source_rank = {source: idx for idx, source in enumerate(_PREFERRED_RESOURCE_SOURCES)}
        sorted_results = sorted(
            search_results,
            key=lambda item: (
                source_rank.get(str(item.get("source", "")).strip(), len(_PREFERRED_RESOURCE_SOURCES)),
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


def normalize_video_brief(
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

    for idx, segment in enumerate(raw_segments[:5], 1):
        title = str(segment.get("title", "")).strip() or f"Segment {idx}"
        content = str(segment.get("content", "")).strip() or f"Technical segment about {topic}."
        talking_points = list(segment.get("talking_points") or [])

        duration_raw = segment.get("duration_sec")
        try:
            duration_sec = int(duration_raw)
        except Exception:
            duration_sec = fallback_duration
        duration_sec = max(20, min(120, duration_sec))

        visual_prompt = str(segment.get("visual_prompt", "")).strip()
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
        bool(segment.get("visual_prompt")) and bool(segment.get("duration_sec"))
        for segment in segments
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


def evaluate_research_quality(
    *,
    facts: List[Dict[str, Any]],
    search_results: List[Dict[str, Any]],
    one_pager: Optional[Dict[str, Any]],
    video_brief: Optional[Dict[str, Any]],
    knowledge_gaps: Optional[List[str]] = None,
    threshold: float = 0.65,
) -> Dict[str, Any]:
    """
    评估研究输出质量（用于 Critic Gate）。

    指标:
    - coverage_score
    - citation_density
    - cross_source_ratio
    - conflict_resolution_rate
    - actionability_score
    """
    facts = list(facts or [])
    search_results = list(search_results or [])
    one_pager = one_pager or {}
    video_brief = video_brief or {}
    knowledge_gaps = list(knowledge_gaps or [])

    source_by_id = {
        str(item.get("id", "")).strip(): str(item.get("source", "")).strip()
        for item in search_results
        if str(item.get("id", "")).strip()
    }

    fact_categories = {str(f.get("category", "")).strip().lower() for f in facts}
    covered_required = sorted(list(_REQUIRED_FACT_CATEGORIES.intersection(fact_categories)))
    missing_required = sorted(list(_REQUIRED_FACT_CATEGORIES - set(covered_required)))
    coverage_score = len(covered_required) / max(1, len(_REQUIRED_FACT_CATEGORIES))

    evidence_sizes: List[int] = []
    cross_source_count = 0
    for fact in facts:
        evidence_ids = [str(item).strip() for item in (fact.get("evidence", []) or []) if str(item).strip()]
        unique_evidence_ids = list(dict.fromkeys(evidence_ids))
        evidence_sizes.append(len(unique_evidence_ids))
        sources = {source_by_id.get(eid, "") for eid in unique_evidence_ids if source_by_id.get(eid, "")}
        if len(sources) >= 2:
            cross_source_count += 1

    avg_citations = (sum(evidence_sizes) / len(evidence_sizes)) if evidence_sizes else 0.0
    citation_density = min(1.0, avg_citations / 2.0)
    cross_source_ratio = (cross_source_count / len(facts)) if facts else 0.0

    conflict_candidates = [
        item
        for item in facts
        if str(item.get("category", "")).strip().lower() in {"comparison", "limitation"}
    ]
    resolved = 0
    for item in conflict_candidates:
        claim = str(item.get("claim", "")).lower()
        if any(token in claim for token in _CONFLICT_HINTS):
            resolved += 1
    conflict_resolution_rate = (resolved / len(conflict_candidates)) if conflict_candidates else 1.0

    implementation_notes = [str(item).strip() for item in (one_pager.get("implementation_notes", []) or []) if str(item).strip()]
    risks = [str(item).strip() for item in (one_pager.get("risks_and_mitigations", []) or []) if str(item).strip()]
    key_findings = [str(item).strip() for item in (one_pager.get("key_findings", []) or []) if str(item).strip()]
    segments = list(video_brief.get("segments", []) or [])

    action_signals = 0
    if len(implementation_notes) >= 2:
        action_signals += 1
    if len(risks) >= 2:
        action_signals += 1
    if len(one_pager.get("metrics", {}) or {}) >= 3:
        action_signals += 1
    if segments and all(bool(seg.get("talking_points")) for seg in segments[:3]):
        action_signals += 1
    if any(any(token in note.lower() for token in _ACTION_HINTS) for note in implementation_notes):
        action_signals += 1
    actionability_score = action_signals / 5.0

    gap_penalty = min(0.25, 0.03 * len(knowledge_gaps))
    overall_score = (
        0.25 * coverage_score
        + 0.20 * citation_density
        + 0.20 * cross_source_ratio
        + 0.15 * conflict_resolution_rate
        + 0.20 * actionability_score
    )
    overall_score = max(0.0, min(1.0, overall_score - gap_penalty))

    recommendations: List[str] = []
    if coverage_score < 0.7 and missing_required:
        recommendations.append(
            "补齐核心维度证据，优先补 architecture/performance/training/comparison/limitation 缺口。"
        )
    if citation_density < 0.6:
        recommendations.append("提高每条关键结论的证据密度，至少 2 个独立 evidence。")
    if cross_source_ratio < 0.5:
        recommendations.append("增加跨来源交叉验证（论文+代码+社区），降低单源偏差。")
    if conflict_resolution_rate < 0.5:
        recommendations.append("补充争议点与取舍分析，明确风险及缓解策略。")
    if actionability_score < 0.6:
        recommendations.append("强化可执行落地信息（配置步骤、监控指标、回滚策略）。")
    if not recommendations:
        recommendations.append("质量门控通过，下一步可针对细分场景扩展样本与对比基线。")

    gate_pass = (
        overall_score >= threshold
        and coverage_score >= 0.6
        and citation_density >= 0.5
        and actionability_score >= 0.5
    )

    return {
        "coverage_score": round(coverage_score, 4),
        "citation_density": round(citation_density, 4),
        "cross_source_ratio": round(cross_source_ratio, 4),
        "conflict_resolution_rate": round(conflict_resolution_rate, 4),
        "actionability_score": round(actionability_score, 4),
        "overall_score": round(overall_score, 4),
        "threshold": float(threshold),
        "pass": gate_pass,
        "covered_required_categories": covered_required,
        "missing_required_categories": missing_required,
        "facts_with_cross_source_evidence": cross_source_count,
        "avg_citations_per_fact": round(avg_citations, 4),
        "knowledge_gap_count": len(knowledge_gaps),
        "recommendations": recommendations,
        "fact_count": len(facts),
        "search_result_count": len(search_results),
        "key_findings_count": len(key_findings),
        "implementation_note_count": len(implementation_notes),
    }
