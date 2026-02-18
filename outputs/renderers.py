"""
Output Renderers
将结构化输出渲染为 Markdown / Mermaid 等可读格式
"""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .models import Timeline, OnePager, VideoBrief


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _format_mmss(value: Any) -> str:
    try:
        total = max(0, int(float(value)))
    except Exception:
        total = 0
    minutes, seconds = divmod(total, 60)
    return f"{minutes:02d}:{seconds:02d}"


def _truncate_text(value: str, max_len: int = 80) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _sanitize_mermaid_label(value: str, max_len: int = 48) -> str:
    text = str(value or "")
    text = re.sub(r"[`|{}\[\]\"'()（）<>]", "", text)
    text = re.sub(r"[:;,#]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return _truncate_text(text, max_len=max_len) or "N/A"


def _parse_iso_like_date(date_str: str) -> Tuple[int, int, int, str]:
    """
    把常见的 ISO-like 日期字符串解析为可排序 key。
    支持: YYYY, YYYY-MM, YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD
    解析失败时把原字符串放到最后做稳定排序。
    """
    s = (date_str or "").strip()
    if not s:
        return (9999, 12, 31, "")

    m = re.match(r"^(\d{4})(?:[-/.](\d{1,2}))?(?:[-/.](\d{1,2}))?", s)
    if not m:
        return (9999, 12, 31, s)

    year = _safe_int(m.group(1), 9999)
    month = _safe_int(m.group(2), 12) if m.group(2) else 1
    day = _safe_int(m.group(3), 31) if m.group(3) else 1
    month = max(1, min(12, month))
    day = max(1, min(31, day))
    return (year, month, day, s)


def _normalize_timeline_events(
    timeline: Optional[Union[Timeline, Sequence[Dict[str, Any]], Dict[str, Any]]],
    topic: str,
) -> Timeline:
    if timeline is None:
        return Timeline(topic=topic, events=[])

    if isinstance(timeline, Timeline):
        return timeline

    # 兼容: ContentAgent 返回 List[Dict]
    if isinstance(timeline, list):
        return Timeline.from_events(topic=topic, events=timeline)

    # 兼容: {"events": [...]}
    if isinstance(timeline, dict):
        return Timeline.from_events(topic=topic, events=timeline.get("events") or [])

    return Timeline(topic=topic, events=[])


def render_timeline_mermaid(timeline: Timeline) -> str:
    """
    渲染 Mermaid timeline 图（尽量兼容常见 Markdown 渲染器）。
    """
    lines = [
        "```mermaid",
        "%%{init: {'theme': 'base', 'themeVariables': {'primaryTextColor': '#1f2d36', 'lineColor': '#506877', 'fontSize': '15px'}}}%%",
        "timeline",
        f"    title {_sanitize_mermaid_label(timeline.topic, max_len=56)}",
    ]

    events_sorted = sorted(
        timeline.events,
        key=lambda e: _parse_iso_like_date(e.date),
    )
    for e in events_sorted:
        date = (e.date or "").strip() or "Unknown"
        title = _sanitize_mermaid_label((e.title or "").strip() or "Untitled")
        lines.append(f"    {date} : {title}")

    lines.append("```")
    return "\n".join(lines)


def render_timeline_markdown(
    topic: str,
    timeline: Optional[Union[Timeline, Sequence[Dict[str, Any]], Dict[str, Any]]],
    *,
    include_mermaid: bool = True,
) -> str:
    """
    渲染 Timeline 为 Markdown。
    """
    tl = _normalize_timeline_events(timeline, topic=topic)

    parts: List[str] = [f"# {topic} Timeline", ""]

    if include_mermaid and tl.events:
        parts.append(render_timeline_mermaid(tl))
        parts.append("")

    if not tl.events:
        parts.append("_No timeline events._")
        return "\n".join(parts).strip() + "\n"

    parts.append("## Events")
    parts.append("")
    parts.append("| Date | Title | Description |")
    parts.append("| --- | --- | --- |")

    events_sorted = sorted(
        tl.events,
        key=lambda e: (_parse_iso_like_date(e.date), -_safe_int(e.importance, 3)),
    )
    for e in events_sorted:
        date = (e.date or "").strip()
        title = (e.title or "").strip().replace("|", "\\|")
        desc = (e.description or "").strip().replace("\n", " ").replace("|", "\\|")
        parts.append(f"| {date} | {title} | {desc} |")

    return "\n".join(parts).strip() + "\n"


def build_knowledge_tree(
    topic: str,
    *,
    timeline: Optional[Union[Timeline, Sequence[Dict[str, Any]], Dict[str, Any]]] = None,
    one_pager: Optional[Union[OnePager, Dict[str, Any]]] = None,
    facts: Optional[Sequence[Dict[str, Any]]] = None,
    search_results: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    tl = _normalize_timeline_events(timeline, topic=topic)
    op = one_pager if isinstance(one_pager, OnePager) else OnePager.from_dict(one_pager or {}, default_title="")
    facts_list = list(facts or [])
    results = list(search_results or [])

    topic_lower = str(topic or "").lower()
    prerequisites: List[str] = []
    if "mcp" in topic_lower or "model context protocol" in topic_lower:
        prerequisites.extend(
            [
                "LLM Tool Calling 与函数调用协议",
                "服务鉴权（OAuth2/JWT/API Key）与密钥管理",
                "分布式系统可靠性（重试、幂等、超时、熔断）",
                "可观测性（日志、指标、链路追踪）",
                "生产部署（灰度发布、回滚、容量规划）",
            ]
        )
    if "reinforcement learning" in topic_lower or "强化学习" in topic_lower:
        prerequisites.extend(
            [
                "马尔可夫决策过程（MDP）",
                "价值函数与策略梯度",
                "探索-利用权衡",
                "离线/在线评估与安全约束",
            ]
        )

    category_to_prereq = {
        "architecture": "系统架构设计与模块边界划分",
        "training": "训练数据与优化目标定义",
        "performance": "性能基准设计与指标口径",
        "comparison": "替代方案对比与取舍分析",
        "limitation": "风险建模与失效模式分析",
        "deployment": "工程部署、监控与回滚策略",
    }
    for fact in facts_list:
        category = str(fact.get("category", "")).strip().lower()
        prereq = category_to_prereq.get(category)
        if prereq:
            prerequisites.append(prereq)

    for item in (op.technical_deep_dive or [])[:4]:
        cleaned = _truncate_text(item, max_len=52)
        if cleaned:
            prerequisites.append(cleaned)

    milestones: List[Dict[str, str]] = []
    events_sorted = sorted(tl.events, key=lambda e: _parse_iso_like_date(e.date))
    for event in events_sorted[:8]:
        milestones.append(
            {
                "date": str(event.date or "").strip(),
                "title": _truncate_text(str(event.title or "").strip(), max_len=52),
                "description": _truncate_text(str(event.description or "").strip(), max_len=110),
            }
        )

    if not milestones:
        for idx, fact in enumerate(facts_list[:6], start=1):
            claim = _truncate_text(str(fact.get("claim", "")).strip(), max_len=90)
            if not claim:
                continue
            milestones.append(
                {
                    "date": f"Step {idx}",
                    "title": claim,
                    "description": claim,
                }
            )

    evidence_priority = {"arxiv": 0, "semantic_scholar": 1, "github": 2, "huggingface": 3}
    results_sorted = sorted(
        results,
        key=lambda item: (
            evidence_priority.get(str(item.get("source", "")).strip(), 9),
            -float(item.get("score", 0) or 0),
        ),
    )
    key_resources: List[Dict[str, str]] = []
    seen_urls = set()
    for item in results_sorted:
        source = str(item.get("source", "")).strip()
        if source not in evidence_priority:
            continue
        url = str(item.get("url", "")).strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        key_resources.append(
            {
                "title": _truncate_text(str(item.get("title", "")).strip() or "Untitled", max_len=70),
                "url": url,
                "source": source,
            }
        )
        if len(key_resources) >= 10:
            break

    if len(key_resources) < 3:
        for resource in list(op.resources or [])[:10]:
            if not isinstance(resource, dict):
                continue
            url = str(resource.get("url", "")).strip()
            if not re.match(r"^https?://", url) or url in seen_urls:
                continue
            title = _truncate_text(str(resource.get("title", "")).strip() or "Resource", max_len=70)
            seen_urls.add(url)
            key_resources.append({"title": title, "url": url, "source": "one_pager"})
            if len(key_resources) >= 10:
                break

    current_focus: List[str] = []
    current_focus.extend([_truncate_text(x, max_len=72) for x in (op.key_findings or [])[:6] if str(x).strip()])
    current_focus.extend([_truncate_text(x, max_len=72) for x in (op.implementation_notes or [])[:4] if str(x).strip()])
    if not current_focus:
        current_focus.extend([_truncate_text(str(f.get("claim", "")).strip(), max_len=72) for f in facts_list[:6]])

    deduped_prereq: List[str] = []
    seen_pre = set()
    for item in prerequisites:
        key = str(item or "").strip()
        if not key or key in seen_pre:
            continue
        seen_pre.add(key)
        deduped_prereq.append(key)

    deduped_focus: List[str] = []
    seen_focus = set()
    for item in current_focus:
        key = str(item or "").strip()
        if not key or key in seen_focus:
            continue
        seen_focus.add(key)
        deduped_focus.append(key)

    return {
        "topic": topic,
        "prerequisites": deduped_prereq[:10],
        "milestones": milestones[:10],
        "key_resources": key_resources,
        "current_focus": deduped_focus[:10],
    }


def render_knowledge_tree_markdown(
    topic: str,
    *,
    timeline: Optional[Union[Timeline, Sequence[Dict[str, Any]], Dict[str, Any]]] = None,
    one_pager: Optional[Union[OnePager, Dict[str, Any]]] = None,
    facts: Optional[Sequence[Dict[str, Any]]] = None,
    search_results: Optional[Sequence[Dict[str, Any]]] = None,
    knowledge_tree: Optional[Dict[str, Any]] = None,
) -> str:
    tree = knowledge_tree or build_knowledge_tree(
        topic=topic,
        timeline=timeline,
        one_pager=one_pager,
        facts=facts,
        search_results=search_results,
    )

    prerequisites = list(tree.get("prerequisites") or [])
    milestones = list(tree.get("milestones") or [])
    resources = list(tree.get("key_resources") or [])
    focus = list(tree.get("current_focus") or [])

    parts: List[str] = [f"# {topic} Knowledge Tree", ""]
    parts.append("```mermaid")
    parts.append(
        "%%{init: {'theme': 'base', 'themeVariables': {'primaryTextColor': '#1f2d36', 'lineColor': '#506877', 'fontSize': '15px'}}}%%"
    )
    parts.append("mindmap")
    parts.append(f"  root(({_sanitize_mermaid_label(topic, max_len=52)}))")
    parts.append("    前置知识")
    for item in prerequisites[:6]:
        parts.append(f"      {_sanitize_mermaid_label(item)}")
    parts.append("    演进路径")
    for item in milestones[:6]:
        date = str(item.get("date", "")).strip()
        title = str(item.get("title", "")).strip()
        label = f"{date} {title}".strip()
        parts.append(f"      {_sanitize_mermaid_label(label)}")
    parts.append("    当前焦点")
    for item in focus[:5]:
        parts.append(f"      {_sanitize_mermaid_label(item)}")
    parts.append("```")
    parts.append("")

    parts.append("## 前置知识")
    parts.append("")
    for item in prerequisites or ["当前检索结果尚未覆盖稳定前置知识，请补充高置信来源后重试。"]:
        parts.append(f"- {item}")
    parts.append("")

    parts.append("## 演进路径（通向当前技术）")
    parts.append("")
    if milestones:
        for idx, item in enumerate(milestones, start=1):
            date = str(item.get("date", "")).strip()
            title = str(item.get("title", "")).strip()
            description = str(item.get("description", "")).strip()
            parts.append(f"{idx}. **{date or 'Unknown'} · {title or 'Untitled'}**")
            if description:
                parts.append(f"   {description}")
    else:
        parts.append("1. 暂无可用里程碑数据。")
    parts.append("")

    parts.append("## 关键论文与工程实现")
    parts.append("")
    if resources:
        for item in resources:
            title = str(item.get("title", "")).strip() or "Untitled"
            url = str(item.get("url", "")).strip()
            source = str(item.get("source", "")).strip() or "unknown"
            parts.append(f"- [{title}]({url})（{source}）")
    else:
        parts.append("- 当前检索结果未提供可验证的公开资源链接。")
    parts.append("")

    parts.append("## 当前技术焦点")
    parts.append("")
    for item in focus or ["当前检索结果尚未形成稳定技术焦点，请优先补充核心证据。"]:
        parts.append(f"- {item}")
    parts.append("")

    return "\n".join(parts).strip() + "\n"


def render_one_pager_markdown(
    one_pager: Optional[Union[OnePager, Dict[str, Any]]],
    *,
    default_title: str = "One-Pager",
) -> str:
    """
    渲染 One-Pager 为 Markdown。
    """
    if one_pager is None:
        return f"# {default_title}\n\n_No one-pager generated._\n"

    op = one_pager if isinstance(one_pager, OnePager) else OnePager.from_dict(one_pager, default_title=default_title)

    parts: List[str] = [f"# {op.title}".strip() or f"# {default_title}", ""]

    if op.executive_summary:
        parts.append(f"> {op.executive_summary.strip()}")
        parts.append("")

    if op.key_findings:
        parts.append("## Key Findings")
        parts.append("")
        parts.extend([f"- {str(x).strip()}" for x in op.key_findings if str(x).strip()])
        parts.append("")

    if op.metrics:
        parts.append("## Metrics")
        parts.append("")
        parts.append("| Metric | Value |")
        parts.append("| --- | --- |")
        for k, v in op.metrics.items():
            metric = str(k).replace("|", "\\|")
            value = str(v).replace("|", "\\|")
            parts.append(f"| {metric} | {value} |")
        parts.append("")

    if op.strengths:
        parts.append("## Strengths")
        parts.append("")
        parts.extend([f"- {str(x).strip()}" for x in op.strengths if str(x).strip()])
        parts.append("")

    if op.weaknesses:
        parts.append("## Weaknesses")
        parts.append("")
        parts.extend([f"- {str(x).strip()}" for x in op.weaknesses if str(x).strip()])
        parts.append("")

    if op.technical_deep_dive:
        parts.append("## Technical Deep Dive")
        parts.append("")
        parts.extend([f"- {str(x).strip()}" for x in op.technical_deep_dive if str(x).strip()])
        parts.append("")

    if op.implementation_notes:
        parts.append("## Implementation Notes")
        parts.append("")
        parts.extend([f"- {str(x).strip()}" for x in op.implementation_notes if str(x).strip()])
        parts.append("")

    if op.risks_and_mitigations:
        parts.append("## Risks and Mitigations")
        parts.append("")
        parts.extend([f"- {str(x).strip()}" for x in op.risks_and_mitigations if str(x).strip()])
        parts.append("")

    if op.resources:
        parts.append("## Resources")
        parts.append("")
        for r in op.resources:
            title = str(r.get("title", "")).strip() or "Resource"
            url = str(r.get("url", "")).strip()
            if url:
                parts.append(f"- [{title}]({url})")
            else:
                parts.append(f"- {title}")
        parts.append("")

    return "\n".join(parts).strip() + "\n"


def render_video_brief_markdown(
    video_brief: Optional[Union[VideoBrief, Dict[str, Any]]],
    *,
    default_title: str = "Video Brief",
) -> str:
    """
    渲染 Video Brief 为 Markdown。
    """
    if video_brief is None:
        return f"# {default_title}\n\n_No video brief generated._\n"

    vb = video_brief if isinstance(video_brief, VideoBrief) else VideoBrief.from_dict(video_brief, default_title=default_title)
    title = vb.title.strip() or default_title

    parts: List[str] = [f"# {title}", ""]

    if vb.duration_estimate:
        parts.append(f"- Duration: {vb.duration_estimate.strip()}")
    if vb.target_audience:
        parts.append(f"- Audience: {vb.target_audience.strip()}")
    if vb.visual_style:
        parts.append(f"- Visual style: {vb.visual_style.strip()}")
    if vb.duration_estimate or vb.target_audience or vb.visual_style:
        parts.append("")

    if vb.hook:
        parts.append("## Hook")
        parts.append("")
        parts.append(vb.hook.strip())
        parts.append("")

    if vb.segments:
        parts.append("## Segments")
        parts.append("")
        for i, seg in enumerate(vb.segments, 1):
            seg_title = str(seg.get("title", "")).strip() or f"Segment {i}"
            seg_content = str(seg.get("content", "")).strip()
            points = seg.get("talking_points") or []
            duration = seg.get("duration_sec")
            start_sec = seg.get("start_sec")
            visual_prompt = str(seg.get("visual_prompt", "")).strip()

            parts.append(f"### {i}. {seg_title}")
            parts.append("")
            if start_sec is not None:
                parts.append(f"- Start: {_format_mmss(start_sec)}")
            if duration:
                parts.append(f"- Duration: {duration}s")
            if visual_prompt:
                parts.append(f"- Visual prompt: {visual_prompt}")
            if start_sec is not None or duration or visual_prompt:
                parts.append("")
            if seg_content:
                parts.append(seg_content)
                parts.append("")

            if points:
                parts.append("- Talking points:")
                parts.extend([f"  - {str(p).strip()}" for p in points if str(p).strip()])
                parts.append("")

    if vb.conclusion:
        parts.append("## Conclusion")
        parts.append("")
        parts.append(vb.conclusion.strip())
        parts.append("")

    if vb.call_to_action:
        parts.append("## Call to Action")
        parts.append("")
        parts.append(vb.call_to_action.strip())
        parts.append("")

    return "\n".join(parts).strip() + "\n"


def render_research_report_markdown(
    topic: str,
    *,
    timeline: Optional[Union[Timeline, Sequence[Dict[str, Any]], Dict[str, Any]]] = None,
    one_pager: Optional[Union[OnePager, Dict[str, Any]]] = None,
    video_brief: Optional[Union[VideoBrief, Dict[str, Any]]] = None,
    facts: Optional[Sequence[Dict[str, Any]]] = None,
    search_results: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    """
    组合渲染：将三种输出整合为一份 Markdown 报告。
    """
    parts = [
        f"# Research Report: {topic}",
        "",
        f"_Generated at (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}_",
        "",
        "---",
        "",
        render_one_pager_markdown(one_pager, default_title=f"{topic} One-Pager").strip(),
        "",
        "---",
        "",
        render_timeline_markdown(topic, timeline).strip(),
        "",
        "---",
        "",
        render_video_brief_markdown(video_brief, default_title=f"{topic} Video Brief").strip(),
        "",
        "---",
        "",
        render_knowledge_tree_markdown(
            topic,
            timeline=timeline,
            one_pager=one_pager,
            facts=facts,
            search_results=search_results,
        ).strip(),
        "",
    ]
    return "\n".join(parts).strip() + "\n"
