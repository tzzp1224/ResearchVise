"""
Output Renderers
将结构化输出渲染为 Markdown / Mermaid 等可读格式
"""

from __future__ import annotations

from datetime import datetime
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .models import Timeline, OnePager, VideoBrief


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _parse_iso_like_date(date_str: str) -> Tuple[int, int, int, str]:
    """
    把常见的 ISO-like 日期字符串解析为可排序 key。
    支持: YYYY, YYYY-MM, YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD
    解析失败时把原字符串放到最后做稳定排序。
    """
    s = (date_str or "").strip()
    if not s:
        return (9999, 12, 31, "")

    m = re.match(r"^(\\d{4})(?:[-/.](\\d{1,2}))?(?:[-/.](\\d{1,2}))?", s)
    if not m:
        return (9999, 12, 31, s)

    year = _safe_int(m.group(1), 9999)
    month = _safe_int(m.group(2), 12) if m.group(2) else 1
    day = _safe_int(m.group(3), 31) if m.group(3) else 1
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
    lines = ["```mermaid", "timeline", f"    title {timeline.topic}"]

    events_sorted = sorted(
        timeline.events,
        key=lambda e: _parse_iso_like_date(e.date),
    )
    for e in events_sorted:
        date = (e.date or "").strip() or "Unknown"
        title = (e.title or "").strip() or "Untitled"
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
    parts.append("| Date | Importance | Title | Description | Sources |")
    parts.append("| --- | ---: | --- | --- | --- |")

    events_sorted = sorted(
        tl.events,
        key=lambda e: (_parse_iso_like_date(e.date), -_safe_int(e.importance, 3)),
    )
    for e in events_sorted:
        date = (e.date or "").strip()
        importance = max(1, min(5, _safe_int(e.importance, 3)))
        title = (e.title or "").strip().replace("|", "\\|")
        desc = (e.description or "").strip().replace("\n", " ").replace("|", "\\|")
        sources = ", ".join([str(s) for s in (e.source_refs or [])]) if e.source_refs else ""
        parts.append(f"| {date} | {importance} | {title} | {desc} | {sources} |")

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
            visual_prompt = str(seg.get("visual_prompt", "")).strip()

            parts.append(f"### {i}. {seg_title}")
            parts.append("")
            if duration:
                parts.append(f"- Duration: {duration}s")
            if visual_prompt:
                parts.append(f"- Visual prompt: {visual_prompt}")
            if duration or visual_prompt:
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
) -> str:
    """
    组合渲染：将三种输出整合为一份 Markdown 报告。
    """
    parts = [
        f"# Research Report: {topic}",
        "",
        f"_Generated at: {datetime.now().isoformat(timespec='seconds')}_",
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
    ]
    return "\n".join(parts).strip() + "\n"
