"""
Output Models
输出层数据结构（Phase 4）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class TimelineEvent:
    """时间轴事件"""

    date: str  # ISO 格式日期，如 "2024-01"
    title: str
    description: str
    importance: int = 3  # 1-5
    source_refs: List[str] = field(default_factory=list)  # 事实 ID

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "title": self.title,
            "description": self.description,
            "importance": self.importance,
            "source_refs": list(self.source_refs),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimelineEvent":
        return cls(
            date=str(data.get("date", "")),
            title=str(data.get("title", "")),
            description=str(data.get("description", "")),
            importance=int(data.get("importance", 3) or 3),
            source_refs=list(data.get("source_refs") or []),
        )


@dataclass
class Timeline:
    """时间线（包含多个事件）"""

    topic: str
    events: List[TimelineEvent] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_events(
        cls,
        topic: str,
        events: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> "Timeline":
        return cls(
            topic=topic,
            events=[TimelineEvent.from_dict(e) for e in (events or [])],
        )


@dataclass
class OnePager:
    """一页纸摘要"""

    title: str
    executive_summary: str
    key_findings: List[str] = field(default_factory=list)
    metrics: Dict[str, str] = field(default_factory=dict)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    technical_deep_dive: List[str] = field(default_factory=list)
    implementation_notes: List[str] = field(default_factory=list)
    risks_and_mitigations: List[str] = field(default_factory=list)
    resources: List[Dict[str, str]] = field(default_factory=list)  # {"title": "", "url": ""}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "executive_summary": self.executive_summary,
            "key_findings": list(self.key_findings),
            "metrics": dict(self.metrics),
            "strengths": list(self.strengths),
            "weaknesses": list(self.weaknesses),
            "technical_deep_dive": list(self.technical_deep_dive),
            "implementation_notes": list(self.implementation_notes),
            "risks_and_mitigations": list(self.risks_and_mitigations),
            "resources": list(self.resources),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, default_title: str = "") -> "OnePager":
        return cls(
            title=str(data.get("title") or default_title),
            executive_summary=str(data.get("executive_summary", "")),
            key_findings=list(data.get("key_findings") or []),
            metrics={str(k): str(v) for k, v in (data.get("metrics") or {}).items()},
            strengths=list(data.get("strengths") or []),
            weaknesses=list(data.get("weaknesses") or []),
            technical_deep_dive=list(data.get("technical_deep_dive") or []),
            implementation_notes=list(data.get("implementation_notes") or []),
            risks_and_mitigations=list(data.get("risks_and_mitigations") or []),
            resources=list(data.get("resources") or []),
        )


@dataclass
class VideoBrief:
    """视频简报脚本"""

    title: str
    duration_estimate: str
    hook: str
    segments: List[Dict[str, Any]] = field(default_factory=list)  # {"title": "", "content": "", "talking_points": []}
    visual_style: str = ""
    target_audience: str = ""
    conclusion: str = ""
    call_to_action: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "duration_estimate": self.duration_estimate,
            "hook": self.hook,
            "segments": list(self.segments),
            "visual_style": self.visual_style,
            "target_audience": self.target_audience,
            "conclusion": self.conclusion,
            "call_to_action": self.call_to_action,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, default_title: str = "") -> "VideoBrief":
        return cls(
            title=str(data.get("title") or default_title),
            duration_estimate=str(data.get("duration_estimate", "")),
            hook=str(data.get("hook", "")),
            segments=list(data.get("segments") or []),
            visual_style=str(data.get("visual_style", "")),
            target_audience=str(data.get("target_audience", "")),
            conclusion=str(data.get("conclusion", "")),
            call_to_action=str(data.get("call_to_action", "")),
        )


@dataclass
class ResearchOutputs:
    """研究输出汇总（Phase 4 对外传递的统一结构）"""

    topic: str
    timeline: Optional[Timeline] = None
    one_pager: Optional[OnePager] = None
    video_brief: Optional[VideoBrief] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "timeline": self.timeline.to_dict() if self.timeline else None,
            "one_pager": self.one_pager.to_dict() if self.one_pager else None,
            "video_brief": self.video_brief.to_dict() if self.video_brief else None,
        }
