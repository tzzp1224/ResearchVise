"""
Output Exporter
把 Phase 4 输出导出为文件（Markdown/JSON）
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

from .models import Timeline, OnePager, VideoBrief
from .renderers import (
    render_research_report_markdown,
    render_timeline_markdown,
    render_one_pager_markdown,
    render_video_brief_markdown,
)


def _normalize_json_obj(obj: Any) -> Any:
    if obj is None:
        return None

    # Project models (dataclass with to_dict)
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        return to_dict()

    # Pydantic v2
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(mode="json")
        except TypeError:
            return model_dump()

    # Pydantic v1
    dict_fn = getattr(obj, "dict", None)
    if callable(dict_fn):
        return dict_fn()

    return obj


def _json_dump(obj: Any) -> str:
    return json.dumps(_normalize_json_obj(obj), ensure_ascii=False, indent=2, default=str) + "\n"


def export_research_outputs(
    out_dir: Union[str, Path],
    *,
    topic: str,
    timeline: Optional[Union[Timeline, Sequence[Dict[str, Any]], Dict[str, Any]]] = None,
    one_pager: Optional[Union[OnePager, Dict[str, Any]]] = None,
    video_brief: Optional[Union[VideoBrief, Dict[str, Any]]] = None,
    write_report: bool = True,
) -> Dict[str, Path]:
    """
    导出研究输出到目录。

    写入文件：
    - timeline.md / timeline.json
    - one_pager.md / one_pager.json
    - video_brief.md / video_brief.json
    - report.md (可选)
    - manifest.json
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    written: Dict[str, Path] = {}

    # Markdown
    timeline_md = render_timeline_markdown(topic, timeline)
    one_pager_md = render_one_pager_markdown(one_pager, default_title=f"{topic} One-Pager")
    video_brief_md = render_video_brief_markdown(video_brief, default_title=f"{topic} Video Brief")

    (out_path / "timeline.md").write_text(timeline_md, encoding="utf-8")
    written["timeline_md"] = out_path / "timeline.md"

    (out_path / "one_pager.md").write_text(one_pager_md, encoding="utf-8")
    written["one_pager_md"] = out_path / "one_pager.md"

    (out_path / "video_brief.md").write_text(video_brief_md, encoding="utf-8")
    written["video_brief_md"] = out_path / "video_brief.md"

    if write_report:
        report_md = render_research_report_markdown(
            topic,
            timeline=timeline,
            one_pager=one_pager,
            video_brief=video_brief,
        )
        (out_path / "report.md").write_text(report_md, encoding="utf-8")
        written["report_md"] = out_path / "report.md"

    # JSON（尽量保留原始结构）
    (out_path / "timeline.json").write_text(_json_dump(timeline), encoding="utf-8")
    written["timeline_json"] = out_path / "timeline.json"

    (out_path / "one_pager.json").write_text(_json_dump(one_pager), encoding="utf-8")
    written["one_pager_json"] = out_path / "one_pager.json"

    (out_path / "video_brief.json").write_text(_json_dump(video_brief), encoding="utf-8")
    written["video_brief_json"] = out_path / "video_brief.json"

    manifest = {
        "topic": topic,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "files": {k: str(v.name) for k, v in written.items()},
    }
    (out_path / "manifest.json").write_text(_json_dump(manifest), encoding="utf-8")
    written["manifest_json"] = out_path / "manifest.json"

    return written
