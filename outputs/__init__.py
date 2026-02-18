"""
Outputs Module (Phase 4)
输出层 - Timeline / One-Pager / Video Brief 的结构化建模、渲染与导出
"""

from .models import (
    TimelineEvent,
    Timeline,
    OnePager,
    VideoBrief,
    ResearchOutputs,
)
from .renderers import (
    render_timeline_markdown,
    render_one_pager_markdown,
    render_video_brief_markdown,
    render_research_report_markdown,
    render_knowledge_tree_markdown,
    build_knowledge_tree,
)
from .exporter import export_research_outputs
from .video_generator import (
    VideoArtifact,
    VideoGenerationError,
    BaseVideoGenerator,
    SlidevVideoGenerator,
    create_video_generator,
    build_video_prompt,
)

__all__ = [
    # Models
    "TimelineEvent",
    "Timeline",
    "OnePager",
    "VideoBrief",
    "ResearchOutputs",
    # Renderers
    "render_timeline_markdown",
    "render_one_pager_markdown",
    "render_video_brief_markdown",
    "render_research_report_markdown",
    "render_knowledge_tree_markdown",
    "build_knowledge_tree",
    # Export
    "export_research_outputs",
    # Video
    "VideoArtifact",
    "VideoGenerationError",
    "BaseVideoGenerator",
    "SlidevVideoGenerator",
    "create_video_generator",
    "build_video_prompt",
]
