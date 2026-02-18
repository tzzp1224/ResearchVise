"""Canonical data contracts for the v2 run/render pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class RunMode(str, Enum):
    """Trigger mode for a run request."""

    DAILY = "daily"
    ONDEMAND = "ondemand"


class ArtifactType(str, Enum):
    """Supported artifact categories."""

    SCRIPT = "script"
    STORYBOARD = "storyboard"
    ONEPAGER = "onepager"
    MP4 = "mp4"
    THUMBNAIL = "thumbnail"
    ZIP = "zip"
    AUDIO = "audio"
    SRT = "srt"


class StatusTimestamps(BaseModel):
    """Lifecycle timestamps for run/render status."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None


class Citation(BaseModel):
    """Evidence citation tied to extracted/normalized content."""

    title: str = ""
    url: str = ""
    snippet: str = ""
    source: str = ""


class RunRequest(BaseModel):
    """Unified run request contract for daily and on-demand triggers."""

    user_id: str
    mode: RunMode
    topic: Optional[str] = None
    time_window: Optional[str] = None
    tz: str = "UTC"
    budget: Optional[Dict[str, Any]] = None
    output_targets: List[str] = Field(default_factory=list)

    @field_validator("user_id", "tz", mode="before")
    @classmethod
    def _non_empty_text(cls, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("value is required")
        return text

    @field_validator("topic", "time_window", mode="before")
    @classmethod
    def _optional_text(cls, value: Any) -> Optional[str]:
        text = str(value or "").strip()
        return text or None


class NormalizedItem(BaseModel):
    """Canonical normalized item used across ranking/generation/render stages."""

    id: str
    source: str
    title: str
    url: str
    author: Optional[str] = None
    published_at: Optional[datetime] = None
    body_md: str = ""
    citations: List[Citation] = Field(default_factory=list)
    tier: Literal["A", "B"]
    lang: str = "en"
    hash: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Shot(BaseModel):
    """Engineering shot spec inside a storyboard."""

    idx: int
    duration: float
    camera: str
    scene: str
    subject_id: Optional[str] = None
    action: str
    overlay_text: Optional[str] = None
    reference_assets: List[str] = Field(default_factory=list)


class Storyboard(BaseModel):
    """Storyboard contract for render orchestration."""

    run_id: str
    item_id: str
    duration_sec: int
    aspect: str
    shots: List[Shot] = Field(default_factory=list)


class PromptSpec(BaseModel):
    """Compiled shot-level prompt ready for renderer adapters."""

    shot_idx: int
    prompt_text: str
    negative_prompt: Optional[str] = None
    references: List[str] = Field(default_factory=list)
    seedance_params: Dict[str, Any] = Field(default_factory=dict)


class Artifact(BaseModel):
    """Output artifact pointer (local path or remote URL)."""

    type: ArtifactType
    path: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RunStatus(BaseModel):
    """Observable run status for orchestration APIs."""

    run_id: str
    state: str
    progress: float = 0.0
    errors: List[str] = Field(default_factory=list)
    timestamps: StatusTimestamps = Field(default_factory=StatusTimestamps)
    cancellation_requested: bool = False
    retry_count: int = 0
    max_retries: int = 1


class RenderStatus(BaseModel):
    """Observable render job status with tracing and retries."""

    render_job_id: str
    run_id: str
    state: str
    progress: float = 0.0
    errors: List[str] = Field(default_factory=list)
    timestamps: StatusTimestamps = Field(default_factory=StatusTimestamps)
    cancellation_requested: bool = False
    retry_count: int = 0
    max_retries: int = 1
    failed_shot_indices: List[int] = Field(default_factory=list)
