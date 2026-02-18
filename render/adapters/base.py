"""Renderer adapter abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from core import PromptSpec


@dataclass
class ShotRenderResult:
    """Result of one rendered shot."""

    shot_idx: int
    success: bool
    output_path: Optional[str] = None
    error: Optional[str] = None
    cost: float = 0.0


class BaseRendererAdapter:
    """Base adapter that can be replaced by Seedance or mocks."""

    provider = "base"

    def render_shot(
        self,
        *,
        prompt_spec: PromptSpec,
        output_dir: Path,
        mode: str,
        budget: Dict[str, Any],
        run_id: str,
        render_job_id: str,
    ) -> ShotRenderResult:
        raise NotImplementedError
