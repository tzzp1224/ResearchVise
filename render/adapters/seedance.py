"""Seedance adapter wrapper (replaceable, network optional)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from core import PromptSpec

from .base import BaseRendererAdapter, ShotRenderResult


class SeedanceAdapter(BaseRendererAdapter):
    """Adapter boundary for Seedance rendering API."""

    provider = "seedance"

    def __init__(self, client: Optional[Callable[..., Dict[str, Any]]] = None) -> None:
        self._client = client

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
        if self._client is None:
            return ShotRenderResult(
                shot_idx=prompt_spec.shot_idx,
                success=False,
                error="seedance client unavailable",
                cost=0.0,
            )

        try:
            payload = self._client(
                prompt_text=prompt_spec.prompt_text,
                negative_prompt=prompt_spec.negative_prompt,
                references=list(prompt_spec.references or []),
                params=dict(prompt_spec.seedance_params or {}),
                mode=mode,
                run_id=run_id,
                render_job_id=render_job_id,
            )
            output_path = str((payload or {}).get("output_path") or "").strip()
            cost = float((payload or {}).get("cost") or 0.0)
            if not output_path:
                return ShotRenderResult(
                    shot_idx=prompt_spec.shot_idx,
                    success=False,
                    error="seedance response missing output_path",
                    cost=cost,
                )
            return ShotRenderResult(
                shot_idx=prompt_spec.shot_idx,
                success=True,
                output_path=output_path,
                cost=cost,
            )
        except Exception as exc:
            return ShotRenderResult(
                shot_idx=prompt_spec.shot_idx,
                success=False,
                error=str(exc),
                cost=0.0,
            )
