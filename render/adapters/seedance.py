"""Seedance adapter wrapper with optional real HTTP client."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from uuid import uuid4

import httpx

from core import PromptSpec

from .base import BaseRendererAdapter, ShotRenderResult


ClientFn = Callable[..., Dict[str, Any]]


def _env_enabled(name: str, *, default: bool = False) -> bool:
    raw = str(os.getenv(name, "")).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(default)


class SeedanceAdapter(BaseRendererAdapter):
    """Adapter boundary for Seedance rendering API."""

    provider = "seedance"

    def __init__(
        self,
        client: Optional[ClientFn] = None,
        *,
        enabled: Optional[bool] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        region: Optional[str] = None,
        timeout_s: Optional[float] = None,
    ) -> None:
        self.enabled = _env_enabled("SEEDANCE_ENABLED", default=False) if enabled is None else bool(enabled)
        self.base_url = str(base_url or os.getenv("SEEDANCE_BASE_URL") or "").strip().rstrip("/")
        self.api_key = str(api_key or os.getenv("SEEDANCE_API_KEY") or "").strip()
        self.region = str(region or os.getenv("SEEDANCE_REGION") or "").strip() or "global"
        try:
            self.timeout_s = float(timeout_s if timeout_s is not None else (os.getenv("SEEDANCE_TIMEOUT_S") or 45.0))
        except Exception:
            self.timeout_s = 45.0

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
        if not self.enabled:
            return ShotRenderResult(
                shot_idx=prompt_spec.shot_idx,
                success=False,
                error="seedance disabled",
                cost=0.0,
            )

        client = self._client or self._http_client
        if client is None:
            return ShotRenderResult(
                shot_idx=prompt_spec.shot_idx,
                success=False,
                error="seedance client unavailable",
                cost=0.0,
            )

        try:
            payload = self._invoke_client(
                client,
                prompt_text=prompt_spec.prompt_text,
                negative_prompt=prompt_spec.negative_prompt,
                references=list(prompt_spec.references or []),
                params=dict(prompt_spec.seedance_params or {}),
                mode=mode,
                run_id=run_id,
                render_job_id=render_job_id,
                output_dir=output_dir,
                shot_idx=int(prompt_spec.shot_idx),
                budget=dict(budget or {}),
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

    def _invoke_client(self, client: ClientFn, **kwargs: Any) -> Dict[str, Any]:
        try:
            return dict(client(**kwargs) or {})
        except TypeError:
            legacy = {
                "prompt_text": kwargs.get("prompt_text"),
                "negative_prompt": kwargs.get("negative_prompt"),
                "references": kwargs.get("references"),
                "params": kwargs.get("params"),
                "mode": kwargs.get("mode"),
                "run_id": kwargs.get("run_id"),
                "render_job_id": kwargs.get("render_job_id"),
            }
            return dict(client(**legacy) or {})

    def _http_client(
        self,
        *,
        prompt_text: str,
        negative_prompt: Optional[str],
        references: list[str],
        params: Dict[str, Any],
        mode: str,
        run_id: str,
        render_job_id: str,
        output_dir: Path,
        shot_idx: int,
        budget: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not self.base_url or not self.api_key:
            raise RuntimeError("seedance config missing: SEEDANCE_BASE_URL/SEEDANCE_API_KEY")

        endpoint = f"{self.base_url}/v1/renders/shots"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Seedance-Region": self.region,
        }
        request_payload = {
            "prompt": prompt_text,
            "negative_prompt": negative_prompt,
            "references": list(references or []),
            "params": dict(params or {}),
            "mode": str(mode or "final"),
            "run_id": run_id,
            "render_job_id": render_job_id,
            "shot_idx": int(shot_idx),
            "budget": dict(budget or {}),
        }

        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                response = client.post(endpoint, headers=headers, json=request_payload)
                if response.status_code in {401, 403}:
                    raise RuntimeError("seedance auth failed")
                if response.status_code == 429:
                    raise RuntimeError("seedance quota exceeded")
                if response.status_code >= 400:
                    raise RuntimeError(f"seedance http {response.status_code}: {response.text[:200]}")

                payload = dict(response.json() or {})
                output_path = self._materialize_output(payload, client=client, output_dir=output_dir, shot_idx=shot_idx)
                return {
                    "output_path": output_path,
                    "cost": float(payload.get("cost") or 0.0),
                }
        except httpx.TimeoutException as exc:
            raise RuntimeError("seedance timeout") from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"seedance request failed: {exc}") from exc

    def _materialize_output(self, payload: Dict[str, Any], *, client: httpx.Client, output_dir: Path, shot_idx: int) -> str:
        output_dir.mkdir(parents=True, exist_ok=True)

        explicit_path = str(payload.get("output_path") or "").strip()
        if explicit_path and Path(explicit_path).exists():
            return explicit_path

        target = output_dir / f"shot_{int(shot_idx):03d}.mp4"

        b64_value = str(payload.get("video_base64") or payload.get("video_b64") or "").strip()
        if b64_value:
            decoded = base64.b64decode(b64_value)
            self._atomic_write_bytes(target, decoded)
            return str(target)

        remote_url = str(payload.get("output_url") or payload.get("video_url") or "").strip()
        if remote_url:
            response = client.get(remote_url, headers={"Authorization": f"Bearer {self.api_key}"})
            if response.status_code >= 400:
                raise RuntimeError(f"seedance download failed: {response.status_code}")
            self._atomic_write_bytes(target, response.content)
            return str(target)

        return ""

    @staticmethod
    def _atomic_write_bytes(path: Path, payload: bytes) -> None:
        tmp = path.parent / f".{path.name}.{uuid4().hex}.tmp"
        tmp.write_bytes(payload)
        os.replace(tmp, path)
