"""Asynchronous-style render manager with retry and fallback controls."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4

from core import PromptSpec, RenderStatus, Shot, StatusTimestamps, Storyboard

from .adapters import BaseRendererAdapter, SeedanceAdapter, ShotRenderResult

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency in runtime environments
    np = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)
_FFMPEG_BIN = shutil.which("ffmpeg")
_FFPROBE_BIN = shutil.which("ffprobe")
_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
]
_CARD_BG_COLORS = [
    "#0b1220",
    "#101a2e",
    "#13233a",
    "#1a2440",
    "#162d3b",
    "#20263a",
]


def _ffmpeg_supports_filter(name: str) -> bool:
    if not _FFMPEG_BIN:
        return False
    proc = subprocess.run([_FFMPEG_BIN, "-hide_banner", "-filters"], check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        return False
    pattern = re.compile(rf"\b{name}\b")
    return bool(pattern.search(proc.stdout or ""))


_HAS_DRAWTEXT = _ffmpeg_supports_filter("drawtext")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_render_job_id() -> str:
    return f"render_{_utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"


def _compact_text(value: str, max_len: int = 180) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _safe_drawtext(value: str) -> str:
    text = str(value or "")
    text = text.replace("\\", "\\\\")
    text = text.replace(":", r"\:")
    text = text.replace("'", r"\'")
    text = text.replace("%", r"\%")
    text = text.replace("\n", r"\n")
    return text


def _safe_fontfile(value: str) -> str:
    text = str(value or "")
    text = text.replace("\\", "\\\\")
    text = text.replace(":", r"\:")
    text = text.replace(" ", r"\ ")
    return text


def _wrap_text(value: str, *, max_chars: int = 34, max_lines: int = 3) -> str:
    words = str(value or "").split()
    if not words:
        return ""
    lines: List[str] = []
    current: List[str] = []
    for word in words:
        candidate = " ".join(current + [word]).strip()
        if current and len(candidate) > max_chars:
            lines.append(" ".join(current))
            current = [word]
            if len(lines) >= max_lines:
                break
        else:
            current.append(word)
    if len(lines) < max_lines and current:
        lines.append(" ".join(current))
    return "\n".join(lines[:max_lines])


_FONT_5X7 = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01110"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "J": ["00001", "00001", "00001", "00001", "10001", "10001", "01110"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "01010", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "01010", "00100", "00100", "00100", "01010", "10001"],
    "Y": ["10001", "01010", "00100", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00010", "00100", "00100", "01000", "10000", "11111"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    ".": ["00000", "00000", "00000", "00000", "00000", "00110", "00110"],
    ":": ["00000", "00110", "00110", "00000", "00110", "00110", "00000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "/": ["00001", "00010", "00100", "00100", "01000", "10000", "00000"],
    "?": ["01110", "10001", "00001", "00010", "00100", "00000", "00100"],
    "!": ["00100", "00100", "00100", "00100", "00100", "00000", "00100"],
    "|": ["00100", "00100", "00100", "00100", "00100", "00100", "00100"],
    "'": ["00100", "00100", "00000", "00000", "00000", "00000", "00000"],
    "(": ["00010", "00100", "01000", "01000", "01000", "00100", "00010"],
    ")": ["01000", "00100", "00010", "00010", "00010", "00100", "01000"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
}


@dataclass
class RenderJob:
    render_job_id: str
    run_id: str
    prompt_specs: List[PromptSpec]
    budget: Dict[str, Any]
    output_dir: Path
    status: RenderStatus
    shot_outputs: Dict[int, str] = field(default_factory=dict)
    failed: Dict[int, str] = field(default_factory=dict)
    total_cost: float = 0.0
    preview_completed: bool = False
    confirmed: bool = False
    preview_output_path: Optional[str] = None
    seedance_success_count: int = 0


class RenderManager:
    """Queue-backed render manager with retries and fallback MP4 output."""

    def __init__(
        self,
        *,
        renderer_adapter: Optional[BaseRendererAdapter] = None,
        work_dir: Optional[Path] = None,
    ) -> None:
        self._adapter = renderer_adapter or SeedanceAdapter()
        self._work_dir = Path(work_dir or (Path("data") / "outputs" / "render_jobs"))
        self._work_dir.mkdir(parents=True, exist_ok=True)

        self._jobs: Dict[str, RenderJob] = {}
        self._queue: List[str] = []

    def enqueue_render(self, run_id: str, prompt_specs: Sequence[PromptSpec], budget: Optional[Dict[str, Any]] = None) -> str:
        """Queue render job and return render_job_id."""
        render_job_id = _new_render_job_id()
        out_dir = self._work_dir / render_job_id
        out_dir.mkdir(parents=True, exist_ok=True)

        budget_map = dict(budget or {})
        max_retries = int(budget_map.get("max_retries", 1) or 1)

        status = RenderStatus(
            render_job_id=render_job_id,
            run_id=str(run_id),
            state="queued",
            progress=0.0,
            timestamps=StatusTimestamps(),
            max_retries=max(0, max_retries),
            seedance_used=False,
        )

        job = RenderJob(
            render_job_id=render_job_id,
            run_id=str(run_id),
            prompt_specs=[PromptSpec(**item.model_dump()) for item in list(prompt_specs or [])],
            budget=budget_map,
            output_dir=out_dir,
            status=status,
        )
        self._jobs[render_job_id] = job
        self._queue.append(render_job_id)
        self._persist_render_status(job)
        logger.info("enqueue_render run_id=%s render_job_id=%s shots=%s", run_id, render_job_id, len(job.prompt_specs))
        return render_job_id

    def poll_render(self, render_job_id: str) -> Optional[RenderStatus]:
        """Get current render job status snapshot."""
        job = self._jobs.get(render_job_id)
        if not job:
            return None
        return RenderStatus(**job.status.model_dump())

    def confirm_render(self, render_job_id: str, *, approved: bool = True) -> Optional[RenderStatus]:
        """Confirm preview and queue final render, or cancel when rejected."""
        job = self._jobs.get(render_job_id)
        if not job:
            return None

        confirm_required = bool(job.budget.get("confirm_required", False))
        if not confirm_required:
            return RenderStatus(**job.status.model_dump())

        if not approved:
            self._set_canceled(job, reason="preview rejected by user")
            self._remove_from_queue(render_job_id)
            self._persist_render_status(job)
            return RenderStatus(**job.status.model_dump())

        if not job.preview_completed:
            return RenderStatus(**job.status.model_dump())

        job.confirmed = True
        job.shot_outputs.clear()
        job.failed.clear()
        job.status.output_path = None
        job.status.state = "queued"
        job.status.progress = max(job.status.progress, 0.65)
        job.status.timestamps.updated_at = _utcnow()
        if render_job_id not in self._queue:
            self._queue.append(render_job_id)
        self._persist_render_status(job)
        return RenderStatus(**job.status.model_dump())

    def cancel_render(self, render_job_id: str) -> bool:
        """Best-effort render cancellation for queued/preview/running jobs."""
        job = self._jobs.get(render_job_id)
        if not job:
            return False
        job.status.cancellation_requested = True
        self._remove_from_queue(render_job_id)
        if job.status.state in {"queued", "retrying", "awaiting_confirmation"}:
            self._set_canceled(job, reason="cancellation requested")
        elif job.status.state == "running":
            job.status.state = "cancel_requested"
            job.status.timestamps.updated_at = _utcnow()
        self._persist_render_status(job)
        return True

    def process_next(self) -> Optional[RenderStatus]:
        """Process next queued render job synchronously (worker-style)."""
        if not self._queue:
            return None
        render_job_id = self._queue.pop(0)
        job = self._jobs.get(render_job_id)
        if job is None:
            return None

        if job.status.cancellation_requested:
            self._set_canceled(job, reason="cancellation requested")
            self._persist_render_status(job)
            return RenderStatus(**job.status.model_dump())

        confirm_required = bool(job.budget.get("confirm_required", False))
        if confirm_required and not job.preview_completed:
            self._set_running(job, phase="preview")
            self._render_all_shots(job, mode="preview")
            if job.failed:
                self.retry_failed_shots(render_job_id, mode="preview")

            if job.failed:
                fallback_board = self._board_from_prompts(job)
                preview_path = self.fallback_render(fallback_board, out_dir=job.output_dir)
                job.status.errors.append("seedance_failed_using_fallback")
            else:
                preview_path = self.stitch_shots(
                    list(job.shot_outputs.values()),
                    out_dir=job.output_dir,
                    board=self._board_from_prompts(job),
                )

            job.preview_completed = True
            job.preview_output_path = preview_path
            job.status.output_path = preview_path
            self._update_output_validation(job)
            job.status.state = "awaiting_confirmation"
            job.status.progress = max(job.status.progress, 0.6)
            job.status.timestamps.updated_at = _utcnow()
            self._persist_render_status(job)
            return RenderStatus(**job.status.model_dump())

        if confirm_required and not job.confirmed:
            job.status.state = "awaiting_confirmation"
            job.status.timestamps.updated_at = _utcnow()
            self._persist_render_status(job)
            return RenderStatus(**job.status.model_dump())

        job.shot_outputs.clear()
        job.failed.clear()
        job.seedance_success_count = 0
        job.status.seedance_used = False
        self._set_running(job, phase="final")
        self._render_all_shots(job, mode="final")

        if job.failed:
            self.retry_failed_shots(render_job_id, mode="final")

        if job.failed:
            fallback_board = self._board_from_prompts(job)
            fallback_path = self.fallback_render(fallback_board, out_dir=job.output_dir)
            job.status.output_path = fallback_path
            job.status.errors.append("seedance_failed_using_fallback")

        if not job.status.output_path:
            stitched = self.stitch_shots(
                list(job.shot_outputs.values()),
                out_dir=job.output_dir,
                board=self._board_from_prompts(job),
            )
            job.status.output_path = stitched

        self._update_output_validation(job)
        if self._seedance_enabled() and job.seedance_success_count <= 0:
            job.status.seedance_used = False
        self._set_completed(job)
        self._persist_render_status(job)
        return RenderStatus(**job.status.model_dump())

    def retry_failed_shots(
        self,
        render_job_id: str,
        max_retries: Optional[int] = None,
        *,
        mode: str = "final",
    ) -> Optional[RenderStatus]:
        """Retry failed shots for a render job within retry budget."""
        job = self._jobs.get(render_job_id)
        if not job:
            return None

        allowed = job.status.max_retries if max_retries is None else max(0, int(max_retries))
        if job.status.retry_count >= allowed:
            return RenderStatus(**job.status.model_dump())

        pending = sorted(list(job.failed.keys()))
        if not pending:
            return RenderStatus(**job.status.model_dump())

        job.status.retry_count += 1
        job.status.state = "retrying"
        job.status.timestamps.updated_at = _utcnow()

        current_failed = dict(job.failed)
        job.failed.clear()
        for idx in pending:
            spec = self._prompt_by_idx(job.prompt_specs, idx)
            if spec is None:
                continue
            result = self._render_one_shot(job, spec, mode=mode)
            if result.success and result.output_path:
                job.shot_outputs[idx] = result.output_path
            else:
                job.failed[idx] = result.error or current_failed.get(idx, "unknown render error")

        job.status.failed_shot_indices = sorted(list(job.failed.keys()))
        job.status.timestamps.updated_at = _utcnow()
        return RenderStatus(**job.status.model_dump())

    def fallback_render(self, board: Storyboard, out_dir: Optional[Path] = None) -> str:
        """Fallback render path that must still emit an MP4 artifact."""
        target_dir = Path(out_dir or self._work_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / "fallback_render.mp4"

        if self._render_motion_graphics(board=board, out_path=out_path):
            return str(out_path)

        if self._synthesize_video(
            out_path=out_path,
            duration_sec=max(1.0, float(board.duration_sec or 30)),
            aspect=str(board.aspect or "9:16"),
        ):
            return str(out_path)

        payload = (
            f"run_id={board.run_id}\nitem_id={board.item_id}\nduration_sec={board.duration_sec}\nshots={len(board.shots)}\n"
        ).encode("utf-8")
        self._atomic_write_mp4ish(out_path, payload)
        return str(out_path)

    def stitch_shots(
        self,
        shot_paths: Sequence[str],
        out_dir: Optional[Path] = None,
        *,
        board: Optional[Storyboard] = None,
    ) -> str:
        """Stitch rendered shots into final MP4 path (concat preferred, synth fallback)."""
        target_dir = Path(out_dir or self._work_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / "rendered_final.mp4"

        existing = [Path(str(item)) for item in list(shot_paths or []) if Path(str(item)).exists()]
        if existing and self._concat_videos(existing, out_path=out_path):
            return str(out_path)

        if board is not None:
            return self.fallback_render(board, out_dir=target_dir)

        duration = max(4.0, float(len(existing) * 4 if existing else 10))
        if self._synthesize_video(out_path=out_path, duration_sec=duration, aspect="9:16"):
            return str(out_path)

        payload = "\n".join(path.name for path in existing).encode("utf-8")
        self._atomic_write_mp4ish(out_path, payload)
        return str(out_path)

    def _concat_videos(self, shot_paths: Sequence[Path], *, out_path: Path) -> bool:
        if not _FFMPEG_BIN or not shot_paths:
            return False

        list_path = out_path.parent / f"{out_path.stem}_concat.txt"
        lines: List[str] = []
        for path in shot_paths:
            escaped = str(path.resolve()).replace("'", "'\\''")
            lines.append("file '" + escaped + "'")
        list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        tmp_out = out_path.parent / f".{out_path.name}.{uuid4().hex}.tmp"
        try:
            cmd = [
                _FFMPEG_BIN,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                "-f",
                "mp4",
                str(tmp_out),
            ]
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if proc.returncode != 0:
                return False
            if not tmp_out.exists() or tmp_out.stat().st_size <= 0:
                return False
            os.replace(tmp_out, out_path)
            return True
        finally:
            if list_path.exists():
                list_path.unlink()
            if tmp_out.exists():
                tmp_out.unlink()

    @staticmethod
    def _pick_fontfile() -> Optional[str]:
        for candidate in _FONT_CANDIDATES:
            if Path(candidate).exists():
                return candidate
        return None

    def _render_text_card(
        self,
        *,
        out_path: Path,
        duration_sec: float,
        aspect: str,
        bg_color: str,
        title: str,
        body: str,
        footer: str,
    ) -> bool:
        if not _FFMPEG_BIN:
            return False
        if not _HAS_DRAWTEXT:
            return self._render_text_card_bitmap(
                out_path=out_path,
                duration_sec=duration_sec,
                aspect=aspect,
                bg_color=bg_color,
                title=title,
                body=body,
                footer=footer,
            )
        width, height = self._resolution_from_aspect(aspect)
        duration = max(0.8, float(duration_sec))
        fontfile = self._pick_fontfile()
        fontfile_escaped = _safe_fontfile(fontfile) if fontfile else None
        title_text = _safe_drawtext(_wrap_text(_compact_text(title, 96), max_chars=24, max_lines=2))
        body_text = _safe_drawtext(_wrap_text(_compact_text(body, 180), max_chars=30, max_lines=4))
        footer_text = _safe_drawtext(_compact_text(footer, 96))

        text_filters = []
        if title_text:
            title_filter = (
                f"drawtext=text='{title_text}':fontcolor=white:fontsize=54:"
                "x=(w-text_w)/2:y=h*0.14:line_spacing=10"
            )
            text_filters.append(title_filter if not fontfile_escaped else title_filter + f":fontfile={fontfile_escaped}")
        if body_text:
            body_filter = (
                f"drawtext=text='{body_text}':fontcolor=white:fontsize=36:"
                "x=(w-text_w)/2:y=h*0.38:line_spacing=8"
            )
            text_filters.append(body_filter if not fontfile_escaped else body_filter + f":fontfile={fontfile_escaped}")
        if footer_text:
            footer_filter = (
                f"drawtext=text='{footer_text}':fontcolor=#cbd5e1:fontsize=24:"
                "x=(w-text_w)/2:y=h*0.88"
            )
            text_filters.append(footer_filter if not fontfile_escaped else footer_filter + f":fontfile={fontfile_escaped}")

        if not text_filters:
            text_filters = ["null"]

        vf = ",".join(
            [
                f"drawbox=x=0:y=0:w=iw:h=ih:color={bg_color}@0.98:t=fill",
                "drawbox=x=0:y=0:w=iw:h=ih:color=#0ea5e9@0.06:t=fill",
                f"drawbox=x=mod(t*110\\,{width}):y={int(height*0.8)}:w={int(width*0.28)}:h=10:color=#22d3ee@0.55:t=fill",
            ]
            + text_filters
        )

        tmp_out = out_path.parent / f".{out_path.name}.{uuid4().hex}.tmp"
        cmd = [
            _FFMPEG_BIN,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c={bg_color}:s={width}x{height}:rate=24:d={duration:.2f}",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-vf",
            vf,
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
            str(tmp_out),
        ]
        try:
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if proc.returncode != 0:
                return False
            if not tmp_out.exists() or tmp_out.stat().st_size <= 0:
                return False
            os.replace(tmp_out, out_path)
            return True
        finally:
            if tmp_out.exists():
                tmp_out.unlink()

    @staticmethod
    def _hex_to_rgb(value: str) -> tuple[int, int, int]:
        text = str(value or "#0f172a").strip().lstrip("#")
        if len(text) != 6:
            return (15, 23, 42)
        try:
            return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))
        except Exception:
            return (15, 23, 42)

    @staticmethod
    def _draw_bitmap_text(
        frame: "np.ndarray",  # type: ignore[name-defined]
        text: str,
        *,
        x: int,
        y: int,
        scale: int,
        color: tuple[int, int, int],
        max_width: int,
    ) -> None:
        if np is None:
            return
        cursor_x = int(x)
        cursor_y = int(y)
        line_height = int(7 * scale + scale)
        width_limit = max(0, int(max_width))
        for raw_char in str(text or "").upper():
            ch = raw_char if raw_char in _FONT_5X7 else "?"
            if ch == "\n":
                cursor_y += line_height
                cursor_x = int(x)
                continue
            glyph = _FONT_5X7.get(ch, _FONT_5X7["?"])
            glyph_width = int(5 * scale + scale)
            if cursor_x + glyph_width > width_limit:
                cursor_y += line_height
                cursor_x = int(x)
            for row_idx, row_bits in enumerate(glyph):
                for col_idx, bit in enumerate(row_bits):
                    if bit != "1":
                        continue
                    x0 = cursor_x + col_idx * scale
                    y0 = cursor_y + row_idx * scale
                    x1 = min(frame.shape[1], x0 + scale)
                    y1 = min(frame.shape[0], y0 + scale)
                    if x0 < 0 or y0 < 0 or x0 >= frame.shape[1] or y0 >= frame.shape[0]:
                        continue
                    frame[y0:y1, x0:x1, 0] = color[0]
                    frame[y0:y1, x0:x1, 1] = color[1]
                    frame[y0:y1, x0:x1, 2] = color[2]
            cursor_x += glyph_width

    def _render_text_card_bitmap(
        self,
        *,
        out_path: Path,
        duration_sec: float,
        aspect: str,
        bg_color: str,
        title: str,
        body: str,
        footer: str,
    ) -> bool:
        if np is None or not _FFMPEG_BIN:
            return False
        width, height = self._resolution_from_aspect(aspect)
        duration = max(0.8, float(duration_sec))
        bg = self._hex_to_rgb(bg_color)

        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = bg[0]
        frame[:, :, 1] = bg[1]
        frame[:, :, 2] = bg[2]
        frame[int(height * 0.8) : int(height * 0.8) + 14, :, :] = (30, 199, 220)
        frame[int(height * 0.12) : int(height * 0.13), :, :] = (14, 165, 233)

        self._draw_bitmap_text(
            frame,
            _wrap_text(_compact_text(title, 72), max_chars=20, max_lines=2),
            x=int(width * 0.08),
            y=int(height * 0.14),
            scale=6,
            color=(245, 248, 255),
            max_width=int(width * 0.92),
        )
        self._draw_bitmap_text(
            frame,
            _wrap_text(_compact_text(body, 160), max_chars=26, max_lines=5),
            x=int(width * 0.08),
            y=int(height * 0.38),
            scale=4,
            color=(240, 245, 250),
            max_width=int(width * 0.92),
        )
        self._draw_bitmap_text(
            frame,
            _wrap_text(_compact_text(footer, 96), max_chars=40, max_lines=2),
            x=int(width * 0.08),
            y=int(height * 0.88),
            scale=3,
            color=(203, 213, 225),
            max_width=int(width * 0.92),
        )

        image_path = out_path.parent / f".{out_path.stem}.{uuid4().hex}.ppm"
        image_path.write_bytes(b"P6\n" + f"{width} {height}\n255\n".encode("ascii") + frame.tobytes())

        tmp_out = out_path.parent / f".{out_path.name}.{uuid4().hex}.tmp"
        cmd = [
            _FFMPEG_BIN,
            "-y",
            "-loglevel",
            "error",
            "-loop",
            "1",
            "-t",
            f"{duration:.2f}",
            "-i",
            str(image_path),
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
            str(tmp_out),
        ]
        try:
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if proc.returncode != 0:
                return False
            if not tmp_out.exists() or tmp_out.stat().st_size <= 0:
                return False
            os.replace(tmp_out, out_path)
            return True
        finally:
            if image_path.exists():
                image_path.unlink()
            if tmp_out.exists():
                tmp_out.unlink()

    def _render_motion_graphics(self, *, board: Storyboard, out_path: Path) -> bool:
        if not _FFMPEG_BIN:
            return False

        sequence = [
            {
                "duration": 1.8,
                "title": "Tech Brief",
                "body": _compact_text(f"Run {board.run_id} | {board.item_id}", 120),
                "footer": "Generated from ranked evidence",
            }
        ]

        for shot in list(board.shots or []):
            overlay = _compact_text(str(shot.overlay_text or shot.action or "Key update"), 170)
            source_hint = str(shot.reference_assets[0]).strip() if shot.reference_assets else "source://curated"
            sequence.append(
                {
                    "duration": max(1.4, float(shot.duration or 3.5)),
                    "title": f"Shot {int(shot.idx)}",
                    "body": overlay,
                    "footer": _compact_text(f"{shot.scene} | {source_hint}", 90),
                }
            )

        sequence.append(
            {
                "duration": 2.0,
                "title": "Summary",
                "body": "Track this topic daily and verify claims using the listed citations.",
                "footer": "CTA: follow for next run",
            }
        )

        segment_dir = out_path.parent / f".segments_{uuid4().hex[:8]}"
        segment_dir.mkdir(parents=True, exist_ok=True)
        segment_paths: List[Path] = []
        try:
            for idx, segment in enumerate(sequence):
                segment_path = segment_dir / f"segment_{idx:03d}.mp4"
                ok = self._render_text_card(
                    out_path=segment_path,
                    duration_sec=float(segment["duration"]),
                    aspect=str(board.aspect or "9:16"),
                    bg_color=_CARD_BG_COLORS[idx % len(_CARD_BG_COLORS)],
                    title=str(segment["title"]),
                    body=str(segment["body"]),
                    footer=str(segment["footer"]),
                )
                if not ok:
                    return False
                segment_paths.append(segment_path)

            return self._concat_videos(segment_paths, out_path=out_path)
        finally:
            shutil.rmtree(segment_dir, ignore_errors=True)

    def _synthesize_video(self, *, out_path: Path, duration_sec: float, aspect: str) -> bool:
        if not _FFMPEG_BIN:
            return False
        width, height = self._resolution_from_aspect(aspect)
        duration = max(1.0, float(duration_sec))
        accent = "0x14b8a6"
        tmp_out = out_path.parent / f".{out_path.name}.{uuid4().hex}.tmp"
        cmd = [
            _FFMPEG_BIN,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=#0f172a:s={width}x{height}:rate=24:d={duration:.2f}",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-vf",
            (
                "drawbox=x=0:y=0:w=iw:h=ih:color=#1f2937@0.35:t=fill,"
                f"drawbox=x=mod(t*90\\,{width}):y={int(height*0.82)}:w={int(width*0.35)}:h=12:color={accent}@0.55:t=fill"
            ),
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
            str(tmp_out),
        ]
        try:
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if proc.returncode != 0:
                return False
            if not tmp_out.exists() or tmp_out.stat().st_size <= 0:
                return False
            os.replace(tmp_out, out_path)
            return True
        finally:
            if tmp_out.exists():
                tmp_out.unlink()

    def _atomic_write_mp4ish(self, out_path: Path, payload: bytes) -> None:
        # Last-resort fallback for environments without ffmpeg.
        tmp = out_path.parent / f".{out_path.name}.{uuid4().hex}.tmp"
        header = b"\x00\x00\x00\x20ftypmp42\x00\x00\x00\x00mp42isom"
        tmp.write_bytes(header + payload)
        os.replace(tmp, out_path)

    @staticmethod
    def _resolution_from_aspect(aspect: str) -> tuple[int, int]:
        value = str(aspect or "").strip()
        if value == "16:9":
            return (1280, 720)
        if value == "1:1":
            return (1080, 1080)
        return (720, 1280)

    def validate_mp4(self, path: str) -> tuple[bool, Optional[str]]:
        candidate = Path(str(path or ""))
        if not candidate.exists() or not candidate.is_file():
            return False, "output file missing"
        if candidate.stat().st_size <= 0:
            return False, "output file is empty"

        if _FFPROBE_BIN:
            cmd = [
                _FFPROBE_BIN,
                "-hide_banner",
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                str(candidate),
            ]
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if proc.returncode != 0:
                return False, (proc.stderr or proc.stdout or "ffprobe failed").strip()
            if "codec_type=video" not in proc.stdout:
                return False, "no video stream found"
            return True, None

        head = candidate.read_bytes()[:64]
        if b"ftyp" in head:
            return True, None
        return False, "ffprobe unavailable and missing ftyp signature"

    def _remove_from_queue(self, render_job_id: str) -> None:
        while render_job_id in self._queue:
            self._queue.remove(render_job_id)

    def _set_running(self, job: RenderJob, *, phase: str) -> None:
        now = _utcnow()
        job.status.state = "running"
        job.status.timestamps.started_at = job.status.timestamps.started_at or now
        job.status.timestamps.updated_at = now
        logger.info("render_start run_id=%s render_job_id=%s phase=%s", job.run_id, job.render_job_id, phase)

    def _set_completed(self, job: RenderJob) -> None:
        now = _utcnow()
        job.status.state = "completed"
        job.status.progress = 1.0
        job.status.timestamps.completed_at = now
        job.status.timestamps.updated_at = now
        job.status.failed_shot_indices = sorted(list(job.failed.keys()))
        logger.info(
            "render_done run_id=%s render_job_id=%s output=%s retries=%s",
            job.run_id,
            job.render_job_id,
            job.status.output_path,
            job.status.retry_count,
        )

    def _update_output_validation(self, job: RenderJob) -> None:
        output = str(job.status.output_path or "").strip()
        if not output:
            job.status.valid_mp4 = False
            job.status.probe_error = "missing output path"
            return
        ok, error = self.validate_mp4(output)
        job.status.valid_mp4 = bool(ok)
        job.status.probe_error = error
        if not ok and error:
            job.status.errors.append(f"invalid_mp4:{error}")

    def _set_canceled(self, job: RenderJob, *, reason: str) -> None:
        now = _utcnow()
        job.status.state = "canceled"
        job.status.cancellation_requested = True
        job.status.timestamps.cancelled_at = job.status.timestamps.cancelled_at or now
        job.status.timestamps.completed_at = job.status.timestamps.completed_at or now
        job.status.timestamps.updated_at = now
        if reason:
            job.status.errors.append(reason)

    def _persist_render_status(self, job: RenderJob) -> None:
        payload = job.status.model_dump(mode="json")
        path = job.output_dir / "render_status.json"
        tmp = job.output_dir / f".{path.name}.{uuid4().hex}.tmp"
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def _render_all_shots(self, job: RenderJob, *, mode: str) -> None:
        total = max(1, len(job.prompt_specs))
        for idx, spec in enumerate(job.prompt_specs, start=1):
            result = self._render_one_shot(job, spec, mode=mode)
            if result.success and result.output_path:
                job.shot_outputs[spec.shot_idx] = result.output_path
            else:
                job.failed[spec.shot_idx] = result.error or "unknown render error"
            job.status.progress = min(0.9, idx / float(total))
            job.status.timestamps.updated_at = _utcnow()
            job.status.failed_shot_indices = sorted(list(job.failed.keys()))

    def _render_one_shot(self, job: RenderJob, spec: PromptSpec, *, mode: str) -> ShotRenderResult:
        budget_limit = float(job.budget.get("max_total_cost", 1e18) or 1e18)
        if job.total_cost > budget_limit:
            return ShotRenderResult(
                shot_idx=spec.shot_idx,
                success=False,
                error="budget exceeded",
                cost=0.0,
            )
        result = self._adapter.render_shot(
            prompt_spec=spec,
            output_dir=job.output_dir,
            mode=mode,
            budget=job.budget,
            run_id=job.run_id,
            render_job_id=job.render_job_id,
        )
        job.total_cost += max(0.0, float(result.cost or 0.0))
        if self._seedance_enabled() and result.success and result.output_path:
            job.seedance_success_count += 1
            job.status.seedance_used = True
        return result

    def _seedance_enabled(self) -> bool:
        return isinstance(self._adapter, SeedanceAdapter) and bool(getattr(self._adapter, "enabled", False))

    @staticmethod
    def _prompt_by_idx(prompt_specs: Sequence[PromptSpec], shot_idx: int) -> Optional[PromptSpec]:
        for spec in prompt_specs:
            if int(spec.shot_idx) == int(shot_idx):
                return spec
        return None

    @staticmethod
    def _board_from_prompts(job: RenderJob) -> Storyboard:
        shots: List[Shot] = []
        for idx, spec in enumerate(job.prompt_specs, start=1):
            duration = float(spec.seedance_params.get("duration_sec", 4.0) or 4.0)
            scene = str(spec.seedance_params.get("scene", "analysis scene") or "analysis scene")
            overlay = _compact_text(spec.prompt_text, max_len=160) or f"Shot {idx}"
            shots.append(
                Shot(
                    idx=idx,
                    duration=max(1.0, duration),
                    camera=str(spec.seedance_params.get("camera", "medium")),
                    scene=scene,
                    subject_id=str(spec.seedance_params.get("character_id", "host_01")),
                    action=spec.prompt_text,
                    overlay_text=overlay,
                    reference_assets=list(spec.references or []),
                )
            )

        total = int(round(sum(shot.duration for shot in shots))) or 30
        return Storyboard(
            run_id=job.run_id,
            item_id=f"item_{job.render_job_id}",
            duration_sec=total,
            aspect="9:16",
            shots=shots,
        )
