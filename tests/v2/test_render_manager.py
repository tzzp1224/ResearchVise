from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

from core import PromptSpec, Shot, Storyboard
from render.adapters import BaseRendererAdapter, ShotRenderResult
from render.manager import RenderManager


class FlakyAdapter(BaseRendererAdapter):
    provider = "mock-seedance"

    def __init__(self, fail_once_indices: set[int] | None = None, always_fail: bool = False) -> None:
        self.fail_once_indices = set(fail_once_indices or set())
        self.always_fail = bool(always_fail)
        self.calls: dict[int, int] = {}

    def render_shot(self, *, prompt_spec: PromptSpec, output_dir: Path, mode: str, budget: dict, run_id: str, render_job_id: str) -> ShotRenderResult:
        _ = mode, budget, run_id, render_job_id
        idx = int(prompt_spec.shot_idx)
        self.calls[idx] = self.calls.get(idx, 0) + 1

        if self.always_fail:
            return ShotRenderResult(shot_idx=idx, success=False, error="adapter failed", cost=0.2)

        if idx in self.fail_once_indices and self.calls[idx] == 1:
            return ShotRenderResult(shot_idx=idx, success=False, error="transient failure", cost=0.1)

        out = output_dir / f"shot_{idx:03d}.mp4"
        out.write_bytes(f"shot-{idx}".encode("utf-8"))
        return ShotRenderResult(shot_idx=idx, success=True, output_path=str(out), cost=0.3)


def _prompts() -> list[PromptSpec]:
    return [
        PromptSpec(shot_idx=1, prompt_text="shot1", seedance_params={"duration_sec": 4.0, "camera": "wide"}),
        PromptSpec(shot_idx=2, prompt_text="shot2", seedance_params={"duration_sec": 4.0, "camera": "medium"}),
        PromptSpec(shot_idx=3, prompt_text="shot3", seedance_params={"duration_sec": 4.0, "camera": "close-up"}),
    ]


def _looks_like_mp4(path: Path) -> bool:
    if not path.exists():
        return False
    head = path.read_bytes()[:64]
    return b"ftyp" in head


def _ffprobe_ok(path: Path) -> bool:
    if not shutil.which("ffprobe"):
        return _looks_like_mp4(path)
    proc = subprocess.run(
        [
            "ffprobe",
            "-hide_banner",
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0 and "codec_type=video" in proc.stdout


def test_render_manager_retries_failed_shot_then_completes(tmp_path: Path) -> None:
    manager = RenderManager(renderer_adapter=FlakyAdapter(fail_once_indices={2}), work_dir=tmp_path)

    render_job_id = manager.enqueue_render(
        run_id="run_1",
        prompt_specs=_prompts(),
        budget={"max_retries": 1, "max_total_cost": 20.0, "preview": True},
    )

    queued = manager.poll_render(render_job_id)
    assert queued is not None
    assert queued.state == "queued"

    completed = manager.process_next()
    assert completed is not None
    assert completed.state == "completed"
    assert completed.retry_count == 1
    assert completed.failed_shot_indices == []
    assert completed.output_path and completed.output_path.endswith(".mp4")
    final_path = Path(completed.output_path)
    assert final_path.exists()
    assert completed.valid_mp4 is True
    assert completed.probe_error is None
    assert _ffprobe_ok(final_path)


def test_render_manager_fallback_when_adapter_unavailable(tmp_path: Path) -> None:
    manager = RenderManager(renderer_adapter=FlakyAdapter(always_fail=True), work_dir=tmp_path)

    render_job_id = manager.enqueue_render(
        run_id="run_2",
        prompt_specs=_prompts(),
        budget={"max_retries": 0, "max_total_cost": 20.0},
    )

    status = manager.process_next()
    assert status is not None
    assert status.state == "completed"
    assert status.output_path is not None
    assert status.output_path.endswith("fallback_render.mp4")
    fallback_path = Path(status.output_path)
    assert fallback_path.exists()
    assert status.valid_mp4 is True
    assert status.probe_error is None
    assert _ffprobe_ok(fallback_path)
    assert "seedance_failed_using_fallback" in status.errors

    polled = manager.poll_render(render_job_id)
    assert polled is not None
    assert polled.output_path == status.output_path


def test_fallback_render_and_stitch_shots_helpers(tmp_path: Path) -> None:
    manager = RenderManager(renderer_adapter=FlakyAdapter(), work_dir=tmp_path)
    board = Storyboard(
        run_id="run_3",
        item_id="item_3",
        duration_sec=30,
        aspect="9:16",
        shots=[
            Shot(
                idx=1,
                duration=5.0,
                camera="wide",
                scene="studio",
                subject_id="host",
                action="explain architecture",
                overlay_text="architecture",
                reference_assets=[],
            )
        ],
    )

    fallback = manager.fallback_render(board, out_dir=tmp_path)
    assert fallback.endswith(".mp4")
    fallback_path = Path(fallback)
    assert fallback_path.exists()
    assert _ffprobe_ok(fallback_path)

    part1 = tmp_path / "part1.mp4"
    part2 = tmp_path / "part2.mp4"
    part1.write_bytes(b"part1")
    part2.write_bytes(b"part2")

    stitched = manager.stitch_shots([str(part1), str(part2)], out_dir=tmp_path)
    assert stitched.endswith(".mp4")
    stitched_path = Path(stitched)
    assert stitched_path.exists()
    assert _ffprobe_ok(stitched_path)


def test_preview_confirm_final_flow(tmp_path: Path) -> None:
    manager = RenderManager(renderer_adapter=FlakyAdapter(), work_dir=tmp_path)
    render_job_id = manager.enqueue_render(
        run_id="run_preview",
        prompt_specs=_prompts(),
        budget={"max_retries": 1, "max_total_cost": 20.0, "preview": True, "confirm_required": True},
    )

    preview = manager.process_next()
    assert preview is not None
    assert preview.state == "awaiting_confirmation"
    assert preview.output_path and preview.output_path.endswith(".mp4")
    assert preview.valid_mp4 is True
    assert preview.probe_error is None

    approved = manager.confirm_render(render_job_id, approved=True)
    assert approved is not None
    assert approved.state == "queued"

    final = manager.process_next()
    assert final is not None
    assert final.state == "completed"
    assert final.output_path and final.output_path.endswith(".mp4")
    assert final.valid_mp4 is True
    assert final.probe_error is None
    assert _ffprobe_ok(Path(final.output_path))


def test_mock_renderer_stitch_ffprobe_passes(tmp_path: Path) -> None:
    manager = RenderManager(renderer_adapter=FlakyAdapter(), work_dir=tmp_path)
    rid = manager.enqueue_render(
        run_id="run_probe",
        prompt_specs=_prompts(),
        budget={"max_retries": 1, "max_total_cost": 20.0},
    )
    status = manager.process_next()
    assert status is not None
    assert status.state == "completed"
    assert status.output_path
    out = Path(status.output_path)
    assert out.exists()
    assert status.valid_mp4 is True
    assert status.probe_error is None
    assert _ffprobe_ok(out)

    polled = manager.poll_render(rid)
    assert polled is not None
    assert polled.valid_mp4 is True


def test_cancel_render_before_start(tmp_path: Path) -> None:
    manager = RenderManager(renderer_adapter=FlakyAdapter(), work_dir=tmp_path)
    render_job_id = manager.enqueue_render(run_id="run_cancel", prompt_specs=_prompts(), budget={"max_retries": 1})

    canceled = manager.cancel_render(render_job_id)
    assert canceled is True

    status = manager.poll_render(render_job_id)
    assert status is not None
    assert status.state == "canceled"

    next_status = manager.process_next()
    assert next_status is None
