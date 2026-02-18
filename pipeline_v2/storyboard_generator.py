"""Storyboard generation and validation utilities."""

from __future__ import annotations

from typing import Dict, List, Tuple

from core import Shot, Storyboard


def _shot_camera(idx: int) -> str:
    cameras = ["wide", "medium", "close-up", "top-down"]
    return cameras[(idx - 1) % len(cameras)]


def _shot_scene(line_text: str) -> str:
    lowered = str(line_text or "").lower()
    if "benchmark" in lowered or "latency" in lowered:
        return "metrics dashboard"
    if "deploy" in lowered or "rollback" in lowered:
        return "production control room"
    if "architecture" in lowered or "pipeline" in lowered:
        return "system diagram wall"
    return "technical studio"


def script_to_storyboard(script: Dict[str, object], constraints: Dict[str, object]) -> Storyboard:
    """Map time-coded script to engineering storyboard format."""
    run_id = str(constraints.get("run_id") or "run_local")
    item_id = str(script.get("item_id") or constraints.get("item_id") or "item_unknown")
    aspect = str(constraints.get("aspect") or "9:16")
    min_shots = max(1, int(constraints.get("min_shots") or 5))
    max_shots = max(min_shots, int(constraints.get("max_shots") or 8))

    lines = list(script.get("lines") or [])
    shots_raw = lines[:max_shots]
    if len(shots_raw) < min_shots and lines:
        # Repeat strongest lines to satisfy min shot count.
        while len(shots_raw) < min_shots:
            shots_raw.append(lines[len(shots_raw) % len(lines)])

    shots: List[Shot] = []
    for idx, line in enumerate(shots_raw, start=1):
        start_sec = float(line.get("start_sec", 0.0) or 0.0)
        end_sec = float(line.get("end_sec", start_sec + 4.0) or (start_sec + 4.0))
        duration = max(1.0, end_sec - start_sec)
        text = str(line.get("text") or "").strip()

        shots.append(
            Shot(
                idx=idx,
                duration=round(duration, 2),
                camera=_shot_camera(idx),
                scene=_shot_scene(text),
                subject_id=f"subject_{idx}",
                action=text or "Explain technical insight",
                overlay_text=text[:72] if text else None,
                reference_assets=[],
            )
        )

    total_duration = int(script.get("duration_sec") or sum(shot.duration for shot in shots) or 30)
    board = Storyboard(
        run_id=run_id,
        item_id=item_id,
        duration_sec=total_duration,
        aspect=aspect,
        shots=shots,
    )
    fixed, _changes = auto_fix_storyboard(board)
    return fixed


def validate_storyboard(board: Storyboard) -> Tuple[bool, List[str]]:
    """Validate storyboard against MVP constraints."""
    errors: List[str] = []
    if not board.run_id:
        errors.append("run_id is required")
    if not board.item_id:
        errors.append("item_id is required")
    if board.aspect not in {"9:16", "16:9", "1:1"}:
        errors.append("aspect must be one of 9:16/16:9/1:1")
    if board.duration_sec <= 0:
        errors.append("duration_sec must be positive")

    shot_count = len(board.shots)
    if shot_count < 5 or shot_count > 8:
        errors.append("shot count must be between 5 and 8 for MVP")

    for shot in board.shots:
        if shot.idx <= 0:
            errors.append("shot idx must be positive")
        if shot.duration <= 0:
            errors.append(f"shot {shot.idx} duration must be positive")
        if not str(shot.camera).strip():
            errors.append(f"shot {shot.idx} camera is required")
        if not str(shot.scene).strip():
            errors.append(f"shot {shot.idx} scene is required")
        if not str(shot.action).strip():
            errors.append(f"shot {shot.idx} action is required")

    return len(errors) == 0, errors


def auto_fix_storyboard(board: Storyboard) -> Tuple[Storyboard, List[str]]:
    """Best-effort fixes for common storyboard violations."""
    changes: List[str] = []
    shots = [Shot(**shot.model_dump()) for shot in board.shots]

    if len(shots) > 8:
        shots = shots[:8]
        changes.append("trimmed_shots_to_8")

    if len(shots) < 5 and shots:
        seed = [Shot(**shot.model_dump()) for shot in shots]
        while len(shots) < 5:
            source = seed[(len(shots) - len(seed)) % len(seed)]
            dup = Shot(**source.model_dump())
            dup.idx = len(shots) + 1
            shots.append(dup)
        changes.append("expanded_shots_to_5")

    if not shots:
        shots = [
            Shot(
                idx=i,
                duration=5.0,
                camera=_shot_camera(i),
                scene="technical studio",
                subject_id=f"subject_{i}",
                action="Explain technical context",
                overlay_text=None,
                reference_assets=[],
            )
            for i in range(1, 6)
        ]
        changes.append("created_default_shots")

    # Reindex and normalize invalid fields.
    for idx, shot in enumerate(shots, start=1):
        if shot.idx != idx:
            changes.append("reindexed_shots")
        shot.idx = idx
        if shot.duration <= 0:
            shot.duration = 4.0
            changes.append("fixed_non_positive_duration")
        if not str(shot.camera).strip():
            shot.camera = _shot_camera(idx)
            changes.append("filled_camera")
        if not str(shot.scene).strip():
            shot.scene = "technical studio"
            changes.append("filled_scene")
        if not str(shot.action).strip():
            shot.action = "Explain technical insight"
            changes.append("filled_action")

    duration_sec = int(board.duration_sec or round(sum(shot.duration for shot in shots)))
    if duration_sec <= 0:
        duration_sec = int(round(sum(shot.duration for shot in shots))) or 30
        changes.append("fixed_duration_sec")

    fixed = Storyboard(
        run_id=board.run_id or "run_local",
        item_id=board.item_id or "item_unknown",
        duration_sec=duration_sec,
        aspect=board.aspect if board.aspect in {"9:16", "16:9", "1:1"} else "9:16",
        shots=shots,
    )
    return fixed, changes
