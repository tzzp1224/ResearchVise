"""Prompt compilation from storyboard to renderer-ready prompt specs."""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List

from core import PromptSpec, Shot, Storyboard


def consistency_pack(character_id: str, style_id: str) -> Dict[str, str]:
    """Build deterministic consistency identifiers for cross-shot coherence."""
    char = str(character_id or "default-character").strip() or "default-character"
    style = str(style_id or "default-style").strip() or "default-style"
    seed_raw = f"{char}|{style}"
    seed = int(hashlib.sha1(seed_raw.encode("utf-8")).hexdigest()[:8], 16)
    return {
        "character_id": char,
        "style_id": style,
        "consistency_seed": str(seed),
    }


def compile_shot_prompt(shot: Shot, style_profile: Dict[str, str]) -> PromptSpec:
    """Compile one shot into a PromptSpec with stable params."""
    profile = dict(style_profile or {})
    style = str(profile.get("style", "cinematic technical explainer")).strip()
    mood = str(profile.get("mood", "confident engineering tone")).strip()
    char_id = str(profile.get("character_id", "host_01")).strip()
    style_id = str(profile.get("style_id", "style_01")).strip()

    def _clean(value: str) -> str:
        text = re.sub(r"<[^>]+>", " ", str(value or ""))
        text = re.sub(r"\s+", " ", text).strip()
        return text

    prompt = (
        f"{style}; {mood}; "
        f"camera={_clean(shot.camera)}; scene={_clean(shot.scene)}; action={_clean(shot.action)}; "
        f"subject={_clean(shot.subject_id or char_id)}; overlay={_clean(shot.overlay_text or 'none')}"
    )
    negative = (
        "blurry, low-detail, random text artifacts, watermark,"
        " inconsistent character identity"
    )

    pack = consistency_pack(character_id=char_id, style_id=style_id)
    params = {
        "duration_sec": shot.duration,
        "aspect": str(profile.get("aspect", "9:16")),
        "camera": shot.camera,
        "seed": int(pack["consistency_seed"]),
        "style_id": style_id,
        "character_id": char_id,
    }

    references = list(shot.reference_assets or [])

    return PromptSpec(
        shot_idx=shot.idx,
        prompt_text=prompt,
        negative_prompt=negative,
        references=references,
        seedance_params=params,
    )


def compile_storyboard(board: Storyboard, style_profile: Dict[str, str] | None = None) -> List[PromptSpec]:
    """Compile full storyboard into ordered prompt specs."""
    profile = dict(style_profile or {})
    profile.setdefault("aspect", board.aspect)
    profile.setdefault("style", "cinematic technical explainer")
    profile.setdefault("mood", "high-clarity engineering narrative")
    profile.setdefault("character_id", "host_01")
    profile.setdefault("style_id", "style_01")

    prompts = [compile_shot_prompt(shot, profile) for shot in board.shots]
    prompts.sort(key=lambda item: item.shot_idx)
    return prompts
