from __future__ import annotations

from core import Shot
from pipeline_v2.prompt_compiler import compile_shot_prompt


def test_compile_shot_prompt_has_required_fields() -> None:
    shot = Shot(
        idx=2,
        duration=5.5,
        camera="wide",
        scene="system diagram wall",
        subject_id="host_02",
        action="explain retrieval pipeline",
        overlay_text="pipeline stages",
        reference_assets=["asset://diagram_2"],
    )

    prompt = compile_shot_prompt(
        shot,
        {
            "style": "technical documentary",
            "mood": "focused",
            "character_id": "host_02",
            "style_id": "style_blueprint",
            "aspect": "9:16",
        },
    )

    assert prompt.shot_idx == 2
    assert prompt.prompt_text
    assert prompt.negative_prompt
    assert prompt.references == []
    assert prompt.seedance_params["render_assets"] == ["asset://diagram_2"]
    assert prompt.seedance_params["character_id"] == "host_02"
    assert prompt.seedance_params["style_id"] == "style_blueprint"
