from __future__ import annotations

from core import Citation, NormalizedItem, Shot, Storyboard
from pipeline_v2.prompt_compiler import compile_shot_prompt, compile_storyboard, consistency_pack
from pipeline_v2.script_generator import generate_script, generate_variants
from pipeline_v2.storyboard_generator import auto_fix_storyboard, overlay_compact, script_to_storyboard, validate_storyboard


def _sample_item() -> NormalizedItem:
    return NormalizedItem(
        id="item_1",
        source="github",
        title="MCP Router Upgrade",
        url="https://example.com/item_1",
        body_md=(
            "Architecture update improves context routing latency by 23%. "
            "Deployment guide includes rollback strategy and canary metrics. "
            "Benchmark chart compares baseline and optimized path."
        ),
        citations=[
            Citation(
                title="Release notes",
                url="https://example.com/item_1/release",
                snippet="Latency improved by 23% in production routing.",
                source="github",
            )
        ],
        tier="A",
        lang="en",
        hash="hash_item_1",
        metadata={"credibility": "high", "clean_text": "Architecture update improves context routing latency by 23% with canary metrics."},
    )


def test_generate_script_and_variants() -> None:
    script = generate_script(_sample_item(), duration_sec=36, platform="reels", tone="professional")

    assert script["duration_sec"] == 36
    assert len(list(script["lines"])) >= 6
    assert float(script["lines"][0]["start_sec"]) == 0.0
    assert script["lines"][0]["section"] == "hook"
    assert float(script["lines"][0]["end_sec"]) <= 3.1
    assert "main_thesis" in script["structure"]
    assert len(list(script["structure"]["key_points"])) == 3
    assert "placeholder" not in str(script).lower()
    assert "facts" in script
    assert str(script["facts"]["what_it_is"]).strip() != ""

    variants = generate_variants(script, ["reels", "youtube"])
    assert "reels" in variants
    assert "youtube" in variants
    assert variants["reels"]["platform"] == "reels"


def test_script_to_storyboard_validate_and_autofix() -> None:
    script = generate_script(_sample_item(), duration_sec=40, platform="shorts", tone="technical")
    board = script_to_storyboard(
        script,
        constraints={"run_id": "run_1", "item_id": "item_1", "aspect": "9:16", "min_shots": 5, "max_shots": 8},
    )

    ok, errors = validate_storyboard(board)
    assert ok is True
    assert errors == []
    assert any(shot.reference_assets for shot in board.shots)
    assert all(len(str(shot.overlay_text or "")) <= 42 for shot in board.shots)

    broken = Storyboard(
        run_id="run_1",
        item_id="item_1",
        duration_sec=0,
        aspect="wrong",
        shots=[Shot(idx=0, duration=-1, camera="", scene="", action="", reference_assets=[])],
    )
    fixed, changes = auto_fix_storyboard(broken)

    ok_fixed, fixed_errors = validate_storyboard(fixed)
    assert ok_fixed is True
    assert fixed_errors == []
    assert changes


def test_prompt_compiler_builds_consistent_prompt_specs() -> None:
    board = Storyboard(
        run_id="run_1",
        item_id="item_1",
        duration_sec=30,
        aspect="9:16",
        shots=[
            Shot(
                idx=1,
                duration=4.0,
                camera="close-up",
                scene="metrics dashboard",
                subject_id="host_01",
                action="show latency benchmark",
                overlay_text="-23% latency",
                reference_assets=["asset://bench_1"],
            )
        ],
    )

    pack = consistency_pack("host_01", "style_neo")
    assert pack["character_id"] == "host_01"
    assert pack["style_id"] == "style_neo"

    shot_prompt = compile_shot_prompt(
        board.shots[0],
        {"style": "cinematic", "mood": "confident", "character_id": "host_01", "style_id": "style_neo", "aspect": "9:16"},
    )
    assert shot_prompt.shot_idx == 1
    assert "camera=close-up" in shot_prompt.prompt_text

    prompts = compile_storyboard(board, style_profile={"style_id": "style_neo", "character_id": "host_01"})
    assert len(prompts) == 1
    assert prompts[0].seedance_params["style_id"] == "style_neo"


def test_overlay_compact_keeps_readability_without_hard_cut() -> None:
    text = "This is a very long sentence that should be compacted into a short overlay for 9:16 cards without ugly truncation artifacts."
    compact = overlay_compact(text, max_chars=42)
    assert len(compact) <= 42
    assert not compact.endswith("...")


def test_script_and_prompt_references_exclude_tooling_links() -> None:
    item = _sample_item()
    item.citations.append(
        Citation(
            title="Tooling endpoint",
            url="https://api.openai.com/v1/chat/completions",
            snippet="endpoint",
            source="docs",
        )
    )
    script = generate_script(item, duration_sec=35, platform="shorts", tone="technical")
    for line in list(script.get("lines") or []):
        refs = [str(ref) for ref in list(line.get("references") or [])]
        assert all("api.openai.com" not in ref for ref in refs)

    board = script_to_storyboard(script, constraints={"run_id": "run_1", "item_id": "item_1", "aspect": "9:16", "min_shots": 5, "max_shots": 8})
    prompts = compile_storyboard(board, style_profile={"style_id": "style_neo", "character_id": "host_01"})
    all_refs = [str(ref) for prompt in prompts for ref in list(prompt.references or [])]
    assert all("api.openai.com" not in ref for ref in all_refs)
