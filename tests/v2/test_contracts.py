from __future__ import annotations

from datetime import datetime, timezone

from core import (
    Artifact,
    ArtifactType,
    Citation,
    NormalizedItem,
    PromptSpec,
    RenderStatus,
    RunMode,
    RunRequest,
    RunStatus,
    Shot,
    Storyboard,
)


def test_run_request_contract_fields() -> None:
    req = RunRequest(
        user_id="user_1",
        mode=RunMode.ONDEMAND,
        topic="mcp",
        time_window="7d",
        tz="America/Los_Angeles",
        output_targets=["script", "storyboard", "mp4"],
    )
    assert req.user_id == "user_1"
    assert req.mode == RunMode.ONDEMAND
    assert req.output_targets == ["script", "storyboard", "mp4"]


def test_normalized_item_storyboard_promptspec_artifact_contracts() -> None:
    item = NormalizedItem(
        id="item_1",
        source="github",
        title="Repo update",
        url="https://example.com/repo",
        author="octocat",
        published_at=datetime.now(timezone.utc),
        body_md="**content**",
        citations=[Citation(title="source", url="https://example.com", snippet="proof")],
        tier="A",
        lang="en",
        hash="abc123",
        metadata={"stars": 123},
    )
    shot = Shot(
        idx=1,
        duration=5.0,
        camera="close-up",
        scene="server room",
        action="show benchmark chart",
        overlay_text="+23% throughput",
        reference_assets=["asset://chart1"],
    )
    board = Storyboard(
        run_id="run_1",
        item_id=item.id,
        duration_sec=35,
        aspect="9:16",
        shots=[shot],
    )
    prompt = PromptSpec(
        shot_idx=1,
        prompt_text="cinematic technical explainer shot",
        references=["asset://chart1"],
        seedance_params={"cfg": 6.5},
    )
    artifact = Artifact(type=ArtifactType.MP4, path="/tmp/video.mp4", metadata={"duration": 35})

    assert board.shots[0].idx == 1
    assert prompt.shot_idx == 1
    assert artifact.type == ArtifactType.MP4


def test_run_and_render_status_have_required_observability_fields() -> None:
    run_status = RunStatus(run_id="run_1", state="queued")
    render_status = RenderStatus(render_job_id="render_1", run_id="run_1", state="queued")

    assert run_status.progress == 0.0
    assert run_status.errors == []
    assert run_status.timestamps.created_at is not None
    assert render_status.progress == 0.0
    assert render_status.errors == []
    assert render_status.timestamps.created_at is not None
