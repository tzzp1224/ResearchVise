from __future__ import annotations

from datetime import datetime, timezone

from core import NormalizedItem
from pipeline_v2.scoring import evaluate_relevance, score_relevance
from pipeline_v2.topic_profile import TopicProfile


def _item(*, item_id: str, title: str, body: str) -> NormalizedItem:
    return NormalizedItem(
        id=item_id,
        source="github",
        title=title,
        url=f"https://example.com/{item_id}",
        author="author",
        published_at=datetime.now(timezone.utc),
        body_md=body,
        citations=[],
        tier="A",
        lang="en",
        hash=f"hash_{item_id}",
        metadata={"body_len": len(body), "citation_count": 2},
    )


def test_topic_profile_ai_agent_hard_pass_for_orchestration_text() -> None:
    profile = TopicProfile.for_topic("AI agent")
    text = "Agent orchestration runtime with MCP tool calling and production workflow automation."

    assert profile.requires_hard_gate is True
    assert profile.hard_match_pass(text) is True
    assert "orchestration" in " ".join(profile.matched_hard_terms(text)).lower()


def test_topic_profile_ai_agent_clip_hard_fail_sets_relevance_zero() -> None:
    item = _item(
        item_id="clip",
        title="openai/clip-vit-base-patch32",
        body=(
            "CLIP ViT model card for image classification and text-image retrieval benchmarks. "
            "This card describes vision transformer checkpoints and dataset metrics."
        ),
    )

    payload = evaluate_relevance(item, "AI agent")
    assert payload["hard_pass"] is False

    relevance = score_relevance(item, "AI agent")
    assert relevance == 0.0
