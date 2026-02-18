from __future__ import annotations

from datetime import datetime, timezone

from core import NormalizedItem
from pipeline_v2.scoring import (
    rank_items,
    score_credibility,
    score_novelty,
    score_talkability,
    score_visual_assets,
)


def _item(
    item_id: str,
    *,
    tier: str,
    source: str,
    title: str,
    metadata: dict,
) -> NormalizedItem:
    return NormalizedItem(
        id=item_id,
        source=source,
        title=title,
        url=f"https://example.com/{item_id}",
        author="author",
        published_at=datetime.now(timezone.utc),
        body_md="architecture diagram benchmark",
        citations=[],
        tier=tier,
        lang="en",
        hash=f"hash_{item_id}",
        metadata=metadata,
    )


def test_score_functions_return_bounded_values() -> None:
    item = _item(
        "a1",
        tier="A",
        source="github",
        title="Repo benchmark vs baseline",
        metadata={"stars": 1200, "points": 60, "comment_count": 22, "has_diagram": True, "credibility": "high"},
    )

    for value in [
        score_novelty(item),
        score_talkability(item),
        score_credibility(item),
        score_visual_assets(item),
    ]:
        assert 0.0 <= value <= 1.0


def test_rank_items_tier_b_not_in_top3_by_default() -> None:
    tier_b_low_talk = _item(
        "b1",
        tier="B",
        source="rss",
        title="Blog summary",
        metadata={"credibility": "low", "points": 1, "comment_count": 0},
    )
    tier_a_items = [
        _item("a1", tier="A", source="github", title="A1", metadata={"stars": 500, "credibility": "high"}),
        _item("a2", tier="A", source="huggingface", title="A2", metadata={"downloads": 900, "credibility": "high"}),
        _item("a3", tier="A", source="hackernews", title="A3", metadata={"points": 80, "comment_count": 30, "credibility": "high"}),
    ]

    ranked = rank_items([tier_b_low_talk] + tier_a_items)
    top3_tiers = [row.item.tier for row in ranked[:3]]
    assert top3_tiers == ["A", "A", "A"]
    assert any(reason.startswith("quality.body_len=") for reason in ranked[0].reasons)
    assert any(reason.startswith("quality.citation_count=") for reason in ranked[0].reasons)
    assert any(reason.startswith("quality.published_recency_days=") for reason in ranked[0].reasons)
    assert any(reason.startswith("quality.link_count=") for reason in ranked[0].reasons)


def test_rank_items_allows_tier_b_top3_when_talkability_is_high() -> None:
    tier_b_high_talk = _item(
        "b2",
        tier="B",
        source="rss",
        title="Breaking: benchmark vs baseline",
        metadata={"credibility": "medium", "points": 3000, "comment_count": 1200},
    )
    tier_a_items = [
        _item("a1", tier="A", source="github", title="A1", metadata={"stars": 120, "credibility": "high"}),
        _item("a2", tier="A", source="huggingface", title="A2", metadata={"downloads": 340, "credibility": "high"}),
        _item("a3", tier="A", source="hackernews", title="A3", metadata={"points": 40, "comment_count": 15, "credibility": "high"}),
    ]

    ranked = rank_items([tier_b_high_talk] + tier_a_items)
    top3_ids = [row.item.id for row in ranked[:3]]
    assert "b2" in top3_ids
