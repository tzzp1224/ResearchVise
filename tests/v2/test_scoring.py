from __future__ import annotations

from datetime import datetime, timezone

from core import NormalizedItem
from pipeline_v2.scoring import (
    evaluate_relevance,
    rank_items,
    score_credibility,
    score_novelty,
    score_relevance,
    score_talkability,
    score_visual_assets,
)
from pipeline_v2.topic_intent import TopicIntent


def _item(
    item_id: str,
    *,
    tier: str,
    source: str,
    title: str,
    metadata: dict,
) -> NormalizedItem:
    body = ("architecture diagram benchmark rollout metrics evidence " * 12).strip()
    return NormalizedItem(
        id=item_id,
        source=source,
        title=title,
        url=f"https://example.com/{item_id}",
        author="author",
        published_at=datetime.now(timezone.utc),
        body_md=body,
        citations=[],
        tier=tier,
        lang="en",
        hash=f"hash_{item_id}",
        metadata={"body_len": len(body), **dict(metadata or {})},
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
        score_relevance(item, "ai agent"),
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


def test_rank_items_defer_short_body_without_evidence() -> None:
    short_body = _item(
        "s1",
        tier="A",
        source="github",
        title="tiny snippet",
        metadata={"body_len": 80, "citation_count": 0, "credibility": "high"},
    )
    strong = _item(
        "a1",
        tier="A",
        source="github",
        title="long and detailed update",
        metadata={"body_len": 760, "citation_count": 2, "credibility": "high"},
    )
    ranked = rank_items([short_body, strong], min_body_len_for_top_picks=300)
    assert ranked[0].item.id == "a1"
    assert any("quality.top_pick_gate=deferred" in reason for reason in ranked[1].reasons)


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


def test_rank_items_topic_relevance_gate_filters_offtopic_items() -> None:
    related = _item(
        "a_rel",
        tier="A",
        source="github",
        title="AI agent orchestration for tool calling",
        metadata={"stars": 500, "credibility": "high", "citation_count": 2},
    )
    unrelated = _item(
        "a_off",
        tier="A",
        source="github",
        title="VSCode dark theme collection",
        metadata={"stars": 800, "credibility": "high", "citation_count": 2},
    )

    ranked = rank_items([unrelated, related], topic="AI agent", relevance_threshold=0.55)
    assert ranked[0].item.id == "a_rel"
    assert any("penalty.relevance_lt_0.55" in reason for reason in ranked[1].reasons)


def test_ranking_keeps_relevance_hard_gate_even_with_recency_and_engagement() -> None:
    related_1 = _item(
        "rel_1",
        tier="A",
        source="github",
        title="AI agent orchestration runtime",
        metadata={"stars": 120, "citation_count": 2, "credibility": "high"},
    )
    related_2 = _item(
        "rel_2",
        tier="A",
        source="hackernews",
        title="Agent tools and MCP workflow",
        metadata={"points": 80, "comment_count": 20, "citation_count": 2, "credibility": "high"},
    )
    off_topic_hot = _item(
        "off_hot",
        tier="A",
        source="github",
        title="VSCode dark islands theme",
        metadata={
            "stars": 50000,
            "points": 9000,
            "comment_count": 3000,
            "citation_count": 3,
            "published_recency": 1.0,
            "credibility": "high",
        },
    )

    ranked = rank_items([off_topic_hot, related_1, related_2], topic="AI agent", relevance_threshold=0.55)
    top_ids = [row.item.id for row in ranked[:2]]
    assert "off_hot" not in top_ids
    off = next(row for row in ranked if row.item.id == "off_hot")
    assert any(reason.startswith("penalty.relevance_lt_0.55") for reason in off.reasons)


def test_agent_generic_term_alone_cannot_reach_perfect_relevance() -> None:
    item = _item(
        "agent_only",
        tier="A",
        source="github",
        title="AI agent updates",
        metadata={
            "credibility": "high",
            "body_len": 820,
            "citation_count": 2,
            "quality_signals": {
                "content_density": 0.16,
                "has_quickstart": False,
                "has_results_or_bench": False,
                "evidence_links_quality": 1,
            },
        },
    )
    item.body_md = (
        "Agent platform update for teams. Agent templates and agent role descriptions are included. "
        "No tool calling, no orchestration details, and no benchmark numbers are provided."
    )
    score = score_relevance(item, "AI agent")
    assert score <= 0.79


def test_agent_generic_term_without_semantic_depth_is_capped_below_selection_floor() -> None:
    item = _item(
        "agent_generic_only",
        tier="A",
        source="github",
        title="Agent update digest",
        metadata={
            "credibility": "high",
            "body_len": 900,
            "citation_count": 2,
            "bucket_hits": [],
            "quality_signals": {
                "content_density": 0.16,
                "has_quickstart": False,
                "has_results_or_bench": False,
                "evidence_links_quality": 1,
            },
        },
    )
    item.body_md = (
        "This post summarizes agent updates and generic assistant examples for teams. "
        "It focuses on general status notes without implementation details or protocol specifics."
    )
    score = score_relevance(item, "AI agent")
    assert score <= 0.74


def test_source_query_terms_do_not_bypass_hard_relevance_gate() -> None:
    item = _item(
        "offtopic_source_query",
        tier="A",
        source="huggingface",
        title="Qwen2.5-VL-3B-Instruct model card",
        metadata={
            "credibility": "high",
            "citation_count": 2,
            "source_query": "agent orchestration tool calling mcp",
        },
    )
    item.body_md = "Vision-language model card with OCR, image understanding, and video reasoning examples."
    score = score_relevance(item, "AI agent")
    assert score == 0.0


def test_cross_source_corroboration_bonus_improves_ranking_order() -> None:
    baseline = _item(
        "baseline",
        tier="A",
        source="github",
        title="Agent orchestration runtime",
        metadata={"credibility": "high", "citation_count": 2},
    )
    corroborated = _item(
        "corroborated",
        tier="A",
        source="github",
        title="Agent orchestration runtime",
        metadata={
            "credibility": "high",
            "citation_count": 2,
            "cross_source_bonus": 0.05,
            "cross_source_corroboration_sources": ["github", "hackernews"],
        },
    )
    ranked = rank_items([baseline, corroborated], topic="AI agent", relevance_threshold=0.55)
    assert ranked[0].item.id == "corroborated"
    assert any("cross_source.corroboration_bonus=0.05" in reason for reason in list(ranked[0].reasons or []))


def test_recent_topic_repeat_penalty_reduces_ranking_priority() -> None:
    repeated = _item(
        "repeated_pick",
        tier="A",
        source="github",
        title="Agent orchestration runtime",
        metadata={
            "credibility": "high",
            "citation_count": 2,
            "recent_topic_repeat_penalty": 0.12,
            "recent_topic_pick_count": 3,
        },
    )
    fresh = _item(
        "fresh_pick",
        tier="A",
        source="github",
        title="Agent orchestration runtime",
        metadata={"credibility": "high", "citation_count": 2},
    )
    ranked = rank_items([repeated, fresh], topic="AI agent", relevance_threshold=0.55)
    assert ranked[0].item.id == "fresh_pick"
    repeated_row = next(row for row in ranked if row.item.id == "repeated_pick")
    assert any("penalty.recent_topic_repeat=0.12" in reason for reason in list(repeated_row.reasons or []))


def test_agent_item_without_engagement_signal_is_capped_below_one() -> None:
    item = _item(
        "agent_low_engagement",
        tier="A",
        source="github",
        title="MCP agent runtime with LangGraph orchestration",
        metadata={
            "credibility": "high",
            "stars": 4,
            "forks": 0,
            "citation_count": 2,
            "quality_signals": {
                "content_density": 0.3,
                "has_quickstart": True,
                "has_results_or_bench": True,
                "evidence_links_quality": 3,
            },
        },
    )
    item.body_md = (
        "This agent runtime supports tool calling, MCP, and LangGraph orchestration. "
        "Quickstart includes pip install and benchmark results."
    )
    ranked = rank_items([item], topic="AI agent", relevance_threshold=0.55)
    assert float(ranked[0].relevance_score) < 0.9


def test_generic_agent_evaluation_terms_fail_hard_gate_and_cannot_reach_one() -> None:
    item = _item(
        "agent_eval_generic_only",
        tier="A",
        source="github",
        title="Agent evaluation digest",
        metadata={
            "credibility": "high",
            "body_len": 980,
            "citation_count": 2,
            "quality_signals": {
                "content_density": 0.16,
                "has_quickstart": False,
                "has_results_or_bench": False,
                "evidence_links_quality": 1,
            },
        },
    )
    item.body_md = (
        "Agent evaluation summary for teams. Agent status update and evaluation checklist only. "
        "No runtime design, no tool-calling details, and no benchmark evidence."
    )
    payload = evaluate_relevance(item, "AI agent")
    assert payload["hard_pass"] is False
    assert float(payload["score"]) < 1.0


def test_hot_new_agent_trend_signal_beats_large_framework_body_size_proxy() -> None:
    intent = TopicIntent.for_request(topic="AI agent", time_window="7d")
    assert intent is not None

    framework = _item(
        "infra_langchain",
        tier="A",
        source="github",
        title="langchain-ai/langchain",
        metadata={
            "credibility": "high",
            "stars": 120000,
            "forks": 32000,
            "citation_count": 3,
            "created_at": "2022-10-17T00:00:00Z",
            "updated_at": "2026-02-18T00:00:00Z",
            "quality_signals": {
                "content_density": 0.32,
                "has_quickstart": True,
                "has_results_or_bench": True,
                "evidence_links_quality": 3,
                "update_recency_days": 1.0,
            },
        },
    )
    framework.body_md = (
        "AI agent framework SDK and orchestration library updates. "
        "Framework notes cover maintenance changes and migration details."
    )

    hot_agent = _item(
        "hot_agent_demo",
        tier="A",
        source="github",
        title="acme/mcp-agent-demo",
        metadata={
            "credibility": "high",
            "stars": 240,
            "forks": 30,
            "citation_count": 3,
            "created_at": "2026-02-10T00:00:00Z",
            "updated_at": "2026-02-18T10:00:00Z",
            "release_published_at": "2026-02-17T09:00:00Z",
            "cross_source_corroboration_count": 3,
            "search_rank": 1,
            "search_pool_size": 30,
            "quality_signals": {
                "content_density": 0.34,
                "has_quickstart": True,
                "has_results_or_bench": True,
                "evidence_links_quality": 3,
                "update_recency_days": 1.0,
            },
        },
    )
    hot_agent.body_md = (
        "MCP agent app demo with tool calling workflow. "
        "Quickstart and benchmark included, plus release notes and community discussion."
    )

    ranked = rank_items(
        [framework, hot_agent],
        topic="AI agent",
        topic_intent=intent,
        relevance_threshold=0.55,
        recall_window="7d",
    )
    assert ranked[0].item.id == "hot_agent_demo"
    top_reasons = list(ranked[0].reasons or [])
    assert any("trend.signal=" in reason for reason in top_reasons)
    assert any("intent.hot_candidate=true" in reason for reason in top_reasons)
