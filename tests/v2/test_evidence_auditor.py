from __future__ import annotations

from core import Citation, NormalizedItem, RankedItem
from pipeline_v2.evidence_auditor import EvidenceAuditor, VERDICT_DOWNGRADE, VERDICT_REJECT


def _row(item_id: str, *, source: str, body: str, citations: list[Citation], metadata: dict, rank: int) -> RankedItem:
    item = NormalizedItem(
        id=item_id,
        source=source,
        title=f"title-{item_id}",
        url=f"https://example.com/{item_id}",
        author="author",
        published_at=None,
        body_md=body,
        citations=citations,
        tier="A",
        lang="en",
        hash=f"hash-{item_id}",
        metadata=metadata,
    )
    return RankedItem(
        rank=rank,
        item=item,
        total_score=0.8,
        novelty_score=0.8,
        talkability_score=0.7,
        credibility_score=0.8,
        visual_assets_score=0.6,
        relevance_score=0.8,
        reasons=[],
    )


def test_evidence_auditor_flags_mostly_duplicate_citations() -> None:
    row = _row(
        "dup",
        source="web_article",
        body=("deploy benchmark workflow " * 120).strip(),
        citations=[
            Citation(
                title="c1",
                url="https://docs.example.com/agent?id=1",
                snippet="repeated evidence prefix for validator and auditor checks alpha",
                source="web",
            ),
            Citation(
                title="c2",
                url="https://docs.example.com/agent?id=2",
                snippet="repeated evidence prefix for validator and auditor checks beta",
                source="web",
            ),
            Citation(
                title="c3",
                url="https://docs.example.com/agent?id=3",
                snippet="repeated evidence prefix for validator and auditor checks gamma",
                source="web",
            ),
        ],
        metadata={
            "body_len": 900,
            "quality_signals": {"evidence_links_quality": 3, "publish_or_update_time": "2026-02-18T10:00:00Z"},
        },
        rank=1,
    )
    auditor = EvidenceAuditor()
    record = auditor.audit_row(row, rank=1)
    assert record.verdict in {VERDICT_REJECT, VERDICT_DOWNGRADE}
    assert record.citation_duplicate_prefix_ratio > 0.6


def test_evidence_auditor_substitutes_rejected_pick_with_pass_candidate() -> None:
    rejected = _row(
        "r1",
        source="hackernews",
        body="short body",
        citations=[
            Citation(title="c1", url="https://docs.example.com/dup1", snippet="same same same", source="web"),
            Citation(title="c2", url="https://docs.example.com/dup2", snippet="same same same", source="web"),
        ],
        metadata={"body_len": 50, "quality_signals": {"evidence_links_quality": 0}},
        rank=1,
    )
    strong = _row(
        "p1",
        source="github",
        body=("cli api orchestration workflow benchmark " * 60).strip(),
        citations=[
            Citation(title="release", url="https://github.com/acme/r/releases/tag/v1.2.0", snippet="release notes", source="github"),
            Citation(title="docs", url="https://docs.acme.dev/agent", snippet="usage docs", source="docs"),
        ],
        metadata={
            "body_len": 1100,
            "quality_signals": {"evidence_links_quality": 3, "publish_or_update_time": "2026-02-18T10:00:00Z"},
            "evidence_links": ["https://github.com/acme/r/releases/tag/v1.2.0", "https://docs.acme.dev/agent"],
        },
        rank=2,
    )
    weak = _row(
        "d1",
        source="github",
        body=("workflow update " * 20).strip(),
        citations=[Citation(title="repo", url="https://github.com/acme/r", snippet="repo readme", source="github")],
        metadata={
            "body_len": 200,
            "quality_signals": {"evidence_links_quality": 1, "publish_or_update_time": ""},
            "evidence_links": ["https://github.com/acme/r"],
        },
        rank=3,
    )

    auditor = EvidenceAuditor()
    selection = auditor.audit_and_select(initial_picks=[rejected], ranked_rows=[rejected, strong, weak], top_count=2)
    selected_ids = [row.item.id for row in selection.selected_rows]
    assert "r1" not in selected_ids
    assert "p1" in selected_ids
    audit_by_id = selection.by_item_id()
    assert audit_by_id["d1"].verdict in {VERDICT_DOWNGRADE, VERDICT_REJECT}


def test_evidence_auditor_rejects_low_engagement_hn_item() -> None:
    row = _row(
        "hn_low",
        source="hackernews",
        body=("agent discussion notes " * 80).strip(),
        citations=[Citation(title="hn", url="https://news.ycombinator.com/item?id=123", snippet="discussion", source="hn")],
        metadata={
            "body_len": 900,
            "points": 2,
            "comment_count": 1,
            "quality_signals": {"evidence_links_quality": 1, "publish_or_update_time": "2026-02-18T10:00:00Z"},
        },
        rank=1,
    )
    auditor = EvidenceAuditor(topic="AI agent")
    record = auditor.audit_row(row, rank=1)
    assert record.verdict == VERDICT_REJECT
    assert any("hn_low_engagement" in reason for reason in list(record.reasons or []))


def test_evidence_auditor_rejects_zero_body_and_hard_gate_fail() -> None:
    row = _row(
        "bad_core",
        source="huggingface",
        body="",
        citations=[],
        metadata={
            "body_len": 0,
            "topic_hard_match_pass": False,
            "quality_signals": {"evidence_links_quality": 2, "publish_or_update_time": "2026-02-18T10:00:00Z"},
        },
        rank=1,
    )
    row.relevance_score = 0.0
    auditor = EvidenceAuditor(topic="AI agent")
    record = auditor.audit_row(row, rank=1)
    assert record.verdict == VERDICT_REJECT
    reasons = " ".join([str(reason) for reason in list(record.reasons or [])]).lower()
    assert "topic_relevance_zero" in reasons
    assert "topic_hard_gate_fail" in reasons
    assert "body_len_zero" in reasons
