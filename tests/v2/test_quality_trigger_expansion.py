from __future__ import annotations

import json
from pathlib import Path

from core import RawItem, RunMode, RunRequest
from orchestrator import RunOrchestrator
from orchestrator.queue import InMemoryRunQueue
from orchestrator.store import InMemoryRunStore
from pipeline_v2.evidence_auditor import AuditRecord
from pipeline_v2.retrieval_controller import SelectionController
from pipeline_v2.runtime import RunPipelineRuntime
from render.adapters import BaseRendererAdapter, ShotRenderResult
from render.manager import RenderManager


class _NoopRenderer(BaseRendererAdapter):
    provider = "mock-noop"

    def render_shot(self, *, prompt_spec, output_dir: Path, mode: str, budget: dict, run_id: str, render_job_id: str) -> ShotRenderResult:
        _ = prompt_spec, output_dir, mode, budget, run_id, render_job_id
        return ShotRenderResult(shot_idx=1, success=True, output_path=str(output_dir / "noop.mp4"), cost=0.0)


def _connectors() -> dict:
    low_body_github = " ".join(
        [
            f"AI powered runtime for autonomous agent using CLIP ViT image classification adapters github step {idx}."
            for idx in range(1, 42)
        ]
    )
    low_body_hf = " ".join(
        [
            f"AI powered toolkit on huggingface for autonomous agent CLIP ViT checkpoints hf step {idx}."
            for idx in range(1, 42)
        ]
    )
    low_body_hn = " ".join(
        [
            f"AI powered discussion thread about autonomous agent CLIP ViT workflows hn step {idx}."
            for idx in range(1, 42)
        ]
    )
    high_body_github = " ".join(
        [f"AI agent orchestration runtime with MCP tool calling github stage {idx}." for idx in range(1, 42)]
    )
    high_body_hf = " ".join(
        [f"AI agent eval toolkit with function calling and MCP datasets hf stage {idx}." for idx in range(1, 42)]
    )
    high_body_hn = " ".join(
        [f"AI agent production incident report on tool routing and orchestration hn stage {idx}." for idx in range(1, 42)]
    )

    async def _github_topic_search(topic: str, time_window: str = "today", limit: int = 20, expanded: bool = False, **kwargs):
        _ = topic, limit, expanded, kwargs
        if time_window == "today":
            return [
                RawItem(
                    id="gh_low",
                    source="github",
                    title="acme/agent-clip-runtime",
                    url="https://github.com/acme/agent-clip-runtime",
                    body=low_body_github,
                    tier="A",
                    metadata={"stars": 1200, "item_type": "repo", "updated_at": "2026-02-18T09:00:00Z"},
                )
            ]
        return [
            RawItem(
                id="gh_high",
                source="github",
                title="acme/agent-orchestrator",
                url="https://github.com/acme/agent-orchestrator",
                body=high_body_github,
                tier="A",
                metadata={"stars": 2200, "item_type": "repo", "updated_at": "2026-02-17T09:00:00Z"},
            )
        ]

    async def _hf_search(topic: str, time_window: str = "today", limit: int = 20, expanded: bool = False, **kwargs):
        _ = topic, limit, expanded, kwargs
        if time_window == "today":
            return [
                RawItem(
                    id="hf_low",
                    source="huggingface",
                    title="acme/agent-clip-toolkit",
                    url="https://huggingface.co/acme/agent-clip-toolkit",
                    body=low_body_hf,
                    tier="A",
                    metadata={"downloads": 4300, "likes": 210, "item_type": "model", "last_modified": "2026-02-18T07:00:00Z"},
                )
            ]
        return [
            RawItem(
                id="hf_high",
                source="huggingface",
                title="acme/agent-tool-runtime",
                url="https://huggingface.co/acme/agent-tool-runtime",
                body=high_body_hf,
                tier="A",
                metadata={"downloads": 9800, "likes": 510, "item_type": "model", "last_modified": "2026-02-17T05:00:00Z"},
            )
        ]

    async def _hn_search(topic: str, time_window: str = "today", limit: int = 20, expanded: bool = False, **kwargs):
        _ = topic, limit, expanded, kwargs
        if time_window == "today":
            return [
                RawItem(
                    id="hn_low",
                    source="hackernews",
                    title="Agent clip runtime notes",
                    url="https://news.ycombinator.com/item?id=9001",
                    body=low_body_hn,
                    tier="A",
                    metadata={"points": 155, "comment_count": 44, "item_type": "story"},
                )
            ]
        return [
            RawItem(
                id="hn_high",
                source="hackernews",
                title="MCP tool calling orchestration in production",
                url="https://news.ycombinator.com/item?id=9002",
                body=high_body_hn,
                tier="A",
                metadata={"points": 240, "comment_count": 66, "item_type": "story"},
            )
        ]

    async def _none(*args, **kwargs):
        _ = args, kwargs
        return []

    return {
        "fetch_github_topic_search": _github_topic_search,
        "fetch_huggingface_search": _hf_search,
        "fetch_hackernews_search": _hn_search,
        "fetch_github_trending": _none,
        "fetch_huggingface_trending": _none,
        "fetch_hackernews_top": _none,
        "fetch_github_releases": _none,
        "fetch_rss_feed": _none,
        "fetch_web_article": _none,
    }


def test_quality_trigger_forces_expansion_even_when_topk_is_filled(tmp_path: Path) -> None:
    orchestrator = RunOrchestrator(store=InMemoryRunStore(), queue=InMemoryRunQueue())
    render_manager = RenderManager(renderer_adapter=_NoopRenderer(), work_dir=tmp_path / "render")
    runtime = RunPipelineRuntime(
        orchestrator=orchestrator,
        render_manager=render_manager,
        output_root=tmp_path / "runs",
        connector_overrides=_connectors(),
    )

    run_id = orchestrator.enqueue_run(
        RunRequest(
            user_id="u_quality",
            mode=RunMode.ONDEMAND,
            topic="AI agent",
            time_window="today",
            tz="UTC",
            budget={"top_k": 3, "include_tier_b": False, "render_enabled": False},
            output_targets=["web"],
        ),
        idempotency_key="u_quality:ai-agent",
    )

    result = runtime.run_next()
    assert result is not None

    run_dir = Path(result.output_dir)
    diagnosis = json.loads((run_dir / "retrieval_diagnosis.json").read_text(encoding="utf-8"))
    attempts = list(diagnosis.get("attempts") or [])
    assert attempts

    base = attempts[0]
    assert int(base.get("pass_count", 0)) < 3
    assert float(base.get("top_picks_min_relevance", 1.0)) < 0.75
    assert bool(base.get("quality_triggered_expansion")) is True

    assert bool(diagnosis.get("quality_triggered_expansion")) is True
    assert str(diagnosis.get("selected_phase") or "") != "base"


def test_controller_triggers_expansion_when_all_candidates_downgraded() -> None:
    from types import SimpleNamespace

    def _row(item_id: str, source: str, bucket: str):
        item = SimpleNamespace(id=item_id, source=source, metadata={"bucket_hits": [bucket]})
        return SimpleNamespace(item=item)

    rows = [_row("a", "github", "Frameworks"), _row("b", "hackernews", "Protocols & tool use")]
    records = [
        AuditRecord(
            item_id="a",
            title="a",
            source="github",
            rank=1,
            verdict="downgrade",
            reasons=["weak_evidence"],
            used_evidence_urls=[],
            evidence_domains=[],
            citation_duplicate_prefix_ratio=0.0,
            evidence_links_quality=1,
            body_len=620,
            min_body_len=600,
            publish_or_update_time="2026-02-18T10:00:00Z",
            machine_action={"action": "downgrade", "reason_code": "weak_evidence", "human_reason": "weak"},
        ),
        AuditRecord(
            item_id="b",
            title="b",
            source="hackernews",
            rank=2,
            verdict="downgrade",
            reasons=["weak_evidence"],
            used_evidence_urls=[],
            evidence_domains=[],
            citation_duplicate_prefix_ratio=0.0,
            evidence_links_quality=1,
            body_len=620,
            min_body_len=600,
            publish_or_update_time="2026-02-18T10:00:00Z",
            machine_action={"action": "downgrade", "reason_code": "weak_evidence", "human_reason": "weak"},
        ),
    ]
    controller = SelectionController(requested_top_k=2)
    outcome = controller.evaluate(ranked_rows=rows, audit_records=records, allow_downgrade_fill=False)
    decision = controller.expansion_decision(outcome=outcome)
    assert decision.should_expand is True
    assert any("pass_count_lt_2" in reason for reason in list(decision.reasons or []))


def test_controller_never_selects_zero_relevance_or_zero_body_even_with_downgrade_fill() -> None:
    from types import SimpleNamespace

    def _row(
        item_id: str,
        *,
        source: str,
        relevance: float,
        body_len: int,
        hard_pass: bool,
    ):
        item = SimpleNamespace(
            id=item_id,
            source=source,
            metadata={"bucket_hits": ["Frameworks"], "body_len": body_len, "topic_hard_match_pass": hard_pass},
            body_md="x" * max(0, body_len),
        )
        return SimpleNamespace(item=item, relevance_score=relevance)

    rows = [
        _row("bad_rel", source="github", relevance=0.0, body_len=900, hard_pass=True),
        _row("bad_body", source="huggingface", relevance=0.86, body_len=0, hard_pass=True),
        _row("bad_hard", source="hackernews", relevance=0.88, body_len=900, hard_pass=False),
        _row("good_dg", source="github", relevance=0.81, body_len=900, hard_pass=True),
    ]
    records = [
        AuditRecord(
            item_id="bad_rel",
            title="bad_rel",
            source="github",
            rank=1,
            verdict="downgrade",
            reasons=["topic_relevance_zero"],
            used_evidence_urls=[],
            evidence_domains=[],
            citation_duplicate_prefix_ratio=0.0,
            evidence_links_quality=2,
            body_len=900,
            min_body_len=400,
            publish_or_update_time="2026-02-18T10:00:00Z",
            machine_action={"action": "reject", "reason_code": "topic_relevance_zero", "human_reason": "bad"},
        ),
        AuditRecord(
            item_id="bad_body",
            title="bad_body",
            source="huggingface",
            rank=2,
            verdict="downgrade",
            reasons=["body_len_zero"],
            used_evidence_urls=[],
            evidence_domains=[],
            citation_duplicate_prefix_ratio=0.0,
            evidence_links_quality=2,
            body_len=0,
            min_body_len=400,
            publish_or_update_time="2026-02-18T10:00:00Z",
            machine_action={"action": "reject", "reason_code": "body_len_zero", "human_reason": "bad"},
        ),
        AuditRecord(
            item_id="bad_hard",
            title="bad_hard",
            source="hackernews",
            rank=3,
            verdict="downgrade",
            reasons=["topic_hard_gate_fail"],
            used_evidence_urls=[],
            evidence_domains=[],
            citation_duplicate_prefix_ratio=0.0,
            evidence_links_quality=2,
            body_len=900,
            min_body_len=600,
            publish_or_update_time="2026-02-18T10:00:00Z",
            machine_action={"action": "reject", "reason_code": "topic_hard_gate_fail", "human_reason": "bad"},
        ),
        AuditRecord(
            item_id="good_dg",
            title="good_dg",
            source="github",
            rank=4,
            verdict="downgrade",
            reasons=["weak_evidence"],
            used_evidence_urls=[],
            evidence_domains=[],
            citation_duplicate_prefix_ratio=0.0,
            evidence_links_quality=2,
            body_len=900,
            min_body_len=400,
            publish_or_update_time="2026-02-18T10:00:00Z",
            machine_action={"action": "downgrade", "reason_code": "weak_evidence", "human_reason": "weak"},
        ),
    ]
    controller = SelectionController(requested_top_k=3)
    outcome = controller.evaluate(
        ranked_rows=rows,
        audit_records=records,
        allow_downgrade_fill=True,
        min_relevance_for_selection=0.55,
    )
    selected_ids = [str(getattr(row.item, "id", "") or "") for row in list(outcome.selected_rows or [])]
    assert "good_dg" in selected_ids
    assert "bad_rel" not in selected_ids
    assert "bad_body" not in selected_ids
    assert "bad_hard" not in selected_ids


def test_controller_keeps_filling_pass_rows_even_when_source_coverage_short() -> None:
    from types import SimpleNamespace

    def _row(item_id: str, buckets: list[str]):
        item = SimpleNamespace(
            id=item_id,
            source="github",
            metadata={"bucket_hits": buckets, "body_len": 1200, "topic_hard_match_pass": True},
            body_md="x" * 1200,
        )
        return SimpleNamespace(item=item, relevance_score=0.92)

    rows = [
        _row("p1", ["Ops & runtime", "Protocols & tool use"]),
        _row("p2", ["Frameworks"]),
        _row("p3", ["Evaluation"]),
    ]
    records = [
        AuditRecord(
            item_id=item_id,
            title=item_id,
            source="github",
            rank=idx,
            verdict="pass",
            reasons=[],
            used_evidence_urls=["https://github.com/acme/repo"],
            evidence_domains=["github.com"],
            citation_duplicate_prefix_ratio=0.0,
            evidence_links_quality=3,
            body_len=1200,
            min_body_len=400,
            publish_or_update_time="2026-02-18T10:00:00Z",
            machine_action={"action": "pass", "reason_code": "ok", "human_reason": "ok"},
        )
        for idx, item_id in enumerate(["p1", "p2", "p3"], start=1)
    ]
    controller = SelectionController(requested_top_k=3, min_source_coverage=2)
    outcome = controller.evaluate(
        ranked_rows=rows,
        audit_records=records,
        allow_downgrade_fill=False,
        min_relevance_for_selection=0.55,
    )
    selected_ids = [str(getattr(row.item, "id", "") or "") for row in list(outcome.selected_rows or [])]
    assert selected_ids == ["p1", "p2", "p3"]
    assert int(outcome.source_coverage) == 1

    decision = controller.expansion_decision(outcome=outcome)
    assert bool(decision.should_expand) is False
    assert not list(decision.reasons or [])


def test_controller_applies_downgrade_fill_cap_and_returns_shortage() -> None:
    from types import SimpleNamespace

    def _row(item_id: str, source: str, relevance: float, verdict: str):
        item = SimpleNamespace(
            id=item_id,
            source=source,
            metadata={"bucket_hits": ["Frameworks"], "body_len": 900, "topic_hard_match_pass": True},
            body_md="agent runtime orchestration quickstart benchmark",
        )
        row = SimpleNamespace(item=item, relevance_score=relevance)
        record = AuditRecord(
            item_id=item_id,
            title=item_id,
            source=source,
            rank=1,
            verdict=verdict,
            reasons=([] if verdict == "pass" else ["weak_evidence"]),
            used_evidence_urls=["https://docs.acme.dev/agent-runtime"],
            evidence_domains=["docs.acme.dev"],
            citation_duplicate_prefix_ratio=0.0,
            evidence_links_quality=(2 if verdict == "pass" else 1),
            body_len=900,
            min_body_len=400,
            publish_or_update_time="2026-02-18T10:00:00Z",
            machine_action={"action": verdict, "reason_code": ("ok" if verdict == "pass" else "weak_evidence"), "human_reason": verdict},
        )
        return row, record

    pass_row, pass_record = _row("pass_1", "github", 0.92, "pass")
    dg_row_1, dg_record_1 = _row("dg_1", "huggingface", 0.88, "downgrade")
    dg_row_2, dg_record_2 = _row("dg_2", "hackernews", 0.87, "downgrade")

    controller = SelectionController(requested_top_k=3)
    outcome = controller.evaluate(
        ranked_rows=[pass_row, dg_row_1, dg_row_2],
        audit_records=[pass_record, dg_record_1, dg_record_2],
        allow_downgrade_fill=True,
        min_relevance_for_selection=0.55,
    )
    assert len(list(outcome.selected_rows or [])) == 2
    assert outcome.max_downgrade_allowed == 1
    assert outcome.downgrade_cap_reached is True


def test_controller_does_not_pick_low_score_bucket_outlier() -> None:
    from types import SimpleNamespace

    def _row(item_id: str, *, relevance: float, total: float, buckets: list[str]):
        item = SimpleNamespace(
            id=item_id,
            source="github",
            metadata={"bucket_hits": buckets, "body_len": 1200, "topic_hard_match_pass": True},
            body_md="agent runtime orchestration quickstart benchmark",
        )
        return SimpleNamespace(item=item, relevance_score=relevance, total_score=total)

    rows = [
        _row("top_a", relevance=0.92, total=0.96, buckets=["Ops & runtime"]),
        _row("low_bucket", relevance=0.75, total=0.74, buckets=["Frameworks"]),
        _row("top_b", relevance=0.91, total=0.92, buckets=[]),
        _row("top_c", relevance=0.89, total=0.90, buckets=["Evaluation"]),
    ]
    records = [
        AuditRecord(
            item_id=str(getattr(row.item, "id", "")),
            title=str(getattr(row.item, "id", "")),
            source="github",
            rank=idx,
            verdict="pass",
            reasons=[],
            used_evidence_urls=["https://docs.acme.dev/agent-runtime"],
            evidence_domains=["docs.acme.dev"],
            citation_duplicate_prefix_ratio=0.0,
            evidence_links_quality=3,
            body_len=1200,
            min_body_len=400,
            publish_or_update_time="2026-02-18T10:00:00Z",
            machine_action={"action": "pass", "reason_code": "ok", "human_reason": "ok"},
        )
        for idx, row in enumerate(rows, start=1)
    ]
    controller = SelectionController(requested_top_k=3, min_bucket_coverage=2, diversity_score_gap=0.12)
    outcome = controller.evaluate(
        ranked_rows=rows,
        audit_records=records,
        allow_downgrade_fill=False,
        min_relevance_for_selection=0.55,
    )
    selected_ids = [str(getattr(row.item, "id", "") or "") for row in list(outcome.selected_rows or [])]
    assert "low_bucket" not in selected_ids
    assert selected_ids[0] == "top_a"
    assert set(selected_ids) == {"top_a", "top_b", "top_c"}


def test_controller_triggers_expansion_when_selection_underfills_topk() -> None:
    from types import SimpleNamespace

    def _row(item_id: str, *, total: float, relevance: float):
        item = SimpleNamespace(
            id=item_id,
            source="github",
            metadata={"bucket_hits": ["Frameworks"], "body_len": 1200, "topic_hard_match_pass": True},
            body_md="agent orchestration runtime",
        )
        return SimpleNamespace(item=item, total_score=total, relevance_score=relevance)

    rows = [
        _row("p1", total=1.0, relevance=0.93),
        _row("p2", total=0.95, relevance=0.92),
        _row("p3", total=0.60, relevance=0.78),
    ]
    records = [
        AuditRecord(
            item_id=str(getattr(row.item, "id", "")),
            title=str(getattr(row.item, "id", "")),
            source="github",
            rank=idx,
            verdict="pass",
            reasons=[],
            used_evidence_urls=["https://docs.acme.dev/agent-runtime"],
            evidence_domains=["docs.acme.dev"],
            citation_duplicate_prefix_ratio=0.0,
            evidence_links_quality=3,
            body_len=1200,
            min_body_len=400,
            publish_or_update_time="2026-02-18T10:00:00Z",
            machine_action={"action": "pass", "reason_code": "ok", "human_reason": "ok"},
        )
        for idx, row in enumerate(rows, start=1)
    ]
    controller = SelectionController(requested_top_k=3, min_bucket_coverage=1, diversity_score_gap=0.12)
    outcome = controller.evaluate(
        ranked_rows=rows,
        audit_records=records,
        allow_downgrade_fill=False,
        min_relevance_for_selection=0.55,
    )
    assert len(list(outcome.selected_rows or [])) == 2
    decision = controller.expansion_decision(outcome=outcome)
    assert decision.should_expand is True
    assert any("selected_lt_3" in reason for reason in list(decision.reasons or []))
