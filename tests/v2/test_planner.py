from __future__ import annotations

from pipeline_v2.planner import build_retrieval_plan


def test_retrieval_plan_builds_distinct_expanded_queries_for_ai_agent() -> None:
    plan = build_retrieval_plan("AI agent", time_window="today")
    assert plan.topic == "AI agent"
    assert plan.base_queries
    assert plan.expanded_queries
    assert set(map(str.lower, plan.expanded_queries)) != set(map(str.lower, plan.base_queries))
    assert "mcp" in " ".join(map(str.lower, plan.expanded_queries))
    assert "vscode theme" in [item.lower() for item in plan.must_exclude_terms]
    assert "retriever" in [item.lower() for item in plan.must_exclude_terms]
    assert {"Frameworks", "Protocols & tool use", "Ops & runtime", "Evaluation"}.issubset(set(plan.query_buckets.keys()))
    github_bucket_queries = dict(plan.bucket_queries_by_source.get("github") or {})
    assert "Frameworks" in github_bucket_queries
    assert "Protocols & tool use" in github_bucket_queries


def test_retrieval_plan_contains_source_weights_and_policies() -> None:
    plan = build_retrieval_plan("copilot agent", time_window="today")
    assert set(plan.source_weights.keys()) == {"github", "hackernews", "huggingface"}
    assert any(rule.phase == "query_expanded" and rule.expanded_queries for rule in plan.time_window_policy)
    assert plan.source_limit_for_phase(source="github", phase="base", fallback=8) >= 1


def test_retrieval_plan_for_7d_never_shrinks_to_3d_window() -> None:
    plan = build_retrieval_plan("AI agent", time_window="7d")
    windows = [str(rule.window or "") for rule in list(plan.time_window_policy or [])]
    assert "3d" not in windows
    assert all(not str(rule.phase or "").startswith("window_3d") for rule in list(plan.time_window_policy or []))
    assert windows[0] == "7d"


def test_retrieval_plan_today_expands_to_3d_then_7d() -> None:
    plan = build_retrieval_plan("AI agent", time_window="today")
    windows = [str(rule.window or "") for rule in list(plan.time_window_policy or [])]
    assert windows[:2] == ["today", "today"]
    assert "3d" in windows
    assert "7d" in windows
