from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys


def test_validate_artifacts_v2_smoke_gate(tmp_path: Path) -> None:
    smoke_dir = tmp_path / "smoke"
    cmd = [sys.executable, "scripts/e2e_smoke_v2.py", "--out-dir", str(smoke_dir)]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(proc.stdout)

    run_id = str(payload["run_id"])
    render_job_id = str(payload["render_job_id"])
    run_dir = smoke_dir / "runs" / run_id
    render_dir = smoke_dir / "render_jobs" / render_job_id

    validate_cmd = [
        sys.executable,
        "scripts/validate_artifacts_v2.py",
        "--run-dir",
        str(run_dir),
        "--render-dir",
        str(render_dir),
    ]
    validated = subprocess.run(validate_cmd, check=False, capture_output=True, text=True)
    assert validated.returncode == 0

    report = json.loads(validated.stdout)
    assert report["ok"] is True
    assert report["checks"]["onepager_top_picks_ge_3"] is True
    assert report["checks"]["onepager_domain_rows_ge_3"] is True
    assert report["checks"]["mp4_duration_ge_10"] is True
    assert report["checks"]["render_status_seedance_flag_present"] is True
    assert report["checks"]["script_no_html_tokens"] is True
    assert report["checks"]["onepager_no_html_tokens"] is True
    assert report["checks"]["topic_relevance_ok"] is True
    assert report["checks"]["onepager_bullets_compact_ok"] is True
    assert report["checks"]["facts_has_why_now_and_proof"] is True
    assert report["checks"]["ranked_items_have_update_signal"] is True
    assert report["checks"]["storyboard_overlay_safe"] is True
    assert report["checks"]["urls_valid"] is True
    assert report["checks"]["evidence_dedup_ok"] is True
    assert report["checks"]["retrieval_diagnosis_parse_ok"] is True
    assert report["checks"]["retrieval_diagnosis_attempts_present"] is True
    assert report["checks"]["retrieval_attempt_quality_fields_present"] is True
    assert report["checks"]["onepager_diagnosis_path_present"] is True
    assert report["checks"]["onepager_evidence_audit_path_present"] is True
    assert report["checks"]["onepager_relevance_summary_fields_present"] is True
    assert report["checks"]["evidence_audit_exists"] is True
    assert report["checks"]["evidence_audit_parse_ok"] is True
    assert report["checks"]["top_picks_all_pass_or_downgrade_reason_present"] is True
    assert report["checks"]["citations_not_mostly_duplicate"] is True
    assert report["checks"]["top_picks_not_all_downgrade"] is True
    assert report["checks"]["retrieval_quality_triggered_expansion_recorded"] is True
    assert report["checks"]["facts_no_truncated_words"] is True
    assert report["checks"]["top_picks_hard_relevance_invariants"] is True
    assert report["checks"]["selected_attempt_has_no_key_connector_timeout"] is True
    assert report["checks"]["top_picks_not_link_heavy_low_alignment"] is True
    assert report["checks"]["relevance_not_all_1.0"] is True
    assert report["checks"]["top_picks_not_infra_dominant"] is True
    assert report["checks"]["onepager_has_hot_new_agents_section"] is True


def test_validate_artifacts_v2_rejects_smoke_tokens_in_live_mode(tmp_path: Path) -> None:
    smoke_dir = tmp_path / "smoke_for_live"
    cmd = [sys.executable, "scripts/e2e_smoke_v2.py", "--out-dir", str(smoke_dir)]
    payload = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)

    run_id = str(payload["run_id"])
    render_job_id = str(payload["render_job_id"])
    run_dir = smoke_dir / "runs" / run_id
    render_dir = smoke_dir / "render_jobs" / render_job_id

    onepager = run_dir / "onepager.md"
    text = onepager.read_text(encoding="utf-8")
    onepager.write_text(text.replace("DataMode: `smoke`", "DataMode: `live`"), encoding="utf-8")

    run_context_path = run_dir / "run_context.json"
    run_context = json.loads(run_context_path.read_text(encoding="utf-8"))
    run_context["data_mode"] = "live"
    run_context_path.write_text(json.dumps(run_context, ensure_ascii=False, indent=2), encoding="utf-8")

    validate_cmd = [
        sys.executable,
        "scripts/validate_artifacts_v2.py",
        "--run-dir",
        str(run_dir),
        "--render-dir",
        str(render_dir),
    ]
    validated = subprocess.run(validate_cmd, check=False, capture_output=True, text=True)
    assert validated.returncode == 1
    report = json.loads(validated.stdout)
    assert report["ok"] is False
    assert any(str(err).startswith("live_smoke_tokens:") for err in report["errors"])


def test_validate_artifacts_v2_fails_when_live_top_picks_below_requested(tmp_path: Path) -> None:
    smoke_dir = tmp_path / "smoke_shortage"
    payload = json.loads(
        subprocess.run([sys.executable, "scripts/e2e_smoke_v2.py", "--out-dir", str(smoke_dir)], check=True, capture_output=True, text=True).stdout
    )
    run_id = str(payload["run_id"])
    render_job_id = str(payload["render_job_id"])
    run_dir = smoke_dir / "runs" / run_id
    render_dir = smoke_dir / "render_jobs" / render_job_id

    run_context_path = run_dir / "run_context.json"
    run_context = json.loads(run_context_path.read_text(encoding="utf-8"))
    run_context["data_mode"] = "live"
    ranking = dict(run_context.get("ranking_stats") or {})
    ranking["requested_top_k"] = 5
    ranking["top_picks_count"] = 2
    ranking["why_not_more"] = []
    run_context["ranking_stats"] = ranking
    retrieval = dict(run_context.get("retrieval") or {})
    retrieval["why_not_more"] = []
    run_context["retrieval"] = retrieval
    run_context_path.write_text(json.dumps(run_context, ensure_ascii=False, indent=2), encoding="utf-8")

    onepager_path = run_dir / "onepager.md"
    text = onepager_path.read_text(encoding="utf-8")
    text = text.replace("DataMode: `smoke`", "DataMode: `live`")
    text = re.sub(r"^- RequestedTopK:\s*`?\d+`?\s*$", "- RequestedTopK: `5`", text, flags=re.MULTILINE)
    text = re.sub(r"^- TopPicksCount:\s*`?\d+`?\s*$", "- TopPicksCount: `2`", text, flags=re.MULTILINE)
    text = re.sub(r"\n## Why not more\?\s*[\s\S]*?(?=\n## |\Z)", "\n", text, flags=re.MULTILINE)
    onepager_path.write_text(text, encoding="utf-8")

    validated = subprocess.run(
        [
            sys.executable,
            "scripts/validate_artifacts_v2.py",
            "--run-dir",
            str(run_dir),
            "--render-dir",
            str(render_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert validated.returncode == 1
    report = json.loads(validated.stdout)
    assert report["ok"] is False
    assert any(str(err).startswith("candidate_shortage_without_explanation:") for err in report["errors"])


def test_validate_artifacts_v2_allows_live_shortage_when_why_not_more_present(tmp_path: Path) -> None:
    smoke_dir = tmp_path / "smoke_shortage_allowed"
    payload = json.loads(
        subprocess.run([sys.executable, "scripts/e2e_smoke_v2.py", "--out-dir", str(smoke_dir)], check=True, capture_output=True, text=True).stdout
    )
    run_id = str(payload["run_id"])
    render_job_id = str(payload["render_job_id"])
    run_dir = smoke_dir / "runs" / run_id
    render_dir = smoke_dir / "render_jobs" / render_job_id

    run_context_path = run_dir / "run_context.json"
    run_context = json.loads(run_context_path.read_text(encoding="utf-8"))
    run_context["data_mode"] = "live"
    ranking = dict(run_context.get("ranking_stats") or {})
    ranking["requested_top_k"] = 5
    ranking["top_picks_count"] = 2
    ranking["why_not_more"] = ["top_picks_lt_5", "source_diversity_lt_2"]
    run_context["ranking_stats"] = ranking
    retrieval = dict(run_context.get("retrieval") or {})
    retrieval["why_not_more"] = ["top_picks_lt_5", "source_diversity_lt_2"]
    run_context["retrieval"] = retrieval
    run_context_path.write_text(json.dumps(run_context, ensure_ascii=False, indent=2), encoding="utf-8")

    # Live mode gate rejects smoke tokens, so remove "-smoke" markers in text artifacts.
    for name in ["script.json", "onepager.md", "storyboard.json", "materials.json", "facts.json"]:
        path = run_dir / name
        content = path.read_text(encoding="utf-8")
        content = content.replace("-smoke", "-live")
        path.write_text(content, encoding="utf-8")

    onepager_path = run_dir / "onepager.md"
    text = onepager_path.read_text(encoding="utf-8")
    text = text.replace("DataMode: `smoke`", "DataMode: `live`")
    text = re.sub(r"^- RequestedTopK:\s*`?\d+`?\s*$", "- RequestedTopK: `5`", text, flags=re.MULTILINE)
    text = re.sub(r"^- TopPicksCount:\s*`?\d+`?\s*$", "- TopPicksCount: `2`", text, flags=re.MULTILINE)
    if "## Why not more?" not in text:
        text += "\n## Why not more?\n\n- top_picks_lt_5\n- source_diversity_lt_2\n"
    onepager_path.write_text(text, encoding="utf-8")

    validated = subprocess.run(
        [
            sys.executable,
            "scripts/validate_artifacts_v2.py",
            "--run-dir",
            str(run_dir),
            "--render-dir",
            str(render_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    report = json.loads(validated.stdout)
    assert report["checks"]["candidate_shortage_explained"] is True
    assert not any(str(err).startswith("candidate_shortage_without_explanation:") for err in list(report.get("errors") or []))


def test_validate_artifacts_v2_flags_selected_key_timeout_attempt(tmp_path: Path) -> None:
    smoke_dir = tmp_path / "smoke_key_timeout"
    payload = json.loads(
        subprocess.run([sys.executable, "scripts/e2e_smoke_v2.py", "--out-dir", str(smoke_dir)], check=True, capture_output=True, text=True).stdout
    )
    run_id = str(payload["run_id"])
    render_job_id = str(payload["render_job_id"])
    run_dir = smoke_dir / "runs" / run_id
    render_dir = smoke_dir / "render_jobs" / render_job_id

    run_context_path = run_dir / "run_context.json"
    run_context = json.loads(run_context_path.read_text(encoding="utf-8"))
    run_context["data_mode"] = "live"
    run_context_path.write_text(json.dumps(run_context, ensure_ascii=False, indent=2), encoding="utf-8")

    onepager_path = run_dir / "onepager.md"
    onepager_path.write_text(
        onepager_path.read_text(encoding="utf-8").replace("DataMode: `smoke`", "DataMode: `live`"),
        encoding="utf-8",
    )

    diagnosis_path = run_dir / "retrieval_diagnosis.json"
    diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    attempts = list(diagnosis.get("attempts") or [])
    if attempts:
        attempts[0]["has_key_connector_timeout"] = True
        diagnosis["attempts"] = attempts
        diagnosis["selected_attempt"] = 1
    diagnosis["selected_attempt_has_key_connector_timeout"] = True
    diagnosis["all_attempts_key_connector_timeout"] = False
    diagnosis["key_connector_timeout_degraded_result"] = False
    diagnosis_path.write_text(json.dumps(diagnosis, ensure_ascii=False, indent=2), encoding="utf-8")

    validated = subprocess.run(
        [
            sys.executable,
            "scripts/validate_artifacts_v2.py",
            "--run-dir",
            str(run_dir),
            "--render-dir",
            str(render_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert validated.returncode == 1
    report = json.loads(validated.stdout)
    assert report["checks"]["selected_attempt_has_no_key_connector_timeout"] is False
    assert "selected_attempt_has_key_connector_timeout" in list(report.get("errors") or [])


def test_validate_artifacts_v2_flags_infra_dominant_top_picks_for_hot_agent_mode(tmp_path: Path) -> None:
    smoke_dir = tmp_path / "smoke_infra_dominant"
    payload = json.loads(
        subprocess.run([sys.executable, "scripts/e2e_smoke_v2.py", "--out-dir", str(smoke_dir)], check=True, capture_output=True, text=True).stdout
    )
    run_id = str(payload["run_id"])
    render_job_id = str(payload["render_job_id"])
    run_dir = smoke_dir / "runs" / run_id
    render_dir = smoke_dir / "render_jobs" / render_job_id

    run_context_path = run_dir / "run_context.json"
    run_context = json.loads(run_context_path.read_text(encoding="utf-8"))
    run_context["data_mode"] = "live"
    run_context["topic"] = "AI agent"
    run_context["time_window"] = "7d"
    ranking = dict(run_context.get("ranking_stats") or {})
    ranking["intent"] = "hot_new_agents"
    ranking["intent_mode"] = "hot_new_agents"
    top_ids = [str(value).strip() for value in list(ranking.get("top_item_ids") or []) if str(value).strip()]
    ranking["top_item_ids"] = top_ids
    run_context["ranking_stats"] = ranking
    run_context_path.write_text(json.dumps(run_context, ensure_ascii=False, indent=2), encoding="utf-8")

    diagnosis_path = run_dir / "retrieval_diagnosis.json"
    diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    candidate_rows = list(diagnosis.get("candidate_rows") or [])
    candidate_map = {
        str((item or {}).get("item_id") or "").strip(): dict(item or {})
        for item in candidate_rows
        if str((item or {}).get("item_id") or "").strip()
    }
    for item_id in top_ids[:3]:
        row = dict(candidate_map.get(item_id) or {"item_id": item_id, "title": item_id, "url": "https://example.com", "source": "github"})
        row["intent_is_infra"] = True
        row["infra_exception_event"] = False
        candidate_map[item_id] = row
    diagnosis["candidate_rows"] = list(candidate_map.values())
    diagnosis_path.write_text(json.dumps(diagnosis, ensure_ascii=False, indent=2), encoding="utf-8")

    onepager_path = run_dir / "onepager.md"
    onepager_path.write_text(
        onepager_path.read_text(encoding="utf-8").replace("DataMode: `smoke`", "DataMode: `live`"),
        encoding="utf-8",
    )

    validated = subprocess.run(
        [
            sys.executable,
            "scripts/validate_artifacts_v2.py",
            "--run-dir",
            str(run_dir),
            "--render-dir",
            str(render_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert validated.returncode == 1
    report = json.loads(validated.stdout)
    assert report["checks"]["top_picks_not_infra_dominant"] is False
    assert any(str(err).startswith("top_picks_infra_dominant:") for err in list(report.get("errors") or []))


def test_validate_artifacts_v2_flags_missing_hot_new_agents_section(tmp_path: Path) -> None:
    smoke_dir = tmp_path / "smoke_missing_hot_section"
    payload = json.loads(
        subprocess.run([sys.executable, "scripts/e2e_smoke_v2.py", "--out-dir", str(smoke_dir)], check=True, capture_output=True, text=True).stdout
    )
    run_id = str(payload["run_id"])
    render_job_id = str(payload["render_job_id"])
    run_dir = smoke_dir / "runs" / run_id
    render_dir = smoke_dir / "render_jobs" / render_job_id

    run_context_path = run_dir / "run_context.json"
    run_context = json.loads(run_context_path.read_text(encoding="utf-8"))
    run_context["data_mode"] = "live"
    run_context["topic"] = "AI agent"
    run_context["time_window"] = "7d"
    ranking = dict(run_context.get("ranking_stats") or {})
    ranking["intent"] = "hot_new_agents"
    ranking["intent_mode"] = "hot_new_agents"
    top_ids = [str(value).strip() for value in list(ranking.get("top_item_ids") or []) if str(value).strip()]
    ranking["top_item_ids"] = top_ids
    run_context["ranking_stats"] = ranking
    run_context_path.write_text(json.dumps(run_context, ensure_ascii=False, indent=2), encoding="utf-8")

    diagnosis_path = run_dir / "retrieval_diagnosis.json"
    diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    candidate_rows = list(diagnosis.get("candidate_rows") or [])
    candidate_map = {
        str((item or {}).get("item_id") or "").strip(): dict(item or {})
        for item in candidate_rows
        if str((item or {}).get("item_id") or "").strip()
    }
    for item_id in top_ids[:3]:
        row = dict(candidate_map.get(item_id) or {"item_id": item_id, "title": item_id, "url": "https://example.com", "source": "github"})
        row["intent_is_infra"] = False
        row["infra_exception_event"] = False
        candidate_map[item_id] = row
    diagnosis["candidate_rows"] = list(candidate_map.values())
    diagnosis_path.write_text(json.dumps(diagnosis, ensure_ascii=False, indent=2), encoding="utf-8")

    onepager_path = run_dir / "onepager.md"
    onepager = onepager_path.read_text(encoding="utf-8")
    onepager = onepager.replace("DataMode: `smoke`", "DataMode: `live`")
    onepager = onepager.replace("## Top Picks: Hot New Agents (Top3)", "## Top Picks")
    onepager_path.write_text(onepager, encoding="utf-8")

    validated = subprocess.run(
        [
            sys.executable,
            "scripts/validate_artifacts_v2.py",
            "--run-dir",
            str(run_dir),
            "--render-dir",
            str(render_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert validated.returncode == 1
    report = json.loads(validated.stdout)
    assert report["checks"]["onepager_has_hot_new_agents_section"] is False
    assert "onepager_missing_hot_new_agents_section" in list(report.get("errors") or [])
