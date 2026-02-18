from __future__ import annotations

import json
from pathlib import Path
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
