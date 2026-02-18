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
