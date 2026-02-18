from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def test_e2e_smoke_command_outputs_artifacts(tmp_path: Path) -> None:
    cmd = [sys.executable, "scripts/e2e_smoke_v2.py", "--out-dir", str(tmp_path / "smoke")]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    payload = json.loads(proc.stdout.strip())

    assert payload["status"]["state"] == "completed"
    artifact_types = {item["type"] for item in payload["artifacts"]}
    assert "script" in artifact_types
    assert "storyboard" in artifact_types
    assert "onepager" in artifact_types
    assert "mp4" in artifact_types
    assert "zip" in artifact_types
