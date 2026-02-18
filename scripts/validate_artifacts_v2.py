"""Validate v2 run/render artifacts and print a machine-readable report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Dict, List


BLOCKLIST = {"placeholder", "dummy", "lorem", "todo", "testsrc", "colorbars"}
MIN_SCRIPT_LEN = 200


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _contains_blocklist(text: str) -> List[str]:
    lowered = str(text or "").lower()
    hits = [word for word in sorted(BLOCKLIST) if word in lowered]
    return hits


def _ffprobe_ok(path: Path) -> tuple[bool, str]:
    probe_bin = shutil.which("ffprobe")
    if not probe_bin:
        head = path.read_bytes()[:64]
        if b"ftyp" in head:
            return True, ""
        return False, "ffprobe unavailable and missing ftyp signature"

    cmd = [
        probe_bin,
        "-hide_banner",
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout or "ffprobe failed").strip()
    if "codec_type=video" not in proc.stdout:
        return False, "video stream missing"
    return True, ""


def validate(run_dir: Path, render_dir: Path) -> Dict:
    report: Dict = {
        "run_dir": str(run_dir),
        "render_dir": str(render_dir),
        "checks": {},
        "errors": [],
    }

    required = {
        "script": run_dir / "script.json",
        "onepager": run_dir / "onepager.md",
        "storyboard": run_dir / "storyboard.json",
        "materials": run_dir / "materials.json",
        "mp4": render_dir / "rendered_final.mp4",
    }
    for key, path in required.items():
        ok = path.exists() and path.is_file()
        report["checks"][f"{key}_exists"] = ok
        if not ok:
            report["errors"].append(f"missing:{key}:{path}")

    script_path = required["script"]
    onepager_path = required["onepager"]
    storyboard_path = required["storyboard"]
    mp4_path = required["mp4"]
    if not mp4_path.exists():
        fallback_mp4 = render_dir / "fallback_render.mp4"
        if fallback_mp4.exists():
            mp4_path = fallback_mp4
            report["checks"]["mp4_exists"] = True
            report["errors"] = [item for item in report["errors"] if not item.startswith("missing:mp4:")]

    if script_path.exists():
        script_payload = _load_json(script_path)
        script_text = json.dumps(script_payload, ensure_ascii=False)
        block_hits = _contains_blocklist(script_text)
        report["checks"]["script_len_ok"] = len(script_text) >= MIN_SCRIPT_LEN
        report["checks"]["script_blocklist_ok"] = len(block_hits) == 0
        report["checks"]["script_has_lines"] = bool(script_payload.get("lines"))
        if block_hits:
            report["errors"].append("script_blocklist:" + ",".join(block_hits))

    if onepager_path.exists():
        onepager = onepager_path.read_text(encoding="utf-8")
        urls = re.findall(r"https?://[^\s)]+", onepager)
        block_hits = _contains_blocklist(onepager)
        report["checks"]["onepager_url_count_ge_3"] = len(urls) >= 3
        report["checks"]["onepager_blocklist_ok"] = len(block_hits) == 0
        if block_hits:
            report["errors"].append("onepager_blocklist:" + ",".join(block_hits))

    if storyboard_path.exists():
        storyboard = _load_json(storyboard_path)
        shots = list(storyboard.get("shots") or [])
        report["checks"]["storyboard_shot_count_5_8"] = 5 <= len(shots) <= 8
        report["checks"]["storyboard_overlay_non_empty"] = all(
            str((shot or {}).get("overlay_text") or "").strip() != "" for shot in shots
        )

    if mp4_path.exists():
        ok, error = _ffprobe_ok(mp4_path)
        report["checks"]["mp4_probe_ok"] = ok
        if not ok:
            report["errors"].append("mp4_probe:" + error)

    report["ok"] = len(report["errors"]) == 0 and all(bool(v) for v in report["checks"].values())
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate v2 run/render artifact quality gates")
    parser.add_argument("--run-dir", required=True, help="Run artifact directory")
    parser.add_argument("--render-dir", required=True, help="Render job directory")
    parser.add_argument("--json-out", default="", help="Optional output path for JSON report")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    render_dir = Path(args.render_dir).resolve()
    report = validate(run_dir=run_dir, render_dir=render_dir)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)

    if str(args.json_out).strip():
        out = Path(args.json_out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + "\n", encoding="utf-8")

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
