"""Validate v2 run/render artifacts and print a machine-readable report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple


BLOCKLIST = {"placeholder", "dummy", "lorem", "todo", "testsrc", "colorbars"}
MIN_SCRIPT_LEN = 260
MIN_MP4_DURATION_SEC = 10.0


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _contains_blocklist(text: str) -> List[str]:
    lowered = str(text or "").lower()
    return [word for word in sorted(BLOCKLIST) if word in lowered]


def _ffprobe_info(path: Path) -> Tuple[bool, Dict, str]:
    probe_bin = shutil.which("ffprobe")
    if not probe_bin:
        head = path.read_bytes()[:64]
        if b"ftyp" in head:
            return True, {"format": {"duration": None}, "streams": []}, ""
        return False, {}, "ffprobe unavailable and missing ftyp signature"

    cmd = [
        probe_bin,
        "-hide_banner",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        return False, {}, (proc.stderr or proc.stdout or "ffprobe failed").strip()

    try:
        payload = json.loads(proc.stdout or "{}")
    except Exception:
        return False, {}, "ffprobe returned non-json payload"

    streams = list(payload.get("streams") or [])
    if not any(str(stream.get("codec_type") or "") == "video" for stream in streams):
        return False, payload, "video stream missing"
    return True, payload, ""


def _extract_video_duration(info: Dict) -> Optional[float]:
    try:
        value = info.get("format", {}).get("duration")
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _frame_bytes(mp4_path: Path, sec: float) -> Tuple[bool, bytes, str]:
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return False, b"", "ffmpeg unavailable"
    cmd = [
        ffmpeg_bin,
        "-v",
        "error",
        "-i",
        str(mp4_path),
        "-ss",
        f"{max(0.0, float(sec)):.3f}",
        "-frames:v",
        "1",
        "-vf",
        "scale=96:96",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True)
    if proc.returncode != 0:
        return False, b"", (proc.stderr.decode("utf-8", errors="ignore")[:240] or "frame extraction failed")
    expected = 96 * 96 * 3
    if len(proc.stdout) < expected:
        return False, b"", f"frame extraction returned {len(proc.stdout)} bytes"
    return True, proc.stdout[:expected], ""


def _frame_delta_ratio(a: bytes, b: bytes) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    total = 0
    for x, y in zip(a, b):
        total += abs(int(x) - int(y))
    return total / float(len(a) * 255)


def _validate_frame_variance(mp4_path: Path, duration: Optional[float]) -> Tuple[bool, str, Optional[float]]:
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return True, "skipped:ffmpeg_unavailable", None

    d = float(duration or 0.0)
    if d <= 2.5:
        return False, "duration too short for frame variance", 0.0

    t1 = 1.0
    t2 = max(1.0, d / 2.0)
    t3 = max(1.0, d - 1.0)

    ok1, f1, err1 = _frame_bytes(mp4_path, t1)
    ok2, f2, err2 = _frame_bytes(mp4_path, t2)
    ok3, f3, err3 = _frame_bytes(mp4_path, t3)
    if not ok1:
        return False, f"frame@{t1:.2f}s:{err1}", None
    if not ok2:
        return False, f"frame@{t2:.2f}s:{err2}", None
    if not ok3:
        return False, f"frame@{t3:.2f}s:{err3}", None

    d12 = _frame_delta_ratio(f1, f2)
    d13 = _frame_delta_ratio(f1, f3)
    d23 = _frame_delta_ratio(f2, f3)
    max_delta = max(d12, d13, d23)
    return max_delta >= 0.01, f"delta={max_delta:.6f}", max_delta


def validate(run_dir: Path, render_dir: Path) -> Dict:
    report: Dict = {
        "run_dir": str(run_dir),
        "render_dir": str(render_dir),
        "checks": {},
        "errors": [],
        "details": {},
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
        structure = dict(script_payload.get("structure") or {})
        key_points = list(structure.get("key_points") or [])

        report["checks"]["script_len_ok"] = len(script_text) >= MIN_SCRIPT_LEN
        report["checks"]["script_blocklist_ok"] = len(block_hits) == 0
        report["checks"]["script_has_lines"] = bool(script_payload.get("lines"))
        report["checks"]["script_structure_ok"] = bool(
            str(structure.get("hook") or "").strip()
            and str(structure.get("main_thesis") or "").strip()
            and len([item for item in key_points if str(item).strip()]) >= 3
            and str(structure.get("cta") or "").strip()
        )
        if block_hits:
            report["errors"].append("script_blocklist:" + ",".join(block_hits))

    if onepager_path.exists():
        onepager = onepager_path.read_text(encoding="utf-8")
        urls = re.findall(r"https?://[^\s)]+", onepager)
        top_pick_headings = re.findall(r"^###\s+\d+\.\s+", onepager, flags=re.MULTILINE)
        domain_rows = re.findall(r"^-\s*Source Domain:\s*`[^`]+`", onepager, flags=re.MULTILINE)
        block_hits = _contains_blocklist(onepager)
        report["checks"]["onepager_url_count_ge_3"] = len(urls) >= 3
        report["checks"]["onepager_top_picks_ge_3"] = len(top_pick_headings) >= 3
        report["checks"]["onepager_domain_rows_ge_3"] = len(domain_rows) >= 3
        report["checks"]["onepager_blocklist_ok"] = len(block_hits) == 0
        report["details"]["onepager_url_count"] = len(urls)
        report["details"]["onepager_top_pick_count"] = len(top_pick_headings)
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
        ok, info, error = _ffprobe_info(mp4_path)
        report["checks"]["mp4_probe_ok"] = ok
        if not ok:
            report["errors"].append("mp4_probe:" + error)
        else:
            duration = _extract_video_duration(info)
            report["details"]["mp4_duration_sec"] = duration
            report["checks"]["mp4_duration_ge_10"] = bool(duration is not None and duration >= MIN_MP4_DURATION_SEC)
            if not report["checks"]["mp4_duration_ge_10"]:
                report["errors"].append(f"mp4_duration_lt_{MIN_MP4_DURATION_SEC}")

            variance_ok, variance_desc, variance_value = _validate_frame_variance(mp4_path, duration)
            report["checks"]["mp4_frame_variance_ok"] = variance_ok
            report["details"]["mp4_frame_variance"] = variance_value
            report["details"]["mp4_frame_variance_desc"] = variance_desc
            if not variance_ok and not variance_desc.startswith("skipped:"):
                report["errors"].append("mp4_frame_variance:" + variance_desc)

    render_status_path = render_dir / "render_status.json"
    if render_status_path.exists():
        try:
            status_payload = _load_json(render_status_path)
            report["checks"]["render_status_seedance_flag_present"] = "seedance_used" in status_payload
            report["details"]["seedance_used"] = status_payload.get("seedance_used")
        except Exception as exc:
            report["checks"]["render_status_seedance_flag_present"] = False
            report["errors"].append(f"render_status_parse:{exc}")
    else:
        report["checks"]["render_status_seedance_flag_present"] = False
        report["errors"].append(f"missing:render_status:{render_status_path}")

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
