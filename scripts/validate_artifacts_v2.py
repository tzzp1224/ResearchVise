"""Validate v2 run/render artifacts and print a machine-readable report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline_v2.sanitize import contains_html_like_tokens, is_allowed_citation_url, is_valid_http_url, normalize_url


BLOCKLIST = {"placeholder", "dummy", "lorem", "todo", "testsrc", "colorbars"}
MIN_SCRIPT_LEN = 260
MIN_MP4_DURATION_SEC = 10.0
MAX_ONEPAGER_BULLET_BYTES = 90
MAX_ONEPAGER_BULLETS_PER_PICK = 6
FIXTURE_HN_IDS = {"1000001", "123456", "999999"}
PLACEHOLDER_REPO_PATTERNS = [
    re.compile(r"https?://github\.com/org/repo(?:[-_/][^\s)]*)?$", flags=re.IGNORECASE),
]


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _contains_blocklist(text: str) -> List[str]:
    lowered = str(text or "").lower()
    return [word for word in sorted(BLOCKLIST) if word in lowered]


def _extract_urls(text: str) -> List[str]:
    urls = []
    for raw in re.findall(r"https?://[^\s)]+", str(text or "")):
        value = normalize_url(str(raw))
        if value:
            urls.append(value)
    deduped = []
    seen = set()
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        deduped.append(url)
    return deduped


def _compact_bullet_issues(onepager: str) -> List[str]:
    issues: List[str] = []
    parts = re.split(r"^###\s+\d+\.\s+", str(onepager or ""), flags=re.MULTILINE)
    if len(parts) <= 1:
        return ["no_pick_sections"]

    required_labels = ("WHAT｜", "WHY NOW｜", "HOW｜", "PROOF｜")
    for idx, section in enumerate(parts[1:], start=1):
        compact = re.search(r"#### Compact Brief\s*(.*?)(?:\n#### |\n### |\Z)", section, flags=re.DOTALL)
        if not compact:
            issues.append(f"pick_{idx}:missing_compact_brief")
            continue
        block = compact.group(1)
        bullets = [line.strip() for line in re.findall(r"^- (.+)$", block, flags=re.MULTILINE) if line.strip()]
        if len(bullets) > MAX_ONEPAGER_BULLETS_PER_PICK:
            issues.append(f"pick_{idx}:too_many_bullets:{len(bullets)}")
        for bullet in bullets:
            if len(bullet.encode("utf-8")) > MAX_ONEPAGER_BULLET_BYTES:
                issues.append(f"pick_{idx}:bullet_too_long:{len(bullet.encode('utf-8'))}")
                break
        for label in required_labels:
            if not any(bullet.startswith(label) for bullet in bullets):
                issues.append(f"pick_{idx}:missing_{label.rstrip('｜').lower().replace(' ', '_')}")
    return issues


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

    run_context_path = run_dir / "run_context.json"
    run_context = {}
    if run_context_path.exists():
        try:
            run_context = _load_json(run_context_path)
            report["checks"]["run_context_exists"] = True
            report["details"]["data_mode"] = str(run_context.get("data_mode") or "")
            report["details"]["connector_stats"] = run_context.get("connector_stats")
            report["details"]["extraction_stats"] = run_context.get("extraction_stats")
        except Exception as exc:
            report["checks"]["run_context_exists"] = False
            report["errors"].append(f"run_context_parse:{exc}")
            run_context = {}
    else:
        report["checks"]["run_context_exists"] = False
        report["errors"].append(f"missing:run_context:{run_context_path}")

    data_mode = str(report["details"].get("data_mode") or "").strip().lower()
    report["checks"]["data_mode_present"] = data_mode in {"live", "smoke"}
    if not report["checks"]["data_mode_present"]:
        report["errors"].append("data_mode_missing")

    required = {
        "script": run_dir / "script.json",
        "facts": run_dir / "facts.json",
        "onepager": run_dir / "onepager.md",
        "storyboard": run_dir / "storyboard.json",
        "prompt_bundle": run_dir / "prompt_bundle.json",
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
    facts_path = required["facts"]
    storyboard_path = required["storyboard"]
    prompt_bundle_path = required["prompt_bundle"]
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
        report["checks"]["script_no_html_tokens"] = not contains_html_like_tokens(script_text)
        report["checks"]["script_structure_ok"] = bool(
            str(structure.get("hook") or "").strip()
            and str(structure.get("main_thesis") or "").strip()
            and len([item for item in key_points if str(item).strip()]) >= 3
            and str(structure.get("cta") or "").strip()
        )
        script_urls = _extract_urls(script_text)
        script_bad_urls = [url for url in script_urls if not is_allowed_citation_url(url)]
        report["checks"]["script_citation_denylist_ok"] = len(script_bad_urls) == 0
        if block_hits:
            report["errors"].append("script_blocklist:" + ",".join(block_hits))
        if not report["checks"]["script_no_html_tokens"]:
            report["errors"].append("script_html_tokens_present")
        if script_bad_urls:
            report["errors"].append("script_citation_denylist:" + ",".join(sorted(set(script_bad_urls))[:6]))

    onepager_urls: List[str] = []
    if onepager_path.exists():
        onepager = onepager_path.read_text(encoding="utf-8")
        urls = _extract_urls(onepager)
        onepager_urls = urls
        top_pick_headings = re.findall(r"^###\s+\d+\.\s+", onepager, flags=re.MULTILINE)
        domain_rows = re.findall(r"^-\s*Source Domain:\s*`[^`]+`", onepager, flags=re.MULTILINE)
        relevance_rows = [float(value) for value in re.findall(r"Topic Relevance:\s*`([0-9.]+)`", onepager)]
        block_hits = _contains_blocklist(onepager)
        bad_urls = [url for url in urls if not is_allowed_citation_url(url)]
        report["checks"]["onepager_url_count_ge_3"] = len(urls) >= 3
        report["checks"]["onepager_top_picks_ge_3"] = len(top_pick_headings) >= 3 if data_mode != "live" else len(top_pick_headings) >= 1
        report["checks"]["onepager_domain_rows_ge_3"] = len(domain_rows) >= 3 if data_mode != "live" else len(domain_rows) >= 1
        report["checks"]["onepager_blocklist_ok"] = len(block_hits) == 0
        report["checks"]["onepager_no_html_tokens"] = not contains_html_like_tokens(onepager)
        report["checks"]["onepager_citation_denylist_ok"] = len(bad_urls) == 0
        report["details"]["onepager_url_count"] = len(urls)
        report["details"]["onepager_top_pick_count"] = len(top_pick_headings)
        report["details"]["onepager_relevance_scores"] = relevance_rows
        if block_hits:
            report["errors"].append("onepager_blocklist:" + ",".join(block_hits))
        if bad_urls:
            report["errors"].append("onepager_citation_denylist:" + ",".join(sorted(set(bad_urls))[:6]))
        if not report["checks"]["onepager_no_html_tokens"]:
            report["errors"].append("onepager_html_tokens_present")
        mode_match = re.search(r"^- DataMode:\s*`?(live|smoke)`?\s*$", onepager, flags=re.IGNORECASE | re.MULTILINE)
        if mode_match:
            report["details"]["data_mode"] = str(mode_match.group(1)).lower()
        report["details"]["onepager_compact_issues"] = _compact_bullet_issues(onepager)

    if facts_path.exists():
        facts_payload = _load_json(facts_path)
        has_why = bool(str(facts_payload.get("why_now") or "").strip())
        proof_list = [str(item).strip() for item in list(facts_payload.get("proof") or []) if str(item).strip()]
        report["checks"]["facts_has_why_now_and_proof"] = bool(has_why and proof_list)
        if not report["checks"]["facts_has_why_now_and_proof"]:
            report["errors"].append("facts_missing_why_now_or_proof")

    if storyboard_path.exists():
        storyboard = _load_json(storyboard_path)
        shots = list(storyboard.get("shots") or [])
        report["checks"]["storyboard_shot_count_5_8"] = 5 <= len(shots) <= 8
        report["checks"]["storyboard_overlay_non_empty"] = all(
            str((shot or {}).get("overlay_text") or "").strip() != "" for shot in shots
        )
        overlay_issues = []
        for idx, shot in enumerate(shots, start=1):
            overlay = str((shot or {}).get("overlay_text") or "")
            if len(overlay) > 42:
                overlay_issues.append(f"shot_{idx}:overlay_too_long:{len(overlay)}")
            if overlay.endswith(("..", "...", "-", "--")):
                overlay_issues.append(f"shot_{idx}:overlay_hard_cut")
        report["checks"]["storyboard_overlay_safe"] = len(overlay_issues) == 0
        if overlay_issues:
            report["errors"].append("overlay_issues:" + ",".join(overlay_issues[:6]))
        report["checks"]["storyboard_no_html_tokens"] = not contains_html_like_tokens(json.dumps(storyboard, ensure_ascii=False))
        if not report["checks"]["storyboard_no_html_tokens"]:
            report["errors"].append("storyboard_html_tokens_present")

    if prompt_bundle_path.exists():
        prompt_bundle = _load_json(prompt_bundle_path)
        bundle_text = json.dumps(prompt_bundle, ensure_ascii=False)
        report["checks"]["prompt_bundle_no_html_tokens"] = not contains_html_like_tokens(bundle_text)
        if not report["checks"]["prompt_bundle_no_html_tokens"]:
            report["errors"].append("prompt_bundle_html_tokens_present")

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

    if data_mode == "live":
        texts = []
        for path in [script_path, onepager_path, storyboard_path, run_dir / "materials.json", run_dir / "facts.json"]:
            if path.exists():
                texts.append(path.read_text(encoding="utf-8", errors="ignore"))
        merged = "\n".join(texts)

        smoke_hits = re.findall(r"(?i)[a-z0-9._/-]*-smoke[a-z0-9._/-]*", merged)
        report["checks"]["live_has_no_smoke_tokens"] = len(smoke_hits) == 0
        if smoke_hits:
            report["errors"].append("live_smoke_tokens:" + ",".join(sorted(set(smoke_hits))[:8]))

        fixture_hits = sorted({fixture for fixture in FIXTURE_HN_IDS if fixture in merged})
        report["checks"]["live_has_no_fixture_hn_ids"] = len(fixture_hits) == 0
        if fixture_hits:
            report["errors"].append("live_fixture_hn_ids:" + ",".join(fixture_hits))

        placeholder_repo_hits = []
        for pattern in PLACEHOLDER_REPO_PATTERNS:
            placeholder_repo_hits.extend(pattern.findall(merged))
        report["checks"]["live_has_no_placeholder_org_repo"] = len(placeholder_repo_hits) == 0
        if placeholder_repo_hits:
            report["errors"].append("live_placeholder_repo_url")

        compact_issues = list(report["details"].get("onepager_compact_issues") or [])
        report["checks"]["onepager_bullets_compact_ok"] = len(compact_issues) == 0
        if compact_issues:
            report["errors"].append("onepager_compact_issues:" + ",".join(compact_issues[:6]))

        if facts_path.exists():
            report["checks"]["facts_has_why_now_and_proof"] = bool(report["checks"].get("facts_has_why_now_and_proof"))
        else:
            report["checks"]["facts_has_why_now_and_proof"] = False
            report["errors"].append(f"missing:facts:{facts_path}")
    elif data_mode == "smoke":
        report["checks"]["smoke_mode_detected"] = True
        report["checks"]["onepager_bullets_compact_ok"] = True
        report["checks"]["facts_has_why_now_and_proof"] = True

    topic_value = ""
    ranking_stats = {}
    if run_context_path.exists():
        try:
            run_context = _load_json(run_context_path)
            topic_value = str(run_context.get("topic") or "").strip()
            ranking_stats = dict(run_context.get("ranking_stats") or {})
        except Exception:
            topic_value = ""
            ranking_stats = {}

    if topic_value and data_mode == "live":
        threshold = float(ranking_stats.get("relevance_threshold", 0.55) or 0.55)
        scores = [float(value) for value in list(ranking_stats.get("top_relevance_scores") or []) if value is not None]
        if not scores and onepager_path.exists():
            onepager = onepager_path.read_text(encoding="utf-8")
            scores = [float(value) for value in re.findall(r"Topic Relevance:\\s*`([0-9.]+)`", onepager)]
        report["checks"]["topic_relevance_ok"] = bool(scores and min(scores) >= threshold)
        report["details"]["topic"] = topic_value
        report["details"]["topic_relevance_scores"] = scores
        report["details"]["topic_relevance_threshold"] = threshold
        if not report["checks"]["topic_relevance_ok"]:
            report["errors"].append("topic_relevance_below_threshold")
    else:
        report["checks"]["topic_relevance_ok"] = True

    if data_mode == "live":
        quality_signals = list(ranking_stats.get("top_quality_signals") or [])
        top_signals = quality_signals[:2]
        has_update_signal = False
        for signal in top_signals:
            payload = dict(signal or {})
            if str(payload.get("publish_or_update_time") or "").strip():
                has_update_signal = True
                break
            if payload.get("update_recency_days") not in (None, "", "unknown"):
                has_update_signal = True
                break
            if int(float(payload.get("evidence_links_quality", 0) or 0)) > 0:
                has_update_signal = True
                break
        report["checks"]["ranked_items_have_update_signal"] = has_update_signal
        if not has_update_signal:
            report["errors"].append("ranked_items_missing_update_signal")
    else:
        report["checks"]["ranked_items_have_update_signal"] = True

    all_urls = []
    for path in [script_path, onepager_path, storyboard_path, prompt_bundle_path, run_dir / "materials.json", facts_path]:
        if path.exists():
            all_urls.extend(_extract_urls(path.read_text(encoding="utf-8", errors="ignore")))
    invalid_urls = sorted({url for url in all_urls if not is_valid_http_url(url)})
    report["checks"]["urls_valid"] = len(invalid_urls) == 0
    if invalid_urls:
        report["errors"].append("invalid_urls:" + ",".join(invalid_urls[:8]))

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
