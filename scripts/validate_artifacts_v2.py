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


def _evidence_dedup_issues(onepager: str) -> List[str]:
    issues: List[str] = []
    text = str(onepager or "")
    match = re.search(r"^## Evidence\s*(.*)$", text, flags=re.MULTILINE | re.DOTALL)
    if not match:
        return ["missing_evidence_section"]

    evidence_block = match.group(1)
    sections = re.split(r"^###\s+Evidence for\s+([^\n:]+):[^\n]*$", evidence_block, flags=re.MULTILINE)
    if len(sections) <= 1:
        return ["missing_item_evidence_groups"]

    global_counts: Dict[str, int] = {}
    for idx in range(1, len(sections), 2):
        item_id = str(sections[idx] or "").strip()
        section_text = str(sections[idx + 1] or "")
        urls = _extract_urls(section_text)
        if len(urls) > 5:
            issues.append(f"item_{item_id}:too_many_evidence:{len(urls)}")
        if len(urls) != len(set(urls)):
            issues.append(f"item_{item_id}:duplicate_evidence_urls")
        for url in urls:
            global_counts[url] = int(global_counts.get(url, 0)) + 1
    repeated = sorted([url for url, count in global_counts.items() if count > 2])
    if repeated:
        issues.append("global_repeats:" + ",".join(repeated[:6]))
    return issues


def _facts_truncated_issues(facts_payload: Dict) -> List[str]:
    issues: List[str] = []
    fields = {
        "what_it_is": [str(facts_payload.get("what_it_is") or "")],
        "how_it_works": [str(value) for value in list(facts_payload.get("how_it_works") or [])],
        "proof": [str(value) for value in list(facts_payload.get("proof") or [])],
    }
    for field, values in fields.items():
        for idx, raw in enumerate(values, start=1):
            text = str(raw or "").strip()
            if not text:
                continue
            if re.search(r"[A-Za-z]{4,}[\"',]\S", text):
                issues.append(f"{field}_{idx}:punct_join_no_space")
            tail = re.search(r"([A-Za-z]{1,12})\s*$", text)
            if tail:
                token = str(tail.group(1) or "")
                if (
                    len(token) < 6
                    and len(text) > 30
                    and not text.endswith((".", "!", "?", "。", "！", "？"))
                ):
                    issues.append(f"{field}_{idx}:short_tail_fragment:{token}")
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
    ranking_stats: Dict = {}
    retrieval_ctx: Dict = {}
    if run_context_path.exists():
        try:
            run_context = _load_json(run_context_path)
            report["checks"]["run_context_exists"] = True
            report["details"]["data_mode"] = str(run_context.get("data_mode") or "")
            report["details"]["connector_stats"] = run_context.get("connector_stats")
            report["details"]["extraction_stats"] = run_context.get("extraction_stats")
            ranking_stats = dict(run_context.get("ranking_stats") or {})
            retrieval_ctx = dict(run_context.get("retrieval") or {})
        except Exception as exc:
            report["checks"]["run_context_exists"] = False
            report["errors"].append(f"run_context_parse:{exc}")
            run_context = {}
            ranking_stats = {}
            retrieval_ctx = {}
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
        "diagnosis": run_dir / "retrieval_diagnosis.json",
        "evidence_audit": run_dir / "evidence_audit.json",
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
    diagnosis_path = required["diagnosis"]
    evidence_audit_path = required["evidence_audit"]
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
    onepager_has_why_not_more = False
    report["checks"]["candidate_shortage_explained"] = True
    if onepager_path.exists():
        onepager = onepager_path.read_text(encoding="utf-8")
        urls = _extract_urls(onepager)
        onepager_urls = urls
        top_pick_headings = re.findall(r"^###\s+\d+\.\s+", onepager, flags=re.MULTILINE)
        domain_rows = re.findall(r"^-\s*Source Domain:\s*`[^`]+`", onepager, flags=re.MULTILINE)
        relevance_rows = [float(value) for value in re.findall(r"Topic Relevance:\s*`([0-9.]+)`", onepager)]
        header_top_match = re.search(r"^- TopPicksCount:\s*`?(\d+)`?\s*$", onepager, flags=re.MULTILINE)
        header_requested_match = re.search(r"^- RequestedTopK:\s*`?(\d+)`?\s*$", onepager, flags=re.MULTILINE)
        header_top_count = int(header_top_match.group(1)) if header_top_match else len(top_pick_headings)
        header_requested_top_k = int(header_requested_match.group(1)) if header_requested_match else 0
        expected_top_count = int(ranking_stats.get("top_picks_count", header_top_count) or header_top_count)
        requested_top_k = int(ranking_stats.get("requested_top_k", header_requested_top_k) or header_requested_top_k)
        block_hits = _contains_blocklist(onepager)
        bad_urls = [url for url in urls if not is_allowed_citation_url(url)]
        report["checks"]["onepager_url_count_ge_3"] = len(urls) >= 3
        report["checks"]["onepager_top_picks_ge_3"] = len(top_pick_headings) >= (1 if data_mode == "live" else 3)
        report["checks"]["onepager_domain_rows_ge_3"] = len(domain_rows) >= 3 if data_mode != "live" else len(domain_rows) >= 1
        report["checks"]["onepager_top_count_consistent"] = bool(
            header_top_count == len(top_pick_headings) == expected_top_count
        )
        report["checks"]["onepager_requested_top_k_consistent"] = bool(
            requested_top_k <= 0 or header_requested_top_k == requested_top_k
        )
        report["checks"]["onepager_blocklist_ok"] = len(block_hits) == 0
        report["checks"]["onepager_no_html_tokens"] = not contains_html_like_tokens(onepager)
        report["checks"]["onepager_citation_denylist_ok"] = len(bad_urls) == 0
        report["details"]["onepager_url_count"] = len(urls)
        report["details"]["onepager_top_pick_count"] = len(top_pick_headings)
        report["details"]["onepager_header_top_count"] = header_top_count
        report["details"]["onepager_header_requested_top_k"] = header_requested_top_k
        report["details"]["requested_top_k"] = requested_top_k
        report["details"]["onepager_relevance_scores"] = relevance_rows
        diagnosis_match = re.search(r"^- DiagnosisPath:\s*`?([^`\n]+)`?\s*$", onepager, flags=re.MULTILINE)
        diagnosis_header_path = str(diagnosis_match.group(1) if diagnosis_match else "").strip()
        evidence_audit_match = re.search(r"^- EvidenceAuditPath:\s*`?([^`\n]+)`?\s*$", onepager, flags=re.MULTILINE)
        evidence_audit_header_path = str(evidence_audit_match.group(1) if evidence_audit_match else "").strip()
        hard_match_terms_match = re.search(r"^- HardMatchTermsUsed:\s*`?([^`\n]+)`?\s*$", onepager, flags=re.MULTILINE)
        hard_match_pass_match = re.search(r"^- HardMatchPassCount:\s*`?(\d+)`?\s*$", onepager, flags=re.MULTILINE)
        min_relevance_match = re.search(r"^- TopPicksMinRelevance:\s*`?([0-9.]+)`?\s*$", onepager, flags=re.MULTILINE)
        top_hard_match_match = re.search(r"^- TopPicksHardMatchCount:\s*`?(\d+)`?\s*$", onepager, flags=re.MULTILINE)
        quality_trigger_match = re.search(r"^- QualityTriggeredExpansion:\s*`?(true|false)`?\s*$", onepager, flags=re.IGNORECASE | re.MULTILINE)
        report["details"]["onepager_diagnosis_path"] = diagnosis_header_path
        report["details"]["onepager_evidence_audit_path"] = evidence_audit_header_path
        report["details"]["onepager_hard_match_terms_used"] = str(hard_match_terms_match.group(1) if hard_match_terms_match else "").strip()
        report["details"]["onepager_hard_match_pass_count"] = int(hard_match_pass_match.group(1)) if hard_match_pass_match else None
        report["details"]["onepager_top_picks_min_relevance"] = float(min_relevance_match.group(1)) if min_relevance_match else None
        report["details"]["onepager_top_picks_hard_match_count"] = int(top_hard_match_match.group(1)) if top_hard_match_match else None
        report["details"]["onepager_quality_triggered_expansion"] = (
            str(quality_trigger_match.group(1)).strip().lower() == "true" if quality_trigger_match else None
        )
        onepager_has_why_not_more = bool(re.search(r"^##\s+Why not more\?\s*$", onepager, flags=re.MULTILINE))
        report["details"]["onepager_has_why_not_more"] = onepager_has_why_not_more
        report["checks"]["onepager_diagnosis_path_present"] = bool(diagnosis_header_path and diagnosis_header_path != "N/A")
        report["checks"]["onepager_evidence_audit_path_present"] = bool(
            evidence_audit_header_path and evidence_audit_header_path != "N/A"
        )
        report["checks"]["onepager_relevance_summary_fields_present"] = bool(
            hard_match_terms_match and hard_match_pass_match and min_relevance_match and top_hard_match_match and quality_trigger_match
        )
        if data_mode == "live" and not report["checks"]["onepager_diagnosis_path_present"]:
            report["errors"].append("onepager_missing_diagnosis_path")
        if data_mode == "live" and not report["checks"]["onepager_evidence_audit_path_present"]:
            report["errors"].append("onepager_missing_evidence_audit_path")
        if data_mode == "live" and not report["checks"]["onepager_relevance_summary_fields_present"]:
            report["errors"].append("onepager_missing_relevance_summary_fields")
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
        evidence_issues = _evidence_dedup_issues(onepager)
        report["checks"]["evidence_dedup_ok"] = len(evidence_issues) == 0
        report["details"]["onepager_evidence_issues"] = evidence_issues
        if evidence_issues:
            report["errors"].append("evidence_dedup:" + ",".join(evidence_issues[:6]))
        shortage_detected = bool(data_mode == "live" and requested_top_k > 0 and len(top_pick_headings) < requested_top_k)
        shortage_reasons = list(retrieval_ctx.get("why_not_more") or [])
        shortage_explained = bool((onepager_has_why_not_more or shortage_reasons) and shortage_detected) or (not shortage_detected)
        report["checks"]["candidate_shortage_explained"] = shortage_explained
        report["details"]["candidate_shortage_detected"] = shortage_detected
        report["details"]["candidate_shortage_reasons"] = shortage_reasons
        if shortage_detected and not shortage_explained:
            report["errors"].append(
                f"candidate_shortage_without_explanation:requested_top_k={requested_top_k},actual_top_picks={len(top_pick_headings)}"
            )
        if not report["checks"]["onepager_top_count_consistent"]:
            report["errors"].append(
                f"onepager_top_count_mismatch:header={header_top_count},headings={len(top_pick_headings)},context={expected_top_count}"
            )
        if not report["checks"]["onepager_requested_top_k_consistent"]:
            report["errors"].append(
                f"onepager_requested_top_k_mismatch:header={header_requested_top_k},context={requested_top_k}"
            )

    diagnosis_payload: Dict = {}
    diagnosis_candidate_map: Dict[str, Dict] = {}
    if diagnosis_path.exists():
        try:
            diagnosis_payload = _load_json(diagnosis_path)
            attempts = list(diagnosis_payload.get("attempts") or [])
            diagnosis_candidate_map = {
                str((row or {}).get("item_id") or "").strip(): dict(row or {})
                for row in list(diagnosis_payload.get("candidate_rows") or [])
                if str((row or {}).get("item_id") or "").strip()
            }
            report["checks"]["retrieval_diagnosis_parse_ok"] = True
            report["checks"]["retrieval_diagnosis_attempts_present"] = len(attempts) >= 1
            required_attempt_fields = {
                "hard_match_terms_used",
                "hard_match_pass_count",
                "top_picks_min_relevance",
                "top_picks_hard_match_count",
                "quality_triggered_expansion",
            }
            report["checks"]["retrieval_attempt_quality_fields_present"] = bool(
                attempts and all(required_attempt_fields.issubset(set(dict(item or {}).keys())) for item in attempts)
            )
            report["details"]["retrieval_attempt_count"] = len(attempts)
            report["details"]["retrieval_selected_phase"] = diagnosis_payload.get("selected_phase")
            report["details"]["retrieval_quality_triggered_expansion"] = diagnosis_payload.get("quality_triggered_expansion")
        except Exception as exc:
            report["checks"]["retrieval_diagnosis_parse_ok"] = False
            report["checks"]["retrieval_diagnosis_attempts_present"] = False
            report["checks"]["retrieval_attempt_quality_fields_present"] = False
            report["errors"].append(f"retrieval_diagnosis_parse:{exc}")
            diagnosis_payload = {}
            diagnosis_candidate_map = {}
    else:
        report["checks"]["retrieval_diagnosis_parse_ok"] = False
        report["checks"]["retrieval_diagnosis_attempts_present"] = False
        report["checks"]["retrieval_attempt_quality_fields_present"] = False
        diagnosis_candidate_map = {}

    evidence_audit_payload: Dict = {}
    if evidence_audit_path.exists():
        try:
            evidence_audit_payload = _load_json(evidence_audit_path)
            records = list(evidence_audit_payload.get("records") or [])
            report["checks"]["evidence_audit_parse_ok"] = isinstance(records, list)
            report["details"]["evidence_audit_record_count"] = len(records)
        except Exception as exc:
            report["checks"]["evidence_audit_parse_ok"] = False
            report["errors"].append(f"evidence_audit_parse:{exc}")
            evidence_audit_payload = {}
    else:
        report["checks"]["evidence_audit_parse_ok"] = False

    top_item_ids = [str(value or "").strip() for value in list(ranking_stats.get("top_item_ids") or []) if str(value or "").strip()]
    if evidence_audit_payload:
        record_map = {}
        for row in list(evidence_audit_payload.get("records") or []):
            item_id = str((row or {}).get("item_id") or "").strip()
            if not item_id:
                continue
            record_map[item_id] = dict(row or {})
        verdict_ok = True
        duplicate_ok = True
        link_heavy_alignment_ok = True
        for item_id in top_item_ids:
            row = dict(record_map.get(item_id) or {})
            verdict = str(row.get("verdict") or "").strip().lower()
            reasons = list(row.get("reasons") or [])
            machine_action = dict(row.get("machine_action") or {})
            reason_code = str(machine_action.get("reason_code") or "").strip().lower()
            if verdict not in {"pass", "downgrade"}:
                verdict_ok = False
                continue
            if verdict == "downgrade" and not reasons:
                verdict_ok = False
            duplicate_ratio = float(row.get("citation_duplicate_prefix_ratio", 0.0) or 0.0)
            if duplicate_ratio > 0.6:
                duplicate_ok = False
            has_link_heavy_alignment_flag = bool(
                reason_code == "link_heavy_low_alignment"
                or any("link_heavy_low_alignment" in str(reason).lower() for reason in list(reasons or []))
            )
            if verdict == "pass" and has_link_heavy_alignment_flag:
                link_heavy_alignment_ok = False
        report["checks"]["top_picks_all_pass_or_downgrade_reason_present"] = verdict_ok
        report["checks"]["citations_not_mostly_duplicate"] = duplicate_ok
        report["checks"]["top_picks_not_link_heavy_low_alignment"] = link_heavy_alignment_ok
        if not verdict_ok:
            report["errors"].append("top_picks_invalid_evidence_verdicts")
        if not duplicate_ok:
            report["errors"].append("top_picks_citations_mostly_duplicate")
        if not link_heavy_alignment_ok:
            report["errors"].append("top_picks_link_heavy_low_alignment_passed")
    else:
        report["checks"]["top_picks_all_pass_or_downgrade_reason_present"] = False
        report["checks"]["citations_not_mostly_duplicate"] = False
        report["checks"]["top_picks_not_link_heavy_low_alignment"] = False
        record_map = {}

    top_relevance_scores = [float(value) for value in list(ranking_stats.get("top_relevance_scores") or []) if value is not None]
    relevance_by_item_id: Dict[str, float] = {}
    if top_item_ids and top_relevance_scores and len(top_item_ids) == len(top_relevance_scores):
        relevance_by_item_id = {item_id: top_relevance_scores[idx] for idx, item_id in enumerate(top_item_ids)}

    invariant_violations: List[str] = []
    for item_id in top_item_ids:
        audit_row = dict(record_map.get(item_id) or {})
        diag_row = dict(diagnosis_candidate_map.get(item_id) or {})

        body_len = int(float(audit_row.get("body_len", 0) or 0))
        if body_len <= 0:
            invariant_violations.append(f"{item_id}:body_len_zero")

        relevance_value = diag_row.get("relevance_score")
        if relevance_value in (None, ""):
            relevance_value = relevance_by_item_id.get(item_id)
        try:
            relevance_float = float(relevance_value)
        except Exception:
            relevance_float = -1.0
        if relevance_float <= 0.0:
            invariant_violations.append(f"{item_id}:relevance_zero")

        hard_match_pass = diag_row.get("hard_match_pass")
        if hard_match_pass is False:
            invariant_violations.append(f"{item_id}:hard_gate_fail")

    report["checks"]["top_picks_hard_relevance_invariants"] = len(invariant_violations) == 0
    report["details"]["top_pick_invariant_violations"] = invariant_violations
    if invariant_violations:
        report["errors"].append("top_picks_hard_invariant_failed:" + ",".join(invariant_violations[:8]))

    if facts_path.exists():
        facts_payload = _load_json(facts_path)
        has_why = bool(str(facts_payload.get("why_now") or "").strip())
        proof_list = [str(item).strip() for item in list(facts_payload.get("proof") or []) if str(item).strip()]
        report["checks"]["facts_has_why_now_and_proof"] = bool(has_why and proof_list)
        truncated_issues = _facts_truncated_issues(facts_payload)
        report["checks"]["facts_no_truncated_words"] = len(truncated_issues) == 0
        report["details"]["facts_truncated_issues"] = truncated_issues
        if not report["checks"]["facts_has_why_now_and_proof"]:
            report["errors"].append("facts_missing_why_now_or_proof")
        if truncated_issues:
            report["errors"].append("facts_truncated_words:" + ",".join(truncated_issues[:6]))
    else:
        report["checks"]["facts_no_truncated_words"] = False

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
        report["checks"]["facts_no_truncated_words"] = True

    topic_value = str(run_context.get("topic") or "").strip()

    if topic_value and data_mode == "live":
        threshold = float(ranking_stats.get("relevance_threshold", 0.55) or 0.55)
        threshold = float(ranking_stats.get("topic_relevance_threshold_used", threshold) or threshold)
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

        diagnosis_attempts = list(diagnosis_payload.get("attempts") or [])
        selected_phase_diag = str(diagnosis_payload.get("selected_phase") or "").strip()
        selected_phase_ctx = str(
            retrieval_ctx.get("selected_phase")
            or ranking_stats.get("selected_recall_phase")
            or ""
        ).strip()
        report["checks"]["retrieval_selected_phase_consistent"] = bool(
            not selected_phase_ctx or not selected_phase_diag or selected_phase_ctx == selected_phase_diag
        )
        if not report["checks"]["retrieval_selected_phase_consistent"]:
            report["errors"].append(
                f"retrieval_selected_phase_mismatch:context={selected_phase_ctx},diagnosis={selected_phase_diag}"
            )

        attempt_count_ctx = int(retrieval_ctx.get("attempt_count", 0) or 0)
        attempt_count_diag = len(diagnosis_attempts)
        report["checks"]["retrieval_attempt_count_consistent"] = bool(
            attempt_count_ctx <= 0 or attempt_count_diag <= 0 or attempt_count_ctx == attempt_count_diag
        )
        if not report["checks"]["retrieval_attempt_count_consistent"]:
            report["errors"].append(
                f"retrieval_attempt_count_mismatch:context={attempt_count_ctx},diagnosis={attempt_count_diag}"
            )

        expansion_steps = list(retrieval_ctx.get("expansion_steps") or [])
        expansion_in_diag = any(bool(item.get("expansion_applied")) for item in diagnosis_attempts)
        candidate_shortage = bool(ranking_stats.get("candidate_shortage"))
        requires_expansion_trace = bool(candidate_shortage or attempt_count_ctx > 1 or attempt_count_diag > 1)
        report["checks"]["retrieval_expansion_recorded"] = bool(
            (not requires_expansion_trace) or expansion_steps or expansion_in_diag
        )
        if not report["checks"]["retrieval_expansion_recorded"]:
            report["errors"].append("retrieval_expansion_trace_missing")
        if not report["checks"].get("retrieval_attempt_quality_fields_present", False):
            report["errors"].append("retrieval_attempt_quality_fields_missing")

        selected_pass_count = int(
            diagnosis_payload.get("selected_pass_count", ranking_stats.get("selected_pass_count", 0)) or 0
        )
        requested_top_k = int(ranking_stats.get("requested_top_k", 0) or 0)
        needs_expansion = bool(requested_top_k > 0 and selected_pass_count < requested_top_k)
        attempts_count = len(diagnosis_attempts)
        report["checks"]["retrieval_quality_triggered_expansion_recorded"] = bool((not needs_expansion) or attempts_count > 1)
        if not report["checks"]["retrieval_quality_triggered_expansion_recorded"]:
            report["errors"].append("retrieval_quality_triggered_expansion_missing")

        selected_attempt_idx = int(diagnosis_payload.get("selected_attempt", 0) or 0) - 1
        selected_attempt_payload = (
            dict(diagnosis_attempts[selected_attempt_idx] or {})
            if 0 <= selected_attempt_idx < len(diagnosis_attempts)
            else {}
        )
        selected_has_key_timeout = bool(
            diagnosis_payload.get(
                "selected_attempt_has_key_connector_timeout",
                selected_attempt_payload.get("has_key_connector_timeout", False),
            )
        )
        all_attempts_key_timeout = bool(diagnosis_payload.get("all_attempts_key_connector_timeout", False))
        key_timeout_degraded_result = bool(diagnosis_payload.get("key_connector_timeout_degraded_result", False))
        timeout_selection_ok = bool(
            (not selected_has_key_timeout)
            or (all_attempts_key_timeout and key_timeout_degraded_result)
        )
        report["checks"]["selected_attempt_has_no_key_connector_timeout"] = timeout_selection_ok
        if not timeout_selection_ok:
            report["errors"].append("selected_attempt_has_key_connector_timeout")

        relevance_scores = [float(value) for value in list(ranking_stats.get("top_relevance_scores") or []) if value is not None]
        if len(relevance_scores) < 3:
            relevance_scores = [float(value) for value in list(report.get("details", {}).get("onepager_relevance_scores") or []) if value is not None]
        top3_scores = relevance_scores[:3]
        relevance_not_all_ones = bool(len(top3_scores) < 3 or any(abs(float(score) - 1.0) > 1e-6 for score in top3_scores))
        report["checks"]["relevance_not_all_1.0"] = relevance_not_all_ones
        if not relevance_not_all_ones:
            report["errors"].append("relevance_scores_all_1.0_top3")

        record_map = {
            str((row or {}).get("item_id") or "").strip(): dict(row or {})
            for row in list(evidence_audit_payload.get("records") or [])
            if str((row or {}).get("item_id") or "").strip()
        }
        top_verdicts = [
            str((record_map.get(item_id) or {}).get("verdict") or "").strip().lower()
            for item_id in top_item_ids
        ]
        all_top_downgrade = bool(top_verdicts) and all(verdict == "downgrade" for verdict in top_verdicts)
        allow_all_downgrade = bool(onepager_has_why_not_more and attempts_count >= 2)
        report["checks"]["top_picks_not_all_downgrade"] = bool((not all_top_downgrade) or allow_all_downgrade)
        if not report["checks"]["top_picks_not_all_downgrade"]:
            report["errors"].append("top_picks_all_downgrade_without_explanation")
    else:
        report["checks"]["ranked_items_have_update_signal"] = True
        report["checks"]["retrieval_selected_phase_consistent"] = True
        report["checks"]["retrieval_attempt_count_consistent"] = True
        report["checks"]["retrieval_expansion_recorded"] = True
        report["checks"]["retrieval_quality_triggered_expansion_recorded"] = True
        report["checks"]["selected_attempt_has_no_key_connector_timeout"] = True
        report["checks"]["relevance_not_all_1.0"] = True
        report["checks"]["top_picks_not_all_downgrade"] = True

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
