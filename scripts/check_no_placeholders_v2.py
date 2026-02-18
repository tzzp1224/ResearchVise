"""Fail if known placeholder implementation markers remain in production code."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


PATTERNS = {
    "legacy_script_filler": r"evidence-backed detail",
    "legacy_render_placeholder_fallback": r"FALLBACK_MP4_PLACEHOLDER",
    "legacy_render_placeholder_stitch": r"STITCHED_MP4_PLACEHOLDER",
    "legacy_ffmpeg_testsrc": r"testsrc2?=",
    "legacy_smoke_fake_shot": r"smoke-shot-",
}

TARGET_GLOBS = [
    "core/**/*.py",
    "orchestrator/**/*.py",
    "pipeline_v2/**/*.py",
    "render/**/*.py",
    "sources/**/*.py",
    "scripts/**/*.py",
    "webapp/**/*.py",
    "main.py",
]

EXCLUDE = {
    "tests/",
    "README.md",
    "scripts/check_no_placeholders_v2.py",
}


def _should_skip(path: Path, root: Path) -> bool:
    rel = str(path.resolve().relative_to(root.resolve()))
    for marker in EXCLUDE:
        if rel.startswith(marker) or rel == marker:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Check placeholder markers in v2 production code")
    parser.add_argument("--root", default=".", help="Repository root")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    candidates = set()
    for pattern in TARGET_GLOBS:
        candidates.update(path for path in root.glob(pattern) if path.is_file())

    hits = []
    compiled = {name: re.compile(expr, flags=re.IGNORECASE) for name, expr in PATTERNS.items()}

    for path in sorted(candidates):
        if _should_skip(path, root):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name, rule in compiled.items():
            for idx, line in enumerate(text.splitlines(), start=1):
                if rule.search(line):
                    rel = str(path.resolve().relative_to(root.resolve()))
                    hits.append((name, rel, idx, line.strip()))

    if not hits:
        print("PASS: no placeholder markers found")
        return 0

    print("FAIL: placeholder markers detected")
    for name, rel, idx, line in hits:
        print(f"- {name} {rel}:{idx}: {line}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
