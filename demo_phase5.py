#!/usr/bin/env python
"""Phase 5 Web UI launcher (FastAPI + SSE)."""

from __future__ import annotations

import argparse

import uvicorn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch Phase 5 web UI")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8765, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    uvicorn.run("webapp.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
