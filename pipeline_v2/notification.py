"""Notification hooks (user/web/email) with local side effects for testing."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _event(
    channel: str,
    payload: Dict[str, Any],
    *,
    out_dir: str | Path | None = None,
) -> Dict[str, Any]:
    entry = {
        "channel": channel,
        "payload": dict(payload or {}),
        "sent_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "ok",
    }
    if out_dir:
        target = Path(out_dir)
        target.mkdir(parents=True, exist_ok=True)
        log_path = target / "notifications.jsonl"
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        entry["log_path"] = str(log_path)
    return entry


def notify_user(user_id: str, message: str, *, out_dir: str | Path | None = None) -> Dict[str, Any]:
    return _event("notify_user", {"user_id": str(user_id), "message": str(message)}, out_dir=out_dir)


def post_to_web(run_id: str, payload: Dict[str, Any], *, out_dir: str | Path | None = None) -> Dict[str, Any]:
    event_payload = {"run_id": str(run_id), "payload": dict(payload or {})}
    return _event("post_to_web", event_payload, out_dir=out_dir)


def send_email(
    to_email: str,
    subject: str,
    body: str,
    *,
    out_dir: str | Path | None = None,
) -> Dict[str, Any]:
    payload = {"to": str(to_email), "subject": str(subject), "body": str(body)}
    return _event("send_email", payload, out_dir=out_dir)
