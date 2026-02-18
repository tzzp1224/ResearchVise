"""Data-mode helpers for live vs smoke execution paths."""

from __future__ import annotations

import os
from typing import Any, Dict, Literal


DataMode = Literal["live", "smoke"]


def resolve_data_mode(*, budget: Dict[str, Any] | None = None, explicit: str | None = None) -> DataMode:
    """Resolve effective data mode with precedence: explicit > budget > env > live."""
    if str(explicit or "").strip():
        value = str(explicit).strip().lower()
    elif isinstance(budget, dict) and str(budget.get("data_mode") or "").strip():
        value = str(budget.get("data_mode") or "").strip().lower()
    else:
        value = str(os.getenv("ARA_DATA_MODE", "live")).strip().lower() or "live"

    if value not in {"live", "smoke"}:
        return "live"
    return value  # type: ignore[return-value]


def should_allow_smoke(*, budget: Dict[str, Any] | None = None, explicit: str | None = None) -> bool:
    """Smoke mode is only allowed when explicitly requested via CLI/API budget."""
    if str(explicit or "").strip().lower() == "smoke":
        return True
    if isinstance(budget, dict) and str(budget.get("data_mode") or "").strip().lower() == "smoke":
        return True
    return False
