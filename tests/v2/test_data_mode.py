from __future__ import annotations

from pipeline_v2.data_mode import resolve_data_mode, should_allow_smoke


def test_resolve_data_mode_defaults_to_live(monkeypatch) -> None:
    monkeypatch.delenv("ARA_DATA_MODE", raising=False)
    mode = resolve_data_mode(budget={})
    assert mode == "live"


def test_resolve_data_mode_prefers_explicit_budget() -> None:
    assert resolve_data_mode(budget={"data_mode": "smoke"}) == "smoke"
    assert resolve_data_mode(budget={"data_mode": "live"}) == "live"


def test_should_allow_smoke_requires_explicit_request() -> None:
    assert should_allow_smoke(budget={"data_mode": "smoke"}) is True
    assert should_allow_smoke(explicit="smoke") is True
    assert should_allow_smoke(budget={}) is False
