from __future__ import annotations

import base64
from pathlib import Path

import httpx
import pytest

from core import PromptSpec
from render.adapters.seedance import SeedanceAdapter


def _prompt() -> PromptSpec:
    return PromptSpec(shot_idx=1, prompt_text="demo shot", seedance_params={"duration_sec": 3.0})


def test_seedance_disabled_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEEDANCE_ENABLED", raising=False)
    adapter = SeedanceAdapter()
    result = adapter.render_shot(
        prompt_spec=_prompt(),
        output_dir=tmp_path,
        mode="preview",
        budget={},
        run_id="run_1",
        render_job_id="render_1",
    )
    assert result.success is False
    assert "disabled" in str(result.error)


def test_seedance_enabled_with_mock_client_success(tmp_path: Path) -> None:
    target = tmp_path / "shot_001.mp4"
    target.write_bytes(b"\x00\x00\x00\x20ftypmp42\x00\x00\x00\x00mp42isom")

    adapter = SeedanceAdapter(enabled=True, client=lambda **_: {"output_path": str(target), "cost": 0.7})
    result = adapter.render_shot(
        prompt_spec=_prompt(),
        output_dir=tmp_path,
        mode="final",
        budget={},
        run_id="run_2",
        render_job_id="render_2",
    )
    assert result.success is True
    assert result.output_path == str(target)
    assert result.cost == 0.7


def test_seedance_http_client_base64_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _Response:
        def __init__(self, status_code: int, payload: dict):
            self.status_code = status_code
            self._payload = payload
            self.text = "ok"
            self.content = b""

        def json(self) -> dict:
            return dict(self._payload)

    class _Client:
        def __init__(self, *args, **kwargs):
            _ = args, kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            _ = exc_type, exc, tb
            return False

        def post(self, *args, **kwargs):
            _ = args, kwargs
            payload = {
                "video_base64": base64.b64encode(b"\x00\x00\x00\x20ftypmp42\x00\x00\x00\x00mp42isom").decode("utf-8"),
                "cost": 0.22,
            }
            return _Response(200, payload)

        def get(self, *args, **kwargs):
            _ = args, kwargs
            raise AssertionError("unexpected download call")

    monkeypatch.setattr("render.adapters.seedance.httpx.Client", _Client)
    monkeypatch.setenv("SEEDANCE_ENABLED", "1")
    monkeypatch.setenv("SEEDANCE_BASE_URL", "https://seedance.example")
    monkeypatch.setenv("SEEDANCE_API_KEY", "test_key")

    adapter = SeedanceAdapter()
    result = adapter.render_shot(
        prompt_spec=_prompt(),
        output_dir=tmp_path,
        mode="final",
        budget={},
        run_id="run_3",
        render_job_id="render_3",
    )
    assert result.success is True
    assert result.output_path
    assert Path(result.output_path).exists()
    assert result.cost == 0.22


def test_seedance_http_client_auth_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _Response:
        def __init__(self, status_code: int, text: str):
            self.status_code = status_code
            self.text = text
            self.content = b""

        def json(self) -> dict:
            return {}

    class _Client:
        def __init__(self, *args, **kwargs):
            _ = args, kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            _ = exc_type, exc, tb
            return False

        def post(self, *args, **kwargs):
            _ = args, kwargs
            return _Response(401, "unauthorized")

        def get(self, *args, **kwargs):
            _ = args, kwargs
            return _Response(404, "not found")

    monkeypatch.setattr("render.adapters.seedance.httpx.Client", _Client)
    monkeypatch.setenv("SEEDANCE_ENABLED", "1")
    monkeypatch.setenv("SEEDANCE_BASE_URL", "https://seedance.example")
    monkeypatch.setenv("SEEDANCE_API_KEY", "test_key")

    adapter = SeedanceAdapter()
    result = adapter.render_shot(
        prompt_spec=_prompt(),
        output_dir=tmp_path,
        mode="final",
        budget={},
        run_id="run_4",
        render_job_id="render_4",
    )
    assert result.success is False
    assert "auth failed" in str(result.error)


def test_seedance_http_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _Client:
        def __init__(self, *args, **kwargs):
            _ = args, kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            _ = exc_type, exc, tb
            return False

        def post(self, *args, **kwargs):
            _ = args, kwargs
            raise httpx.TimeoutException("timeout")

        def get(self, *args, **kwargs):
            _ = args, kwargs
            raise AssertionError("unexpected")

    monkeypatch.setattr("render.adapters.seedance.httpx.Client", _Client)
    monkeypatch.setenv("SEEDANCE_ENABLED", "1")
    monkeypatch.setenv("SEEDANCE_BASE_URL", "https://seedance.example")
    monkeypatch.setenv("SEEDANCE_API_KEY", "test_key")

    adapter = SeedanceAdapter()
    result = adapter.render_shot(
        prompt_spec=_prompt(),
        output_dir=tmp_path,
        mode="final",
        budget={},
        run_id="run_5",
        render_job_id="render_5",
    )
    assert result.success is False
    assert "timeout" in str(result.error)
