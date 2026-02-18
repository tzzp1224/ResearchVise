"""Tests for Phase 5 FastAPI + SSE web interface."""

from __future__ import annotations

import importlib
from pathlib import Path
import shutil
import time

from fastapi.testclient import TestClient

webapp_module = importlib.import_module("webapp.app")
input_ingest_module = importlib.import_module("webapp.input_ingest")


def _wait_for_job(client: TestClient, run_id: str, timeout_sec: float = 2.0) -> dict:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        status = client.get(f"/api/research/{run_id}").json()
        if status.get("status") != "running":
            return status
        time.sleep(0.02)
    return client.get(f"/api/research/{run_id}").json()


def test_research_job_streams_documents(monkeypatch):
    out_dir = webapp_module.ROOT_DIR / "data" / "outputs" / "test_webapp_api"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    one_pager_path = out_dir / "one_pager.md"
    one_pager_path.write_text("# One Pager\n\nstreamed content\n", encoding="utf-8")
    timeline_path = out_dir / "timeline.md"
    timeline_path.write_text("# Timeline\n\n- event\n", encoding="utf-8")

    written_files = {
        "one_pager_md": str(one_pager_path),
        "timeline_md": str(timeline_path),
    }

    async def _fake_run_research_end_to_end(**kwargs):
        assert kwargs["allow_cache_hit"] is False
        progress_callback = kwargs.get("progress_callback")
        if progress_callback:
            await progress_callback(
                {
                    "event": "search_tool_call",
                    "iteration": 1,
                    "tool": "github_search",
                    "query": "MCP deployment architecture",
                }
            )
            await progress_callback(
                {
                    "event": "search_tool_result",
                    "iteration": 1,
                    "tool": "github_search",
                    "query": "MCP deployment architecture",
                    "result_count": 2,
                    "source_breakdown": {"github": 2},
                }
            )
            await progress_callback(
                {"event": "documents_exported", "written_files": written_files}
            )
            await progress_callback(
                {"event": "research_finished", "output_dir": str(out_dir)}
            )
        return {
            "topic": kwargs["topic"],
            "input_topic": kwargs["topic"],
            "search_results_count": 2,
            "facts": [{"id": "f1"}, {"id": "f2"}],
            "cache_hit": False,
            "quality_metrics": {"overall_score": 0.8},
            "quality_gate_pass": True,
            "quality_recommendations": ["add more benchmarks"],
            "knowledge_gaps": [],
            "written_files": written_files,
            "video_artifact": None,
            "video_error": None,
            "output_dir": str(out_dir),
            "search_strategy": "react_agent",
        }

    monkeypatch.setattr(
        webapp_module, "run_research_end_to_end", _fake_run_research_end_to_end
    )
    webapp_module.JOBS.clear()

    client = TestClient(webapp_module.app)

    create_resp = client.post(
        "/api/research",
        json={"topic": "MCP deployment", "generate_video": False},
    )
    assert create_resp.status_code == 200
    run_id = create_resp.json()["run_id"]

    status = _wait_for_job(client, run_id)
    assert status["status"] == "completed"
    assert status["result"]["facts_count"] == 2

    stream_resp = client.get(f"/api/research/{run_id}/events")
    assert stream_resp.status_code == 200
    body = stream_resp.text
    assert "event: search_tool_call" in body
    assert "event: search_tool_result" in body
    assert "event: document_chunk" in body
    assert "event: result" in body
    assert "event: stream_end" in body
    shutil.rmtree(out_dir, ignore_errors=True)


def test_chat_endpoint_deduplicates_citations(monkeypatch):
    async def _fake_chat_over_kb(**kwargs):
        assert kwargs["question"] == "风险有哪些？"
        return {
            "answer": "风险主要在安全和稳定性。",
            "citations": [
                {"id": "1", "source": "github", "url": "https://example.com/a"},
                {"id": "1", "source": "github", "url": "https://example.com/a"},
            ],
        }

    monkeypatch.setattr(webapp_module, "chat_over_kb", _fake_chat_over_kb)

    client = TestClient(webapp_module.app)
    response = client.post("/api/chat", json={"question": "风险有哪些？"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"]
    assert len(payload["citations"]) == 1


def test_input_modes_are_mutually_exclusive():
    client = TestClient(webapp_module.app)
    response = client.post(
        "/api/research",
        json={"topic": "MCP", "arxiv_url": "https://arxiv.org/abs/2401.00001", "generate_video": False},
    )
    assert response.status_code == 422


def test_pdf_upload_endpoint_starts_job(monkeypatch):
    out_dir = webapp_module.ROOT_DIR / "data" / "outputs" / "test_webapp_upload"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    async def _fake_ingest_uploaded_pdf(**kwargs):
        return input_ingest_module.InputSeedBundle(
            topic="Uploaded Paper Topic",
            search_results=[
                {
                    "id": "user_pdf_seed",
                    "source": "user_document",
                    "title": "Uploaded Paper",
                    "content": "seed content",
                    "url": "",
                    "metadata": {"input_mode": "pdf_upload"},
                }
            ],
            notes=["ok"],
        )

    async def _fake_run_research_end_to_end(**kwargs):
        assert kwargs["allow_cache_hit"] is False
        return {
            "topic": kwargs["topic"],
            "input_topic": kwargs["topic"],
            "search_results": kwargs.get("seed_search_results") or [],
            "search_results_count": 1,
            "facts": [{"id": "f1"}],
            "cache_hit": False,
            "quality_metrics": {"overall_score": 0.8},
            "quality_gate_pass": True,
            "quality_recommendations": [],
            "knowledge_gaps": [],
            "written_files": {},
            "video_artifact": None,
            "video_error": None,
            "output_dir": str(out_dir),
            "search_strategy": "react_agent+seed_input",
        }

    monkeypatch.setattr(webapp_module, "ingest_uploaded_pdf", _fake_ingest_uploaded_pdf)
    monkeypatch.setattr(webapp_module, "run_research_end_to_end", _fake_run_research_end_to_end)

    client = TestClient(webapp_module.app)
    files = {"file": ("paper.pdf", b"%PDF-1.4\nfake", "application/pdf")}
    data = {"config_json": "{\"generate_video\": false}"}
    response = client.post("/api/research/upload", data=data, files=files)
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "running"
    shutil.rmtree(out_dir, ignore_errors=True)


def test_cache_suggestions_filters_low_quality_temp_and_low_fact_candidates(monkeypatch):
    class _FakeCacheCfg:
        similarity_threshold = 0.82
        collection_name = "research_artifacts"
        min_quality_score = 0.35
        require_quality_gate_pass = False
        min_facts_count = 4

    class _FakeStore:
        def __init__(self, collection_name: str):
            self.collection_name = collection_name
            self.closed = False

        def find_similar(self, *, query: str, score_threshold: float, top_k: int):
            assert query == "kimi 2.5"
            assert score_threshold <= 0.40
            assert top_k >= 4
            return [
                {
                    "artifact_id": "good",
                    "topic": "Kimi Chat 2.5 模型技术分析",
                    "score": 0.61,
                    "quality_score": 0.82,
                    "quality_gate_pass": True,
                    "snapshot_path": "data/outputs/20260209_120600_Kimi_Chat_2.5_模型技术分析/research_result_snapshot.json",
                    "created_at": "2026-02-09T12:06:00",
                    "search_results_count": 20,
                },
                {
                    "artifact_id": "low_quality",
                    "topic": "Kimi Chat 2.5 low quality",
                    "score": 0.59,
                    "quality_score": 0.12,
                    "quality_gate_pass": False,
                    "snapshot_path": "data/outputs/20260209_110050_Kimi_Chat_2.5_模型技术分析/research_result_snapshot.json",
                    "created_at": "2026-02-09T11:00:50",
                    "search_results_count": 8,
                },
                {
                    "artifact_id": "tmp_debug",
                    "topic": "Temporary Debug Run",
                    "score": 0.57,
                    "quality_score": 0.91,
                    "quality_gate_pass": True,
                    "snapshot_path": "data/outputs/tmp_debug_kimi/research_result_snapshot.json",
                    "created_at": "2026-02-09T09:00:00",
                    "search_results_count": 10,
                },
                {
                    "artifact_id": "low_facts",
                    "topic": "Kimi Chat 2.5 low facts",
                    "score": 0.55,
                    "quality_score": 0.83,
                    "quality_gate_pass": True,
                    "snapshot_path": "data/outputs/20260209_110050_Kimi_Chat_2.5_模型技术分析/research_result_snapshot_low_facts.json",
                    "created_at": "2026-02-09T11:00:50",
                    "search_results_count": 6,
                },
            ]

        def close(self):
            self.closed = True

    def _fake_resolve(path_text: str) -> Path:
        return Path(path_text)

    def _fake_read_snapshot(path: Path) -> dict:
        text = str(path)
        if "low_facts" in text:
            return {
                "topic": "Kimi Chat 2.5 low facts",
                "facts": [{"claim": "a"}, {"claim": "b"}],
                "search_results": [],
                "output_dir": "data/outputs/20260209_110050_Kimi_Chat_2.5_模型技术分析",
            }
        return {
            "topic": "Kimi Chat 2.5 模型技术分析",
            "facts": [{"claim": "a"}, {"claim": "b"}, {"claim": "c"}, {"claim": "d"}],
            "search_results": [{"id": "x"}],
            "output_dir": "data/outputs/20260209_120600_Kimi_Chat_2.5_模型技术分析",
            "one_pager": {"executive_summary": "Kimi 2.5 summary"},
            "video_artifact": {"output_path": "data/outputs/20260209_120600_Kimi_Chat_2.5_模型技术分析/video_brief.mp4"},
        }

    monkeypatch.setattr(webapp_module, "get_research_cache_settings", lambda: _FakeCacheCfg())
    monkeypatch.setattr(webapp_module, "ResearchArtifactStore", _FakeStore)
    monkeypatch.setattr(webapp_module, "_resolve_artifact_path", _fake_resolve)
    monkeypatch.setattr(webapp_module, "_read_snapshot_file", _fake_read_snapshot)
    monkeypatch.setattr(webapp_module, "_video_artifact_exists", lambda snapshot: bool(snapshot.get("video_artifact")))

    client = TestClient(webapp_module.app)
    response = client.get("/api/cache/suggestions", params={"topic": "kimi 2.5", "top_k": 4})
    assert response.status_code == 200
    candidates = response.json()["candidates"]
    assert len(candidates) == 1
    assert candidates[0]["artifact_id"] == "good"
    assert candidates[0]["facts_count"] == 4
