from __future__ import annotations

import json
from pathlib import Path
import zipfile

from core import Artifact, ArtifactType, Citation, NormalizedItem
from pipeline_v2.notification import notify_user, post_to_web, send_email
from pipeline_v2.report_export import export_package, generate_onepager, generate_thumbnail


def _item() -> NormalizedItem:
    return NormalizedItem(
        id="n_1",
        source="github",
        title="Model Context Protocol",
        url="https://example.com/mcp",
        author="dexter",
        published_at=None,
        body_md="MCP architecture benchmark and deployment plan.",
        citations=[
            Citation(
                title="MCP Spec",
                url="https://spec.example.com",
                snippet="Spec excerpt",
                source="github",
            )
        ],
        tier="A",
        lang="en",
        hash="abc123",
        metadata={
            "credibility": "high",
            "quality_signals": {
                "content_density": 0.42,
                "has_quickstart": True,
                "has_results_or_bench": True,
                "has_images_non_badge": True,
                "publish_or_update_time": "2026-02-17T12:00:00+00:00",
                "update_recency_days": 1.2,
                "evidence_links_quality": 2,
            },
            "facts": {
                "what_it_is": "MCP 路由层升级，主打多代理工具编排。",
                "why_now": "最近 2 天更新，社区讨论集中在生产可用性。",
                "how_it_works": ["把上下文路由拆成可观测阶段。", "提供可回滚发布策略。"],
                "proof": ["HN 讨论持续升温。"],
            },
        },
    )


def test_generate_onepager_thumbnail_and_export_package(tmp_path: Path) -> None:
    item = _item()

    onepager = generate_onepager([item], item.citations, out_dir=tmp_path)
    assert onepager.endswith("onepager.md")
    assert Path(onepager).exists()
    content = Path(onepager).read_text(encoding="utf-8")
    assert "Top Picks" in content
    assert "RequestedTopK" in content
    assert "EvidenceAuditPath" in content
    assert "Source Domain" in content
    assert "#### Compact Brief" in content
    assert "#### Citations" in content
    assert "Quality Metrics" in content
    assert "WHAT｜" in content
    assert "WHY NOW｜" in content
    assert "HOW｜" in content
    assert "PROOF｜" in content
    assert "### Evidence for n_1:" in content

    compact_bullets = [line[2:] for line in content.splitlines() if line.startswith("- WHAT｜") or line.startswith("- WHY NOW｜") or line.startswith("- HOW｜") or line.startswith("- PROOF｜") or line.startswith("- CTA｜")]
    assert len(compact_bullets) <= 6
    assert all(len(line.encode("utf-8")) <= 90 for line in compact_bullets)

    thumbnail = generate_thumbnail(item.title, ["mcp", "agent"], {"fg": "#000", "bg": "#fff"}, out_dir=tmp_path)
    assert thumbnail.endswith(".svg")
    assert Path(thumbnail).exists()

    artifacts = [
        Artifact(type=ArtifactType.ONEPAGER, path=onepager, metadata={}),
        Artifact(type=ArtifactType.THUMBNAIL, path=thumbnail, metadata={}),
    ]
    package_path = export_package(tmp_path, artifacts, package_name="demo_package")
    assert package_path.endswith(".zip")
    assert Path(package_path).exists()

    with zipfile.ZipFile(package_path) as zf:
        members = set(zf.namelist())
        assert "manifest.json" in members
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        assert manifest["artifact_count"] == 2


def test_notifications_write_jsonl(tmp_path: Path) -> None:
    out_dir = tmp_path / "notify"
    event1 = notify_user("u_1", "run done", out_dir=out_dir)
    event2 = post_to_web("run_1", {"status": "ok"}, out_dir=out_dir)
    event3 = send_email("a@b.com", "subject", "body", out_dir=out_dir)

    assert event1["status"] == "ok"
    assert event2["channel"] == "post_to_web"
    assert event3["channel"] == "send_email"

    log_path = out_dir / "notifications.jsonl"
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3


def test_onepager_evidence_section_prefers_audit_urls(tmp_path: Path) -> None:
    item = _item()
    item.id = "audit_1"
    item.citations = [
        Citation(
            title="fallback",
            url="https://fallback.example.com/weak",
            snippet="fallback evidence",
            source="web",
        )
    ]
    run_context = {
        "ranking_stats": {
            "requested_top_k": 1,
            "top_evidence_urls": {
                "audit_1": [
                    "https://github.com/acme/agent-runtime/releases/tag/v1.0.0",
                    "https://docs.acme.dev/agent-runtime",
                ]
            },
            "top_evidence_audit_verdicts": {"audit_1": "pass"},
            "top_evidence_audit_reasons": {"audit_1": []},
        }
    }
    onepager = generate_onepager([item], item.citations, out_dir=tmp_path, run_context=run_context)
    text = Path(onepager).read_text(encoding="utf-8")
    assert "https://github.com/acme/agent-runtime/releases/tag/v1.0.0" in text
    assert "https://docs.acme.dev/agent-runtime" in text
