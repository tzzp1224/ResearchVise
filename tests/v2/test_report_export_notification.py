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
        metadata={"credibility": "high"},
    )


def test_generate_onepager_thumbnail_and_export_package(tmp_path: Path) -> None:
    item = _item()

    onepager = generate_onepager([item], item.citations, out_dir=tmp_path)
    assert onepager.endswith("onepager.md")
    assert Path(onepager).exists()
    assert "Top Picks" in Path(onepager).read_text(encoding="utf-8")

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
