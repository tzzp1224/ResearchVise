from __future__ import annotations

from core import Citation, NormalizedItem
from pipeline_v2.dedup_cluster import cluster, dedup_exact, embed, merge_cluster


def _item(
    item_id: str,
    title: str,
    body: str,
    item_hash: str,
    tier: str = "A",
    *,
    source: str = "github",
    url: str | None = None,
    item_type: str = "repo",
    citations: list[Citation] | None = None,
) -> NormalizedItem:
    return NormalizedItem(
        id=item_id,
        source=source,
        title=title,
        url=url or f"https://example.com/{item_id}",
        body_md=body,
        citations=list(citations or []),
        tier=tier,
        lang="en",
        hash=item_hash,
        metadata={"item_type": item_type},
    )


def test_dedup_exact_removes_same_hash() -> None:
    items = [
        _item("a1", "Repo A", "router cache", "hash_a"),
        _item("a2", "Repo A mirror", "router cache", "hash_a"),
        _item("b1", "Repo B", "benchmark chart", "hash_b"),
    ]

    unique = dedup_exact(items)
    assert [item.id for item in unique] == ["a1", "b1"]


def test_cluster_and_merge_cluster() -> None:
    items = [
        _item("a1", "Router cache policy", "latency and routing benchmark", "hash_a1"),
        _item("a2", "Router cache policy v2", "routing benchmark with latency", "hash_a2"),
        _item("b1", "TTS subtitle alignment", "forced alignment for subtitles", "hash_b1"),
    ]

    vectors = embed(items)
    groups = cluster(items, vectors, similarity_threshold=0.70)

    assert len(groups) == 2

    merged = [merge_cluster(group) for group in groups]
    merged_sizes = sorted(item.cluster_size for item in merged)
    assert merged_sizes == [1, 2]


def test_cluster_keeps_different_github_repos_separate_even_with_similar_text() -> None:
    shared_body = (
        "ai agent orchestration workflow with mcp tool calling and benchmark summary "
        "for production control plane deployments "
    ) * 20
    items = [
        _item(
            "gh_a",
            "org-a/agent-runtime",
            shared_body,
            "hash_a",
            url="https://github.com/org-a/agent-runtime",
            source="github",
            item_type="repo",
        ),
        _item(
            "gh_b",
            "org-b/agent-runtime",
            shared_body,
            "hash_b",
            url="https://github.com/org-b/agent-runtime",
            source="github",
            item_type="repo",
        ),
    ]

    vectors = embed(items)
    groups = cluster(items, vectors, similarity_threshold=0.70)
    assert len(groups) == 2


def test_cluster_merges_same_github_repo_url_variants() -> None:
    body = "langgraph orchestration quickstart with tool calling examples " * 12
    items = [
        _item(
            "gh_repo",
            "org/agent-repo",
            body,
            "hash_repo_1",
            url="https://github.com/org/agent-repo",
            source="github",
            item_type="repo",
        ),
        _item(
            "gh_repo_git",
            "org/agent-repo mirror",
            body,
            "hash_repo_2",
            url="https://github.com/org/agent-repo.git",
            source="github",
            item_type="repo",
        ),
    ]

    vectors = embed(items)
    groups = cluster(items, vectors, similarity_threshold=0.95)
    assert len(groups) == 1
    assert len(groups[0]) == 2
