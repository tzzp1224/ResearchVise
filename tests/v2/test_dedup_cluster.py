from __future__ import annotations

from core import NormalizedItem
from pipeline_v2.dedup_cluster import cluster, dedup_exact, embed, merge_cluster


def _item(item_id: str, title: str, body: str, item_hash: str, tier: str = "A") -> NormalizedItem:
    return NormalizedItem(
        id=item_id,
        source="github",
        title=title,
        url=f"https://example.com/{item_id}",
        body_md=body,
        citations=[],
        tier=tier,
        lang="en",
        hash=item_hash,
        metadata={},
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
