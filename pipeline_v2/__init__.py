"""v2 pipeline stages: normalize, dedup/cluster, scoring."""

from .dedup_cluster import cluster, dedup_exact, embed, merge_cluster
from .normalize import content_hash, extract_citations, normalize
from .scoring import (
    rank_items,
    score_credibility,
    score_novelty,
    score_talkability,
    score_visual_assets,
)

__all__ = [
    "cluster",
    "content_hash",
    "dedup_exact",
    "embed",
    "extract_citations",
    "merge_cluster",
    "normalize",
    "rank_items",
    "score_credibility",
    "score_novelty",
    "score_talkability",
    "score_visual_assets",
]
