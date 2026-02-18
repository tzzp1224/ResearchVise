"""v2 pipeline stages: normalize, dedup/cluster, scoring."""

from .dedup_cluster import cluster, dedup_exact, embed, merge_cluster
from .normalize import content_hash, extract_citations, normalize
from .prompt_compiler import compile_shot_prompt, compile_storyboard, consistency_pack
from .scoring import (
    rank_items,
    score_credibility,
    score_novelty,
    score_talkability,
    score_visual_assets,
)
from .script_generator import generate_script, generate_variants
from .storyboard_generator import auto_fix_storyboard, script_to_storyboard, validate_storyboard

__all__ = [
    "auto_fix_storyboard",
    "cluster",
    "compile_shot_prompt",
    "compile_storyboard",
    "content_hash",
    "consistency_pack",
    "dedup_exact",
    "embed",
    "extract_citations",
    "generate_script",
    "generate_variants",
    "merge_cluster",
    "normalize",
    "rank_items",
    "score_credibility",
    "score_novelty",
    "score_talkability",
    "score_visual_assets",
    "script_to_storyboard",
    "validate_storyboard",
]
