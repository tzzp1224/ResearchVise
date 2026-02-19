"""v2 pipeline stages: normalize, dedup/cluster, scoring."""

from .dedup_cluster import cluster, dedup_exact, embed, merge_cluster
from .evidence_auditor import EvidenceAuditor, LLMEvidenceAuditor
from .normalize import content_hash, extract_citations, normalize
from .planner import LLMPlanner, ResearchPlanner, RetrievalPlan, build_retrieval_plan
from .prompt_compiler import compile_shot_prompt, compile_storyboard, consistency_pack
from .report_export import export_package, generate_onepager, generate_thumbnail
from .runtime import RunExecutionResult, RunPipelineRuntime
from .scoring import (
    evaluate_relevance,
    rank_items,
    score_credibility,
    score_novelty,
    score_relevance,
    score_talkability,
    score_visual_assets,
)
from .script_generator import build_facts, generate_script, generate_variants
from .storyboard_generator import auto_fix_storyboard, overlay_compact, script_to_storyboard, validate_storyboard
from .notification import notify_user, post_to_web, send_email
from .topic_profile import TopicProfile

__all__ = [
    "auto_fix_storyboard",
    "cluster",
    "compile_shot_prompt",
    "compile_storyboard",
    "content_hash",
    "consistency_pack",
    "dedup_exact",
    "EvidenceAuditor",
    "LLMEvidenceAuditor",
    "embed",
    "evaluate_relevance",
    "export_package",
    "extract_citations",
    "build_facts",
    "generate_onepager",
    "generate_script",
    "generate_thumbnail",
    "generate_variants",
    "merge_cluster",
    "normalize",
    "notify_user",
    "post_to_web",
    "rank_items",
    "ResearchPlanner",
    "LLMPlanner",
    "RetrievalPlan",
    "build_retrieval_plan",
    "RunExecutionResult",
    "RunPipelineRuntime",
    "send_email",
    "score_credibility",
    "score_novelty",
    "score_relevance",
    "score_talkability",
    "score_visual_assets",
    "TopicProfile",
    "overlay_compact",
    "script_to_storyboard",
    "validate_storyboard",
]
