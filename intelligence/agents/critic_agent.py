"""
Critic Agent
质量门控：基于可解释指标评估研究输出质量。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from intelligence.pipeline_helpers import evaluate_research_quality


class CriticAgent:
    """Quality gate agent for research outputs."""

    def __init__(self, *, quality_threshold: float = 0.65):
        self.quality_threshold = float(max(0.0, min(1.0, quality_threshold)))

    async def evaluate(
        self,
        *,
        facts: List[Dict[str, Any]],
        search_results: List[Dict[str, Any]],
        one_pager: Optional[Dict[str, Any]],
        video_brief: Optional[Dict[str, Any]],
        knowledge_gaps: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        quality = evaluate_research_quality(
            facts=facts,
            search_results=search_results,
            one_pager=one_pager,
            video_brief=video_brief,
            knowledge_gaps=knowledge_gaps,
            threshold=self.quality_threshold,
        )
        return {
            "pass": bool(quality.get("pass", False)),
            "quality_metrics": quality,
            "recommendations": list(quality.get("recommendations", [])),
        }
