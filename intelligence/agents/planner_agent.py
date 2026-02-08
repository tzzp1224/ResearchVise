"""
Planner Agent
研究规划层：过滤非技术请求、重写查询、拆解研究问题。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
import logging
import re

from intelligence.llm import BaseLLM, get_llm


logger = logging.getLogger(__name__)

_TECH_KEYWORDS = {
    "model",
    "llm",
    "agent",
    "ai",
    "machine learning",
    "deep learning",
    "algorithm",
    "architecture",
    "benchmark",
    "latency",
    "throughput",
    "retrieval",
    "rag",
    "mcp",
    "api",
    "deployment",
    "infra",
    "framework",
    "向量",
    "检索",
    "推理",
    "模型",
    "算法",
    "架构",
    "延迟",
    "吞吐",
    "部署",
    "性能",
    "基准",
}

_NON_TECH_KEYWORDS = {
    "菜谱",
    "吃什么",
    "星座",
    "运势",
    "恋爱",
    "八卦",
    "穿搭",
    "旅游攻略",
    "减肥",
    "天气",
    "双色球",
    "电影推荐",
    "recipe",
    "horoscope",
    "dating",
    "celebrity",
    "fashion",
    "travel tips",
    "lottery",
    "movie recommendation",
}

_DOMAIN_EXPANSIONS = {
    "mcp": "Model Context Protocol",
    "rag": "Retrieval Augmented Generation",
    "moe": "Mixture of Experts",
    "llm": "Large Language Model",
}

_PLAN_DIMS = [
    ("architecture", "architecture design internals"),
    ("performance", "benchmark latency throughput cost"),
    ("comparison", "comparison versus alternatives trade-offs"),
    ("limitation", "failure modes limitations risks"),
    ("deployment", "production deployment observability reliability"),
]

_PLANNER_SYSTEM_PROMPT = """你是研究规划 Agent。
你的目标：判断请求是否属于技术研究任务，并生成可执行的检索计划。

输出 JSON:
{
  "is_technical": true,
  "reason": "简短理由",
  "normalized_topic": "归一化主题",
  "query_rewrites": ["关键词1", "关键词2"],
  "research_questions": ["问题1", "问题2"],
  "search_plan": [
    {"dimension": "architecture", "query": "xxx"},
    {"dimension": "performance", "query": "xxx"}
  ]
}

要求:
1) 如果是非技术请求，is_technical=false，并给 reason。
2) query_rewrites 需提升命中率，保留领域术语、缩写、主流同义词。
3) research_questions 必须覆盖机制、指标、对比、局限、工程落地。"""


class PlannerAgent:
    """Research planning agent."""

    def __init__(self, llm: Optional[BaseLLM] = None):
        self.llm = llm or get_llm()

    def _heuristic_is_technical(self, text: str) -> bool:
        lowered = text.lower()
        if any(k in lowered for k in _NON_TECH_KEYWORDS) and not any(
            k in lowered for k in _TECH_KEYWORDS
        ):
            return False
        if any(k in lowered for k in _TECH_KEYWORDS):
            return True
        return bool(re.search(r"\b[A-Z]{2,8}\b", text))

    def _heuristic_rewrites(self, topic: str) -> List[str]:
        base = re.sub(r"\s+", " ", str(topic or "").strip())
        if not base:
            return []

        rewrites: List[str] = [base]
        lowered = base.lower()

        tokens = re.findall(r"[A-Za-z0-9#+-]+", base)
        filtered = [
            token
            for token in tokens
            if token.lower()
            not in {
                "production",
                "deployment",
                "design",
                "analysis",
                "study",
                "for",
                "with",
                "and",
                "the",
                "in",
                "of",
            }
        ]
        if filtered:
            rewrites.append(" ".join(filtered[:4]))

        for token in tokens:
            lower_token = token.lower()
            if lower_token in _DOMAIN_EXPANSIONS:
                rewrites.append(_DOMAIN_EXPANSIONS[lower_token])
            if token.isupper() and 2 <= len(token) <= 8:
                rewrites.append(token)

        deduped: List[str] = []
        seen = set()
        for item in rewrites:
            key = item.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(item.strip())
        return deduped[:6]

    def _heuristic_questions(self, topic: str) -> List[str]:
        return [
            f"{topic} 的核心机制与系统架构是什么？",
            f"{topic} 的关键性能指标（延迟/吞吐/准确率/成本）如何？",
            f"{topic} 与主流替代方案相比，收益与代价分别是什么？",
            f"{topic} 在真实生产环境中的失效模式和风险点有哪些？",
            f"{topic} 的工程落地路径、监控指标与回滚策略是什么？",
        ]

    def _heuristic_plan(self, topic: str) -> List[Dict[str, str]]:
        root = self._heuristic_rewrites(topic)
        seed = root[0] if root else topic
        return [
            {"dimension": dim, "query": f"{seed} {suffix}".strip()} for dim, suffix in _PLAN_DIMS
        ]

    def _extract_json(self, text: str) -> Dict[str, Any]:
        raw = (text or "").strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group())
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    async def plan(self, *, topic: str, user_query: Optional[str] = None) -> Dict[str, Any]:
        text = f"{topic}\n{user_query or ''}".strip()
        heuristic_is_technical = self._heuristic_is_technical(text)
        heuristic_rewrites = self._heuristic_rewrites(topic)
        heuristic_questions = self._heuristic_questions(topic)
        heuristic_plan = self._heuristic_plan(topic)

        fallback = {
            "is_technical": heuristic_is_technical,
            "reason": "heuristic",
            "normalized_topic": topic.strip(),
            "query_rewrites": heuristic_rewrites,
            "research_questions": heuristic_questions,
            "search_plan": heuristic_plan,
        }

        try:
            prompt = (
                f"用户主题: {topic}\n"
                f"用户补充问题: {user_query or ''}\n"
                "请输出规划 JSON。"
            )
            response = await self.llm.achat(prompt, system_prompt=_PLANNER_SYSTEM_PROMPT)
            parsed = self._extract_json(response)
            if not parsed:
                return fallback

            normalized_topic = str(parsed.get("normalized_topic") or topic).strip() or topic.strip()
            query_rewrites = [
                str(item).strip()
                for item in (parsed.get("query_rewrites") or [])
                if str(item).strip()
            ]
            research_questions = [
                str(item).strip()
                for item in (parsed.get("research_questions") or [])
                if str(item).strip()
            ]
            search_plan: List[Dict[str, str]] = []
            for item in parsed.get("search_plan", []) or []:
                if not isinstance(item, dict):
                    continue
                dimension = str(item.get("dimension", "")).strip().lower()
                query = str(item.get("query", "")).strip()
                if dimension and query:
                    search_plan.append({"dimension": dimension, "query": query})

            llm_is_technical = bool(parsed.get("is_technical"))
            is_technical = llm_is_technical or heuristic_is_technical

            return {
                "is_technical": is_technical,
                "reason": str(parsed.get("reason") or ("technical request" if is_technical else "non-technical request")).strip(),
                "normalized_topic": normalized_topic,
                "query_rewrites": query_rewrites or heuristic_rewrites,
                "research_questions": research_questions or heuristic_questions,
                "search_plan": search_plan or heuristic_plan,
            }
        except Exception as e:
            logger.warning(f"PlannerAgent fallback due to error: {e}")
            return fallback
