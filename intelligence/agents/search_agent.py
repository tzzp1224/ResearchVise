"""
Search Agent
搜索情报员 - ReAct 驱动的多源信息搜集。
"""

from __future__ import annotations

from typing import Awaitable, Callable, Dict, Iterable, List, Any, Optional, Sequence, Set, Tuple
import logging
import re

from intelligence.llm import BaseLLM, get_llm, Message
from intelligence.state import AgentPhase
from intelligence.tools.search_tools import create_search_tools, execute_search_tool


logger = logging.getLogger(__name__)

_SOURCE_ALIASES = {
    "semantic-scholar": "semantic_scholar",
    "semantic scholar": "semantic_scholar",
    "stack overflow": "stackoverflow",
    "stack-overflow": "stackoverflow",
    "hn": "hackernews",
}

_TOOL_PRIORITY = [
    "arxiv_search",
    "semantic_scholar_search",
    "huggingface_search",
    "github_search",
    "stackoverflow_search",
    "hackernews_search",
    "reddit_search",
    "twitter_search",
]

_DIMENSION_KEYWORDS = {
    "architecture": ["architecture", "design", "system", "framework", "机制", "架构"],
    "performance": ["benchmark", "latency", "throughput", "score", "性能", "延迟", "吞吐"],
    "training": ["train", "dataset", "fine-tune", "loss", "训练", "数据集"],
    "comparison": ["compare", "versus", "vs", "trade-off", "对比", "取舍"],
    "limitation": ["limitation", "failure", "risk", "issue", "局限", "风险", "失效"],
    "deployment": ["deploy", "production", "infra", "monitor", "部署", "生产", "监控"],
}

_DEFAULT_REQUIRED_DIMENSIONS = [
    "architecture",
    "performance",
    "comparison",
    "limitation",
    "deployment",
]


class SearchAgent:
    """ReAct-style search agent with coverage-aware iteration."""

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        max_iterations: int = 8,
        min_total_results: int = 18,
        allowed_sources: Optional[Iterable[str]] = None,
        tool_executor: Optional[
            Callable[[str, Dict[str, Any]], Awaitable[List[Dict[str, Any]]]]
        ] = None,
    ):
        self.llm = llm or get_llm()
        self.max_iterations = max_iterations
        self.min_total_results = max(6, int(min_total_results))
        self.allowed_sources = self._normalize_allowed_sources(allowed_sources)
        self.tools = create_search_tools(allowed_sources=self.allowed_sources)
        self._tool_names = {
            tool.get("function", {}).get("name", "") for tool in self.tools if tool.get("function")
        }
        self._tool_executor = tool_executor or execute_search_tool

    def _normalize_allowed_sources(self, values: Optional[Iterable[str]]) -> Optional[Set[str]]:
        if values is None:
            return None
        normalized: Set[str] = set()
        for item in values:
            key = str(item or "").strip().lower().replace("-", "_")
            if not key:
                continue
            normalized.add(_SOURCE_ALIASES.get(key, key))
        return normalized or None

    def _default_queries(self, topic: str, query_rewrites: Optional[Sequence[str]]) -> List[str]:
        queries = [str(q).strip() for q in (query_rewrites or []) if str(q).strip()]
        if topic.strip() and topic.strip() not in queries:
            queries.insert(0, topic.strip())
        return queries[:8] if queries else [topic.strip()]

    def _normalize_search_plan(self, raw_plan: Any, queries: List[str]) -> List[Dict[str, str]]:
        plan: List[Dict[str, str]] = []
        for item in raw_plan or []:
            if not isinstance(item, dict):
                continue
            dimension = str(item.get("dimension", "")).strip().lower()
            query = str(item.get("query", "")).strip()
            if dimension and query:
                plan.append({"dimension": dimension, "query": query})
        if plan:
            return plan[:8]
        seed = queries[0] if queries else ""
        return [
            {"dimension": dim, "query": f"{seed} {dim}".strip()} for dim in _DEFAULT_REQUIRED_DIMENSIONS
        ]

    def _build_system_prompt(
        self,
        *,
        topic: str,
        user_query: str,
        queries: List[str],
        research_questions: List[str],
        plan: List[Dict[str, str]],
    ) -> str:
        plan_block = "\n".join(
            [f"- [{item['dimension']}] {item['query']}" for item in plan]
        )
        question_block = "\n".join([f"- {q}" for q in research_questions]) or "- (none)"
        query_block = "\n".join([f"- {q}" for q in queries])
        tools_hint = ", ".join(sorted(self._tool_names))

        return (
            "你是研究型搜索 Agent。你必须像学者一样进行多轮 ReAct 检索："
            "先提出假设，再调用工具验证，再根据缺口补搜，直到覆盖核心维度。\n\n"
            f"主题: {topic}\n"
            f"用户问题: {user_query or '(none)'}\n"
            f"可用工具: {tools_hint}\n"
            "候选查询:\n"
            f"{query_block}\n"
            "研究问题:\n"
            f"{question_block}\n"
            "建议计划:\n"
            f"{plan_block}\n\n"
            "执行原则:\n"
            "1) 每轮至少调用 1-2 个工具；\n"
            "2) 优先论文/代码，再补社区；\n"
            "3) 避免重复查询；\n"
            "4) 当信息足够时给出总结并停止。"
        )

    def _sanitize_tool_args(
        self,
        *,
        tool_name: str,
        tool_args: Any,
        default_query: str,
        max_results_per_source: int,
    ) -> Dict[str, Any]:
        args = dict(tool_args or {}) if isinstance(tool_args, dict) else {}

        query = str(args.get("query") or default_query).strip()
        if not query:
            query = default_query

        try:
            max_results = int(args.get("max_results", max_results_per_source))
        except Exception:
            max_results = max_results_per_source
        max_results = max(1, min(max_results, max_results_per_source))

        payload: Dict[str, Any] = {
            "query": query,
            "max_results": max_results,
        }

        if tool_name == "huggingface_search":
            search_type = str(args.get("search_type", "all")).strip().lower()
            payload["search_type"] = search_type if search_type in {"all", "models", "datasets"} else "all"
        elif tool_name == "reddit_search":
            subreddits = args.get("subreddits")
            if isinstance(subreddits, list):
                clean_subreddits = [str(item).strip() for item in subreddits if str(item).strip()]
                if clean_subreddits:
                    payload["subreddits"] = clean_subreddits[:6]
        elif tool_name == "stackoverflow_search":
            tags = args.get("tags")
            if isinstance(tags, list):
                clean_tags = [str(item).strip() for item in tags if str(item).strip()]
                if clean_tags:
                    payload["tags"] = clean_tags[:6]
        elif tool_name == "hackernews_search":
            sort_by = str(args.get("sort_by", "relevance")).strip().lower()
            payload["sort_by"] = sort_by if sort_by in {"relevance", "date"} else "relevance"

        return payload

    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        unique: List[Dict[str, Any]] = []
        seen = set()
        for item in results:
            rid = str(item.get("id") or "").strip()
            key = rid or str(item.get("url") or "").strip() or str(item.get("title") or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique

    def _covered_dimensions(self, results: List[Dict[str, Any]], required: List[str]) -> List[str]:
        covered: List[str] = []
        for dim in required:
            keywords = _DIMENSION_KEYWORDS.get(dim, [dim])
            hit = False
            for item in results:
                haystack = f"{item.get('title', '')} {item.get('content', '')}".lower()
                if any(k.lower() in haystack for k in keywords):
                    hit = True
                    break
            if hit:
                covered.append(dim)
        return covered

    def _coverage_snapshot(self, results: List[Dict[str, Any]], required: List[str]) -> Dict[str, Any]:
        sources = sorted({str(item.get("source", "")).strip() for item in results if str(item.get("source", "")).strip()})
        covered_dims = self._covered_dimensions(results, required)
        return {
            "result_count": len(results),
            "source_coverage": sources,
            "dimension_coverage": covered_dims,
            "missing_dimensions": [dim for dim in required if dim not in covered_dims],
        }

    def _should_stop(self, results: List[Dict[str, Any]], required: List[str], *, soft: bool = False) -> bool:
        if not results:
            return False
        coverage = self._coverage_snapshot(results, required)
        source_target = 2 if soft else 3
        dim_target = max(2, min(4, len(required)))
        result_target = max(8, self.min_total_results // (2 if soft else 1))
        return (
            coverage["result_count"] >= result_target
            and len(coverage["source_coverage"]) >= min(source_target, max(1, len(self._tool_names)))
            and len(coverage["dimension_coverage"]) >= min(dim_target, len(required))
        )

    def _is_search_complete(self, content: str) -> bool:
        text = (content or "").lower()
        return any(
            token in text
            for token in [
                "搜索完成",
                "信息收集完毕",
                "总结如下",
                "search complete",
                "finished searching",
            ]
        )

    def _format_observation(self, results: List[Dict[str, Any]], tool_name: str) -> str:
        if not results:
            return f"{tool_name}: 未找到结果。"
        lines = [f"{tool_name}: 找到 {len(results)} 条结果"]
        for idx, item in enumerate(results[:4], 1):
            lines.append(
                f"{idx}. [{item.get('source', 'unknown')}] {item.get('title', 'Untitled')}"
            )
        return "\n".join(lines)

    async def _fallback_step(
        self,
        *,
        queries: List[str],
        plan: List[Dict[str, str]],
        used_actions: Set[Tuple[str, str]],
        max_results_per_source: int,
        iteration: int,
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        candidates: List[str] = []
        if plan:
            idx = min(iteration - 1, len(plan) - 1)
            candidates.append(plan[idx]["query"])
        candidates.extend(queries)

        for query in candidates:
            normalized_query = str(query).strip()
            if not normalized_query:
                continue
            for tool_name in _TOOL_PRIORITY:
                if tool_name not in self._tool_names:
                    continue
                action_key = (tool_name, normalized_query.lower())
                if action_key in used_actions:
                    continue

                payload = self._sanitize_tool_args(
                    tool_name=tool_name,
                    tool_args={"query": normalized_query, "max_results": max_results_per_source},
                    default_query=normalized_query,
                    max_results_per_source=max_results_per_source,
                )

                try:
                    results = await self._tool_executor(tool_name, payload)
                except Exception as exc:
                    logger.warning(f"Fallback search failed: {tool_name}({normalized_query}): {exc}")
                    used_actions.add(action_key)
                    continue

                used_actions.add(action_key)
                trace = {
                    "iteration": iteration,
                    "mode": "fallback",
                    "tool": tool_name,
                    "query": payload.get("query", ""),
                    "result_count": len(results),
                }
                return results, trace

        return [], None

    async def search(
        self,
        topic: str,
        user_query: Optional[str] = None,
        query_rewrites: Optional[List[str]] = None,
        research_questions: Optional[List[str]] = None,
        search_plan: Optional[List[Dict[str, str]]] = None,
        max_results_per_source: int = 8,
    ) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "topic": topic,
            "user_query": user_query,
            "query_rewrites": query_rewrites or [],
            "research_questions": research_questions or [],
            "search_plan": search_plan or [],
            "max_results_per_source": max_results_per_source,
        }
        return await self.run(state)

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """执行 ReAct 搜索任务。"""
        topic = str(state.get("topic") or "").strip()
        user_query = str(state.get("user_query") or "").strip()
        max_results_per_source = max(2, min(int(state.get("max_results_per_source") or 8), 20))

        queries = self._default_queries(topic, state.get("query_rewrites"))
        research_questions = [
            str(item).strip() for item in (state.get("research_questions") or []) if str(item).strip()
        ]
        plan = self._normalize_search_plan(state.get("search_plan"), queries)
        required_dimensions = [item["dimension"] for item in plan if item.get("dimension")] or list(
            _DEFAULT_REQUIRED_DIMENSIONS
        )

        messages = [
            Message.system(
                self._build_system_prompt(
                    topic=topic,
                    user_query=user_query,
                    queries=queries,
                    research_questions=research_questions,
                    plan=plan,
                )
            ),
            Message.user("请开始第一轮检索。"),
        ]

        all_results: List[Dict[str, Any]] = []
        trace: List[Dict[str, Any]] = []
        used_actions: Set[Tuple[str, str]] = set()

        for iteration in range(1, self.max_iterations + 1):
            coverage_before = self._coverage_snapshot(all_results, required_dimensions)
            logger.info(
                "Search Agent iteration %s/%s (results=%s sources=%s)",
                iteration,
                self.max_iterations,
                coverage_before["result_count"],
                len(coverage_before["source_coverage"]),
            )

            response = await self.llm.acomplete(messages, tools=self.tools)
            executed = False

            if response.has_tool_calls:
                for tool_call in response.tool_calls or []:
                    tool_name = str(tool_call.name or "").strip()
                    if tool_name not in self._tool_names:
                        continue

                    default_query = queries[min(iteration - 1, len(queries) - 1)]
                    payload = self._sanitize_tool_args(
                        tool_name=tool_name,
                        tool_args=tool_call.arguments,
                        default_query=default_query,
                        max_results_per_source=max_results_per_source,
                    )
                    action_key = (tool_name, str(payload.get("query", "")).lower())
                    used_actions.add(action_key)

                    try:
                        results = await self._tool_executor(tool_name, payload)
                    except Exception as exc:
                        logger.warning(f"Tool execution failed: {tool_name}({payload}): {exc}")
                        continue

                    executed = True
                    all_results.extend(results)
                    trace.append(
                        {
                            "iteration": iteration,
                            "mode": "react",
                            "tool": tool_name,
                            "query": payload.get("query", ""),
                            "result_count": len(results),
                        }
                    )
                    messages.append(Message.assistant(response.content or f"调用 {tool_name}"))
                    messages.append(Message.user(self._format_observation(results, tool_name)))

            else:
                assistant_content = response.content or ""
                messages.append(Message.assistant(assistant_content))
                if self._is_search_complete(assistant_content) and self._should_stop(
                    all_results,
                    required_dimensions,
                    soft=True,
                ):
                    break

            if not executed:
                fallback_results, fallback_trace = await self._fallback_step(
                    queries=queries,
                    plan=plan,
                    used_actions=used_actions,
                    max_results_per_source=max_results_per_source,
                    iteration=iteration,
                )
                if fallback_trace:
                    trace.append(fallback_trace)
                    all_results.extend(fallback_results)
                    messages.append(
                        Message.user(
                            self._format_observation(
                                fallback_results,
                                str(fallback_trace.get("tool", "fallback")),
                            )
                        )
                    )

            all_results = self._deduplicate_results(all_results)
            if self._should_stop(all_results, required_dimensions):
                break

            coverage_after = self._coverage_snapshot(all_results, required_dimensions)
            messages.append(
                Message.user(
                    "当前覆盖情况: "
                    f"results={coverage_after['result_count']}, "
                    f"sources={coverage_after['source_coverage']}, "
                    f"missing_dimensions={coverage_after['missing_dimensions']}。"
                    "请继续补齐缺口。"
                )
            )

        final_results = self._deduplicate_results(all_results)
        final_coverage = self._coverage_snapshot(final_results, required_dimensions)

        return {
            "phase": AgentPhase.SEARCHING,
            "iteration": min(self.max_iterations, len(trace) if trace else 1),
            "search_results": final_results,
            "messages": [message.to_dict() for message in messages],
            "search_trace": trace,
            "coverage": final_coverage,
            "strategy": "react_agent",
        }

    async def plan_search(self, topic: str) -> List[str]:
        """保留兼容接口：返回基础关键词列表。"""
        topic = str(topic or "").strip()
        if not topic:
            return []
        tokens = re.findall(r"[A-Za-z0-9#+-]+", topic)
        candidates = [topic]
        if tokens:
            candidates.append(" ".join(tokens[:4]))
        deduped: List[str] = []
        seen = set()
        for item in candidates:
            key = item.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped[:5]
