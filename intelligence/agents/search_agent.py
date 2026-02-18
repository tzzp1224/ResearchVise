"""
Search Agent
搜索情报员 - ReAct 驱动的多源信息搜集。
"""

from __future__ import annotations

import asyncio
import inspect
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
    "arxiv_rss_search",
    "openreview_search",
    "semantic_scholar_search",
    "huggingface_search",
    "github_search",
    "stackoverflow_search",
    "hackernews_search",
    "reddit_search",
    "twitter_search",
]

_SOURCE_TO_TOOL = {
    "arxiv": "arxiv_search",
    "arxiv_rss": "arxiv_rss_search",
    "openreview": "openreview_search",
    "semantic_scholar": "semantic_scholar_search",
    "huggingface": "huggingface_search",
    "github": "github_search",
    "stackoverflow": "stackoverflow_search",
    "hackernews": "hackernews_search",
    "reddit": "reddit_search",
    "twitter": "twitter_search",
}

_DIMENSION_KEYWORDS = {
    "architecture": ["architecture", "design", "system", "framework", "method", "formula", "equation", "机制", "架构", "公式"],
    "performance": ["benchmark", "latency", "throughput", "score", "sota", "性能", "延迟", "吞吐", "准确率"],
    "training": ["train", "dataset", "fine-tune", "loss", "训练", "数据集"],
    "comparison": ["compare", "versus", "vs", "trade-off", "对比", "取舍"],
    "limitation": ["limitation", "failure", "risk", "issue", "局限", "风险", "失效"],
    "deployment": ["deploy", "production", "infra", "monitor", "hardware", "gpu", "部署", "生产", "监控", "硬件", "显存"],
}

_DEFAULT_REQUIRED_DIMENSIONS = [
    "architecture",
    "performance",
    "comparison",
    "limitation",
    "deployment",
]

_CN_QUERY_EXPANSIONS = {
    "模型上下文协议": "model context protocol",
    "强化学习": "reinforcement learning",
    "深度强化学习": "deep reinforcement learning",
    "检索增强生成": "retrieval augmented generation",
    "向量数据库": "vector database",
    "知识图谱": "knowledge graph",
    "大语言模型": "large language model",
    "多模态": "multimodal model",
    "生产环境部署": "production deployment",
    "高可用": "high availability",
    "可观测性": "observability",
}


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
        progress_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
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
        self._progress_callback = progress_callback

    async def _emit_progress(self, event: str, **payload: Any) -> None:
        if self._progress_callback is None:
            return
        message: Dict[str, Any] = {"event": event}
        message.update(payload)
        try:
            result = self._progress_callback(message)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.debug("Search progress callback failed", exc_info=True)

    def _source_breakdown(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        breakdown: Dict[str, int] = {}
        for item in results:
            source = str(item.get("source", "")).strip() or "unknown"
            breakdown[source] = breakdown.get(source, 0) + 1
        return breakdown

    def _empty_result_disable_threshold(self, tool_name: str) -> int:
        if tool_name == "semantic_scholar_search":
            return 1
        return 3

    def _required_tools(self) -> Set[str]:
        if not self.allowed_sources:
            return set()
        required: Set[str] = set()
        for source in sorted(self.allowed_sources):
            tool_name = _SOURCE_TO_TOOL.get(source)
            if tool_name and tool_name in self._tool_names:
                required.add(tool_name)
        return required

    def _required_tools_covered(
        self,
        *,
        trace: List[Dict[str, Any]],
        disabled_tools: Set[str],
        required_tools: Set[str],
    ) -> bool:
        if not required_tools:
            return True
        attempted_tools = {
            str(item.get("tool", "")).strip()
            for item in trace
            if str(item.get("tool", "")).strip()
        }
        return required_tools.issubset(attempted_tools.union(set(disabled_tools)))

    async def _execute_tool_call(
        self,
        *,
        tool_name: str,
        payload: Dict[str, Any],
        timeout_sec: int,
    ) -> List[Dict[str, Any]]:
        timeout = max(3, int(timeout_sec))
        return await asyncio.wait_for(
            self._tool_executor(tool_name, payload),
            timeout=timeout,
        )

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
        topic_text = topic.strip()
        if topic_text and topic_text not in queries:
            queries.insert(0, topic_text)

        bilingual_map = {
            "技术研究": "technical research",
            "生产部署": "production deployment",
            "部署": "deployment",
            "架构": "architecture",
            "性能": "performance benchmark",
            "对比": "comparison",
            "风险": "security risk",
        }

        expanded: List[str] = []
        seen = set()
        for item in queries or [topic_text]:
            text = str(item or "").strip()
            if not text:
                continue
            key = text.lower()
            if key not in seen:
                seen.add(key)
                expanded.append(text)

            if re.search(r"[\u4e00-\u9fff]", text):
                ascii_tokens = re.findall(r"[A-Za-z0-9#+-]+(?:\.[0-9]+)?", text)
                if ascii_tokens:
                    ascii_query = " ".join(ascii_tokens[:6]).strip()
                    if ascii_query and ascii_query.lower() not in seen:
                        seen.add(ascii_query.lower())
                        expanded.append(ascii_query)
                translated = text
                for cn, en in bilingual_map.items():
                    translated = translated.replace(cn, en)
                for cn, en in sorted(_CN_QUERY_EXPANSIONS.items(), key=lambda pair: len(pair[0]), reverse=True):
                    if cn in text:
                        if en.lower() not in seen:
                            seen.add(en.lower())
                            expanded.append(en)
                        translated = translated.replace(cn, en)
                translated = re.sub(r"\s+", " ", translated).strip()
                if translated and translated.lower() not in seen:
                    seen.add(translated.lower())
                    expanded.append(translated)

        return expanded[:10] if expanded else [topic_text]

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

    async def _run_tool_batch(
        self,
        *,
        batch: Sequence[Tuple[str, Dict[str, Any]]],
        iteration: int,
        trace_mode: str,
        progress_mode: str,
        tool_failures: Dict[str, int],
        tool_empty_streak: Dict[str, int],
        disabled_tools: Set[str],
        failure_threshold: int,
        tool_timeout_sec: int,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Tuple[str, List[Dict[str, Any]]]]]:
        if not batch:
            return [], [], []

        tasks = [
            self._execute_tool_call(
                tool_name=tool_name,
                payload=payload,
                timeout_sec=tool_timeout_sec,
            )
            for tool_name, payload in batch
        ]
        outputs = await asyncio.gather(*tasks, return_exceptions=True)

        merged_results: List[Dict[str, Any]] = []
        traces: List[Dict[str, Any]] = []
        observations: List[Tuple[str, List[Dict[str, Any]]]] = []

        for (tool_name, payload), output in zip(batch, outputs):
            if isinstance(output, Exception):
                failure_count = int(tool_failures.get(tool_name, 0)) + 1
                tool_failures[tool_name] = failure_count
                disabled = failure_count >= failure_threshold
                if disabled:
                    disabled_tools.add(tool_name)
                if isinstance(output, asyncio.TimeoutError):
                    error_text = f"timeout>{max(3, int(tool_timeout_sec))}s"
                else:
                    error_text = str(output)
                await self._emit_progress(
                    "search_tool_error",
                    iteration=iteration,
                    mode=progress_mode,
                    tool=tool_name,
                    query=payload.get("query", ""),
                    error=error_text,
                    failure_count=failure_count,
                    disabled=disabled,
                )
                if disabled:
                    await self._emit_progress(
                        "search_tool_disabled",
                        iteration=iteration,
                        tool=tool_name,
                        reason=f"failed {failure_count} times",
                    )
                continue

            results = list(output or [])
            tool_failures[tool_name] = 0
            if results:
                tool_empty_streak[tool_name] = 0
            else:
                streak = int(tool_empty_streak.get(tool_name, 0)) + 1
                tool_empty_streak[tool_name] = streak
                disable_threshold = max(1, int(self._empty_result_disable_threshold(tool_name)))
                if streak >= disable_threshold:
                    disabled_tools.add(tool_name)
                    await self._emit_progress(
                        "search_tool_disabled",
                        iteration=iteration,
                        tool=tool_name,
                        reason=f"no_results_{streak}_times",
                    )

            traces.append(
                {
                    "iteration": iteration,
                    "mode": trace_mode,
                    "tool": tool_name,
                    "query": payload.get("query", ""),
                    "result_count": len(results),
                }
            )
            observations.append((tool_name, results))
            await self._emit_progress(
                "search_tool_result",
                iteration=iteration,
                mode=progress_mode,
                tool=tool_name,
                query=payload.get("query", ""),
                result_count=len(results),
                source_breakdown=self._source_breakdown(results),
            )
            if results:
                merged_results.extend(results)

        return merged_results, traces, observations

    def _build_search_output(
        self,
        *,
        iteration: int,
        results: List[Dict[str, Any]],
        messages: List[Message],
        trace: List[Dict[str, Any]],
        coverage: Dict[str, Any],
        disabled_tools: Set[str],
    ) -> Dict[str, Any]:
        return {
            "phase": AgentPhase.SEARCHING,
            "iteration": iteration,
            "search_results": results,
            "messages": [message.to_dict() for message in messages],
            "search_trace": trace,
            "coverage": coverage,
            "strategy": "react_agent",
            "disabled_tools": sorted(disabled_tools),
        }

    async def _fallback_step(
        self,
        *,
        queries: List[str],
        plan: List[Dict[str, str]],
        used_actions: Set[Tuple[str, str]],
        max_results_per_source: int,
        iteration: int,
        tool_failures: Dict[str, int],
        tool_empty_streak: Dict[str, int],
        disabled_tools: Set[str],
        failure_threshold: int,
        tool_timeout_sec: int,
        mode: str = "fallback",
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        candidates: List[str] = []
        if plan:
            idx = min(iteration - 1, len(plan) - 1)
            candidates.append(plan[idx]["query"])
        candidates.extend(queries)

        all_merged_results: List[Dict[str, Any]] = []
        all_traces: List[Dict[str, Any]] = []
        required_tools = self._required_tools()

        for query in candidates:
            normalized_query = str(query).strip()
            if not normalized_query:
                continue
            batch: List[Tuple[str, Dict[str, Any]]] = []
            for tool_name in _TOOL_PRIORITY:
                if tool_name not in self._tool_names or tool_name in disabled_tools:
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
                used_actions.add(action_key)
                batch.append((tool_name, payload))
                await self._emit_progress(
                    "search_bootstrap_attempt" if mode == "bootstrap" else "search_fallback_attempt",
                    iteration=iteration,
                    mode=mode,
                    tool=tool_name,
                    query=payload.get("query", ""),
                )

            if not batch:
                continue

            merged_results, traces, _ = await self._run_tool_batch(
                batch=batch,
                iteration=iteration,
                trace_mode=mode,
                progress_mode="fallback",
                tool_failures=tool_failures,
                tool_empty_streak=tool_empty_streak,
                disabled_tools=disabled_tools,
                failure_threshold=failure_threshold,
                tool_timeout_sec=tool_timeout_sec,
            )

            if traces:
                all_traces.extend(traces)
                if merged_results:
                    all_merged_results.extend(merged_results)
                all_merged_results = self._deduplicate_results(all_merged_results)

            # 如果当前覆盖已经足够，提前结束 fallback。
            if self._should_stop(all_merged_results, _DEFAULT_REQUIRED_DIMENSIONS, soft=True):
                break
            if required_tools and self._required_tools_covered(
                trace=all_traces,
                disabled_tools=disabled_tools,
                required_tools=required_tools,
            ):
                break

        return all_merged_results, all_traces

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
        tool_timeout_sec = max(3, min(int(state.get("tool_timeout_sec") or 12), 60))
        time_budget_sec = max(20, min(int(state.get("time_budget_sec") or 90), 900))
        react_thought_timeout_sec = max(4, min(int(state.get("react_thought_timeout_sec") or 9), 45))
        loop = asyncio.get_running_loop()
        started_at = loop.time()

        queries = self._default_queries(topic, state.get("query_rewrites"))
        research_questions = [
            str(item).strip() for item in (state.get("research_questions") or []) if str(item).strip()
        ]
        plan = self._normalize_search_plan(state.get("search_plan"), queries)
        required_dimensions = [item["dimension"] for item in plan if item.get("dimension")] or list(
            _DEFAULT_REQUIRED_DIMENSIONS
        )
        required_tools = self._required_tools()

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
        tool_failures: Dict[str, int] = {}
        tool_empty_streak: Dict[str, int] = {}
        disabled_tools: Set[str] = set()
        failure_threshold = 2
        no_progress_rounds = 0
        topup_performed = False

        await self._emit_progress(
            "search_agent_started",
            topic=topic,
            max_iterations=self.max_iterations,
            queries=queries,
            required_dimensions=required_dimensions,
            required_tools=sorted(required_tools),
            plan=plan,
            tool_timeout_sec=tool_timeout_sec,
            time_budget_sec=time_budget_sec,
            react_thought_timeout_sec=react_thought_timeout_sec,
        )

        start_iteration = 1
        if queries:
            bootstrap_results, bootstrap_traces = await self._fallback_step(
                queries=[queries[0]],
                plan=[],
                used_actions=used_actions,
                max_results_per_source=max_results_per_source,
                iteration=1,
                tool_failures=tool_failures,
                tool_empty_streak=tool_empty_streak,
                disabled_tools=disabled_tools,
                failure_threshold=failure_threshold,
                tool_timeout_sec=tool_timeout_sec,
                mode="bootstrap",
            )
            if bootstrap_traces:
                start_iteration = 2
                trace.extend(bootstrap_traces)
                all_results.extend(bootstrap_results)
                all_results = self._deduplicate_results(all_results)
                coverage_bootstrap = self._coverage_snapshot(all_results, required_dimensions)
                required_tools_covered = self._required_tools_covered(
                    trace=trace,
                    disabled_tools=disabled_tools,
                    required_tools=required_tools,
                )
                await self._emit_progress(
                    "search_bootstrap_completed",
                    query=queries[0],
                    coverage=coverage_bootstrap,
                    disabled_tools=sorted(disabled_tools),
                    required_tools_covered=required_tools_covered,
                )
                messages.append(
                    Message.user(
                        "Bootstrap 并发检索结果: "
                        + ", ".join(
                            [
                                f"{item.get('tool', 'unknown')}={item.get('result_count', 0)}"
                                for item in bootstrap_traces
                            ]
                        )
                    )
                )
                if self._should_stop(all_results, required_dimensions, soft=True) and required_tools_covered:
                    final_results = self._deduplicate_results(all_results)
                    final_coverage = self._coverage_snapshot(final_results, required_dimensions)
                    await self._emit_progress(
                        "search_agent_completed",
                        result_count=len(final_results),
                        coverage=final_coverage,
                        trace_steps=len(trace),
                        disabled_tools=sorted(disabled_tools),
                        required_tools=sorted(required_tools),
                        required_tools_covered=True,
                        elapsed_sec=round(loop.time() - started_at, 2),
                    )
                    return self._build_search_output(
                        iteration=1,
                        results=final_results,
                        messages=messages,
                        trace=trace,
                        coverage=final_coverage,
                        disabled_tools=disabled_tools,
                    )

        for iteration in range(start_iteration, self.max_iterations + 1):
            elapsed_sec = loop.time() - started_at
            if elapsed_sec >= float(time_budget_sec):
                await self._emit_progress(
                    "search_agent_terminated",
                    iteration=iteration,
                    reason="time_budget_exceeded",
                    elapsed_sec=round(elapsed_sec, 2),
                )
                break
            coverage_before = self._coverage_snapshot(all_results, required_dimensions)
            logger.info(
                "Search Agent iteration %s/%s (results=%s sources=%s)",
                iteration,
                self.max_iterations,
                coverage_before["result_count"],
                len(coverage_before["source_coverage"]),
            )
            await self._emit_progress(
                "search_iteration_started",
                iteration=iteration,
                max_iterations=self.max_iterations,
                coverage=coverage_before,
                disabled_tools=sorted(disabled_tools),
                required_tools=sorted(required_tools),
                elapsed_sec=round(elapsed_sec, 2),
            )

            before_result_count = len(all_results)
            response = None
            try:
                response = await asyncio.wait_for(
                    self.llm.acomplete(messages, tools=self.tools),
                    timeout=react_thought_timeout_sec,
                )
            except asyncio.TimeoutError:
                await self._emit_progress(
                    "search_thought_timeout",
                    iteration=iteration,
                    timeout_sec=react_thought_timeout_sec,
                )
            except Exception as exc:
                await self._emit_progress(
                    "search_thought_error",
                    iteration=iteration,
                    error=str(exc),
                )
            executed = False

            if response and response.has_tool_calls:
                batch: List[Tuple[str, Dict[str, Any]]] = []
                for tool_call in response.tool_calls or []:
                    tool_name = str(tool_call.name or "").strip()
                    if tool_name not in self._tool_names:
                        continue
                    if tool_name in disabled_tools:
                        continue

                    default_query = queries[min(iteration - 1, len(queries) - 1)]
                    payload = self._sanitize_tool_args(
                        tool_name=tool_name,
                        tool_args=tool_call.arguments,
                        default_query=default_query,
                        max_results_per_source=max_results_per_source,
                    )
                    action_key = (tool_name, str(payload.get("query", "")).lower())
                    if action_key in used_actions:
                        continue
                    used_actions.add(action_key)
                    await self._emit_progress(
                        "search_tool_call",
                        iteration=iteration,
                        mode="react",
                        tool=tool_name,
                        query=payload.get("query", ""),
                        max_results=payload.get("max_results", max_results_per_source),
                    )
                    batch.append((tool_name, payload))

                if batch:
                    merged_results, batch_traces, observations = await self._run_tool_batch(
                        batch=batch,
                        iteration=iteration,
                        trace_mode="react",
                        progress_mode="react",
                        tool_failures=tool_failures,
                        tool_empty_streak=tool_empty_streak,
                        disabled_tools=disabled_tools,
                        failure_threshold=failure_threshold,
                        tool_timeout_sec=tool_timeout_sec,
                    )
                    if batch_traces:
                        trace.extend(batch_traces)
                    if merged_results:
                        executed = True
                        all_results.extend(merged_results)
                    for tool_name, results in observations:
                        messages.append(Message.assistant(response.content or f"调用 {tool_name}"))
                        messages.append(Message.user(self._format_observation(results, tool_name)))

            elif response:
                assistant_content = response.content or ""
                messages.append(Message.assistant(assistant_content))
                required_tools_covered = self._required_tools_covered(
                    trace=trace,
                    disabled_tools=disabled_tools,
                    required_tools=required_tools,
                )
                if self._is_search_complete(assistant_content) and self._should_stop(
                    all_results,
                    required_dimensions,
                    soft=True,
                ) and required_tools_covered:
                    break

            if not executed:
                fallback_results, fallback_traces = await self._fallback_step(
                    queries=queries,
                    plan=plan,
                    used_actions=used_actions,
                    max_results_per_source=max_results_per_source,
                    iteration=iteration,
                    tool_failures=tool_failures,
                    tool_empty_streak=tool_empty_streak,
                    disabled_tools=disabled_tools,
                    failure_threshold=failure_threshold,
                    tool_timeout_sec=tool_timeout_sec,
                    mode="fallback",
                )
                if fallback_traces:
                    trace.extend(fallback_traces)
                    all_results.extend(fallback_results)
                    messages.append(
                        Message.user(
                            "Fallback 批量检索结果: "
                            + ", ".join(
                                [
                                    f"{item.get('tool', 'unknown')}={item.get('result_count', 0)}"
                                    for item in fallback_traces
                                ]
                            )
                        )
                    )

            all_results = self._deduplicate_results(all_results)
            if len(all_results) <= before_result_count:
                no_progress_rounds += 1
            else:
                no_progress_rounds = 0
            coverage_after = self._coverage_snapshot(all_results, required_dimensions)
            required_tools_covered = self._required_tools_covered(
                trace=trace,
                disabled_tools=disabled_tools,
                required_tools=required_tools,
            )
            await self._emit_progress(
                "search_coverage_update",
                iteration=iteration,
                coverage=coverage_after,
                disabled_tools=sorted(disabled_tools),
                required_tools_covered=required_tools_covered,
            )

            min_source_target = min(3, max(1, len(self._tool_names)))
            low_coverage = (
                coverage_after["result_count"] < max(8, self.min_total_results // 2)
                or len(coverage_after["source_coverage"]) < min_source_target
                or len(coverage_after["dimension_coverage"]) < min(3, len(required_dimensions))
            )
            should_topup = (
                not topup_performed
                and low_coverage
                and not required_tools_covered
                and iteration >= max(2, start_iteration)
            )
            if should_topup:
                await self._emit_progress(
                    "search_topup_started",
                    iteration=iteration,
                    coverage=coverage_after,
                    candidate_queries=queries[:3],
                )
                topup_results, topup_traces = await self._fallback_step(
                    queries=queries[:3],
                    plan=plan[:2],
                    used_actions=used_actions,
                    max_results_per_source=max_results_per_source,
                    iteration=iteration,
                    tool_failures=tool_failures,
                    tool_empty_streak=tool_empty_streak,
                    disabled_tools=disabled_tools,
                    failure_threshold=failure_threshold,
                    tool_timeout_sec=tool_timeout_sec,
                    mode="fallback",
                )
                topup_performed = True
                if topup_traces:
                    trace.extend(topup_traces)
                    all_results.extend(topup_results)
                    all_results = self._deduplicate_results(all_results)
                    coverage_after = self._coverage_snapshot(all_results, required_dimensions)
                    required_tools_covered = self._required_tools_covered(
                        trace=trace,
                        disabled_tools=disabled_tools,
                        required_tools=required_tools,
                    )
                    await self._emit_progress(
                        "search_topup_completed",
                        iteration=iteration,
                        added_results=len(topup_results),
                        coverage=coverage_after,
                        required_tools_covered=required_tools_covered,
                    )

            if self._should_stop(all_results, required_dimensions) and required_tools_covered:
                break
            if no_progress_rounds >= 2 and iteration >= 2 and required_tools_covered:
                await self._emit_progress(
                    "search_agent_terminated",
                    iteration=iteration,
                    reason="no_progress",
                )
                break
            if self._tool_names and len(disabled_tools) >= len(self._tool_names):
                await self._emit_progress(
                    "search_agent_terminated",
                    iteration=iteration,
                    reason="all_tools_disabled",
                )
                break

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
        await self._emit_progress(
            "search_agent_completed",
            result_count=len(final_results),
            coverage=final_coverage,
            trace_steps=len(trace),
            disabled_tools=sorted(disabled_tools),
            required_tools=sorted(required_tools),
            required_tools_covered=self._required_tools_covered(
                trace=trace,
                disabled_tools=disabled_tools,
                required_tools=required_tools,
            ),
            elapsed_sec=round(loop.time() - started_at, 2),
        )

        return self._build_search_output(
            iteration=min(self.max_iterations, len(trace) if trace else 1),
            results=final_results,
            messages=messages,
            trace=trace,
            coverage=final_coverage,
            disabled_tools=disabled_tools,
        )

    async def plan_search(self, topic: str) -> List[str]:
        """保留兼容接口：返回基础关键词列表。"""
        topic = str(topic or "").strip()
        if not topic:
            return []
        tokens = re.findall(r"[A-Za-z0-9#+-]+(?:\.[0-9]+)?", topic)
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
