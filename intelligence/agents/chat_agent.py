"""
Chat Agent
基于知识库检索结果进行问答（Chat over KB）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
import logging
import re

from intelligence.llm import BaseLLM, Message, get_llm
from intelligence.tools.rag_tools import hybrid_search, vector_search


logger = logging.getLogger(__name__)

_CHAT_SYSTEM_PROMPT = """你是研究助手，基于给定证据回答问题。
要求：
1) 只依据给定证据回答，不要编造；
2) 明确给出结论、证据依据和不确定性；
3) 输出 JSON:
{
  "answer": "回答",
  "highlights": ["要点1", "要点2"],
  "limitations": ["局限1"],
  "follow_up_queries": ["后续查询词1"]
}"""


class ChatAgent:
    """Knowledge-base grounded chat agent."""

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        *,
        default_top_k: int = 6,
        score_threshold: float = 0.15,
    ):
        self.llm = llm or get_llm()
        self.default_top_k = max(2, min(int(default_top_k), 20))
        self.score_threshold = float(max(0.0, min(score_threshold, 1.0)))

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
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

    @staticmethod
    def _format_context(docs: List[Dict[str, Any]], *, limit: int = 8) -> str:
        lines: List[str] = []
        for idx, item in enumerate(docs[:limit], 1):
            meta = item.get("metadata", {}) or {}
            source = str(meta.get("source", "unknown")).strip() or "unknown"
            url = str(meta.get("url", "")).strip()
            score = item.get("score")
            score_text = f"{float(score):.3f}" if isinstance(score, (int, float)) else "n/a"
            content = str(item.get("content", "")).strip().replace("\n", " ")
            lines.append(
                f"[{idx}] source={source} score={score_text} id={item.get('id', '')} url={url}\n"
                f"snippet: {content[:700]}"
            )
        return "\n\n".join(lines)

    @staticmethod
    def _dedupe_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        unique: List[Dict[str, Any]] = []
        seen = set()
        for item in docs:
            meta = item.get("metadata", {}) or {}
            doc_id = str(item.get("id", "")).strip()
            url = str(meta.get("url", "")).strip()
            if doc_id or url:
                key = (doc_id, url)
            else:
                key = ("", str(item.get("content", "")).strip()[:180])
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique

    async def ask(
        self,
        *,
        question: str,
        top_k: Optional[int] = None,
        sources: Optional[List[str]] = None,
        year_filter: Optional[int] = None,
        use_hybrid: bool = True,
        namespace: Optional[str] = None,
        topic_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        query = str(question or "").strip()
        if not query:
            return {
                "question": question,
                "answer": "问题为空，无法检索知识库。",
                "citations": [],
                "retrieved_count": 0,
                "highlights": [],
                "limitations": ["empty question"],
                "follow_up_queries": [],
            }

        k = self.default_top_k if top_k is None else max(1, min(int(top_k), 20))
        try:
            if use_hybrid:
                docs = await hybrid_search(
                    query=query,
                    sources=sources,
                    year_filter=year_filter,
                    top_k=k,
                    score_threshold=self.score_threshold,
                    namespace=namespace,
                    topic_hash=topic_hash,
                )
            else:
                docs = await vector_search(
                    query=query,
                    top_k=k,
                    score_threshold=self.score_threshold,
                    namespace=namespace,
                    topic_hash=topic_hash,
                )
        except Exception as exc:
            logger.warning(f"KB search failed: {exc}")
            docs = []

        docs = self._dedupe_docs(docs)
        if not docs and use_hybrid:
            try:
                docs = await vector_search(
                    query=query,
                    top_k=max(k, 8),
                    score_threshold=max(0.0, min(self.score_threshold, 0.15)),
                    namespace=namespace,
                    topic_hash=topic_hash,
                )
                docs = self._dedupe_docs(docs)
                if docs:
                    logger.info(
                        "KB hybrid fallback activated: retrieved=%s namespace=%s",
                        len(docs),
                        namespace or "*",
                    )
            except Exception as exc:
                logger.warning(f"KB hybrid fallback search failed: {exc}")

        if not docs:
            return {
                "question": query,
                "answer": "知识库中未检索到足够相关内容，请先运行研究流程建立索引。",
                "citations": [],
                "retrieved_count": 0,
                "highlights": [],
                "limitations": ["no evidence retrieved"],
                "follow_up_queries": [query],
            }

        context = self._format_context(docs)
        response = await self.llm.acomplete(
            [
                Message.system(_CHAT_SYSTEM_PROMPT),
                Message.user(
                    f"用户问题: {query}\n\n"
                    "证据片段如下：\n"
                    f"{context}\n\n"
                    "请输出 JSON。"
                ),
            ]
        )
        parsed = self._extract_json(response.content)

        answer = str(parsed.get("answer", "")).strip() or "基于当前检索证据，无法给出稳定结论。"
        highlights = [
            str(item).strip()
            for item in (parsed.get("highlights") or [])
            if str(item).strip()
        ]
        limitations = [
            str(item).strip()
            for item in (parsed.get("limitations") or [])
            if str(item).strip()
        ]
        follow_up = [
            str(item).strip()
            for item in (parsed.get("follow_up_queries") or [])
            if str(item).strip()
        ]

        citations: List[Dict[str, Any]] = []
        for item in docs[: min(8, len(docs))]:
            meta = item.get("metadata", {}) or {}
            citations.append(
                {
                    "id": item.get("id"),
                    "source": meta.get("source"),
                    "url": meta.get("url"),
                    "score": item.get("score"),
                }
            )

        return {
            "question": query,
            "answer": answer,
            "highlights": highlights,
            "limitations": limitations,
            "follow_up_queries": follow_up,
            "citations": citations,
            "retrieved_count": len(docs),
            "strategy": "chat_over_kb",
        }
