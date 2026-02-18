"""
Content Agent
多模态内容官 - 并行生成 Timeline, One-Pager, Video Brief
"""
from typing import List, Dict, Any, Optional
import json
import logging
import asyncio
import re

from intelligence.llm import BaseLLM, get_llm, Message
from intelligence.state import (
    AgentPhase,
    TimelineEvent,
    OnePager,
    VideoBrief,
)


logger = logging.getLogger(__name__)


# Timeline 生成 Prompt
TIMELINE_PROMPT = """你是一个技术历史学家。请根据给定事实与证据，梳理 "{topic}" 的发展时间轴。

## 事实列表
{facts}

## 检索证据摘录
{evidence}

## 要求
1. 提取关键里程碑事件，按时间排序。
2. 每个事件包含 date/title/description/importance/source_refs。
3. date 只允许使用可验证日期（YYYY 或 YYYY-MM 或 YYYY-MM-DD）；无法确认时填写 "Unknown"。
4. description 必须体现“技术变化点”（架构、训练范式、指标、部署）。
5. source_refs 只能引用给定 evidence_ids/fact_id；不允许编造来源。
6. 如果证据不足，请显式写出“证据不足点”，不要补全臆测内容。

请只输出 JSON：
{{
  "events": [
    {{
      "date": "2024-01",
      "title": "事件标题",
      "description": "详细描述（含证据边界）",
      "importance": 5,
      "source_refs": ["fact_id_or_evidence_id"]
    }}
  ]
}}
"""


# One-Pager 生成 Prompt
ONE_PAGER_PROMPT = """你是一个资深研究工程师。请根据给定事实与证据，生成 "{topic}" 的技术一页纸摘要。

## 事实列表
{facts}

## 检索证据摘录
{evidence}

## 生成约束
1. 只使用可验证证据，不得编造论文结论、指标、链接。
2. executive_summary 必须是可核验结论，不允许空泛口号。
3. key_findings 至少 6 条，优先定量结论；其中至少 2 条是横向对比（收益 + 代价）。
4. technical_deep_dive 至少 4 条，必须覆盖“实现细节 + 设计考量 + 失效边界”。
5. implementation_notes 至少 4 条，必须是可执行步骤（配置/依赖/监控/回滚）。
6. risks_and_mitigations 至少 3 条，格式建议“风险 -> 缓解方案”。
7. metrics 必须包含：SOTA_Metric, Hardware_Requirement, Core_Formula, Key_Optimization。
8. resources 仅允许真实可访问 http(s) URL；无法确认就省略，不要占位符。
9. 如果某项证据不足，明确写“证据不足 + 需要补充的信息”，不要硬填。

请只输出 JSON：
{{
  "title": "...",
  "executive_summary": "一句话总结",
  "key_findings": ["发现1", "发现2", "发现3", "发现4", "发现5", "发现6"],
  "metrics": {{
    "SOTA_Metric": "...",
    "Hardware_Requirement": "...",
    "Core_Formula": "...",
    "Key_Optimization": "..."
  }},
  "strengths": ["优势1", "优势2"],
  "weaknesses": ["劣势1", "劣势2"],
  "technical_deep_dive": ["机制/公式/系统细节1", "细节2", "细节3", "细节4"],
  "implementation_notes": ["工程建议1", "工程建议2", "工程建议3", "工程建议4"],
  "risks_and_mitigations": ["风险1 -> 缓解1", "风险2 -> 缓解2", "风险3 -> 缓解3"],
  "resources": [{{"title": "资源名", "url": "https://..."}}]
}}
"""


# Video Brief 生成 Prompt
VIDEO_BRIEF_PROMPT = """你是一个技术视频导演+编导。请根据给定事实与证据，生成 "{topic}" 的视频简报脚本。

## 事实列表
{facts}

## 检索证据摘录
{evidence}

## 生成约束
1. 只使用可验证证据，不得补全缺失信息。
2. segments 数量 2-6（证据不足时允许 1 个“证据缺口与补证路径”段落）。
3. 每个 segment 必须包含 title/content/talking_points/duration_sec/visual_prompt。
4. talking_points 必须尽量量化或可执行（指标、参数、代价、决策依据）。
5. 至少 1 个段落覆盖“与替代方案的对比与取舍”；若证据不足须明确说明不足点。
6. 总时长建议 120-360 秒；证据不足时可更短，但必须显式说明原因。
7. 禁止营销话术和想象性叙述，优先工程细节与证据边界。

请只输出 JSON：
{{
  "title": "视频标题",
  "duration_estimate": "3-6 minutes",
  "hook": "开场钩子（可验证问题）",
  "target_audience": "目标受众",
  "visual_style": "视觉风格说明",
  "segments": [
    {{
      "title": "段落标题",
      "content": "主要内容（含证据边界）",
      "talking_points": ["要点1", "要点2"],
      "duration_sec": 45,
      "visual_prompt": "cinematic technical explainer shot ..."
    }}
  ],
  "conclusion": "总结",
  "call_to_action": "行动号召"
}}
"""


class ContentAgent:
    """
    多模态内容官
    
    负责并行生成：
    1. Timeline - 技术演进时间轴
    2. One-Pager - 一页纸摘要
    3. Video Brief - 视频简报脚本
    """
    
    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        *,
        request_timeout_sec: int = 35,
    ):
        self.llm = llm or get_llm()
        self.request_timeout_sec = max(10, int(request_timeout_sec))

    async def generate(
        self,
        topic: str,
        facts: List[Dict[str, Any]],
        *,
        search_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        便捷方法：根据事实生成 Timeline / One-Pager / Video Brief

        Args:
            topic: 主题
            facts: 事实列表（dict）

        Returns:
            {"timeline": ..., "one_pager": ..., "video_brief": ...}
        """
        return await self.run(
            {
                "topic": topic,
                "facts": facts,
                "search_results": list(search_results or []),
            }
        )
    
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行内容生成任务
        
        使用 asyncio.gather 并行生成三种内容
        
        Args:
            state: AgentState 字典
            
        Returns:
            更新后的状态
        """
        topic = state["topic"]
        facts = state.get("facts", [])
        search_results = list(state.get("search_results", []) or [])

        if not facts and not search_results:
            logger.warning("No facts or search results to generate content from")
            return {
                "phase": AgentPhase.GENERATING,
                "timeline": None,
                "one_pager": None,
                "video_brief": None,
            }

        # 准备事实上下文与证据上下文
        facts_context = self._format_facts(facts)
        evidence_context = self._format_search_evidence(search_results)

        # 并行生成三种内容
        logger.info("Generating content in parallel...")

        timeline_task = self._generate_timeline(topic, facts_context, evidence_context)
        one_pager_task = self._generate_one_pager(topic, facts_context, evidence_context)
        video_brief_task = self._generate_video_brief(topic, facts_context, evidence_context)
        
        results = await asyncio.gather(
            timeline_task,
            one_pager_task,
            video_brief_task,
            return_exceptions=True,
        )
        
        timeline = self._unwrap_result("Timeline", results[0])
        one_pager = self._unwrap_result("One-pager", results[1])
        video_brief = self._unwrap_result("Video brief", results[2])

        if not timeline or not one_pager or not video_brief:
            logger.info("Retrying missing content sections sequentially...")
            if not timeline:
                timeline = await self._generate_timeline(topic, facts_context, evidence_context)
            if not one_pager:
                one_pager = await self._generate_one_pager(topic, facts_context, evidence_context)
            if not video_brief:
                video_brief = await self._generate_video_brief(topic, facts_context, evidence_context)

        if not timeline:
            timeline = self._fallback_timeline(topic=topic, facts=facts, search_results=search_results)
        if not one_pager:
            one_pager = self._fallback_one_pager(topic=topic, facts=facts, search_results=search_results)
        if not video_brief:
            video_brief = self._fallback_video_brief(topic=topic, facts=facts, search_results=search_results)
        
        logger.info("Content generation completed")
        
        return {
            "phase": AgentPhase.GENERATING,
            "timeline": timeline,
            "one_pager": one_pager,
            "video_brief": video_brief,
        }

    @staticmethod
    def _unwrap_result(label: str, result: Any) -> Any:
        if isinstance(result, Exception):
            logger.error(f"{label} generation failed: {result}")
            return None
        return result

    async def _request_json(self, prompt: str, label: str) -> Optional[Dict[str, Any]]:
        max_tokens_candidates = [1800, 1200]
        for idx, max_tokens in enumerate(max_tokens_candidates, start=1):
            prompt_body = prompt
            if idx > 1:
                prompt_body = (
                    f"{prompt}\n\n"
                    "请只返回一个合法 JSON 对象，"
                    "不要包含 ```、注释、解释文本、前后缀。"
                )
            try:
                response = await asyncio.wait_for(
                    self.llm.acomplete(
                        [Message.user(prompt_body)],
                        temperature=0.2,
                        max_tokens=max_tokens,
                    ),
                    timeout=float(self.request_timeout_sec),
                )
            except Exception as exc:
                logger.warning(f"{label} request failed (attempt={idx}): {exc}")
                continue
            try:
                content = response.content or ""
                parsed = self._extract_json_dict(content)
                if parsed is not None:
                    return parsed
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning(f"{label} parse error (attempt={idx}): {exc}")
        return None

    def _extract_json_dict(self, content: str) -> Optional[Dict[str, Any]]:
        text = str(content or "").strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        starts = [idx for idx, ch in enumerate(text) if ch == "{"]
        for start in starts:
            depth = 0
            for end in range(start, len(text)):
                ch = text[end]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : end + 1]
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, dict):
                                return parsed
                        except Exception:
                            break
        return None
    
    def _format_facts(self, facts: List[Dict]) -> str:
        """格式化事实列表"""
        formatted = []
        for i, f in enumerate(facts, 1):
            evidence = ", ".join([str(x).strip() for x in (f.get("evidence", []) or []) if str(x).strip()]) or "N/A"
            formatted.append(
                f"{i}. [{f.get('category', 'other')}] {f.get('claim', '')}\n"
                f"   置信度: {f.get('confidence', 0):.1%} | 来源: {f.get('source_type', 'unknown')} | evidence_ids: {evidence}"
            )
        return "\n".join(formatted) if formatted else "N/A"

    @staticmethod
    def _source_priority(source: str) -> int:
        ranking = {
            "arxiv": 0,
            "semantic_scholar": 1,
            "openreview": 2,
            "github": 3,
            "huggingface": 4,
            "stackoverflow": 5,
            "hackernews": 6,
            "reddit": 7,
            "twitter": 8,
        }
        return ranking.get(str(source or "").strip().lower(), 9)

    def _infer_category(self, text: Any) -> str:
        lowered = str(text or "").lower()
        hints = {
            "architecture": ("architecture", "mechanism", "pipeline", "协议", "架构", "机制", "链路"),
            "performance": ("benchmark", "latency", "throughput", "speed", "性能", "延迟", "吞吐"),
            "training": ("train", "dataset", "optimizer", "loss", "训练", "数据集", "损失"),
            "comparison": ("versus", "vs", "trade-off", "compare", "对比", "取舍"),
            "deployment": ("deploy", "production", "rollback", "monitor", "部署", "回滚", "监控"),
            "limitation": ("risk", "limitation", "failure", "issue", "风险", "局限", "失效"),
        }
        for category, tokens in hints.items():
            if any(token in lowered for token in tokens):
                return category
        return "community"

    def _metadata_signals(self, metadata: Dict[str, Any]) -> List[str]:
        pairs: List[str] = []
        if not metadata:
            return pairs
        fields = (
            ("citation_count", "citations"),
            ("stars", "stars"),
            ("downloads", "downloads"),
            ("score", "score"),
            ("points", "points"),
            ("published_date", "date"),
            ("year", "year"),
            ("language", "language"),
        )
        for key, label in fields:
            value = metadata.get(key)
            text = str(value).strip()
            if not text or text.lower() in {"none", "null", "nan"}:
                continue
            pairs.append(f"{label}={text}")
            if len(pairs) >= 4:
                break
        return pairs

    def _search_item_score(self, item: Dict[str, Any]) -> float:
        def _to_float(value: Any, default: float = 0.0) -> float:
            try:
                return float(value)
            except Exception:
                return default

        source = str(item.get("source", "")).strip().lower()
        metadata = dict(item.get("metadata", {}) or {})
        title = str(item.get("title", "")).strip()
        content = str(item.get("content", "")).strip()
        score = float(10 - self._source_priority(source))
        if re.search(r"\d", f"{title} {content}"):
            score += 1.2
        if len(content) >= 160:
            score += 0.6
        score += min(3.5, _to_float(metadata.get("citation_count"), 0.0) / 120.0)
        score += min(2.5, _to_float(metadata.get("stars"), 0.0) / 900.0)
        score += min(2.0, _to_float(metadata.get("downloads"), 0.0) / 20000.0)
        return score

    def _search_result_claim(self, item: Dict[str, Any]) -> str:
        source = str(item.get("source", "")).strip().lower() or "unknown"
        title = self._clean_claim(item.get("title", ""), max_len=130)
        content = self._clean_claim(item.get("content", ""), max_len=260)
        snippet = content
        if snippet:
            snippet = re.split(r"[。.!?；;\n]+", snippet)[0]
            snippet = self._clean_claim(snippet, max_len=180)
        metadata = dict(item.get("metadata", {}) or {})
        signals = self._metadata_signals(metadata)
        signal_text = f" | {', '.join(signals)}" if signals else ""
        prefix = f"[{source}] {title}" if title else f"[{source}]"
        body = snippet or self._clean_claim(content, max_len=180)
        return self._clean_claim(f"{prefix}{signal_text}: {body}", max_len=240)

    def _format_search_evidence(self, search_results: List[Dict[str, Any]], max_items: int = 28) -> str:
        if not search_results:
            return "N/A"
        ranked = sorted(
            [item for item in search_results if isinstance(item, dict)],
            key=lambda item: (
                self._source_priority(str(item.get("source", ""))),
                -self._search_item_score(item),
            ),
        )
        lines: List[str] = []
        seen = set()
        for item in ranked:
            claim = self._search_result_claim(item)
            if not claim:
                continue
            key = re.sub(r"\s+", "", claim.lower())
            if not key or key in seen:
                continue
            seen.add(key)
            rid = str(item.get("id", "")).strip()
            rid_text = f"id={rid} | " if rid else ""
            lines.append(f"- {rid_text}{claim}")
            if len(lines) >= max_items:
                break
        return "\n".join(lines) if lines else "N/A"

    def _grouped_claims_for_fallback(
        self,
        *,
        facts: List[Dict[str, Any]],
        search_results: List[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        grouped = self._facts_by_category(facts)
        ranked_results = sorted(
            [item for item in search_results if isinstance(item, dict)],
            key=lambda item: (-self._search_item_score(item), self._source_priority(str(item.get("source", "")))),
        )
        for item in ranked_results[:24]:
            claim = self._search_result_claim(item)
            if not claim:
                continue
            category = self._infer_category(f"{item.get('title', '')} {item.get('content', '')}")
            grouped.setdefault(category, [])
            if claim not in grouped[category]:
                grouped[category].append(claim)
        return grouped

    def _clean_claim(self, value: Any, max_len: int = 220) -> str:
        text = str(value or "").strip()
        text = re.sub(r"\s+", " ", text)
        if len(text) > max_len:
            text = text[: max_len - 3] + "..."
        return text

    def _facts_by_category(self, facts: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        grouped: Dict[str, List[str]] = {}
        for item in facts:
            category = str(item.get("category", "other")).strip().lower() or "other"
            claim = self._clean_claim(item.get("claim", ""))
            if not claim:
                continue
            grouped.setdefault(category, [])
            if claim not in grouped[category]:
                grouped[category].append(claim)
        return grouped

    def _extract_metrics(self, claims: List[str]) -> Dict[str, str]:
        metrics: Dict[str, str] = {}
        patterns = [
            ("latency", r"(\d+(?:\.\d+)?)\s*(ms|s|秒)"),
            ("throughput", r"(\d+(?:\.\d+)?)\s*(rps|qps|req/s|requests/s)"),
            ("cost", r"(\$ ?\d+(?:\.\d+)?)"),
            ("accuracy", r"(\d+(?:\.\d+)?)\s*%"),
            ("context_window", r"(\d+(?:\.\d+)?)\s*(k|K)\s*token"),
        ]
        for claim in claims:
            lower = claim.lower()
            for key, pattern in patterns:
                if key in metrics:
                    continue
                match = re.search(pattern, lower)
                if not match:
                    continue
                metrics[key] = match.group(0)
        return metrics

    def _fallback_timeline(
        self,
        *,
        topic: str,
        facts: List[Dict[str, Any]],
        search_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        grouped = self._grouped_claims_for_fallback(facts=facts, search_results=search_results)
        category_order = [
            ("architecture", "架构机制"),
            ("deployment", "部署路径"),
            ("performance", "性能验证"),
            ("comparison", "方案对比"),
            ("limitation", "风险与边界"),
        ]
        events: List[Dict[str, Any]] = []
        phase = 1
        for category, title in category_order:
            for claim in grouped.get(category, [])[:1]:
                events.append(
                    TimelineEvent(
                        date=f"Phase-{phase}",
                        title=title,
                        description=claim,
                        importance=max(1, 6 - phase),
                        source_refs=[],
                    ).to_dict()
                )
                phase += 1
        if not events:
            events.append(
                TimelineEvent(
                    date="Phase-1",
                    title="核心结论",
                    description=f"{topic} 的结构化证据不足，建议补充工程实现与性能数据后重试。",
                    importance=3,
                    source_refs=[],
                ).to_dict()
            )
        return events

    def _fallback_one_pager(
        self,
        *,
        topic: str,
        facts: List[Dict[str, Any]],
        search_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        grouped = self._grouped_claims_for_fallback(facts=facts, search_results=search_results)
        all_claims = [claim for claims in grouped.values() for claim in claims]
        key_findings = all_claims[:6] or [f"{topic} 缺少可验证事实，建议先扩展检索源。"]
        metrics = self._extract_metrics(all_claims)
        strengths = grouped.get("architecture", [])[:2] + grouped.get("performance", [])[:1]
        weaknesses = grouped.get("limitation", [])[:3]
        deep_dive = (
            grouped.get("architecture", [])[:2]
            + grouped.get("deployment", [])[:1]
            + grouped.get("training", [])[:1]
        )
        implementation_notes: List[str] = []
        for category in ("deployment", "architecture", "performance", "comparison", "limitation"):
            for claim in grouped.get(category, [])[:2]:
                cleaned = self._clean_claim(claim, max_len=180)
                if cleaned and cleaned not in implementation_notes:
                    implementation_notes.append(cleaned)
                if len(implementation_notes) >= 4:
                    break
            if len(implementation_notes) >= 4:
                break
        if not implementation_notes:
            implementation_notes = ["证据不足，建议优先补齐部署指标、性能口径和可复现实验步骤。"]
        risks = weaknesses or grouped.get("community", [])[:2]
        if not risks:
            risks = ["证据覆盖不足，容易出现结论漂移 -> 通过补充多源对齐和复验缓解。"]

        return OnePager(
            title=f"{topic} One-Pager",
            executive_summary=key_findings[0],
            key_findings=key_findings,
            metrics=metrics,
            strengths=strengths[:3],
            weaknesses=weaknesses[:3],
            technical_deep_dive=deep_dive[:4] or key_findings[:2],
            implementation_notes=implementation_notes[:4],
            risks_and_mitigations=risks[:4],
            resources=[],
        ).to_dict()

    def _fallback_video_brief(
        self,
        *,
        topic: str,
        facts: List[Dict[str, Any]],
        search_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        grouped = self._grouped_claims_for_fallback(facts=facts, search_results=search_results)
        segment_plan = [
            ("architecture", "架构与协议流程"),
            ("deployment", "部署拓扑与运行时策略"),
            ("performance", "性能指标与容量规划"),
            ("comparison", "替代方案对比与取舍"),
            ("limitation", "风险、攻击面与防护"),
        ]
        segments: List[Dict[str, Any]] = []
        for category, title in segment_plan:
            claims = grouped.get(category, [])
            if not claims:
                continue
            core = claims[0]
            segments.append(
                {
                    "title": title,
                    "content": core,
                    "talking_points": claims[:3],
                    "duration_sec": 45,
                    "visual_prompt": (
                        f"{topic} technical explainer, {title}, "
                        "architecture diagram, production dashboard, metrics overlay"
                    ),
                }
            )
        if not segments:
            segments = [
                {
                    "title": "核心问题与边界",
                    "content": f"{topic} 当前证据不足，需要继续补充工程实现与性能验证。",
                    "talking_points": ["补齐检索源", "增强证据交叉验证", "先做小规模演练"],
                    "duration_sec": 45,
                    "visual_prompt": "technical dashboard and architecture whiteboard",
                }
            ]

        if len(segments) <= 1:
            segments.append(
                {
                    "title": "证据缺口与补证路径",
                    "content": f"{topic} 当前证据链不完整，需要补齐实验条件、对照基线与生产监控数据。",
                    "talking_points": [
                        "补充可复现实验与对照组口径",
                        "补齐线上指标与回滚策略",
                        "明确结论适用边界后再扩展场景",
                    ],
                    "duration_sec": 35,
                    "visual_prompt": "evidence board, missing data map, technical checklist",
                }
            )

        return VideoBrief(
            title=f"{topic} Video Brief",
            duration_estimate="2-6 minutes",
            hook=f"{topic} 在生产环境中，哪些设计会直接决定可用性与成本？",
            segments=segments[:5],
            target_audience="平台工程师 / 研究工程师",
            visual_style="evidence-driven technical explainer",
            conclusion="先用可验证指标固化决策，再逐步放量。",
            call_to_action="按模块执行灰度验证并记录回归结果。",
        ).to_dict()
    
    async def _generate_timeline(
        self,
        topic: str,
        facts_context: str,
        evidence_context: str,
    ) -> Optional[List[Dict]]:
        """生成时间轴"""
        prompt = TIMELINE_PROMPT.format(
            topic=topic,
            facts=facts_context,
            evidence=evidence_context,
        )
        data = await self._request_json(prompt, "Timeline")
        if data:
            events = data.get("events", [])
            return [
                TimelineEvent(
                    date=e.get("date", ""),
                    title=e.get("title", ""),
                    description=e.get("description", ""),
                    importance=int(e.get("importance", 3)),
                    source_refs=e.get("source_refs", []),
                ).to_dict()
                for e in events
            ]
        return None
    
    async def _generate_one_pager(
        self,
        topic: str,
        facts_context: str,
        evidence_context: str,
    ) -> Optional[Dict]:
        """生成一页纸摘要"""
        prompt = ONE_PAGER_PROMPT.format(
            topic=topic,
            facts=facts_context,
            evidence=evidence_context,
        )
        data = await self._request_json(prompt, "One-pager")
        if data:
            return OnePager(
                title=data.get("title", topic),
                executive_summary=data.get("executive_summary", ""),
                key_findings=data.get("key_findings", []),
                metrics=data.get("metrics", {}),
                strengths=data.get("strengths", []),
                weaknesses=data.get("weaknesses", []),
                technical_deep_dive=data.get("technical_deep_dive", []),
                implementation_notes=data.get("implementation_notes", []),
                risks_and_mitigations=data.get("risks_and_mitigations", []),
                resources=data.get("resources", []),
            ).to_dict()
        return None
    
    async def _generate_video_brief(
        self,
        topic: str,
        facts_context: str,
        evidence_context: str,
    ) -> Optional[Dict]:
        """生成视频简报"""
        prompt = VIDEO_BRIEF_PROMPT.format(
            topic=topic,
            facts=facts_context,
            evidence=evidence_context,
        )
        data = await self._request_json(prompt, "Video brief")
        if data:
            return VideoBrief(
                title=data.get("title", topic),
                duration_estimate=data.get("duration_estimate", "5-7 minutes"),
                hook=data.get("hook", ""),
                segments=data.get("segments", []),
                target_audience=data.get("target_audience", ""),
                visual_style=data.get("visual_style", ""),
                conclusion=data.get("conclusion", ""),
                call_to_action=data.get("call_to_action", ""),
            ).to_dict()
        return None
    
    async def generate_single(
        self,
        output_type: str,
        topic: str,
        facts: List[Dict],
        *,
        search_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict]:
        """
        生成单个输出类型
        
        Args:
            output_type: timeline, one_pager, video_brief
            topic: 主题
            facts: 事实列表
            
        Returns:
            生成的内容
        """
        facts_context = self._format_facts(facts)
        evidence_context = self._format_search_evidence(list(search_results or []))
        handlers = {
            "timeline": self._generate_timeline,
            "one_pager": self._generate_one_pager,
            "video_brief": self._generate_video_brief,
        }
        handler = handlers.get(output_type)
        if not handler:
            raise ValueError(f"Unknown output type: {output_type}")
        return await handler(topic, facts_context, evidence_context)
