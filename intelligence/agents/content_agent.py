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
TIMELINE_PROMPT = """你是一个技术历史学家。请根据以下事实，梳理 "{topic}" 的发展时间轴。

## 事实列表
{facts}

## 要求
1. 提取关键里程碑事件
2. 按时间排序
3. 标注重要性 (1-5, 5最重要)
4. 每个事件包含日期、标题、描述
5. 描述中要体现“技术变化点”（如架构改动、训练范式变化、部署影响）

请以 JSON 格式输出：
{{
    "events": [
        {{
            "date": "2024-01",
            "title": "事件标题",
            "description": "详细描述",
            "importance": 5,
            "source_refs": ["fact_id"]
        }}
    ]
}}
"""


# One-Pager 生成 Prompt  
ONE_PAGER_PROMPT = """你是一个资深研究工程师。请根据以下事实，生成关于 "{topic}" 的技术一页纸摘要。

## 事实列表
{facts}

## 要求
1. 开头一句话总结（executive summary）
2. 至少 5 个核心发现（优先包含定量结论）
3. 关键性能指标（尽量给 benchmark、吞吐、延迟、成本、参数规模）
4. 优势和劣势分析
5. 技术深潜：关键机制、失败模式、可扩展边界
6. 实现建议：工程落地步骤、硬件与系统依赖
7. 风险与缓解：至少 3 条
8. 相关资源链接（论文/代码/文档）
9. resources 里只能使用真实可访问的 http(s) 链接；不要输出“请搜索/待补充/占位符”文本
10. 若无法确认链接，请省略该条资源，不要编造
11. technical_deep_dive 必须包含“实现细节 + 设计考量 + 失效边界”，至少 3 条
12. key_findings 至少 2 条为“横向对比”（与其他技术/方案相比的差异、收益与代价）
13. implementation_notes 必须包含可执行步骤（配置、依赖、监控指标、回滚策略）
14. 严禁泛泛而谈或营销措辞，优先写可验证、可复现、可度量结论

请以 JSON 格式输出：
{{
    "title": "...",
    "executive_summary": "一句话总结",
    "key_findings": ["发现1", "发现2", "发现3", "发现4", "发现5"],
    "metrics": {{"指标名": "指标值"}},
    "strengths": ["优势1", "优势2"],
    "weaknesses": ["劣势1", "劣势2"],
    "technical_deep_dive": ["机制/公式/系统细节1", "细节2"],
    "implementation_notes": ["工程建议1", "工程建议2"],
    "risks_and_mitigations": ["风险1 -> 缓解方案1", "风险2 -> 缓解方案2"],
    "resources": [{{"title": "资源名", "url": "链接"}}]
}}
"""


# Video Brief 生成 Prompt
VIDEO_BRIEF_PROMPT = """你是一个技术视频导演+编导。请根据以下事实，生成关于 "{topic}" 的视频简报脚本。

## 事实列表
{facts}

## 要求
1. 设计一个吸引人的开场钩子
2. 分成 3-5 个内容段落
3. 每个段落有标题、核心内容、talking points、镜头/视觉提示词
4. 结尾有总结和行动号召
5. 风格：专业但易懂，强调技术细节和工程取舍
6. 每个段落给出建议时长（秒），总时长 120-360 秒
7. visual_prompt 要可直接用于文生视频模型，包含镜头语言、场景、风格、画面元素
8. 每个 segment 必须包含 duration_sec(int) 和 visual_prompt(non-empty string)
9. 段落结构必须覆盖：机制实现 -> 指标/实验 -> 横向对比 -> 局限性/风险 -> 工程落地
10. talking_points 必须包含可量化或可执行信息（指标、参数、代价、决策依据），避免空泛描述
11. 至少 1 个段落明确说明“与主流替代方案的对比与取舍”

请以 JSON 格式输出：
{{
    "title": "视频标题",
    "duration_estimate": "3-6 minutes",
    "hook": "开场钩子（吸引观众注意力的问题或观点）",
    "target_audience": "目标受众",
    "visual_style": "视觉风格说明",
    "segments": [
        {{
            "title": "段落标题",
            "content": "主要内容",
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
    ):
        self.llm = llm or get_llm()

    async def generate(
        self,
        topic: str,
        facts: List[Dict[str, Any]],
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
        
        if not facts:
            logger.warning("No facts to generate content from")
            return {
                "phase": AgentPhase.GENERATING,
                "timeline": None,
                "one_pager": None,
                "video_brief": None,
            }
        
        # 准备事实上下文
        facts_context = self._format_facts(facts)
        
        # 并行生成三种内容
        logger.info("Generating content in parallel...")
        
        timeline_task = self._generate_timeline(topic, facts_context)
        one_pager_task = self._generate_one_pager(topic, facts_context)
        video_brief_task = self._generate_video_brief(topic, facts_context)
        
        results = await asyncio.gather(
            timeline_task,
            one_pager_task,
            video_brief_task,
            return_exceptions=True,
        )
        
        timeline = self._unwrap_result("Timeline", results[0])
        one_pager = self._unwrap_result("One-pager", results[1])
        video_brief = self._unwrap_result("Video brief", results[2])
        
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
        response = await self.llm.achat(prompt)
        try:
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning(f"{label} parse error: {exc}")
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
        return "\n".join(formatted)
    
    async def _generate_timeline(
        self, 
        topic: str, 
        facts_context: str
    ) -> Optional[List[Dict]]:
        """生成时间轴"""
        prompt = TIMELINE_PROMPT.format(topic=topic, facts=facts_context)
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
        facts_context: str
    ) -> Optional[Dict]:
        """生成一页纸摘要"""
        prompt = ONE_PAGER_PROMPT.format(topic=topic, facts=facts_context)
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
        facts_context: str
    ) -> Optional[Dict]:
        """生成视频简报"""
        prompt = VIDEO_BRIEF_PROMPT.format(topic=topic, facts=facts_context)
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
        facts: List[Dict]
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
        handlers = {
            "timeline": self._generate_timeline,
            "one_pager": self._generate_one_pager,
            "video_brief": self._generate_video_brief,
        }
        handler = handlers.get(output_type)
        if not handler:
            raise ValueError(f"Unknown output type: {output_type}")
        return await handler(topic, facts_context)
