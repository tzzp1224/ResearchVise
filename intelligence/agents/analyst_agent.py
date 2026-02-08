"""
Analyst Agent
学术分析师 - RAG 驱动的事实核验和分析
"""
from typing import Awaitable, Callable, List, Dict, Any, Optional
import json
import logging
import hashlib

from intelligence.llm import BaseLLM, get_llm, Message
from intelligence.state import AgentPhase, Fact
from intelligence.tools.rag_tools import (
    add_to_knowledge_base,
    vector_search,
    hybrid_search,
)


logger = logging.getLogger(__name__)


ANALYST_SYSTEM_PROMPT = """你是一个严谨的学术分析师。你的任务是：
1. 分析搜集到的多源信息
2. 提取可验证的事实
3. 评估每个事实的置信度
4. 区分"论文证据"和"社区观点"

## 分析原则
- 优先采信学术论文（arxiv）的内容
- 社交媒体内容需要与论文交叉验证
- 标注每个事实的来源类型和置信度
- 识别信息之间的矛盾或不一致

## 事实分类
请将提取的事实归类为：
- architecture: 模型架构相关
- performance: 性能指标相关
- training: 训练方法相关
- deployment: 部署应用相关
- comparison: 与其他方法的比较
- limitation: 已知限制或问题
- community: 社区反馈和观点

## 输出格式
对于每个事实，请提供：
1. claim: 事实陈述（一句话）
2. evidence: 支持该事实的来源ID列表（必须来自给定 source_id）
3. confidence: 置信度 (0.0-1.0)
4. source_type: paper/social/code
5. category: 分类

请优先输出“可操作、可复现、可量化”的事实。
"""


class AnalystAgent:
    """
    学术分析师
    
    负责：
    1. 将搜索结果存入知识库
    2. 分析信息，提取事实
    3. 交叉验证，评估置信度
    """
    
    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        *,
        enable_knowledge_indexing: bool = True,
        knowledge_indexer: Optional[
            Callable[[List[Dict[str, Any]]], Awaitable[int]]
        ] = None,
    ):
        self.llm = llm or get_llm()
        self.enable_knowledge_indexing = enable_knowledge_indexing
        self._knowledge_indexer = knowledge_indexer or add_to_knowledge_base

    async def analyze(
        self,
        topic: str,
        search_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        便捷方法：对给定搜索结果进行事实提取与缺口分析

        Args:
            topic: 研究主题
            search_results: 搜索结果列表（dict）

        Returns:
            {"facts": [...], "knowledge_gaps": [...]} 等字段
        """
        return await self.run(
            {
                "topic": topic,
                "search_results": search_results,
            }
        )
    
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行分析任务
        
        Args:
            state: AgentState 字典
            
        Returns:
            更新后的状态
        """
        topic = state["topic"]
        search_results = state.get("search_results", [])
        
        if not search_results:
            logger.warning("No search results to analyze")
            return {
                "phase": AgentPhase.ANALYZING,
                "facts": [],
                "knowledge_gaps": ["未找到任何搜索结果"],
            }
        
        # 1. 将搜索结果存入知识库（可选，失败时降级）
        if self.enable_knowledge_indexing:
            logger.info(f"Indexing {len(search_results)} search results...")
            documents = [
                {
                    "id": r.get("id", ""),
                    "content": f"{r.get('title', '')}\n\n{r.get('content', '')}",
                    "metadata": {
                        "source": r.get("source", "unknown"),
                        "url": r.get("url", ""),
                        **{k: v for k, v in r.get("metadata", {}).items() if v is not None},
                    },
                }
                for r in search_results
            ]
            try:
                await self._knowledge_indexer(documents)
            except Exception as e:
                logger.warning(f"Knowledge indexing skipped due to error: {e}")
        
        # 2. 准备分析上下文
        context = self._prepare_context(search_results)
        
        # 3. 调用 LLM 分析
        messages = [
            Message.system(ANALYST_SYSTEM_PROMPT),
            Message.user(f"""
请分析以下关于 "{topic}" 的多源信息，提取关键事实。

## 搜集到的信息
{context}

## 任务
1. 提取 8-20 个关键事实，至少覆盖 architecture/performance/training/comparison/limitation 五类
2. 为每个事实评估置信度，并引用 1-3 个 evidence source_id
3. 指出信息缺口（还需要了解什么）
4. 优先保留定量陈述（参数规模、benchmark、吞吐/延迟、成本、数据规模）

请以 JSON 格式返回：
{{
    "facts": [
        {{
            "claim": "...",
            "evidence": ["source_id_1", "source_id_2"],
            "confidence": 0.9,
            "source_type": "paper",
            "category": "architecture"
        }}
    ],
    "knowledge_gaps": ["...", "..."]
}}
"""),
        ]
        
        response = await self.llm.acomplete(messages)
        
        # 4. 解析响应
        facts, knowledge_gaps = self._parse_analysis(response.content, search_results)
        
        logger.info(f"Extracted {len(facts)} facts, {len(knowledge_gaps)} knowledge gaps")
        
        return {
            "phase": AgentPhase.ANALYZING,
            "facts": [f.to_dict() for f in facts],
            "knowledge_gaps": knowledge_gaps,
        }
    
    def _prepare_context(self, results: List[Dict], max_tokens: int = 8000) -> str:
        """准备分析上下文"""
        # 按来源分组
        by_source = {}
        for r in results:
            source = r.get("source", "unknown")
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(r)
        
        # 格式化
        context_parts = []
        
        # 优先显示学术来源
        source_order = ["arxiv", "huggingface", "github", "reddit", "twitter"]
        
        for source in source_order:
            if source not in by_source:
                continue
            
            items = by_source[source]
            context_parts.append(f"\n### {source.upper()} ({len(items)} 条)")
            
            for item in items[:5]:  # 每个来源最多5条
                metadata = item.get("metadata", {}) or {}
                meta_pairs = [
                    f"{k}={v}"
                    for k, v in metadata.items()
                    if v is not None and str(v).strip() != ""
                ][:6]
                metadata_line = f"\nmetadata: {', '.join(meta_pairs)}" if meta_pairs else ""
                context_parts.append(f"""
**[{item.get('id')}] {item.get('title', 'Untitled')}**
{item.get('content', '')[:700]}{metadata_line}
""")
        
        context = "\n".join(context_parts)
        
        # 简单截断 (实际应该用 token 计数)
        if len(context) > max_tokens * 4:  # 粗略估计
            context = context[:max_tokens * 4] + "\n\n...(内容已截断)"
        
        return context
    
    def _parse_analysis(
        self, 
        content: str, 
        search_results: List[Dict]
    ) -> tuple[List[Fact], List[str]]:
        """解析分析结果"""
        facts = []
        knowledge_gaps = []
        
        # 尝试提取 JSON
        try:
            # 查找 JSON 块
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                data = json.loads(json_match.group())
                
                # 解析事实
                for i, f in enumerate(data.get("facts", [])):
                    fact_id = hashlib.md5(f.get("claim", str(i)).encode()).hexdigest()[:8]
                    facts.append(Fact(
                        id=f"fact_{fact_id}",
                        claim=f.get("claim", ""),
                        evidence=f.get("evidence", []),
                        confidence=float(f.get("confidence", 0.5)),
                        source_type=f.get("source_type", "unknown"),
                        category=f.get("category", "other"),
                    ))
                
                knowledge_gaps = data.get("knowledge_gaps", [])
        
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse analysis JSON: {e}")
            # 回退：尝试从文本提取
            facts = self._extract_facts_from_text(content)
        
        return facts, knowledge_gaps
    
    def _extract_facts_from_text(self, content: str) -> List[Fact]:
        """从纯文本提取事实 (回退方案)"""
        facts = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line and len(line) > 20 and not line.startswith('#'):
                # 简单启发式：较长的句子可能是事实
                if any(keyword in line.lower() for keyword in 
                       ['是', '使用', '采用', '达到', '实现', 'is', 'uses', 'achieves']):
                    fact_id = hashlib.md5(line.encode()).hexdigest()[:8]
                    facts.append(Fact(
                        id=f"fact_{fact_id}",
                        claim=line[:200],
                        evidence=[],
                        confidence=0.5,
                        source_type="unknown",
                        category="other",
                    ))
                    
                    if len(facts) >= 10:
                        break
        
        return facts
    
    async def verify_claim(self, claim: str) -> Dict[str, Any]:
        """
        验证单个声明
        
        Args:
            claim: 需要验证的声明
            
        Returns:
            验证结果
        """
        # 在知识库中搜索相关内容
        results = await vector_search(claim, top_k=5)
        
        if not results:
            return {
                "claim": claim,
                "verified": False,
                "confidence": 0.0,
                "reason": "未找到相关证据",
            }
        
        # 构建验证 prompt
        context = "\n".join([
            f"[{r['metadata'].get('source', 'unknown')}] {r['content'][:300]}"
            for r in results
        ])
        
        prompt = f"""请验证以下声明是否有证据支持：

声明: {claim}

相关信息:
{context}

请判断：
1. 该声明是否能被上述信息支持？
2. 支持程度如何？(完全支持/部分支持/无法确认/与证据矛盾)
3. 给出置信度 (0.0-1.0)

请以 JSON 格式返回：
{{"verified": true/false, "confidence": 0.8, "reason": "..."}}
"""
        
        response = await self.llm.achat(prompt)
        
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                result = json.loads(json_match.group())
                result["claim"] = claim
                return result
        except:
            pass
        
        return {
            "claim": claim,
            "verified": False,
            "confidence": 0.3,
            "reason": "解析失败",
        }
