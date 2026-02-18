"""
Agent State
全局状态定义 - 用于 LangGraph 状态管理
"""
from typing import List, Optional, Dict, Any, Annotated, TypedDict
from dataclasses import dataclass, field
from enum import Enum

from outputs.models import TimelineEvent, OnePager, VideoBrief


class AgentPhase(str, Enum):
    """Agent 执行阶段"""
    INIT = "init"                    # 初始化
    PLANNING = "planning"            # 规划阶段
    SEARCHING = "searching"          # 搜索情报
    ANALYZING = "analyzing"          # 分析事实
    GENERATING = "generating"        # 生成内容
    COMPLETED = "completed"          # 完成
    ERROR = "error"                  # 错误


@dataclass
class SearchResult:
    """搜索结果"""
    id: str
    source: str  # arxiv, huggingface, twitter, reddit, github
    title: str
    content: str
    url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    relevance_score: float = 0.0  # 相关性分数
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "title": self.title,
            "content": self.content,
            "url": self.url,
            "metadata": self.metadata,
            "relevance_score": self.relevance_score,
        }


@dataclass
class Fact:
    """分析后的事实"""
    id: str
    claim: str  # 事实陈述
    evidence: List[str]  # 证据来源 (搜索结果 ID)
    confidence: float  # 置信度 0-1
    source_type: str  # paper, social, code
    category: str  # architecture, performance, training, deployment, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "claim": self.claim,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "source_type": self.source_type,
            "category": self.category,
        }

# 用于 LangGraph 的状态累加器
def add_search_results(
    existing: List[SearchResult], 
    new: List[SearchResult]
) -> List[SearchResult]:
    """累加搜索结果 (去重)"""
    existing_ids = {r.id for r in existing}
    for r in new:
        if r.id not in existing_ids:
            existing.append(r)
            existing_ids.add(r.id)
    return existing


def add_facts(
    existing: List[Fact], 
    new: List[Fact]
) -> List[Fact]:
    """累加事实 (去重)"""
    existing_ids = {f.id for f in existing}
    for f in new:
        if f.id not in existing_ids:
            existing.append(f)
            existing_ids.add(f.id)
    return existing


def add_messages(
    existing: List[Dict[str, Any]], 
    new: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """累加消息历史"""
    return existing + new


class AgentState:
    """
    Agent 全局状态
    
    用于 LangGraph 在整个工作流中传递状态
    
    注意: 这是一个 TypedDict 兼容的类，用于 LangGraph
    """
    
    # 输入
    topic: str  # 用户输入的主题
    user_query: Optional[str]  # 用户的具体问题 (可选)
    
    # 执行状态
    phase: AgentPhase
    iteration: int  # 当前迭代次数
    max_iterations: int  # 最大迭代次数
    
    # 搜索阶段产出
    search_plan: List[str]  # 搜索计划 (关键词列表)
    search_results: Annotated[List[SearchResult], add_search_results]
    
    # 分析阶段产出
    facts: Annotated[List[Fact], add_facts]
    knowledge_gaps: List[str]  # 信息缺口
    
    # 生成阶段产出
    timeline: Optional[List[TimelineEvent]]
    one_pager: Optional[OnePager]
    video_brief: Optional[VideoBrief]
    
    # 对话历史 (用于 ReAct)
    messages: Annotated[List[Dict[str, Any]], add_messages]
    
    # 错误处理
    errors: List[str]
    
    @classmethod
    def create_initial(
        cls,
        topic: str,
        user_query: Optional[str] = None,
        max_iterations: int = 5,
    ) -> Dict[str, Any]:
        """创建初始状态"""
        return {
            "topic": topic,
            "user_query": user_query,
            "phase": AgentPhase.INIT,
            "iteration": 0,
            "max_iterations": max_iterations,
            "search_plan": [],
            "search_results": [],
            "facts": [],
            "knowledge_gaps": [],
            "timeline": None,
            "one_pager": None,
            "video_brief": None,
            "messages": [],
            "errors": [],
        }


class AgentStateDict(TypedDict, total=False):
    """LangGraph 兼容的状态字典"""
    # 输入
    topic: str
    user_query: Optional[str]
    
    # 执行状态
    phase: AgentPhase
    iteration: int
    max_iterations: int
    
    # 搜索阶段
    search_plan: List[str]
    search_results: List[Dict[str, Any]]  # SearchResult.to_dict()
    
    # 分析阶段
    facts: List[Dict[str, Any]]  # Fact.to_dict()
    knowledge_gaps: List[str]
    
    # 生成阶段
    timeline: Optional[List[Dict[str, Any]]]  # TimelineEvent.to_dict()
    one_pager: Optional[Dict[str, Any]]  # OnePager.to_dict()
    video_brief: Optional[Dict[str, Any]]  # VideoBrief.to_dict()
    
    # 对话
    messages: List[Dict[str, Any]]
    
    # 错误
    errors: List[str]
