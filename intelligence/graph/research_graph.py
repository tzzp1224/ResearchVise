"""
Research Graph
LangGraph 主图定义 - Plan-and-Execute 模式
"""
from typing import Dict, Any, Optional, Literal, Annotated, TypedDict, List, AsyncIterator
import logging
import operator

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from intelligence.llm import get_llm
from intelligence.agents import SearchAgent, AnalystAgent, ContentAgent
from intelligence.state import AgentPhase


logger = logging.getLogger(__name__)


# 定义状态结构 (LangGraph 兼容)
class ResearchState(TypedDict, total=False):
    """研究流程状态"""
    # 输入
    topic: str
    user_query: Optional[str]
    
    # 控制
    phase: str
    iteration: int
    max_iterations: int
    should_continue: bool
    
    # 搜索阶段输出
    search_plan: List[str]
    search_results: List[Dict[str, Any]]
    
    # 分析阶段输出
    facts: List[Dict[str, Any]]
    knowledge_gaps: List[str]
    
    # 生成阶段输出
    timeline: Optional[List[Dict[str, Any]]]
    one_pager: Optional[Dict[str, Any]]
    video_brief: Optional[Dict[str, Any]]
    
    # 对话历史
    messages: List[Dict[str, Any]]
    
    # 错误
    errors: List[str]


def create_initial_state(
    topic: str,
    user_query: Optional[str] = None,
    max_iterations: int = 5,
) -> ResearchState:
    """创建初始状态"""
    return {
        "topic": topic,
        "user_query": user_query,
        "phase": AgentPhase.INIT.value,
        "iteration": 0,
        "max_iterations": max_iterations,
        "should_continue": True,
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


# ===== 节点函数 =====

async def planning_node(state: ResearchState) -> Dict[str, Any]:
    """
    规划节点 - 分析主题，制定搜索策略
    """
    logger.info(f"[Planning] Topic: {state['topic']}")
    
    search_agent = SearchAgent()
    search_plan = await search_agent.plan_search(state["topic"])
    
    return {
        "phase": AgentPhase.PLANNING.value,
        "search_plan": search_plan,
    }


async def searching_node(state: ResearchState) -> Dict[str, Any]:
    """
    搜索节点 - 执行 ReAct 搜索循环
    """
    logger.info(f"[Searching] Iteration: {state['iteration'] + 1}")
    
    search_agent = SearchAgent(max_iterations=state.get("max_iterations", 5))
    result = await search_agent.run(state)
    
    return {
        "phase": AgentPhase.SEARCHING.value,
        "iteration": state["iteration"] + 1,
        "search_results": result.get("search_results", []),
        "messages": result.get("messages", []),
    }


async def analyzing_node(state: ResearchState) -> Dict[str, Any]:
    """
    分析节点 - 提取事实，评估置信度
    """
    logger.info(f"[Analyzing] {len(state.get('search_results', []))} results to analyze")
    
    analyst_agent = AnalystAgent()
    result = await analyst_agent.run(state)
    
    return {
        "phase": AgentPhase.ANALYZING.value,
        "facts": result.get("facts", []),
        "knowledge_gaps": result.get("knowledge_gaps", []),
    }


async def generating_node(state: ResearchState) -> Dict[str, Any]:
    """
    生成节点 - 并行生成三种输出
    """
    logger.info(f"[Generating] {len(state.get('facts', []))} facts to generate from")
    
    content_agent = ContentAgent()
    result = await content_agent.run(state)
    
    return {
        "phase": AgentPhase.GENERATING.value,
        "timeline": result.get("timeline"),
        "one_pager": result.get("one_pager"),
        "video_brief": result.get("video_brief"),
    }


async def refine_node(state: ResearchState) -> Dict[str, Any]:
    """
    修正节点 - 检查信息缺口，决定是否继续搜索
    """
    knowledge_gaps = state.get("knowledge_gaps", [])
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 5)
    
    # 决定是否继续
    should_continue = (
        len(knowledge_gaps) > 0 
        and iteration < max_iterations
        and len(state.get("search_results", [])) < 30  # 限制总结果数
    )
    
    logger.info(f"[Refine] Gaps: {len(knowledge_gaps)}, Continue: {should_continue}")
    
    return {
        "should_continue": should_continue,
    }


def completion_node(state: ResearchState) -> Dict[str, Any]:
    """
    完成节点 - 标记完成
    """
    logger.info("[Completed] Research finished")
    return {
        "phase": AgentPhase.COMPLETED.value,
    }


# ===== 边函数 (条件路由) =====

def should_refine_search(state: ResearchState) -> Literal["refine", "generate"]:
    """
    决定是否需要继续搜索
    
    条件：
    - 如果信息充足，直接生成
    - 如果有明显信息缺口，继续搜索
    """
    facts = state.get("facts", [])
    
    # 如果有足够多的高置信度事实，直接生成
    high_confidence_facts = [f for f in facts if f.get("confidence", 0) >= 0.7]
    
    if len(high_confidence_facts) >= 5:
        return "generate"
    
    return "refine"


def should_continue_searching(state: ResearchState) -> Literal["search", "generate"]:
    """
    修正后决定下一步
    """
    if state.get("should_continue", False):
        return "search"
    return "generate"


# ===== 图构建 =====

def create_research_graph() -> StateGraph:
    """
    创建研究工作流图
    
    流程：
    1. Planning -> Searching (ReAct 循环)
    2. Searching -> Analyzing
    3. Analyzing -> Refine (决定是否继续搜索)
    4. Refine -> Searching (如果需要) 或 Generating
    5. Generating -> Completed
    
    Returns:
        编译后的 LangGraph
    """
    # 创建图
    workflow = StateGraph(ResearchState)
    
    # 添加节点
    workflow.add_node("planning", planning_node)
    workflow.add_node("searching", searching_node)
    workflow.add_node("analyzing", analyzing_node)
    workflow.add_node("refine", refine_node)
    workflow.add_node("generating", generating_node)
    workflow.add_node("completion", completion_node)
    
    # 添加边
    # START -> Planning
    workflow.add_edge(START, "planning")
    
    # Planning -> Searching
    workflow.add_edge("planning", "searching")
    
    # Searching -> Analyzing
    workflow.add_edge("searching", "analyzing")
    
    # Analyzing -> Refine (条件路由)
    workflow.add_conditional_edges(
        "analyzing",
        should_refine_search,
        {
            "refine": "refine",
            "generate": "generating",
        }
    )
    
    # Refine -> Searching or Generating
    workflow.add_conditional_edges(
        "refine",
        should_continue_searching,
        {
            "search": "searching",
            "generate": "generating",
        }
    )
    
    # Generating -> Completion
    workflow.add_edge("generating", "completion")
    
    # Completion -> END
    workflow.add_edge("completion", END)
    
    return workflow


class ResearchGraph:
    """
    研究图封装类
    
    提供更友好的 API
    """
    
    def __init__(self, checkpointer: bool = False):
        """
        初始化研究图
        
        Args:
            checkpointer: 是否启用状态检查点
        """
        self.workflow = create_research_graph()
        
        if checkpointer:
            self.memory = MemorySaver()
            self.graph = self.workflow.compile(checkpointer=self.memory)
        else:
            self.graph = self.workflow.compile()
    
    async def run(
        self,
        topic: str,
        user_query: Optional[str] = None,
        max_iterations: int = 5,
        thread_id: Optional[str] = None,
    ) -> ResearchState:
        """
        执行研究流程
        
        Args:
            topic: 研究主题
            user_query: 用户问题
            max_iterations: 最大迭代次数
            thread_id: 会话 ID (用于状态持久化)
            
        Returns:
            最终状态
        """
        initial_state = create_initial_state(
            topic=topic,
            user_query=user_query,
            max_iterations=max_iterations,
        )
        
        config = {}
        if thread_id:
            config["configurable"] = {"thread_id": thread_id}
        
        # 执行图
        final_state = await self.graph.ainvoke(initial_state, config)
        
        return final_state
    
    async def stream(
        self,
        topic: str,
        user_query: Optional[str] = None,
        max_iterations: int = 5,
    ):
        """
        流式执行，实时返回状态更新
        
        Yields:
            (node_name, state_update) 元组
        """
        initial_state = create_initial_state(
            topic=topic,
            user_query=user_query,
            max_iterations=max_iterations,
        )
        
        async for event in self.graph.astream(initial_state):
            yield event


async def run_research(
    topic: str,
    user_query: Optional[str] = None,
    max_iterations: int = 5,
) -> ResearchState:
    """
    便捷函数 - 执行研究流程 (非流式)
    
    Args:
        topic: 研究主题
        user_query: 用户问题
        max_iterations: 最大迭代次数
        
    Returns:
        最终状态
    """
    graph = ResearchGraph()
    return await graph.run(topic, user_query, max_iterations)


async def stream_research(
    topic: str,
    user_query: Optional[str] = None,
    max_iterations: int = 5,
) -> AsyncIterator[dict]:
    """
    便捷函数 - 执行研究流程 (流式)
    
    Args:
        topic: 研究主题
        user_query: 用户问题
        max_iterations: 最大迭代次数
        
    Yields:
        状态更新事件
    """
    graph = ResearchGraph()
    async for event in graph.stream(topic, user_query, max_iterations):
        yield event
