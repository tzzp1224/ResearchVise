"""
Search Agent
搜索情报员 - ReAct 驱动的多源信息搜集
"""
from typing import Awaitable, Callable, List, Dict, Any, Optional
import json
import logging

from intelligence.llm import BaseLLM, get_llm, Message
from intelligence.state import AgentState, SearchResult, AgentPhase
from intelligence.tools.search_tools import (
    create_search_tools,
    execute_search_tool,
)


logger = logging.getLogger(__name__)


SEARCH_AGENT_SYSTEM_PROMPT = """你是一个学术研究情报搜集专家。你的任务是针对给定的研究主题，制定搜索策略并执行多源信息搜集。

## 你的能力
你可以使用以下工具搜索信息：
1. arxiv_search - 搜索学术论文，获取技术细节、模型架构、实验结果
2. huggingface_search - 搜索开源模型和数据集，了解实现和使用情况
3. twitter_search - 搜索社交媒体讨论，了解社区反馈和专家观点
4. reddit_search - 搜索深度讨论，了解使用经验和问题
5. github_search - 搜索代码仓库，了解开源实现和项目活跃度

## 搜索策略
1. 首先分析主题，确定需要了解的维度（技术原理、性能表现、社区反馈、开源实现等）
2. 为每个维度设计针对性的搜索关键词
3. 优先搜索学术论文（arxiv）获取权威信息
4. 用社交媒体（twitter, reddit）补充社区观点
5. 用 GitHub 了解实际应用情况

## 输出要求
- 每次搜索后，评估结果质量
- 如果某个维度信息不足，调整关键词重新搜索
- 搜索完成后，汇总你认为最有价值的发现

## 当前任务
研究主题: {topic}
{user_query_section}

请开始你的搜索任务。每次行动后，我会告诉你搜索结果，你再决定下一步。
"""


class SearchAgent:
    """
    搜索情报员
    
    使用 ReAct 模式进行多源信息搜集
    """
    
    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        max_iterations: int = 10,
        tool_executor: Optional[
            Callable[[str, Dict[str, Any]], Awaitable[List[Dict[str, Any]]]]
        ] = None,
    ):
        self.llm = llm or get_llm()
        self.max_iterations = max_iterations
        self.tools = create_search_tools()
        self._tool_executor = tool_executor or execute_search_tool

    async def search(
        self,
        topic: str,
        user_query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        便捷方法：仅返回搜索结果列表

        Args:
            topic: 研究主题
            user_query: 用户具体问题（可选）

        Returns:
            搜索结果列表（dict）
        """
        state: Dict[str, Any] = {"topic": topic}
        if user_query:
            state["user_query"] = user_query

        result = await self.run(state)
        return result.get("search_results", [])
    
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行搜索任务
        
        Args:
            state: AgentState 字典
            
        Returns:
            更新后的状态
        """
        topic = state["topic"]
        user_query = state.get("user_query", "")
        
        # 构建系统提示
        user_query_section = f"用户问题: {user_query}" if user_query else ""
        system_prompt = SEARCH_AGENT_SYSTEM_PROMPT.format(
            topic=topic,
            user_query_section=user_query_section,
        )
        
        messages = [Message.system(system_prompt)]
        messages.append(Message.user(f"请开始搜索关于 '{topic}' 的信息。"))
        
        all_results = []
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"Search Agent iteration {iteration}/{self.max_iterations}")
            
            # 调用 LLM
            response = await self.llm.acomplete(messages, tools=self.tools)
            
            # 检查是否有工具调用
            if response.has_tool_calls:
                # 执行工具调用
                for tool_call in response.tool_calls:
                    tool_name = tool_call.name
                    tool_args = tool_call.arguments
                    
                    logger.info(f"Executing tool: {tool_name}({tool_args})")
                    
                    try:
                        results = await self._tool_executor(tool_name, tool_args)
                        all_results.extend(results)
                        
                        # 构建观察消息
                        observation = self._format_results(results, tool_name)
                        messages.append(Message.assistant(response.content or f"调用 {tool_name}"))
                        messages.append(Message.user(f"工具执行结果:\n{observation}"))
                        
                    except Exception as e:
                        logger.error(f"Tool execution failed: {e}")
                        messages.append(Message.assistant(response.content or f"调用 {tool_name}"))
                        messages.append(Message.user(f"工具执行失败: {str(e)}"))
            
            else:
                # 没有工具调用，可能是总结或完成
                messages.append(Message.assistant(response.content))
                
                # 检查是否完成
                if self._is_search_complete(response.content):
                    logger.info("Search Agent completed")
                    break
                
                # 询问下一步
                messages.append(Message.user("请继续搜索，或者如果你认为信息已经足够，请总结发现。"))
        
        # 去重结果
        unique_results = self._deduplicate_results(all_results)
        
        # 更新状态
        return {
            "phase": AgentPhase.SEARCHING,
            "iteration": iteration,
            "search_results": unique_results,
            "messages": [m.to_dict() for m in messages],
        }
    
    def _format_results(self, results: List[Dict], tool_name: str) -> str:
        """格式化搜索结果"""
        if not results:
            return "未找到相关结果。"
        
        formatted = f"找到 {len(results)} 条结果:\n\n"
        for i, r in enumerate(results[:5], 1):  # 只显示前5条
            formatted += f"{i}. **{r.get('title', 'Untitled')}**\n"
            formatted += f"   来源: {r.get('source', 'unknown')}\n"
            content = r.get('content', '')[:200]
            formatted += f"   摘要: {content}...\n"
            if r.get('url'):
                formatted += f"   链接: {r.get('url')}\n"
            formatted += "\n"
        
        if len(results) > 5:
            formatted += f"(还有 {len(results) - 5} 条结果未显示)\n"
        
        return formatted
    
    def _is_search_complete(self, content: str) -> bool:
        """判断搜索是否完成"""
        completion_signals = [
            "搜索完成",
            "信息收集完毕",
            "以上是我的发现",
            "总结如下",
            "综上所述",
            "search complete",
            "finished searching",
        ]
        content_lower = content.lower()
        return any(signal.lower() in content_lower for signal in completion_signals)
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """去重搜索结果"""
        seen_ids = set()
        unique = []
        for r in results:
            rid = r.get("id")
            if rid and rid not in seen_ids:
                seen_ids.add(rid)
                unique.append(r)
        return unique
    
    async def plan_search(self, topic: str) -> List[str]:
        """
        生成搜索计划 (关键词列表)
        
        Args:
            topic: 研究主题
            
        Returns:
            搜索关键词列表
        """
        prompt = f"""针对研究主题 "{topic}"，请生成搜索关键词列表。

要求：
1. 涵盖技术原理、性能表现、社区反馈等维度
2. 包含英文和中文关键词
3. 每个关键词应该能用于搜索引擎

请直接返回 JSON 数组格式，例如：
["keyword1", "keyword2", "keyword3"]
"""
        
        response = await self.llm.achat(prompt)
        
        try:
            # 尝试解析 JSON
            keywords = json.loads(response)
            if isinstance(keywords, list):
                return keywords[:10]  # 最多10个关键词
        except json.JSONDecodeError:
            pass
        
        # 解析失败，简单分割
        return [topic]
