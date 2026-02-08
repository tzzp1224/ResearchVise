"""
Intelligence Module (Phase 3)
智能层 - LLM抽象 + LangGraph编排 + 多Agent协作
"""
from .llm import (
    BaseLLM,
    OpenAILLM,
    AnthropicLLM,
    DeepSeekLLM,
    GeminiLLM,
    get_llm,
)
from .state import AgentState, SearchResult as StateSearchResult
from .agents import SearchAgent, AnalystAgent, ContentAgent
from .graph import create_research_graph, run_research, stream_research, ResearchGraph
from .pipeline import (
    run_research_end_to_end,
    run_research_from_search_results,
    aggregated_result_to_search_results,
    evaluate_output_depth,
)

__all__ = [
    # LLM
    "BaseLLM",
    "OpenAILLM",
    "AnthropicLLM",
    "DeepSeekLLM",
    "GeminiLLM",
    "get_llm",
    # State
    "AgentState",
    "StateSearchResult",
    # Agents
    "SearchAgent",
    "AnalystAgent",
    "ContentAgent",
    # Graph
    "create_research_graph",
    "run_research",
    "stream_research",
    "ResearchGraph",
    # Pipeline
    "run_research_end_to_end",
    "run_research_from_search_results",
    "aggregated_result_to_search_results",
    "evaluate_output_depth",
]
