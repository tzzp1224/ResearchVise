"""Function-call tool schema definitions for search tools."""

from __future__ import annotations

from typing import Any, Dict


def _tool_definition(
    *,
    name: str,
    description: str,
    properties: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": ["query"],
            },
        },
    }


SEARCH_TOOL_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "arxiv_search": _tool_definition(
        name="arxiv_search",
        description="搜索 ArXiv 学术论文。用于获取最新的学术研究成果、技术论文、模型架构细节等。",
        properties={
            "query": {
                "type": "string",
                "description": "搜索关键词，可以是论文标题、作者、主题等",
            },
            "max_results": {
                "type": "integer",
                "description": "最大返回结果数",
                "default": 10,
            },
        },
    ),
    "huggingface_search": _tool_definition(
        name="huggingface_search",
        description="搜索 HuggingFace 上的模型、数据集。用于了解开源模型、预训练权重、数据集信息。",
        properties={
            "query": {
                "type": "string",
                "description": "搜索关键词",
            },
            "search_type": {
                "type": "string",
                "enum": ["all", "models", "datasets"],
                "description": "搜索类型",
                "default": "all",
            },
            "max_results": {
                "type": "integer",
                "description": "最大返回结果数",
                "default": 10,
            },
        },
    ),
    "twitter_search": _tool_definition(
        name="twitter_search",
        description="搜索 Twitter/X 上的讨论。用于了解社区反馈、实时热点、专家观点。",
        properties={
            "query": {
                "type": "string",
                "description": "搜索关键词",
            },
            "max_results": {
                "type": "integer",
                "description": "最大返回结果数",
                "default": 20,
            },
        },
    ),
    "reddit_search": _tool_definition(
        name="reddit_search",
        description="搜索 Reddit 上的讨论。用于了解社区深度讨论、使用经验、问题反馈。",
        properties={
            "query": {
                "type": "string",
                "description": "搜索关键词",
            },
            "subreddits": {
                "type": "array",
                "items": {"type": "string"},
                "description": "限定的子版块",
            },
            "max_results": {
                "type": "integer",
                "description": "最大返回结果数",
                "default": 20,
            },
        },
    ),
    "github_search": _tool_definition(
        name="github_search",
        description="搜索 GitHub 仓库。用于了解开源实现、代码质量、项目活跃度。",
        properties={
            "query": {
                "type": "string",
                "description": "搜索关键词",
            },
            "max_results": {
                "type": "integer",
                "description": "最大返回结果数",
                "default": 10,
            },
        },
    ),
    "semantic_scholar_search": _tool_definition(
        name="semantic_scholar_search",
        description="搜索 Semantic Scholar 论文。用于补充引用关系与更广学术覆盖。",
        properties={
            "query": {
                "type": "string",
                "description": "搜索关键词",
            },
            "max_results": {
                "type": "integer",
                "description": "最大返回结果数",
                "default": 10,
            },
        },
    ),
    "stackoverflow_search": _tool_definition(
        name="stackoverflow_search",
        description="搜索 Stack Overflow 技术问答。用于获取工程实践问题与解决方案。",
        properties={
            "query": {
                "type": "string",
                "description": "搜索关键词",
            },
            "max_results": {
                "type": "integer",
                "description": "最大返回结果数",
                "default": 10,
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "可选标签过滤",
            },
        },
    ),
    "hackernews_search": _tool_definition(
        name="hackernews_search",
        description="搜索 Hacker News 讨论。用于获取社区趋势、生产经验和争议点。",
        properties={
            "query": {
                "type": "string",
                "description": "搜索关键词",
            },
            "max_results": {
                "type": "integer",
                "description": "最大返回结果数",
                "default": 10,
            },
            "sort_by": {
                "type": "string",
                "enum": ["relevance", "date"],
                "description": "排序方式",
                "default": "relevance",
            },
        },
    ),
}
