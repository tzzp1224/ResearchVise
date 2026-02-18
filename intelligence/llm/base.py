"""
Base LLM
LLM 抽象基类
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum


class MessageRole(str, Enum):
    """消息角色"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """对话消息"""
    role: MessageRole
    content: str
    name: Optional[str] = None  # 用于 tool 消息
    tool_call_id: Optional[str] = None  # 用于 tool 响应
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典 (用于 API 调用)"""
        d = {"role": self.role.value, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d
    
    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role=MessageRole.SYSTEM, content=content)
    
    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role=MessageRole.USER, content=content)
    
    @classmethod
    def assistant(cls, content: str) -> "Message":
        return cls(role=MessageRole.ASSISTANT, content=content)


@dataclass
class ToolCall:
    """工具调用"""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMResponse:
    """LLM 响应"""
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)  # prompt_tokens, completion_tokens, total_tokens
    tool_calls: Optional[List[ToolCall]] = None
    finish_reason: Optional[str] = None
    raw_response: Optional[Any] = None  # 原始响应对象
    
    @property
    def has_tool_calls(self) -> bool:
        return self.tool_calls is not None and len(self.tool_calls) > 0


class BaseLLM(ABC):
    """
    LLM 抽象基类
    
    所有 LLM 供应商实现需继承此类
    """
    
    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 60.0,
        **kwargs,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra_config = kwargs
    
    @property
    @abstractmethod
    def provider(self) -> str:
        """返回供应商名称"""
        pass
    
    @abstractmethod
    async def acomplete(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        异步生成响应
        
        Args:
            messages: 对话消息列表
            tools: 工具定义 (OpenAI function calling 格式)
            **kwargs: 额外参数
            
        Returns:
            LLMResponse
        """
        pass
    
    @abstractmethod
    async def astream(
        self,
        messages: List[Message],
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        异步流式生成
        
        Args:
            messages: 对话消息列表
            **kwargs: 额外参数
            
        Yields:
            生成的文本片段
        """
        pass
    
    def complete(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        **kwargs,
    ) -> LLMResponse:
        """同步生成响应 (包装异步方法)"""
        import asyncio
        return asyncio.run(self.acomplete(messages, tools, **kwargs))
    
    def chat(self, user_message: str, system_prompt: Optional[str] = None) -> str:
        """
        简单对话接口
        
        Args:
            user_message: 用户消息
            system_prompt: 系统提示词 (可选)
            
        Returns:
            助手回复内容
        """
        messages = []
        if system_prompt:
            messages.append(Message.system(system_prompt))
        messages.append(Message.user(user_message))
        
        response = self.complete(messages)
        return response.content
    
    async def achat(self, user_message: str, system_prompt: Optional[str] = None) -> str:
        """异步简单对话接口"""
        messages = []
        if system_prompt:
            messages.append(Message.system(system_prompt))
        messages.append(Message.user(user_message))
        
        response = await self.acomplete(messages)
        return response.content

    async def aclose(self) -> None:
        """
        关闭底层客户端资源（默认 no-op）。
        子类可覆盖以释放 HTTP 连接池，避免事件循环关闭时的析构警告。
        """
        return None
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model}, provider={self.provider})"
