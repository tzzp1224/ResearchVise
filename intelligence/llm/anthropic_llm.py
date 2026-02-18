"""
Anthropic LLM
支持 Claude 3.5 Sonnet, Claude 3 Opus 等模型
"""
from typing import List, Optional, Dict, AsyncGenerator
import logging
import inspect

from .base import BaseLLM, Message, MessageRole, LLMResponse, ToolCall


logger = logging.getLogger(__name__)


class AnthropicLLM(BaseLLM):
    """
    Anthropic Claude LLM 实现
    
    支持模型:
    - claude-3-5-sonnet-20241022 (推荐)
    - claude-3-opus-20240229
    - claude-3-sonnet-20240229
    - claude-3-haiku-20240307
    """
    
    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 60.0,
        **kwargs,
    ):
        super().__init__(model, temperature, max_tokens, timeout, **kwargs)
        self.api_key = api_key
        self._client = None
        self._async_client = None
    
    @property
    def provider(self) -> str:
        return "anthropic"
    
    def _get_async_client(self):
        """获取异步客户端"""
        if self._async_client is None:
            from anthropic import AsyncAnthropic
            self._async_client = AsyncAnthropic(
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self._async_client
    
    def _convert_messages(self, messages: List[Message]) -> tuple:
        """
        转换消息格式 (Anthropic 格式不同)
        
        Returns:
            (system_prompt, messages_list)
        """
        system_prompt = None
        converted = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_prompt = msg.content
            else:
                converted.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })
        
        return system_prompt, converted
    
    def _convert_tools(self, tools: Optional[List[Dict]]) -> Optional[List[Dict]]:
        """将 OpenAI 格式工具转换为 Anthropic 格式"""
        if not tools:
            return None
        
        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                })
        
        return anthropic_tools if anthropic_tools else None
    
    async def acomplete(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        **kwargs,
    ) -> LLMResponse:
        """异步生成响应"""
        client = self._get_async_client()
        
        system_prompt, converted_messages = self._convert_messages(messages)
        
        # 构建请求参数
        request_params = {
            "model": self.model,
            "messages": converted_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        
        if system_prompt:
            request_params["system"] = system_prompt
        
        anthropic_tools = self._convert_tools(tools)
        if anthropic_tools:
            request_params["tools"] = anthropic_tools
        
        # 调用 API
        response = await client.messages.create(**request_params)
        
        # 解析响应
        content = ""
        tool_calls = []
        
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))
        
        return LLMResponse(
            content=content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=response.stop_reason,
            raw_response=response,
        )
    
    async def astream(
        self,
        messages: List[Message],
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """异步流式生成"""
        client = self._get_async_client()
        
        system_prompt, converted_messages = self._convert_messages(messages)
        
        request_params = {
            "model": self.model,
            "messages": converted_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        
        if system_prompt:
            request_params["system"] = system_prompt
        
        async with client.messages.stream(**request_params) as stream:
            async for text in stream.text_stream:
                yield text

    async def aclose(self) -> None:
        for client_attr in ("_async_client", "_client"):
            client = getattr(self, client_attr, None)
            if client is None:
                continue
            try:
                close_fn = getattr(client, "close", None)
                if callable(close_fn):
                    maybe_awaitable = close_fn()
                    if inspect.isawaitable(maybe_awaitable):
                        await maybe_awaitable
            except Exception:
                pass
            setattr(self, client_attr, None)
