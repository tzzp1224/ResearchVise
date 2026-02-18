"""
OpenAI LLM
支持 GPT-4, GPT-4o, GPT-4o-mini 等模型
"""
from typing import List, Optional, Dict, AsyncGenerator
import json
import logging
import inspect

from .base import BaseLLM, Message, LLMResponse, ToolCall


logger = logging.getLogger(__name__)


class OpenAILLM(BaseLLM):
    """
    OpenAI LLM 实现
    
    支持模型:
    - gpt-4o (推荐)
    - gpt-4o-mini (经济)
    - gpt-4-turbo
    - gpt-4
    - gpt-3.5-turbo
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 60.0,
        **kwargs,
    ):
        super().__init__(model, temperature, max_tokens, timeout, **kwargs)
        self.api_key = api_key
        self.base_url = base_url
        self._client = None
        self._async_client = None
    
    @property
    def provider(self) -> str:
        return "openai"
    
    def _get_client(self):
        """获取同步客户端"""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client
    
    def _get_async_client(self):
        """获取异步客户端"""
        if self._async_client is None:
            from openai import AsyncOpenAI
            self._async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._async_client
    
    async def acomplete(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        **kwargs,
    ) -> LLMResponse:
        """异步生成响应"""
        client = self._get_async_client()
        
        # 构建请求参数
        request_params = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        
        if tools:
            request_params["tools"] = tools
            request_params["tool_choice"] = kwargs.get("tool_choice", "auto")
        
        # 调用 API
        response = await client.chat.completions.create(**request_params)
        
        # 解析响应
        choice = response.choices[0]
        content = choice.message.content or ""
        
        # 解析工具调用
        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                )
                for tc in choice.message.tool_calls
            ]
        
        return LLMResponse(
            content=content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            raw_response=response,
        )
    
    async def astream(
        self,
        messages: List[Message],
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """异步流式生成"""
        client = self._get_async_client()
        
        request_params = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": True,
        }
        
        stream = await client.chat.completions.create(**request_params)
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

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
