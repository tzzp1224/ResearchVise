"""
DeepSeek LLM
支持 DeepSeek-V3, DeepSeek-R1 等模型
"""
from typing import List, Optional, Dict, AsyncGenerator
import json
import logging
import inspect

from .base import BaseLLM, Message, LLMResponse, ToolCall


logger = logging.getLogger(__name__)


class DeepSeekLLM(BaseLLM):
    """
    DeepSeek LLM 实现
    
    使用 OpenAI 兼容接口
    
    支持模型:
    - deepseek-chat (DeepSeek-V3, 推荐)
    - deepseek-reasoner (DeepSeek-R1, 推理增强)
    """
    
    DEFAULT_BASE_URL = "https://api.deepseek.com"
    
    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 120.0,  # DeepSeek 可能需要更长时间
        **kwargs,
    ):
        super().__init__(model, temperature, max_tokens, timeout, **kwargs)
        self.api_key = api_key
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self._async_client = None
    
    @property
    def provider(self) -> str:
        return "deepseek"
    
    def _get_async_client(self):
        """获取异步客户端 (使用 OpenAI SDK)"""
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
        
        # DeepSeek 支持 function calling
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
        if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                )
                for tc in choice.message.tool_calls
            ]
        
        # DeepSeek-R1 可能包含推理过程
        reasoning_content = ""
        if hasattr(choice.message, 'reasoning_content'):
            reasoning_content = choice.message.reasoning_content or ""
        
        # 如果有推理内容，附加到响应中
        if reasoning_content:
            content = f"<reasoning>\n{reasoning_content}\n</reasoning>\n\n{content}"
        
        return LLMResponse(
            content=content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
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
        client = self._async_client
        if client is None:
            return
        try:
            close_fn = getattr(client, "close", None)
            if callable(close_fn):
                maybe_awaitable = close_fn()
                if inspect.isawaitable(maybe_awaitable):
                    await maybe_awaitable
        except Exception:
            pass
        self._async_client = None
