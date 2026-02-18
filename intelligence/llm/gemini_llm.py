"""
Google Gemini LLM
支持 Gemini 2.0, Gemini 1.5 Pro 等模型
"""
from typing import List, Optional, Dict, AsyncGenerator
import logging

from .base import BaseLLM, Message, MessageRole, LLMResponse, ToolCall


logger = logging.getLogger(__name__)


class GeminiLLM(BaseLLM):
    """
    Google Gemini LLM 实现
    
    支持模型:
    - gemini-2.0-flash-exp (推荐)
    - gemini-1.5-pro
    - gemini-1.5-flash
    """
    
    def __init__(
        self,
        model: str = "gemini-2.0-flash-exp",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 60.0,
        **kwargs,
    ):
        super().__init__(model, temperature, max_tokens, timeout, **kwargs)
        self.api_key = api_key
        self._client = None
    
    @property
    def provider(self) -> str:
        return "gemini"
    
    def _get_client(self):
        """获取 Gemini 客户端"""
        if self._client is None:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(
                model_name=self.model,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                },
            )
        return self._client
    
    def _convert_messages(self, messages: List[Message]) -> tuple:
        """
        转换消息格式 (Gemini 格式)
        
        Returns:
            (system_instruction, history, last_message)
        """
        system_instruction = None
        history = []
        last_message = None
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_instruction = msg.content
            elif msg.role == MessageRole.USER:
                last_message = msg.content
            elif msg.role == MessageRole.ASSISTANT:
                # 添加到历史
                if last_message:
                    history.append({"role": "user", "parts": [last_message]})
                    last_message = None
                history.append({"role": "model", "parts": [msg.content]})
        
        return system_instruction, history, last_message
    
    def _convert_tools(self, tools: Optional[List[Dict]]) -> Optional[List]:
        """将 OpenAI 格式工具转换为 Gemini 格式"""
        if not tools:
            return None
        
        try:
            from google.generativeai.types import Tool, FunctionDeclaration
            
            function_declarations = []
            for tool in tools:
                if tool.get("type") == "function":
                    func = tool["function"]
                    function_declarations.append(FunctionDeclaration(
                        name=func["name"],
                        description=func.get("description", ""),
                        parameters=func.get("parameters", {}),
                    ))
            
            return [Tool(function_declarations=function_declarations)] if function_declarations else None
        except ImportError:
            logger.warning("google-generativeai not installed, tools not available")
            return None
    
    async def acomplete(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        **kwargs,
    ) -> LLMResponse:
        """异步生成响应"""
        import google.generativeai as genai
        
        # 重新配置客户端 (支持动态参数)
        genai.configure(api_key=self.api_key)
        
        system_instruction, history, last_message = self._convert_messages(messages)
        
        # 构建模型配置
        generation_config = {
            "temperature": kwargs.get("temperature", self.temperature),
            "max_output_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        
        model = genai.GenerativeModel(
            model_name=self.model,
            generation_config=generation_config,
            system_instruction=system_instruction,
        )
        
        # 创建聊天会话
        chat = model.start_chat(history=history)
        
        # 发送消息
        gemini_tools = self._convert_tools(tools)
        if gemini_tools:
            response = await chat.send_message_async(last_message or "", tools=gemini_tools)
        else:
            response = await chat.send_message_async(last_message or "")
        
        # 解析响应
        content = response.text if response.text else ""
        
        # 解析工具调用
        tool_calls = None
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    fc = part.function_call
                    if tool_calls is None:
                        tool_calls = []
                    tool_calls.append(ToolCall(
                        id=f"call_{fc.name}",
                        name=fc.name,
                        arguments=dict(fc.args),
                    ))
        
        # 解析 usage
        usage = {}
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count,
            }
        
        return LLMResponse(
            content=content,
            model=self.model,
            usage=usage,
            tool_calls=tool_calls,
            finish_reason=response.candidates[0].finish_reason.name if response.candidates else None,
            raw_response=response,
        )
    
    async def astream(
        self,
        messages: List[Message],
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """异步流式生成"""
        import google.generativeai as genai
        
        genai.configure(api_key=self.api_key)
        
        system_instruction, history, last_message = self._convert_messages(messages)
        
        generation_config = {
            "temperature": kwargs.get("temperature", self.temperature),
            "max_output_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        
        model = genai.GenerativeModel(
            model_name=self.model,
            generation_config=generation_config,
            system_instruction=system_instruction,
        )
        
        chat = model.start_chat(history=history)
        
        response = await chat.send_message_async(last_message or "", stream=True)
        
        async for chunk in response:
            if chunk.text:
                yield chunk.text

    async def aclose(self) -> None:
        self._client = None
