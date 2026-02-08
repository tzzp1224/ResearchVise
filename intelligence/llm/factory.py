"""
LLM Factory
工厂函数 - 根据配置自动创建 LLM 实例
"""
from typing import Optional
import logging

from .base import BaseLLM
from .openai_llm import OpenAILLM
from .anthropic_llm import AnthropicLLM
from .deepseek_llm import DeepSeekLLM
from .gemini_llm import GeminiLLM


logger = logging.getLogger(__name__)


# 默认模型配置
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-20241022",
    "deepseek": "deepseek-chat",
    "gemini": "gemini-2.0-flash-exp",
}


def get_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs,
) -> BaseLLM:
    """
    获取 LLM 实例
    
    自动从 .env 读取配置，也可手动指定
    
    Args:
        provider: LLM 供应商 (openai, anthropic, deepseek, gemini)
        model: 模型名称 (不传则使用默认)
        **kwargs: 额外参数 (temperature, max_tokens 等)
        
    Returns:
        BaseLLM 实例
        
    Example:
        # 使用 .env 配置
        llm = get_llm()
        
        # 指定供应商
        llm = get_llm(provider="deepseek")
        
        # 指定模型
        llm = get_llm(provider="openai", model="gpt-4o")
        
        # 传入自定义参数
        llm = get_llm(provider="anthropic", temperature=0.5)
    """
    from config import get_llm_settings
    
    settings = get_llm_settings()
    
    # 确定供应商
    provider = provider or settings.provider
    
    # 确定模型
    model = model or settings.model_name or DEFAULT_MODELS.get(provider)
    
    # 获取 API Key
    api_keys = {
        "openai": settings.openai_api_key,
        "anthropic": settings.anthropic_api_key,
        "deepseek": settings.deepseek_api_key,
        "gemini": settings.gemini_api_key,
    }
    
    api_key = kwargs.pop("api_key", None) or api_keys.get(provider)
    
    # 合并默认参数
    default_params = {
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
    }
    for key, value in default_params.items():
        if key not in kwargs:
            kwargs[key] = value
    
    # 创建实例
    if provider == "openai":
        return OpenAILLM(
            model=model,
            api_key=api_key,
            base_url=kwargs.pop("base_url", None),
            **kwargs,
        )
    elif provider == "anthropic":
        return AnthropicLLM(
            model=model,
            api_key=api_key,
            **kwargs,
        )
    elif provider == "deepseek":
        return DeepSeekLLM(
            model=model,
            api_key=api_key,
            base_url=kwargs.pop("base_url", None),
            **kwargs,
        )
    elif provider == "gemini":
        return GeminiLLM(
            model=model,
            api_key=api_key,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
