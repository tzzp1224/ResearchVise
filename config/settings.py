"""
Settings Configuration
使用 Pydantic 进行配置验证和管理
"""
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class ArxivSettings(BaseSettings):
    """ArXiv API 配置"""
    max_results: int = Field(default=50, description="最大返回结果数")
    sort_by: str = Field(default="submittedDate", description="排序方式")
    
    class Config:
        env_prefix = "ARXIV_"


class HuggingFaceSettings(BaseSettings):
    """Hugging Face API 配置"""
    token: Optional[str] = Field(default=None, description="HuggingFace API Token")
    max_results: int = Field(default=30, description="最大返回结果数")
    
    class Config:
        env_prefix = "HUGGINGFACE_"


class TwitterSettings(BaseSettings):
    """Twitter/X API 配置"""
    bearer_token: Optional[str] = Field(default=None, description="Twitter Bearer Token")
    max_results: int = Field(default=100, description="最大返回结果数")
    
    class Config:
        env_prefix = "TWITTER_"


class RedditSettings(BaseSettings):
    """Reddit API 配置"""
    client_id: Optional[str] = Field(default=None, description="Reddit Client ID")
    client_secret: Optional[str] = Field(default=None, description="Reddit Client Secret")
    user_agent: str = Field(default="AcademicResearchAgent/1.0", description="User Agent")
    max_results: int = Field(default=50, description="最大返回结果数")
    
    class Config:
        env_prefix = "REDDIT_"


class GitHubSettings(BaseSettings):
    """GitHub API 配置"""
    token: Optional[str] = Field(default=None, description="GitHub Token")
    max_results: int = Field(default=30, description="最大返回结果数")
    
    class Config:
        env_prefix = "GITHUB_"


class SemanticScholarSettings(BaseSettings):
    """Semantic Scholar API 配置"""
    api_key: Optional[str] = Field(default=None, description="Semantic Scholar API Key (可选)")
    max_results: int = Field(default=30, description="最大返回结果数")
    
    class Config:
        env_prefix = "SEMANTIC_SCHOLAR_"


class StackOverflowSettings(BaseSettings):
    """Stack Overflow API 配置"""
    api_key: Optional[str] = Field(default=None, description="Stack Exchange API Key (可选)")
    max_results: int = Field(default=30, description="最大返回结果数")
    
    class Config:
        env_prefix = "STACKOVERFLOW_"


class HackerNewsSettings(BaseSettings):
    """Hacker News 配置"""
    max_results: int = Field(default=30, description="最大返回结果数")
    
    class Config:
        env_prefix = "HACKERNEWS_"


class GeneralSettings(BaseSettings):
    """通用设置"""
    request_timeout: int = Field(default=30, description="请求超时时间(秒)")
    max_retries: int = Field(default=3, description="最大重试次数")
    retry_delay: int = Field(default=1, description="重试延迟(秒)")


class EmbeddingSettings(BaseSettings):
    """Embedding 服务配置"""
    provider: str = Field(default="siliconflow", description="Embedding提供商: siliconflow, jina, openai, sentence_transformers")
    model_name: Optional[str] = Field(default=None, description="模型名称(不填则使用默认)")
    
    # API Keys (通过环境变量设置)
    siliconflow_api_key: Optional[str] = Field(default=None, description="SiliconFlow API Key")
    jina_api_key: Optional[str] = Field(default=None, description="Jina API Key")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API Key")
    
    class Config:
        env_prefix = "EMBEDDING_"


class StorageSettings(BaseSettings):
    """存储配置"""
    vector_db_provider: str = Field(default="qdrant", description="向量数据库 (仅支持 qdrant)")
    vector_db_path: str = Field(default="./data/qdrant_db", description="向量数据库存储路径")
    cache_path: str = Field(default="./data/cache", description="缓存目录")
    cache_ttl: int = Field(default=3600, description="缓存过期时间(秒)")
    
    # Qdrant 远程连接配置 (可选)
    qdrant_url: Optional[str] = Field(default=None, description="Qdrant Cloud URL")
    qdrant_api_key: Optional[str] = Field(default=None, description="Qdrant Cloud API Key")
    
    class Config:
        env_prefix = "STORAGE_"


class LLMSettings(BaseSettings):
    """LLM 配置 (Phase 3)"""
    provider: str = Field(default="deepseek", description="LLM提供商: openai, anthropic, deepseek, gemini")
    model_name: Optional[str] = Field(default=None, description="模型名称(不填则使用默认)")
    temperature: float = Field(default=0.7, description="生成温度")
    max_tokens: int = Field(default=4096, description="最大生成token数")
    
    # API Keys
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API Key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API Key")
    deepseek_api_key: Optional[str] = Field(default=None, description="DeepSeek API Key")
    gemini_api_key: Optional[str] = Field(default=None, description="Google Gemini API Key")
    
    class Config:
        env_prefix = "LLM_"


class Settings(BaseSettings):
    """主配置类 - 聚合所有子配置"""
    
    # 子配置
    arxiv: ArxivSettings = Field(default_factory=ArxivSettings)
    huggingface: HuggingFaceSettings = Field(default_factory=HuggingFaceSettings)
    twitter: TwitterSettings = Field(default_factory=TwitterSettings)
    reddit: RedditSettings = Field(default_factory=RedditSettings)
    github: GitHubSettings = Field(default_factory=GitHubSettings)
    semantic_scholar: SemanticScholarSettings = Field(default_factory=SemanticScholarSettings)
    stackoverflow: StackOverflowSettings = Field(default_factory=StackOverflowSettings)
    hackernews: HackerNewsSettings = Field(default_factory=HackerNewsSettings)
    general: GeneralSettings = Field(default_factory=GeneralSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
    
    @classmethod
    def load_from_env_file(cls, env_path: Optional[Path] = None) -> "Settings":
        """从指定的 .env 文件加载配置"""
        if env_path is None:
            # 默认查找 config/.env
            env_path = Path(__file__).parent / ".env"
        
        if env_path.exists():
            from dotenv import load_dotenv
            load_dotenv(env_path)
        
        return cls(
            arxiv=ArxivSettings(),
            huggingface=HuggingFaceSettings(),
            twitter=TwitterSettings(),
            reddit=RedditSettings(),
            github=GitHubSettings(),
            general=GeneralSettings(),
            embedding=EmbeddingSettings(),
            storage=StorageSettings(),
            llm=LLMSettings(),
        )


@lru_cache()
def get_settings() -> Settings:
    """获取全局配置单例"""
    return Settings.load_from_env_file()


# 便捷访问
def get_arxiv_settings() -> ArxivSettings:
    return get_settings().arxiv


def get_huggingface_settings() -> HuggingFaceSettings:
    return get_settings().huggingface


def get_twitter_settings() -> TwitterSettings:
    return get_settings().twitter


def get_reddit_settings() -> RedditSettings:
    return get_settings().reddit


def get_github_settings() -> GitHubSettings:
    return get_settings().github


def get_embedding_settings() -> EmbeddingSettings:
    return get_settings().embedding


def get_storage_settings() -> StorageSettings:
    return get_settings().storage


def get_llm_settings() -> LLMSettings:
    return get_settings().llm
