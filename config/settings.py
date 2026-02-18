"""
Settings Configuration
使用 Pydantic 进行配置验证和管理
"""
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _env_prefixed(prefix: str) -> SettingsConfigDict:
    return SettingsConfigDict(env_prefix=prefix, extra="ignore")


class ArxivSettings(BaseSettings):
    """ArXiv API 配置"""
    model_config = _env_prefixed("ARXIV_")

    max_results: int = Field(default=50, description="最大返回结果数")
    sort_by: str = Field(default="submittedDate", description="排序方式")


class HuggingFaceSettings(BaseSettings):
    """Hugging Face API 配置"""
    model_config = _env_prefixed("HUGGINGFACE_")

    token: Optional[str] = Field(default=None, description="HuggingFace API Token")
    max_results: int = Field(default=30, description="最大返回结果数")


class TwitterSettings(BaseSettings):
    """Twitter/X API 配置"""
    model_config = _env_prefixed("TWITTER_")

    bearer_token: Optional[str] = Field(default=None, description="Twitter Bearer Token")
    max_results: int = Field(default=100, description="最大返回结果数")


class RedditSettings(BaseSettings):
    """Reddit API 配置"""
    model_config = _env_prefixed("REDDIT_")

    client_id: Optional[str] = Field(default=None, description="Reddit Client ID")
    client_secret: Optional[str] = Field(default=None, description="Reddit Client Secret")
    user_agent: str = Field(default="AcademicResearchAgent/1.0", description="User Agent")
    max_results: int = Field(default=50, description="最大返回结果数")


class GitHubSettings(BaseSettings):
    """GitHub API 配置"""
    model_config = _env_prefixed("GITHUB_")

    token: Optional[str] = Field(default=None, description="GitHub Token")
    max_results: int = Field(default=30, description="最大返回结果数")


class SemanticScholarSettings(BaseSettings):
    """Semantic Scholar API 配置"""
    model_config = _env_prefixed("SEMANTIC_SCHOLAR_")

    api_key: Optional[str] = Field(default=None, description="Semantic Scholar API Key (可选)")
    max_results: int = Field(default=30, description="最大返回结果数")
    requests_per_second: float = Field(
        default=0.8,
        description="请求速率（建议 <= 1.0，含所有 endpoint 的累计请求）",
    )


class StackOverflowSettings(BaseSettings):
    """Stack Overflow API 配置"""
    model_config = _env_prefixed("STACKOVERFLOW_")

    api_key: Optional[str] = Field(default=None, description="Stack Exchange API Key (可选)")
    max_results: int = Field(default=30, description="最大返回结果数")


class HackerNewsSettings(BaseSettings):
    """Hacker News 配置"""
    model_config = _env_prefixed("HACKERNEWS_")

    max_results: int = Field(default=30, description="最大返回结果数")


class GeneralSettings(BaseSettings):
    """通用设置"""
    model_config = SettingsConfigDict(extra="ignore")

    request_timeout: int = Field(default=30, description="请求超时时间(秒)")
    max_retries: int = Field(default=3, description="最大重试次数")
    retry_delay: int = Field(default=1, description="重试延迟(秒)")


class EmbeddingSettings(BaseSettings):
    """Embedding 服务配置"""
    model_config = _env_prefixed("EMBEDDING_")

    provider: str = Field(default="siliconflow", description="Embedding提供商: siliconflow, jina, openai, sentence_transformers")
    model_name: Optional[str] = Field(default=None, description="模型名称(不填则使用默认)")
    
    # API Keys (通过环境变量设置)
    siliconflow_api_key: Optional[str] = Field(default=None, description="SiliconFlow API Key")
    jina_api_key: Optional[str] = Field(default=None, description="Jina API Key")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API Key")
    
class StorageSettings(BaseSettings):
    """存储配置"""
    model_config = _env_prefixed("STORAGE_")

    vector_db_provider: str = Field(default="qdrant", description="向量数据库 (仅支持 qdrant)")
    vector_db_path: str = Field(default="./data/qdrant_db", description="向量数据库存储路径")
    cache_path: str = Field(default="./data/cache", description="缓存目录")
    cache_ttl: int = Field(default=3600, description="缓存过期时间(秒)")
    
    # Qdrant 远程连接配置 (可选)
    qdrant_url: Optional[str] = Field(default=None, description="Qdrant Cloud URL")
    qdrant_api_key: Optional[str] = Field(default=None, description="Qdrant Cloud API Key")
    
class LLMSettings(BaseSettings):
    """LLM 配置 (Phase 3)"""
    model_config = _env_prefixed("LLM_")

    provider: str = Field(default="deepseek", description="LLM提供商: openai, anthropic, deepseek, gemini")
    model_name: Optional[str] = Field(default=None, description="模型名称(不填则使用默认)")
    temperature: float = Field(default=0.7, description="生成温度")
    max_tokens: int = Field(default=4096, description="最大生成token数")
    
    # API Keys
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API Key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API Key")
    deepseek_api_key: Optional[str] = Field(default=None, description="DeepSeek API Key")
    gemini_api_key: Optional[str] = Field(default=None, description="Google Gemini API Key")
    
class ResearchCacheSettings(BaseSettings):
    """研究产物缓存配置（相似查询复用）"""
    model_config = _env_prefixed("RESEARCH_CACHE_")

    enabled: bool = Field(default=True, description="是否启用研究产物缓存")
    similarity_threshold: float = Field(default=0.82, description="命中相似缓存的最低相似度阈值")
    top_k: int = Field(default=3, description="相似查询检索候选数")
    min_quality_score: float = Field(default=0.35, description="可复用缓存的最小质量分")
    require_quality_gate_pass: bool = Field(
        default=False,
        description="仅复用通过 quality gate 的产物",
    )
    min_facts_count: int = Field(default=4, description="可复用缓存要求的最少事实数")
    require_video_for_video_request: bool = Field(
        default=True,
        description="当用户请求视频时，缓存必须包含可用视频产物",
    )
    collection_name: str = Field(default="research_artifacts", description="缓存索引集合名")

class RetrievalSettings(BaseSettings):
    """深度抓取/内容增强配置"""
    model_config = _env_prefixed("RETRIEVAL_")

    deep_enrichment_enabled: bool = Field(
        default=True,
        description="是否启用深度抓取增强（论文PDF/README/代码片段）",
    )
    deep_max_items_per_source: int = Field(default=5, description="每个来源最多深度抓取条数")
    deep_concurrency: int = Field(default=5, description="深度抓取并发度")
    deep_request_timeout_sec: float = Field(default=12.0, description="深度抓取单请求超时")
    deep_max_pdf_pages: int = Field(default=14, description="PDF最多提取页数")
    deep_max_chars_per_item: int = Field(default=12000, description="单条结果最多追加字符数")

class Settings(BaseSettings):
    """主配置类 - 聚合所有子配置"""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
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
    research_cache: ResearchCacheSettings = Field(default_factory=ResearchCacheSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    
    @classmethod
    def load_from_env_file(cls, env_path: Optional[Path] = None) -> "Settings":
        """从指定的 .env 文件加载配置"""
        if env_path is None:
            # 默认查找 config/.env
            env_path = Path(__file__).parent / ".env"
        
        if env_path.exists():
            from dotenv import load_dotenv
            load_dotenv(env_path)

        return cls()


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


def get_research_cache_settings() -> ResearchCacheSettings:
    return get_settings().research_cache


def get_retrieval_settings() -> RetrievalSettings:
    return get_settings().retrieval
