"""Minimal configuration for v2 connectors and scraper adapters."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _env_prefixed(prefix: str) -> SettingsConfigDict:
    return SettingsConfigDict(env_prefix=prefix, extra="ignore")


class ArxivSettings(BaseSettings):
    model_config = _env_prefixed("ARXIV_")
    max_results: int = Field(default=50)
    sort_by: str = Field(default="submittedDate")


class HuggingFaceSettings(BaseSettings):
    model_config = _env_prefixed("HUGGINGFACE_")
    token: Optional[str] = Field(default=None)
    max_results: int = Field(default=30)


class TwitterSettings(BaseSettings):
    model_config = _env_prefixed("TWITTER_")
    bearer_token: Optional[str] = Field(default=None)
    max_results: int = Field(default=100)


class RedditSettings(BaseSettings):
    model_config = _env_prefixed("REDDIT_")
    client_id: Optional[str] = Field(default=None)
    client_secret: Optional[str] = Field(default=None)
    user_agent: str = Field(default="AcademicResearchAgent/2.0")
    max_results: int = Field(default=50)


class GitHubSettings(BaseSettings):
    model_config = _env_prefixed("GITHUB_")
    token: Optional[str] = Field(default=None)
    max_results: int = Field(default=30)


class SemanticScholarSettings(BaseSettings):
    model_config = _env_prefixed("SEMANTIC_SCHOLAR_")
    api_key: Optional[str] = Field(default=None)
    max_results: int = Field(default=30)
    requests_per_second: float = Field(default=0.8)


class StackOverflowSettings(BaseSettings):
    model_config = _env_prefixed("STACKOVERFLOW_")
    api_key: Optional[str] = Field(default=None)
    max_results: int = Field(default=30)


class HackerNewsSettings(BaseSettings):
    model_config = _env_prefixed("HACKERNEWS_")
    max_results: int = Field(default=30)


class GeneralSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")
    request_timeout: int = Field(default=30)
    max_retries: int = Field(default=3)
    retry_delay: int = Field(default=1)


class Settings(BaseSettings):
    """Aggregated settings used by scraper-backed connectors."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    arxiv: ArxivSettings = Field(default_factory=ArxivSettings)
    huggingface: HuggingFaceSettings = Field(default_factory=HuggingFaceSettings)
    twitter: TwitterSettings = Field(default_factory=TwitterSettings)
    reddit: RedditSettings = Field(default_factory=RedditSettings)
    github: GitHubSettings = Field(default_factory=GitHubSettings)
    semantic_scholar: SemanticScholarSettings = Field(default_factory=SemanticScholarSettings)
    stackoverflow: StackOverflowSettings = Field(default_factory=StackOverflowSettings)
    hackernews: HackerNewsSettings = Field(default_factory=HackerNewsSettings)
    general: GeneralSettings = Field(default_factory=GeneralSettings)

    @classmethod
    def load_from_env_file(cls, env_path: Optional[Path] = None) -> "Settings":
        if env_path is None:
            env_path = Path(__file__).parent / ".env"
        if env_path.exists():
            from dotenv import load_dotenv

            load_dotenv(env_path)
        return cls()


@lru_cache
def get_settings() -> Settings:
    return Settings.load_from_env_file()


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
