"""
Scrapers Module
"""
from .base import BaseScraper, RateLimitedScraper
from .arxiv_scraper import ArxivScraper, search_arxiv
from .huggingface_scraper import HuggingFaceScraper, search_hf_models, search_hf_datasets
from .semantic_scholar_scraper import SemanticScholarScraper
from .stackoverflow_scraper import StackOverflowScraper
from .hackernews_scraper import HackerNewsScraper
from .social import (
    TwitterScraper,
    RedditScraper,
    GitHubScraper,
    search_twitter,
    search_reddit,
    search_github,
)

__all__ = [
    # Base
    "BaseScraper",
    "RateLimitedScraper",
    # ArXiv
    "ArxivScraper",
    "search_arxiv",
    # HuggingFace
    "HuggingFaceScraper",
    "search_hf_models",
    "search_hf_datasets",
    # Semantic Scholar
    "SemanticScholarScraper",
    # Stack Overflow
    "StackOverflowScraper",
    # Hacker News
    "HackerNewsScraper",
    # Social
    "TwitterScraper",
    "RedditScraper",
    "GitHubScraper",
    "search_twitter",
    "search_reddit",
    "search_github",
]
