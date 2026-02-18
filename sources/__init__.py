"""Source connectors for v2 ingestion."""

from .connectors import (
    fetch_github_releases,
    fetch_github_trending,
    fetch_hackernews_top,
    fetch_huggingface_trending,
    fetch_rss_feed,
    fetch_web_article,
)

__all__ = [
    "fetch_github_releases",
    "fetch_github_trending",
    "fetch_hackernews_top",
    "fetch_huggingface_trending",
    "fetch_rss_feed",
    "fetch_web_article",
]
