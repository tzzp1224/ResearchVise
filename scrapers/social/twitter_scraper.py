"""
Twitter/X Scraper
从 Twitter/X 抓取社交媒体讨论
"""
import asyncio
from datetime import datetime
from typing import List, Optional
import logging

from tenacity import retry, stop_after_attempt, wait_exponential

from scrapers.base import RateLimitedScraper
from models import SocialPost, SourceType
from config import get_settings


logger = logging.getLogger(__name__)


class TwitterScraper(RateLimitedScraper[SocialPost]):
    """
    Twitter/X 抓取器
    使用 Twitter API v2 (需要 Bearer Token)
    """
    
    def __init__(self):
        super().__init__(requests_per_second=0.5)  # Twitter API 有严格限制
        self._twitter_settings = get_settings().twitter
        self._client = None
    
    @property
    def source_type(self) -> SourceType:
        return SourceType.TWITTER
    
    @property
    def name(self) -> str:
        return "Twitter/X"
    
    def is_configured(self) -> bool:
        return bool(self._twitter_settings.bearer_token)
    
    def _get_client(self):
        """获取 Twitter 客户端"""
        if self._client is None:
            try:
                import tweepy
                self._client = tweepy.Client(
                    bearer_token=self._twitter_settings.bearer_token,
                    wait_on_rate_limit=True,
                )
            except ImportError:
                raise ImportError("Please install tweepy: pip install tweepy")
        return self._client
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30)
    )
    async def search(
        self, 
        query: str, 
        max_results: Optional[int] = None,
    ) -> List[SocialPost]:
        """
        搜索推文
        
        Args:
            query: 搜索关键词
            max_results: 最大返回结果数 (Twitter API限制每次最多100)
            
        Returns:
            推文列表
        """
        if not self.is_configured():
            logger.warning("[Twitter] API not configured, skipping...")
            return []
        
        if max_results is None:
            max_results = min(self._twitter_settings.max_results, 100)
        
        # Twitter API 限制
        max_results = min(max_results, 100)
        
        logger.info(f"[Twitter] Searching: {query}")
        
        await self._wait_for_rate_limit()
        
        loop = asyncio.get_event_loop()
        
        try:
            results = await loop.run_in_executor(
                None,
                self._sync_search,
                query,
                max_results,
            )
            
            posts = [self._convert_to_post(tweet) for tweet in results]
            self._log_search(query, len(posts))
            
            return posts
            
        except Exception as e:
            self._log_error(f"Search failed for '{query}'", e)
            return []
    
    def _sync_search(self, query: str, max_results: int) -> list:
        """同步搜索"""
        client = self._get_client()
        
        # 构建搜索查询 (排除 retweets，只搜索英文和中文)
        search_query = f"{query} -is:retweet lang:en OR lang:zh"
        
        response = client.search_recent_tweets(
            query=search_query,
            max_results=max_results,
            tweet_fields=["created_at", "public_metrics", "author_id", "conversation_id"],
            user_fields=["username", "name", "url"],
            expansions=["author_id"],
        )
        
        tweets = response.data or []
        
        # 创建用户ID到用户信息的映射
        users_map = {}
        if response.includes and "users" in response.includes:
            for user in response.includes["users"]:
                users_map[user.id] = user
        
        # 将用户信息附加到推文
        result = []
        for tweet in tweets:
            tweet._user = users_map.get(tweet.author_id)
            result.append(tweet)
        
        return result
    
    async def get_details(self, tweet_id: str) -> Optional[SocialPost]:
        """
        获取单条推文详情
        
        Args:
            tweet_id: 推文ID
            
        Returns:
            推文详情
        """
        if not self.is_configured():
            return None
        
        loop = asyncio.get_event_loop()
        
        try:
            await self._wait_for_rate_limit()
            
            result = await loop.run_in_executor(
                None,
                self._sync_get_tweet,
                tweet_id,
            )
            
            if result:
                return self._convert_to_post(result)
            return None
            
        except Exception as e:
            self._log_error(f"Failed to get tweet {tweet_id}", e)
            return None
    
    def _sync_get_tweet(self, tweet_id: str):
        """同步获取推文"""
        client = self._get_client()
        
        response = client.get_tweet(
            tweet_id,
            tweet_fields=["created_at", "public_metrics", "author_id"],
            user_fields=["username", "name", "url"],
            expansions=["author_id"],
        )
        
        if response.data:
            tweet = response.data
            if response.includes and "users" in response.includes:
                tweet._user = response.includes["users"][0]
            return tweet
        return None
    
    def _convert_to_post(self, tweet) -> SocialPost:
        """将 Tweet 转换为 SocialPost"""
        # 获取用户信息
        author = "Unknown"
        author_url = None
        
        if hasattr(tweet, '_user') and tweet._user:
            author = f"@{tweet._user.username}"
            author_url = f"https://twitter.com/{tweet._user.username}"
        
        # 获取互动数据
        metrics = tweet.public_metrics or {}
        
        return SocialPost(
            id=str(tweet.id),
            content=tweet.text,
            author=author,
            author_url=author_url,
            created_at=tweet.created_at,
            url=f"https://twitter.com/i/web/status/{tweet.id}",
            likes=metrics.get("like_count", 0),
            reposts=metrics.get("retweet_count", 0),
            comments=metrics.get("reply_count", 0),
            source=SourceType.TWITTER,
            extra={
                "quote_count": metrics.get("quote_count", 0),
                "impression_count": metrics.get("impression_count", 0),
            }
        )


# 便捷函数
async def search_twitter(query: str, max_results: int = 50) -> List[SocialPost]:
    """便捷函数：搜索 Twitter"""
    async with TwitterScraper() as scraper:
        return await scraper.search(query, max_results)
