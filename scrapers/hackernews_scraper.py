"""
Hacker News Scraper
极客社区热门讨论搜索
使用 Algolia HN Search API (推荐)
API 文档: https://hn.algolia.com/api
"""
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging

import aiohttp

from .base import RateLimitedScraper
from models import HackerNewsItem, SourceType


logger = logging.getLogger(__name__)


class HackerNewsScraper(RateLimitedScraper[HackerNewsItem]):
    """
    Hacker News 抓取器 (Algolia API)
    
    特性:
    - 全文搜索 (标题 + URL + 内容)
    - 时间范围过滤
    - 按时间/热度排序
    - 无需 API Key，无严格限制
    
    官方 Firebase API (备用):
    - https://hacker-news.firebaseio.com/v0/
    - 仅支持获取最新/热门列表，不支持搜索
    """
    
    # Algolia HN Search API
    ALGOLIA_URL = "https://hn.algolia.com/api/v1"
    
    # Firebase API (备用)
    FIREBASE_URL = "https://hacker-news.firebaseio.com/v0"
    
    def __init__(self):
        # Algolia API 无严格限制，但保持合理频率
        super().__init__(requests_per_second=5.0)
    
    @property
    def source_type(self) -> SourceType:
        return SourceType.HACKERNEWS
    
    @property
    def name(self) -> str:
        return "Hacker News"
    
    def is_configured(self) -> bool:
        # 无需配置，完全开放
        return True
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建 HTTP 会话"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return self._session
    
    async def search(
        self, 
        query: str, 
        max_results: Optional[int] = None,
        sort_by: str = "relevance",
        time_range: Optional[str] = None,
    ) -> List[HackerNewsItem]:
        """
        搜索 Hacker News
        
        Args:
            query: 搜索关键词
            max_results: 最大结果数
            sort_by: 排序方式 (relevance, date)
            time_range: 时间范围 (last_24h, past_week, past_month, past_year)
            
        Returns:
            帖子列表
        """
        if max_results is None:
            max_results = self.settings.hackernews.max_results if hasattr(self.settings, 'hackernews') else 30
        
        logger.info(f"[Hacker News] Searching: {query}")
        
        await self._wait_for_rate_limit()
        
        session = await self._get_session()
        
        # 选择 API 端点
        if sort_by == "date":
            endpoint = "search_by_date"
        else:
            endpoint = "search"
        
        params = {
            "query": query,
            "tags": "story",  # 仅搜索故事 (不含评论)
            "hitsPerPage": min(max_results, 100),
        }
        
        # 时间范围过滤
        if time_range:
            now = int(datetime.now().timestamp())
            time_filters = {
                "last_24h": now - 86400,
                "past_week": now - 604800,
                "past_month": now - 2592000,
                "past_year": now - 31536000,
            }
            if time_range in time_filters:
                params["numericFilters"] = f"created_at_i>{time_filters[time_range]}"
        
        try:
            async with session.get(
                f"{self.ALGOLIA_URL}/{endpoint}",
                params=params,
            ) as response:
                response.raise_for_status()
                data = await response.json()
                
                items = []
                for hit in data.get("hits", []):
                    item = self._convert_algolia_to_item(hit)
                    if item:
                        items.append(item)
                
                self._log_search(query, len(items))
                return items
                
        except aiohttp.ClientError as e:
            self._log_error("Search failed", e)
            return []
    
    async def get_front_page(self, max_results: int = 30) -> List[HackerNewsItem]:
        """
        获取首页热门帖子 (使用 Firebase API)
        
        Args:
            max_results: 最大结果数
        """
        logger.info("[Hacker News] Fetching front page")
        
        session = await self._get_session()
        
        try:
            # 获取热门帖子 ID 列表
            async with session.get(f"{self.FIREBASE_URL}/topstories.json") as response:
                response.raise_for_status()
                story_ids = await response.json()
            
            # 限制数量
            story_ids = story_ids[:max_results]
            
            # 并行获取详情 (分批以避免过多并发)
            items = []
            batch_size = 10
            
            for i in range(0, len(story_ids), batch_size):
                batch_ids = story_ids[i:i + batch_size]
                batch_items = await asyncio.gather(
                    *[self._get_firebase_item(str(sid)) for sid in batch_ids],
                    return_exceptions=True,
                )
                
                for item in batch_items:
                    if isinstance(item, HackerNewsItem):
                        items.append(item)
            
            logger.info(f"[Hacker News] Fetched {len(items)} front page items")
            return items
            
        except aiohttp.ClientError as e:
            self._log_error("Get front page failed", e)
            return []
    
    async def get_new_stories(self, max_results: int = 30) -> List[HackerNewsItem]:
        """
        获取最新帖子
        """
        logger.info("[Hacker News] Fetching new stories")
        
        session = await self._get_session()
        
        try:
            async with session.get(f"{self.FIREBASE_URL}/newstories.json") as response:
                response.raise_for_status()
                story_ids = await response.json()
            
            story_ids = story_ids[:max_results]
            
            items = []
            batch_size = 10
            
            for i in range(0, len(story_ids), batch_size):
                batch_ids = story_ids[i:i + batch_size]
                batch_items = await asyncio.gather(
                    *[self._get_firebase_item(str(sid)) for sid in batch_ids],
                    return_exceptions=True,
                )
                
                for item in batch_items:
                    if isinstance(item, HackerNewsItem):
                        items.append(item)
            
            return items
            
        except aiohttp.ClientError as e:
            self._log_error("Get new stories failed", e)
            return []
    
    async def get_details(self, item_id: str) -> Optional[HackerNewsItem]:
        """
        获取帖子详情
        """
        return await self._get_firebase_item(item_id)
    
    async def _get_firebase_item(self, item_id: str) -> Optional[HackerNewsItem]:
        """通过 Firebase API 获取单个帖子"""
        await self._wait_for_rate_limit()
        
        session = await self._get_session()
        
        try:
            async with session.get(
                f"{self.FIREBASE_URL}/item/{item_id}.json"
            ) as response:
                response.raise_for_status()
                data = await response.json()
                
                if data:
                    return self._convert_firebase_to_item(data)
                return None
                
        except aiohttp.ClientError as e:
            self._log_error(f"Get item {item_id} failed", e)
            return None
    
    def _convert_algolia_to_item(self, data: Dict[str, Any]) -> Optional[HackerNewsItem]:
        """转换 Algolia API 响应为 HackerNewsItem"""
        if not data or not data.get("objectID"):
            return None
        
        item_id = data["objectID"]
        
        return HackerNewsItem(
            id=item_id,
            title=data.get("title", "Untitled"),
            url=data.get("url"),
            text=data.get("story_text"),
            author=data.get("author", "Unknown"),
            points=data.get("points", 0),
            comment_count=data.get("num_comments", 0),
            hn_url=f"https://news.ycombinator.com/item?id={item_id}",
            item_type=self._parse_item_type(data.get("_tags", [])),
            created_at=datetime.fromtimestamp(data["created_at_i"]) if data.get("created_at_i") else None,
            source=SourceType.HACKERNEWS,
            extra={
                "relevancy_score": data.get("relevancy_score"),
            },
        )
    
    def _convert_firebase_to_item(self, data: Dict[str, Any]) -> Optional[HackerNewsItem]:
        """转换 Firebase API 响应为 HackerNewsItem"""
        if not data or not data.get("id"):
            return None
        
        item_id = str(data["id"])
        item_type = data.get("type", "story")
        
        # 跳过评论和已删除内容
        if item_type == "comment" or data.get("deleted") or data.get("dead"):
            return None
        
        return HackerNewsItem(
            id=item_id,
            title=data.get("title", "Untitled"),
            url=data.get("url"),
            text=data.get("text"),
            author=data.get("by", "Unknown"),
            points=data.get("score", 0),
            comment_count=len(data.get("kids", [])),
            hn_url=f"https://news.ycombinator.com/item?id={item_id}",
            item_type=item_type,
            created_at=datetime.fromtimestamp(data["time"]) if data.get("time") else None,
            source=SourceType.HACKERNEWS,
        )
    
    def _parse_item_type(self, tags: List[str]) -> str:
        """从 Algolia tags 解析帖子类型"""
        type_map = {
            "ask_hn": "ask",
            "show_hn": "show",
            "job": "job",
            "poll": "poll",
            "story": "story",
        }
        
        for tag in tags:
            if tag in type_map:
                return type_map[tag]
        
        return "story"
