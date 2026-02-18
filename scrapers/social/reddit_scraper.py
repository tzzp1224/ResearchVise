"""
Reddit Scraper
从 Reddit 抓取社区讨论
"""
from datetime import datetime
from typing import List, Optional
import logging

from tenacity import retry, stop_after_attempt, wait_exponential

from scrapers.base import RateLimitedScraper
from models import SocialPost, SourceType
from config import get_settings


logger = logging.getLogger(__name__)


class RedditScraper(RateLimitedScraper[SocialPost]):
    """
    Reddit 抓取器
    使用 PRAW (Python Reddit API Wrapper)
    """
    
    # 与AI/ML相关的热门 Subreddits
    DEFAULT_SUBREDDITS = [
        "MachineLearning",
        "deeplearning",
        "artificial",
        "LocalLLaMA",
        "learnmachinelearning",
        "MLQuestions",
        "datascience",
        "LanguageTechnology",
    ]
    
    def __init__(self):
        super().__init__(requests_per_second=1.0)
        self._reddit_settings = get_settings().reddit
        self._reddit = None
    
    @property
    def source_type(self) -> SourceType:
        return SourceType.REDDIT
    
    @property
    def name(self) -> str:
        return "Reddit"
    
    def is_configured(self) -> bool:
        return bool(
            self._reddit_settings.client_id and 
            self._reddit_settings.client_secret
        )
    
    def _get_reddit(self):
        """获取 Reddit 客户端"""
        if self._reddit is None:
            try:
                import praw
                self._reddit = praw.Reddit(
                    client_id=self._reddit_settings.client_id,
                    client_secret=self._reddit_settings.client_secret,
                    user_agent=self._reddit_settings.user_agent,
                )
            except ImportError:
                raise ImportError("Please install praw: pip install praw")
        return self._reddit
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def search(
        self, 
        query: str, 
        max_results: Optional[int] = None,
        subreddits: Optional[List[str]] = None,
        time_filter: str = "month",  # hour, day, week, month, year, all
        sort: str = "relevance",  # relevance, hot, top, new, comments
    ) -> List[SocialPost]:
        """
        搜索 Reddit 帖子
        
        Args:
            query: 搜索关键词
            max_results: 最大返回结果数
            subreddits: 限制搜索的 subreddit 列表
            time_filter: 时间过滤
            sort: 排序方式
            
        Returns:
            帖子列表
        """
        if not self.is_configured():
            logger.warning("[Reddit] API not configured, skipping...")
            return []
        
        if max_results is None:
            max_results = self._reddit_settings.max_results
        
        if subreddits is None:
            subreddits = self.DEFAULT_SUBREDDITS
        
        logger.info(f"[Reddit] Searching: {query} in {len(subreddits)} subreddits")
        
        await self._wait_for_rate_limit()

        try:
            results = await self._run_blocking(
                self._sync_search,
                query,
                max_results,
                subreddits,
                time_filter,
                sort,
            )
            
            posts = [self._convert_to_post(submission) for submission in results]
            self._log_search(query, len(posts))
            
            return posts
            
        except Exception as exc:
            self._log_error(f"Search failed for '{query}'", exc)
            return []
    
    def _sync_search(
        self,
        query: str,
        max_results: int,
        subreddits: List[str],
        time_filter: str,
        sort: str,
    ) -> list:
        """同步搜索"""
        reddit = self._get_reddit()
        
        # 在多个 subreddit 中搜索
        subreddit_str = "+".join(subreddits)
        subreddit = reddit.subreddit(subreddit_str)
        
        submissions = list(subreddit.search(
            query=query,
            sort=sort,
            time_filter=time_filter,
            limit=max_results,
        ))
        
        return submissions
    
    async def get_hot_posts(
        self,
        subreddits: Optional[List[str]] = None,
        max_results: int = 25,
    ) -> List[SocialPost]:
        """
        获取热门帖子
        
        Args:
            subreddits: subreddit 列表
            max_results: 最大返回结果数
            
        Returns:
            热门帖子列表
        """
        if not self.is_configured():
            return []
        
        if subreddits is None:
            subreddits = self.DEFAULT_SUBREDDITS

        try:
            await self._wait_for_rate_limit()
            
            results = await self._run_blocking(
                self._sync_get_hot,
                subreddits,
                max_results,
            )
            
            return [self._convert_to_post(s) for s in results]
            
        except Exception as exc:
            self._log_error("Failed to get hot posts", exc)
            return []
    
    def _sync_get_hot(self, subreddits: List[str], max_results: int) -> list:
        """同步获取热门帖子"""
        reddit = self._get_reddit()
        subreddit_str = "+".join(subreddits)
        subreddit = reddit.subreddit(subreddit_str)
        return list(subreddit.hot(limit=max_results))
    
    async def get_details(self, post_id: str) -> Optional[SocialPost]:
        """
        获取帖子详情
        
        Args:
            post_id: Reddit 帖子ID
            
        Returns:
            帖子详情
        """
        if not self.is_configured():
            return None

        try:
            await self._wait_for_rate_limit()
            
            result = await self._run_blocking(
                self._sync_get_submission,
                post_id,
            )
            
            if result:
                return self._convert_to_post(result)
            return None
            
        except Exception as exc:
            self._log_error(f"Failed to get post {post_id}", exc)
            return None
    
    def _sync_get_submission(self, post_id: str):
        """同步获取帖子"""
        reddit = self._get_reddit()
        return reddit.submission(id=post_id)
    
    def _convert_to_post(self, submission) -> SocialPost:
        """将 Reddit Submission 转换为 SocialPost"""
        # 组合标题和内容
        content = submission.title
        if submission.selftext:
            content += f"\n\n{submission.selftext[:500]}..."  # 截断长文本
        
        return SocialPost(
            id=submission.id,
            content=content,
            author=f"u/{submission.author.name}" if submission.author else "[deleted]",
            author_url=f"https://reddit.com/u/{submission.author.name}" if submission.author else None,
            created_at=datetime.fromtimestamp(submission.created_utc),
            url=f"https://reddit.com{submission.permalink}",
            likes=submission.score,
            reposts=0,  # Reddit doesn't have reposts
            comments=submission.num_comments,
            source=SourceType.REDDIT,
            extra={
                "subreddit": submission.subreddit.display_name,
                "upvote_ratio": submission.upvote_ratio,
                "is_self": submission.is_self,
                "link_flair_text": submission.link_flair_text,
                "num_crossposts": submission.num_crossposts,
            }
        )


# 便捷函数
async def search_reddit(
    query: str, 
    max_results: int = 50,
    subreddits: Optional[List[str]] = None,
) -> List[SocialPost]:
    """便捷函数：搜索 Reddit"""
    async with RedditScraper() as scraper:
        return await scraper.search(query, max_results, subreddits)
