"""
Stack Overflow Scraper
技术问答搜索
API 文档: https://api.stackexchange.com/docs
"""
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging
import gzip
import json

import aiohttp

from .base import RateLimitedScraper
from models import StackOverflowQuestion, SourceType


logger = logging.getLogger(__name__)


class StackOverflowScraper(RateLimitedScraper[StackOverflowQuestion]):
    """
    Stack Overflow 抓取器
    
    特性:
    - 问题搜索 (关键词、标签)
    - 热门问答
    - 按投票/活跃度/最新排序
    - API 返回压缩数据 (gzip)
    
    限制:
    - 无 API Key: 300 requests/day
    - 有 API Key: 10,000 requests/day
    """
    
    BASE_URL = "https://api.stackexchange.com/2.3"
    
    def __init__(self):
        # Stack Exchange API 限制
        super().__init__(requests_per_second=2.0)
        self._api_key = self.settings.stackoverflow.api_key if hasattr(self.settings, 'stackoverflow') else None
    
    @property
    def source_type(self) -> SourceType:
        return SourceType.STACKOVERFLOW
    
    @property
    def name(self) -> str:
        return "Stack Overflow"
    
    def is_configured(self) -> bool:
        # 无需 API Key 也可使用 (有限额)
        return True
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建 HTTP 会话"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return self._session
    
    def _build_params(self, extra: Dict[str, Any] = None) -> Dict[str, Any]:
        """构建请求参数"""
        params = {
            "site": "stackoverflow",
            "filter": "withbody",  # 包含问题内容
        }
        if self._api_key:
            params["key"] = self._api_key
        if extra:
            params.update(extra)
        return params
    
    async def search(
        self, 
        query: str, 
        max_results: Optional[int] = None,
        sort: str = "votes",  # 默认按投票排序，获取高质量帖子
        tags: Optional[List[str]] = None,
        min_score: int = 0,  # 最低得分过滤
        min_answers: int = 0,  # 最少回答数
    ) -> List[StackOverflowQuestion]:
        """
        搜索问题
        
        Args:
            query: 搜索关键词
            max_results: 最大结果数
            sort: 排序方式 (votes, relevance, creation, activity)
            tags: 标签过滤
            min_score: 最低得分 (过滤低质量帖子)
            min_answers: 最少回答数
            
        Returns:
            问题列表
        """
        if max_results is None:
            max_results = self.settings.stackoverflow.max_results if hasattr(self.settings, 'stackoverflow') else 30
        
        # API 限制每页最多 100 条
        max_results = min(max_results, 100)
        
        logger.info(f"[Stack Overflow] Searching: {query}")
        
        await self._wait_for_rate_limit()
        
        session = await self._get_session()
        
        params = self._build_params({
            "order": "desc",
            "sort": sort,
            "q": query,  # 全文搜索
            "pagesize": max_results,
        })
        
        # 标签过滤
        if tags:
            params["tagged"] = ";".join(tags)
        
        # 质量过滤 (API 不直接支持，后置过滤)
        
        try:
            async with session.get(
                f"{self.BASE_URL}/search/advanced",
                params=params,
            ) as response:
                response.raise_for_status()
                
                # Stack Exchange API 返回 gzip 压缩数据
                raw_data = await response.read()
                try:
                    data = json.loads(gzip.decompress(raw_data))
                except gzip.BadGzipFile:
                    data = json.loads(raw_data)
                
                questions = []
                for item in data.get("items", []):
                    question = self._convert_to_question(item)
                    if question:
                        # 后置过滤：低质量帖子
                        if question.score >= min_score and question.answer_count >= min_answers:
                            questions.append(question)
                
                # 检查 API 配额
                quota_remaining = data.get("quota_remaining", 0)
                if quota_remaining < 50:
                    logger.warning(f"[Stack Overflow] Low quota remaining: {quota_remaining}")
                
                self._log_search(query, len(questions))
                return questions
                
        except aiohttp.ClientError as e:
            self._log_error("Search failed", e)
            return []
    
    async def search_by_tags(
        self, 
        tags: List[str], 
        max_results: int = 30,
        sort: str = "votes",
    ) -> List[StackOverflowQuestion]:
        """
        按标签搜索热门问题
        
        Args:
            tags: 标签列表
            max_results: 最大结果数
            sort: 排序方式
        """
        logger.info(f"[Stack Overflow] Searching by tags: {tags}")
        
        await self._wait_for_rate_limit()
        
        session = await self._get_session()
        
        params = self._build_params({
            "order": "desc",
            "sort": sort,
            "tagged": ";".join(tags),
            "pagesize": min(max_results, 100),
        })
        
        try:
            async with session.get(
                f"{self.BASE_URL}/questions",
                params=params,
            ) as response:
                response.raise_for_status()
                
                raw_data = await response.read()
                try:
                    data = json.loads(gzip.decompress(raw_data))
                except gzip.BadGzipFile:
                    data = json.loads(raw_data)
                
                questions = []
                for item in data.get("items", []):
                    question = self._convert_to_question(item)
                    if question:
                        questions.append(question)
                
                return questions
                
        except aiohttp.ClientError as e:
            self._log_error("Search by tags failed", e)
            return []
    
    async def get_details(self, question_id: str) -> Optional[StackOverflowQuestion]:
        """
        获取问题详情
        
        Args:
            question_id: 问题ID
        """
        await self._wait_for_rate_limit()
        
        session = await self._get_session()
        
        params = self._build_params({
            "filter": "withbody",
        })
        
        try:
            async with session.get(
                f"{self.BASE_URL}/questions/{question_id}",
                params=params,
            ) as response:
                response.raise_for_status()
                
                raw_data = await response.read()
                try:
                    data = json.loads(gzip.decompress(raw_data))
                except gzip.BadGzipFile:
                    data = json.loads(raw_data)
                
                items = data.get("items", [])
                if items:
                    return self._convert_to_question(items[0])
                return None
                
        except aiohttp.ClientError as e:
            self._log_error(f"Get details failed for {question_id}", e)
            return None
    
    async def get_answers(
        self, 
        question_id: str, 
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        获取问题的答案
        
        Args:
            question_id: 问题ID
            max_results: 最大结果数
        """
        await self._wait_for_rate_limit()
        
        session = await self._get_session()
        
        params = self._build_params({
            "order": "desc",
            "sort": "votes",
            "filter": "withbody",
            "pagesize": min(max_results, 100),
        })
        
        try:
            async with session.get(
                f"{self.BASE_URL}/questions/{question_id}/answers",
                params=params,
            ) as response:
                response.raise_for_status()
                
                raw_data = await response.read()
                try:
                    data = json.loads(gzip.decompress(raw_data))
                except gzip.BadGzipFile:
                    data = json.loads(raw_data)
                
                answers = []
                for item in data.get("items", []):
                    answers.append({
                        "id": str(item.get("answer_id")),
                        "body": item.get("body", ""),
                        "score": item.get("score", 0),
                        "is_accepted": item.get("is_accepted", False),
                        "author": item.get("owner", {}).get("display_name", "Unknown"),
                        "created_at": datetime.fromtimestamp(item.get("creation_date", 0)),
                    })
                
                return answers
                
        except aiohttp.ClientError as e:
            self._log_error(f"Get answers failed for {question_id}", e)
            return []
    
    def _convert_to_question(self, data: Dict[str, Any]) -> Optional[StackOverflowQuestion]:
        """转换 API 响应为 StackOverflowQuestion 模型"""
        if not data or not data.get("question_id"):
            return None
        
        owner = data.get("owner", {})
        
        return StackOverflowQuestion(
            id=str(data["question_id"]),
            title=data.get("title", "Untitled"),
            body=data.get("body"),
            tags=data.get("tags", []),
            author=owner.get("display_name", "Unknown"),
            author_reputation=owner.get("reputation", 0),
            url=data.get("link", f"https://stackoverflow.com/questions/{data['question_id']}"),
            score=data.get("score", 0),
            view_count=data.get("view_count", 0),
            answer_count=data.get("answer_count", 0),
            is_answered=data.get("is_answered", False),
            accepted_answer_id=str(data["accepted_answer_id"]) if data.get("accepted_answer_id") else None,
            created_at=datetime.fromtimestamp(data["creation_date"]) if data.get("creation_date") else None,
            last_activity_at=datetime.fromtimestamp(data["last_activity_date"]) if data.get("last_activity_date") else None,
            source=SourceType.STACKOVERFLOW,
            extra={
                "owner_user_id": owner.get("user_id"),
                "owner_profile_image": owner.get("profile_image"),
            },
        )
