"""
Semantic Scholar Scraper
学术论文搜索 + 引用关系分析
API 文档: https://api.semanticscholar.org/
"""
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging

import aiohttp

from .base import RateLimitedScraper
from models import Paper, Author, SourceType


logger = logging.getLogger(__name__)


class SemanticScholarScraper(RateLimitedScraper[Paper]):
    """
    Semantic Scholar 抓取器
    
    特性:
    - 学术论文搜索
    - 引用关系 (引用/被引用)
    - 作者信息
    - 未认证: 1 req/s, 认证后: 10 req/s
    """
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    # 请求的论文字段
    PAPER_FIELDS = [
        "paperId",
        "title",
        "abstract",
        "year",
        "authors",
        "citationCount",
        "referenceCount",
        "influentialCitationCount",
        "url",
        "openAccessPdf",
        "fieldsOfStudy",
        "publicationDate",
        "venue",
        "externalIds",
    ]
    
    def __init__(self):
        # 未认证用户限制 1 req/s
        super().__init__(requests_per_second=1.0)
        self._api_key = self.settings.semantic_scholar.api_key if hasattr(self.settings, 'semantic_scholar') else None
        
        # 有 API Key 可以提升到 10 req/s
        if self._api_key:
            self._rate_limit = 10.0
    
    @property
    def source_type(self) -> SourceType:
        return SourceType.SEMANTIC_SCHOLAR
    
    @property
    def name(self) -> str:
        return "Semantic Scholar"
    
    def is_configured(self) -> bool:
        # 无需 API Key 也可使用 (有限额)
        return True
    
    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        headers = {
            "Accept": "application/json",
        }
        if self._api_key:
            headers["x-api-key"] = self._api_key
        return headers
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建 HTTP 会话"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers=self._get_headers(),
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return self._session
    
    async def search(
        self, 
        query: str, 
        max_results: Optional[int] = None,
        year_range: Optional[tuple] = None,
        fields_of_study: Optional[List[str]] = None,
    ) -> List[Paper]:
        """
        搜索论文
        
        Args:
            query: 搜索关键词
            max_results: 最大结果数
            year_range: 年份范围 (start_year, end_year)
            fields_of_study: 领域过滤
            
        Returns:
            论文列表
        """
        if max_results is None:
            max_results = self.settings.semantic_scholar.max_results if hasattr(self.settings, 'semantic_scholar') else 30
        
        # Semantic Scholar API 限制每次最多 100 条
        max_results = min(max_results, 100)
        
        logger.info(f"[Semantic Scholar] Searching: {query}")
        
        await self._wait_for_rate_limit()
        
        session = await self._get_session()
        
        params = {
            "query": query,
            "limit": max_results,
            "fields": ",".join(self.PAPER_FIELDS),
        }
        
        # 年份过滤
        if year_range:
            params["year"] = f"{year_range[0]}-{year_range[1]}"
        
        # 领域过滤
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)
        
        try:
            async with session.get(
                f"{self.BASE_URL}/paper/search",
                params=params,
            ) as response:
                if response.status == 429:
                    logger.warning("[Semantic Scholar] Rate limit exceeded")
                    return []
                
                response.raise_for_status()
                data = await response.json()
                
                papers = []
                for item in data.get("data", []):
                    paper = self._convert_to_paper(item)
                    if paper:
                        papers.append(paper)
                
                self._log_search(query, len(papers))
                return papers
                
        except aiohttp.ClientError as e:
            self._log_error("Search failed", e)
            return []
    
    async def get_details(self, paper_id: str) -> Optional[Paper]:
        """
        获取论文详情
        
        Args:
            paper_id: 论文ID (Semantic Scholar ID 或 DOI/ArXiv ID)
        """
        await self._wait_for_rate_limit()
        
        session = await self._get_session()
        
        try:
            async with session.get(
                f"{self.BASE_URL}/paper/{paper_id}",
                params={"fields": ",".join(self.PAPER_FIELDS)},
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return self._convert_to_paper(data)
                
        except aiohttp.ClientError as e:
            self._log_error(f"Get details failed for {paper_id}", e)
            return None
    
    async def get_citations(
        self, 
        paper_id: str, 
        max_results: int = 50,
    ) -> List[Paper]:
        """
        获取引用该论文的论文列表
        
        Args:
            paper_id: 论文ID
            max_results: 最大结果数
        """
        await self._wait_for_rate_limit()
        
        session = await self._get_session()
        
        try:
            async with session.get(
                f"{self.BASE_URL}/paper/{paper_id}/citations",
                params={
                    "fields": "paperId,title,abstract,year,authors,citationCount,url",
                    "limit": min(max_results, 1000),
                },
            ) as response:
                response.raise_for_status()
                data = await response.json()
                
                papers = []
                for item in data.get("data", []):
                    citing_paper = item.get("citingPaper", {})
                    paper = self._convert_to_paper(citing_paper)
                    if paper:
                        papers.append(paper)
                
                logger.info(f"[Semantic Scholar] Found {len(papers)} citations for {paper_id}")
                return papers
                
        except aiohttp.ClientError as e:
            self._log_error(f"Get citations failed for {paper_id}", e)
            return []
    
    async def get_references(
        self, 
        paper_id: str, 
        max_results: int = 50,
    ) -> List[Paper]:
        """
        获取该论文引用的论文列表
        
        Args:
            paper_id: 论文ID
            max_results: 最大结果数
        """
        await self._wait_for_rate_limit()
        
        session = await self._get_session()
        
        try:
            async with session.get(
                f"{self.BASE_URL}/paper/{paper_id}/references",
                params={
                    "fields": "paperId,title,abstract,year,authors,citationCount,url",
                    "limit": min(max_results, 1000),
                },
            ) as response:
                response.raise_for_status()
                data = await response.json()
                
                papers = []
                for item in data.get("data", []):
                    cited_paper = item.get("citedPaper", {})
                    paper = self._convert_to_paper(cited_paper)
                    if paper:
                        papers.append(paper)
                
                logger.info(f"[Semantic Scholar] Found {len(papers)} references for {paper_id}")
                return papers
                
        except aiohttp.ClientError as e:
            self._log_error(f"Get references failed for {paper_id}", e)
            return []
    
    def _convert_to_paper(self, data: Dict[str, Any]) -> Optional[Paper]:
        """转换 API 响应为 Paper 模型"""
        if not data or not data.get("paperId"):
            return None
        
        # 解析作者
        authors = []
        for author_data in data.get("authors", []):
            authors.append(Author(
                name=author_data.get("name", "Unknown"),
                url=f"https://www.semanticscholar.org/author/{author_data.get('authorId', '')}",
            ))
        
        # 解析日期
        pub_date = None
        if data.get("publicationDate"):
            try:
                pub_date = datetime.strptime(data["publicationDate"], "%Y-%m-%d")
            except ValueError:
                pass
        elif data.get("year"):
            pub_date = datetime(int(data["year"]), 1, 1)
        
        # PDF URL
        pdf_url = None
        if data.get("openAccessPdf"):
            pdf_url = data["openAccessPdf"].get("url")
        
        # 外部 ID
        external_ids = data.get("externalIds", {})
        
        return Paper(
            id=data["paperId"],
            title=data.get("title", "Untitled"),
            abstract=data.get("abstract") or "",  # 处理 None
            authors=authors,
            published_date=pub_date,
            categories=data.get("fieldsOfStudy") or [],
            url=data.get("url", f"https://www.semanticscholar.org/paper/{data['paperId']}"),
            pdf_url=pdf_url,
            source=SourceType.SEMANTIC_SCHOLAR,
            citation_count=data.get("citationCount", 0),
            extra={
                "venue": data.get("venue"),
                "reference_count": data.get("referenceCount", 0),
                "influential_citation_count": data.get("influentialCitationCount", 0),
                "arxiv_id": external_ids.get("ArXiv"),
                "doi": external_ids.get("DOI"),
                "pubmed_id": external_ids.get("PubMed"),
            },
        )
