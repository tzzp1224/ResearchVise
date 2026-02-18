"""
ArXiv Scraper
从 ArXiv 抓取学术论文
"""
from typing import List, Optional
import logging

import arxiv
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BaseScraper
from models import Paper, Author, SourceType
from config import get_settings


logger = logging.getLogger(__name__)


class ArxivScraper(BaseScraper[Paper]):
    """
    ArXiv 论文抓取器
    使用官方 arxiv Python 库
    """
    
    def __init__(self):
        super().__init__()
        self._arxiv_settings = get_settings().arxiv
        
        # 排序方式映射
        self._sort_map = {
            "relevance": arxiv.SortCriterion.Relevance,
            "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
            "submittedDate": arxiv.SortCriterion.SubmittedDate,
        }
    
    @property
    def source_type(self) -> SourceType:
        return SourceType.ARXIV
    
    @property
    def name(self) -> str:
        return "ArXiv"
    
    def is_configured(self) -> bool:
        # ArXiv 使用公开API，无需配置
        return True
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def search(
        self, 
        query: str, 
        max_results: Optional[int] = None,
        sort_by: Optional[str] = None,
        categories: Optional[List[str]] = None,
    ) -> List[Paper]:
        """
        搜索 ArXiv 论文
        
        Args:
            query: 搜索关键词
            max_results: 最大返回结果数
            sort_by: 排序方式 (relevance, lastUpdatedDate, submittedDate)
            categories: 限制搜索的分类 (如 ["cs.AI", "cs.LG"])
            
        Returns:
            论文列表
        """
        if max_results is None:
            max_results = self._arxiv_settings.max_results
        
        if sort_by is None:
            sort_by = self._arxiv_settings.sort_by
        
        sort_criterion = self._sort_map.get(sort_by, arxiv.SortCriterion.Relevance)
        
        # 构建查询
        search_query = query
        if categories:
            cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
            search_query = f"({query}) AND ({cat_query})"
        
        logger.info(f"[ArXiv] Searching: {search_query}")
        
        results = await self._run_blocking(
            self._sync_search,
            search_query,
            max_results,
            sort_criterion,
        )
        
        papers = [self._convert_to_paper(result) for result in results]
        self._log_search(query, len(papers))
        
        return papers
    
    def _sync_search(
        self,
        query: str,
        max_results: int,
        sort_criterion: arxiv.SortCriterion,
    ) -> List[arxiv.Result]:
        """同步搜索方法"""
        client = arxiv.Client(
            page_size=min(max_results, 100),
            delay_seconds=1.0,
            num_retries=3,
        )
        
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_criterion,
        )
        
        return list(client.results(search))
    
    async def get_details(self, paper_id: str) -> Optional[Paper]:
        """
        获取单篇论文详情
        
        Args:
            paper_id: ArXiv 论文ID (如 "2301.07041")
            
        Returns:
            论文详情
        """
        try:
            result = await self._run_blocking(
                self._sync_get_paper,
                paper_id,
            )
            
            if result:
                return self._convert_to_paper(result)
            return None
            
        except Exception as exc:
            self._log_error(f"Failed to get paper {paper_id}", exc)
            return None
    
    def _sync_get_paper(self, paper_id: str) -> Optional[arxiv.Result]:
        """同步获取单篇论文"""
        client = arxiv.Client()
        search = arxiv.Search(id_list=[paper_id])
        results = list(client.results(search))
        return results[0] if results else None
    
    def _convert_to_paper(self, result: arxiv.Result) -> Paper:
        """
        将 arxiv.Result 转换为 Paper 模型
        """
        # 提取作者信息
        authors = [
            Author(name=author.name)
            for author in result.authors
        ]
        
        # 提取分类
        categories = list(result.categories) if result.categories else []
        
        return Paper(
            id=result.entry_id,
            title=result.title,
            abstract=result.summary,
            authors=authors,
            published_date=result.published,
            updated_date=result.updated,
            categories=categories,
            url=result.entry_id,
            pdf_url=result.pdf_url,
            source=SourceType.ARXIV,
            extra={
                "primary_category": result.primary_category,
                "comment": result.comment,
                "journal_ref": result.journal_ref,
                "doi": result.doi,
            }
        )
    
    async def search_by_author(self, author_name: str, max_results: Optional[int] = None) -> List[Paper]:
        """
        按作者搜索论文
        
        Args:
            author_name: 作者名称
            max_results: 最大返回结果数
            
        Returns:
            论文列表
        """
        query = f'au:"{author_name}"'
        return await self.search(query, max_results)
    
    async def search_by_category(
        self, 
        category: str, 
        keywords: Optional[str] = None,
        max_results: Optional[int] = None
    ) -> List[Paper]:
        """
        按分类搜索论文
        
        Args:
            category: 分类ID (如 "cs.AI")
            keywords: 可选的关键词
            max_results: 最大返回结果数
            
        Returns:
            论文列表
        """
        if keywords:
            query = f'cat:{category} AND ({keywords})'
        else:
            query = f'cat:{category}'
        
        return await self.search(query, max_results)


# 便捷函数
async def search_arxiv(query: str, max_results: int = 50) -> List[Paper]:
    """
    便捷函数：搜索 ArXiv 论文
    
    Usage:
        papers = await search_arxiv("transformer attention mechanism")
    """
    async with ArxivScraper() as scraper:
        return await scraper.search(query, max_results)
