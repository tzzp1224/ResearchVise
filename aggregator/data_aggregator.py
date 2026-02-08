"""
Data Aggregator
统一聚合多个数据源的结果
"""
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from models import (
    AggregatedResult,
    Paper,
    Model,
    Dataset,
    SocialPost,
    GitHubRepo,
    StackOverflowQuestion,
    HackerNewsItem,
)
from scrapers import (
    ArxivScraper,
    HuggingFaceScraper,
    TwitterScraper,
    RedditScraper,
    GitHubScraper,
    SemanticScholarScraper,
    StackOverflowScraper,
    HackerNewsScraper,
)


logger = logging.getLogger(__name__)
console = Console()


class DataAggregator:
    """
    数据聚合器
    统一管理多个数据源的抓取和聚合
    """
    
    def __init__(
        self,
        enable_arxiv: bool = True,
        enable_huggingface: bool = True,
        enable_twitter: bool = True,
        enable_reddit: bool = True,
        enable_github: bool = True,
        enable_semantic_scholar: bool = True,
        enable_stackoverflow: bool = True,
        enable_hackernews: bool = True,
    ):
        """
        初始化聚合器
        
        Args:
            enable_*: 控制各个数据源的启用状态
        """
        self.enable_arxiv = enable_arxiv
        self.enable_huggingface = enable_huggingface
        self.enable_twitter = enable_twitter
        self.enable_reddit = enable_reddit
        self.enable_github = enable_github
        self.enable_semantic_scholar = enable_semantic_scholar
        self.enable_stackoverflow = enable_stackoverflow
        self.enable_hackernews = enable_hackernews
        
        # 初始化抓取器
        self._scrapers: Dict[str, Any] = {}
        
        if enable_arxiv:
            self._scrapers['arxiv'] = ArxivScraper()
        if enable_huggingface:
            self._scrapers['huggingface'] = HuggingFaceScraper()
        if enable_twitter:
            self._scrapers['twitter'] = TwitterScraper()
        if enable_reddit:
            self._scrapers['reddit'] = RedditScraper()
        if enable_github:
            self._scrapers['github'] = GitHubScraper()
        if enable_semantic_scholar:
            self._scrapers['semantic_scholar'] = SemanticScholarScraper()
        if enable_stackoverflow:
            self._scrapers['stackoverflow'] = StackOverflowScraper()
        if enable_hackernews:
            self._scrapers['hackernews'] = HackerNewsScraper()
    
    async def aggregate(
        self,
        topic: str,
        max_results_per_source: Optional[int] = None,
        show_progress: bool = True,
        arxiv_sort_by: Optional[str] = None,
    ) -> AggregatedResult:
        """
        聚合搜索所有启用的数据源
        
        Args:
            topic: 搜索主题
            max_results_per_source: 每个源的最大结果数
            show_progress: 是否显示进度条
            arxiv_sort_by: ArXiv排序方式 (relevance, submittedDate, lastUpdatedDate)
            
        Returns:
            聚合结果
        """
        result = AggregatedResult(topic=topic)
        
        if show_progress:
            console.print(f"\n🔍 [bold blue]Searching for:[/bold blue] {topic}\n")
        
        # 创建所有搜索任务
        tasks = []
        task_names = []
        
        if self.enable_arxiv:
            tasks.append(self._search_arxiv(topic, max_results_per_source, arxiv_sort_by))
            task_names.append("ArXiv")
        
        if self.enable_huggingface:
            tasks.append(self._search_huggingface(topic, max_results_per_source))
            task_names.append("HuggingFace")
        
        if self.enable_twitter:
            tasks.append(self._search_twitter(topic, max_results_per_source))
            task_names.append("Twitter")
        
        if self.enable_reddit:
            tasks.append(self._search_reddit(topic, max_results_per_source))
            task_names.append("Reddit")
        
        if self.enable_github:
            tasks.append(self._search_github(topic, max_results_per_source))
            task_names.append("GitHub")
        
        if self.enable_semantic_scholar:
            tasks.append(self._search_semantic_scholar(topic, max_results_per_source))
            task_names.append("SemanticScholar")
        
        if self.enable_stackoverflow:
            tasks.append(self._search_stackoverflow(topic, max_results_per_source))
            task_names.append("StackOverflow")
        
        if self.enable_hackernews:
            tasks.append(self._search_hackernews(topic, max_results_per_source))
            task_names.append("HackerNews")
        
        # 并行执行所有搜索
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"[cyan]Fetching from {len(tasks)} sources...",
                    total=None
                )
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                progress.update(task, completed=True)
        else:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        for i, (name, res) in enumerate(zip(task_names, results)):
            if isinstance(res, Exception):
                logger.error(f"Error fetching from {name}: {res}")
                continue
            
            if name == "ArXiv":
                result.papers.extend(res)
            elif name == "HuggingFace":
                models, datasets = res
                result.models.extend(models)
                result.datasets.extend(datasets)
            elif name == "Twitter":
                result.social_posts.extend(res)
            elif name == "Reddit":
                result.social_posts.extend(res)
            elif name == "GitHub":
                repos, discussions = res
                result.github_repos.extend(repos)
                result.social_posts.extend(discussions)
            elif name == "SemanticScholar":
                result.papers.extend(res)
            elif name == "StackOverflow":
                result.stackoverflow_questions.extend(res)
            elif name == "HackerNews":
                result.hackernews_items.extend(res)
        
        if show_progress:
            self._print_summary(result)
        
        return result
    
    async def _search_arxiv(
        self, 
        topic: str, 
        max_results: Optional[int],
        sort_by: Optional[str] = None,
    ) -> List[Paper]:
        """搜索 ArXiv"""
        scraper = self._scrapers.get('arxiv')
        if not scraper:
            return []
        
        try:
            return await scraper.search(topic, max_results, sort_by=sort_by)
        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")
            return []
    
    async def _search_huggingface(
        self, 
        topic: str, 
        max_results: Optional[int]
    ) -> tuple:
        """搜索 HuggingFace (模型和数据集)"""
        scraper = self._scrapers.get('huggingface')
        if not scraper:
            return [], []
        
        try:
            # 并行搜索模型和数据集
            models, datasets = await asyncio.gather(
                scraper.search_models(topic, max_results),
                scraper.search_datasets(topic, max_results),
            )
            return models, datasets
        except Exception as e:
            logger.error(f"HuggingFace search failed: {e}")
            return [], []
    
    async def _search_twitter(
        self, 
        topic: str, 
        max_results: Optional[int]
    ) -> List[SocialPost]:
        """搜索 Twitter"""
        scraper = self._scrapers.get('twitter')
        if not scraper or not scraper.is_configured():
            return []
        
        try:
            return await scraper.search(topic, max_results)
        except Exception as e:
            logger.error(f"Twitter search failed: {e}")
            return []
    
    async def _search_reddit(
        self, 
        topic: str, 
        max_results: Optional[int]
    ) -> List[SocialPost]:
        """搜索 Reddit"""
        scraper = self._scrapers.get('reddit')
        if not scraper or not scraper.is_configured():
            return []
        
        try:
            return await scraper.search(topic, max_results)
        except Exception as e:
            logger.error(f"Reddit search failed: {e}")
            return []
    
    async def _search_github(
        self, 
        topic: str, 
        max_results: Optional[int]
    ) -> tuple:
        """搜索 GitHub (仓库和讨论)"""
        scraper = self._scrapers.get('github')
        if not scraper:
            return [], []
        
        try:
            # 并行搜索仓库和讨论
            repos, discussions = await asyncio.gather(
                scraper.search_repos(topic, max_results),
                scraper.search_discussions(topic, max_results // 2 if max_results else None),
            )
            return repos, discussions
        except Exception as e:
            logger.error(f"GitHub search failed: {e}")
            return [], []
    
    def _print_summary(self, result: AggregatedResult):
        """打印结果摘要"""
        console.print()
        
        table = Table(title="📊 Aggregation Summary", show_header=True)
        table.add_column("Source", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Count", justify="right", style="green")
        
        if self.enable_arxiv:
            table.add_row("ArXiv", "Papers", str(len(result.papers)))
        
        if self.enable_huggingface:
            table.add_row("HuggingFace", "Models", str(len(result.models)))
            table.add_row("HuggingFace", "Datasets", str(len(result.datasets)))
        
        # 按来源统计社交帖子
        twitter_posts = [p for p in result.social_posts if p.source.value == "twitter"]
        reddit_posts = [p for p in result.social_posts if p.source.value == "reddit"]
        github_posts = [p for p in result.social_posts if p.source.value == "github"]
        
        if self.enable_twitter:
            table.add_row("Twitter", "Posts", str(len(twitter_posts)))
        
        if self.enable_reddit:
            table.add_row("Reddit", "Posts", str(len(reddit_posts)))
        
        if self.enable_github:
            table.add_row("GitHub", "Repos", str(len(result.github_repos)))
            table.add_row("GitHub", "Issues", str(len(github_posts)))
        
        if self.enable_semantic_scholar:
            # Semantic Scholar papers 已包含在 papers 中，这里显示来源统计
            ss_papers = [p for p in result.papers if p.source.value == "semantic_scholar"]
            table.add_row("SemanticScholar", "Papers", str(len(ss_papers)))
        
        if self.enable_stackoverflow:
            table.add_row("StackOverflow", "Questions", str(len(result.stackoverflow_questions)))
        
        if self.enable_hackernews:
            table.add_row("HackerNews", "Posts", str(len(result.hackernews_items)))
        
        table.add_row("", "", "")
        table.add_row("[bold]Total[/bold]", "", f"[bold]{result.total_count}[/bold]")
        
        console.print(table)
        console.print()
    
    async def _search_semantic_scholar(
        self, 
        topic: str, 
        max_results: Optional[int]
    ) -> List[Paper]:
        """搜索 Semantic Scholar"""
        scraper = self._scrapers.get('semantic_scholar')
        if not scraper:
            return []
        
        try:
            return await scraper.search(topic, max_results)
        except Exception as e:
            logger.error(f"Semantic Scholar search failed: {e}")
            return []
    
    async def _search_stackoverflow(
        self, 
        topic: str, 
        max_results: Optional[int]
    ) -> List[StackOverflowQuestion]:
        """搜索 Stack Overflow"""
        scraper = self._scrapers.get('stackoverflow')
        if not scraper:
            return []
        
        try:
            return await scraper.search(topic, max_results)
        except Exception as e:
            logger.error(f"Stack Overflow search failed: {e}")
            return []
    
    async def _search_hackernews(
        self, 
        topic: str, 
        max_results: Optional[int]
    ) -> List[HackerNewsItem]:
        """搜索 Hacker News"""
        scraper = self._scrapers.get('hackernews')
        if not scraper:
            return []
        
        try:
            return await scraper.search(topic, max_results)
        except Exception as e:
            logger.error(f"Hacker News search failed: {e}")
            return []
    
    async def close(self):
        """关闭所有抓取器"""
        for scraper in self._scrapers.values():
            await scraper.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# 便捷函数
async def aggregate_research(
    topic: str,
    max_results_per_source: int = 30,
    **kwargs
) -> AggregatedResult:
    """
    便捷函数：聚合研究信息
    
    Usage:
        result = await aggregate_research("DeepSeek")
        print(f"Found {result.total_count} items")
    """
    async with DataAggregator(**kwargs) as aggregator:
        return await aggregator.aggregate(topic, max_results_per_source)
