"""
Data Aggregator
统一聚合多个数据源的结果
"""
import asyncio
from typing import Any, Awaitable, Callable, Dict, List, Optional
import logging
import re

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from models import (
    AggregatedResult,
    Paper,
    SocialPost,
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
        source_timeout_sec: int = 18,
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
        self.source_timeout_sec = max(5, int(source_timeout_sec))
        
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

    async def _run_source_task(self, source_name: str, task_coro):
        try:
            return await asyncio.wait_for(task_coro, timeout=float(self.source_timeout_sec))
        except Exception as exc:
            logger.warning(f"{source_name} source skipped: {exc}")
            return exc

    @staticmethod
    def _payload_has_items(payload: Any) -> bool:
        if isinstance(payload, tuple):
            return any(bool(part) for part in payload)
        return bool(payload)

    async def _search_with_fallback(
        self,
        *,
        source_name: str,
        topic: str,
        search_fn: Callable[[str], Awaitable[Any]],
        empty_result: Any,
    ) -> Any:
        for query in self._topic_query_candidates(topic):
            try:
                payload = await search_fn(query)
            except Exception as exc:
                logger.warning(f"{source_name} search failed for query '{query}': {exc}")
                continue

            if self._payload_has_items(payload):
                if query != topic:
                    logger.info(f"{source_name} fallback query hit: '{query}'")
                return payload

        return empty_result

    def _merge_source_payload(
        self,
        aggregated: AggregatedResult,
        source_name: str,
        payload: Any,
    ) -> None:
        if source_name == "ArXiv":
            aggregated.papers.extend(payload)
            return
        if source_name == "HuggingFace":
            models, datasets = payload
            aggregated.models.extend(models)
            aggregated.datasets.extend(datasets)
            return
        if source_name in {"Twitter", "Reddit"}:
            aggregated.social_posts.extend(payload)
            return
        if source_name == "GitHub":
            repos, discussions = payload
            aggregated.github_repos.extend(repos)
            aggregated.social_posts.extend(discussions)
            return
        if source_name == "SemanticScholar":
            aggregated.papers.extend(payload)
            return
        if source_name == "StackOverflow":
            aggregated.stackoverflow_questions.extend(payload)
            return
        if source_name == "HackerNews":
            aggregated.hackernews_items.extend(payload)
    
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
        
        source_jobs: List[tuple[str, Awaitable[Any]]] = []

        if self.enable_arxiv:
            source_jobs.append(
                ("ArXiv", self._search_arxiv(topic, max_results_per_source, arxiv_sort_by))
            )
        if self.enable_huggingface:
            source_jobs.append(("HuggingFace", self._search_huggingface(topic, max_results_per_source)))
        if self.enable_twitter:
            source_jobs.append(("Twitter", self._search_twitter(topic, max_results_per_source)))
        if self.enable_reddit:
            source_jobs.append(("Reddit", self._search_reddit(topic, max_results_per_source)))
        if self.enable_github:
            source_jobs.append(("GitHub", self._search_github(topic, max_results_per_source)))
        if self.enable_semantic_scholar:
            source_jobs.append(
                ("SemanticScholar", self._search_semantic_scholar(topic, max_results_per_source))
            )
        if self.enable_stackoverflow:
            source_jobs.append(
                ("StackOverflow", self._search_stackoverflow(topic, max_results_per_source))
            )
        if self.enable_hackernews:
            source_jobs.append(("HackerNews", self._search_hackernews(topic, max_results_per_source)))

        tasks = [
            self._run_source_task(source_name, source_coro)
            for source_name, source_coro in source_jobs
        ]
        task_names = [name for name, _ in source_jobs]
        
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
        for name, res in zip(task_names, results):
            if isinstance(res, Exception):
                logger.error(f"Error fetching from {name}: {res}")
                continue
            self._merge_source_payload(result, name, res)
        
        if show_progress:
            self._print_summary(result)
        
        return result

    def _topic_query_candidates(self, topic: str) -> List[str]:
        """
        生成查询降级候选词，提升长查询在多源 API 的召回率。
        例如: "MCP production deployment" -> ["MCP production deployment", "MCP", "Model Context Protocol"]
        """
        base = str(topic or "").strip()
        if not base:
            return []

        candidates: List[str] = [base]
        lower_base = base.lower()

        tokens = re.findall(r"[A-Za-z0-9#+-]+(?:\.[0-9]+)?", base)
        stopwords = {
            "production",
            "deployment",
            "deploy",
            "prod",
            "system",
            "systems",
            "implementation",
            "architecture",
            "analysis",
            "research",
            "study",
            "with",
            "for",
            "and",
            "the",
            "in",
            "of",
            "to",
        }

        core_tokens = [t for t in tokens if t.lower() not in stopwords]
        if core_tokens:
            core = " ".join(core_tokens[:4]).strip()
            if core and core.lower() != lower_base:
                candidates.append(core)

        version_tokens = re.findall(r"\d+(?:\.\d+)+", base)
        if version_tokens:
            core = " ".join(
                [
                    token
                    for token in core_tokens
                    if not re.fullmatch(r"\d+(?:\.\d+)+", token)
                ][:4]
            ).strip()
            for version in version_tokens:
                candidate = f"{core} {version}".strip() if core else version
                if candidate and candidate.lower() != lower_base:
                    candidates.append(candidate)

        acronyms = [t for t in tokens if t.isupper() and 2 <= len(t) <= 8]
        for token in acronyms:
            if token.lower() != lower_base and token not in candidates:
                candidates.append(token)

        token_set = {t.lower() for t in tokens}
        if "mcp" in token_set and "model context protocol" not in {c.lower() for c in candidates}:
            candidates.append("Model Context Protocol")

        deduped: List[str] = []
        seen = set()
        for item in candidates:
            key = item.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(item.strip())

        return deduped[:3]
    
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

        return await self._search_with_fallback(
            source_name="ArXiv",
            topic=topic,
            search_fn=lambda query: scraper.search(query, max_results, sort_by=sort_by),
            empty_result=[],
        )
    
    async def _search_huggingface(
        self, 
        topic: str, 
        max_results: Optional[int]
    ) -> tuple:
        """搜索 HuggingFace (模型和数据集)"""
        scraper = self._scrapers.get('huggingface')
        if not scraper:
            return [], []

        return await self._search_with_fallback(
            source_name="HuggingFace",
            topic=topic,
            search_fn=lambda query: asyncio.gather(
                scraper.search_models(query, max_results),
                scraper.search_datasets(query, max_results),
            ),
            empty_result=([], []),
        )
    
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

        return await self._search_with_fallback(
            source_name="GitHub",
            topic=topic,
            search_fn=lambda query: asyncio.gather(
                scraper.search_repos(query, max_results),
                scraper.search_discussions(query, max_results // 2 if max_results else None),
            ),
            empty_result=([], []),
        )
    
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

        return await self._search_with_fallback(
            source_name="Semantic Scholar",
            topic=topic,
            search_fn=lambda query: scraper.search(query, max_results),
            empty_result=[],
        )
    
    async def _search_stackoverflow(
        self, 
        topic: str, 
        max_results: Optional[int]
    ) -> List[StackOverflowQuestion]:
        """搜索 Stack Overflow"""
        scraper = self._scrapers.get('stackoverflow')
        if not scraper:
            return []

        return await self._search_with_fallback(
            source_name="StackOverflow",
            topic=topic,
            search_fn=lambda query: scraper.search(query, max_results),
            empty_result=[],
        )
    
    async def _search_hackernews(
        self, 
        topic: str, 
        max_results: Optional[int]
    ) -> List[HackerNewsItem]:
        """搜索 Hacker News"""
        scraper = self._scrapers.get('hackernews')
        if not scraper:
            return []

        return await self._search_with_fallback(
            source_name="HackerNews",
            topic=topic,
            search_fn=lambda query: scraper.search(query, max_results),
            empty_result=[],
        )
    
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
