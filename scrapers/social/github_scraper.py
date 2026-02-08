"""
GitHub Scraper
从 GitHub 抓取代码仓库和讨论
"""
import asyncio
from datetime import datetime
from typing import List, Optional
import logging
from itertools import islice

from tenacity import retry, stop_after_attempt, wait_exponential

from scrapers.base import RateLimitedScraper
from models import GitHubRepo, SocialPost, SourceType
from config import get_settings


logger = logging.getLogger(__name__)


class GitHubScraper(RateLimitedScraper[GitHubRepo]):
    """
    GitHub 抓取器
    使用 PyGithub 库
    """
    
    def __init__(self):
        super().__init__(requests_per_second=2.0)  # GitHub API限制较宽松
        self._github_settings = get_settings().github
        self._github = None
    
    @property
    def source_type(self) -> SourceType:
        return SourceType.GITHUB
    
    @property
    def name(self) -> str:
        return "GitHub"
    
    def is_configured(self) -> bool:
        # GitHub 可以不需要 token 进行基础搜索，但有更严格的速率限制
        return True
    
    def _get_github(self):
        """获取 GitHub 客户端"""
        if self._github is None:
            try:
                from github import Github
                
                if self._github_settings.token:
                    self._github = Github(self._github_settings.token)
                else:
                    self._github = Github()  # 未认证，速率限制更严格
                    
            except ImportError:
                raise ImportError("Please install PyGithub: pip install PyGithub")
        return self._github
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def search(
        self, 
        query: str, 
        max_results: Optional[int] = None,
    ) -> List[GitHubRepo]:
        """
        搜索 GitHub 仓库
        
        Args:
            query: 搜索关键词
            max_results: 最大返回结果数
            
        Returns:
            仓库列表
        """
        return await self.search_repos(query, max_results)
    
    async def search_repos(
        self,
        query: str,
        max_results: Optional[int] = None,
        language: Optional[str] = None,
        sort: str = "stars",  # stars, forks, help-wanted-issues, updated
        order: str = "desc",
    ) -> List[GitHubRepo]:
        """
        搜索仓库
        
        Args:
            query: 搜索关键词
            max_results: 最大返回结果数
            language: 编程语言过滤
            sort: 排序方式
            order: 排序顺序
            
        Returns:
            仓库列表
        """
        if max_results is None:
            max_results = self._github_settings.max_results
        
        # 构建查询
        search_query = query
        if language:
            search_query += f" language:{language}"
        
        logger.info(f"[GitHub] Searching repos: {search_query}")
        
        await self._wait_for_rate_limit()
        
        loop = asyncio.get_event_loop()
        
        try:
            results = await loop.run_in_executor(
                None,
                self._sync_search_repos,
                search_query,
                max_results,
                sort,
                order,
            )
            
            repos = [self._convert_to_repo(r) for r in results]
            self._log_search(query, len(repos))
            
            return repos
            
        except Exception as e:
            self._log_error(f"Search failed for '{query}'", e)
            return []
    
    def _sync_search_repos(
        self,
        query: str,
        max_results: int,
        sort: str,
        order: str,
    ) -> list:
        """同步搜索仓库"""
        github = self._get_github()
        
        repos = github.search_repositories(
            query=query,
            sort=sort,
            order=order,
        )

        # 使用 islice 避免 PaginatedList 切片在边界情况下触发索引错误
        return list(islice(repos, max_results))
    
    async def search_code(
        self,
        query: str,
        max_results: Optional[int] = None,
        language: Optional[str] = None,
    ) -> List[dict]:
        """
        搜索代码
        
        Args:
            query: 搜索关键词
            max_results: 最大返回结果数
            language: 编程语言过滤
            
        Returns:
            代码搜索结果列表
        """
        if max_results is None:
            max_results = self._github_settings.max_results
        
        # 构建查询
        search_query = query
        if language:
            search_query += f" language:{language}"
        
        logger.info(f"[GitHub] Searching code: {search_query}")
        
        await self._wait_for_rate_limit()
        
        loop = asyncio.get_event_loop()
        
        try:
            results = await loop.run_in_executor(
                None,
                self._sync_search_code,
                search_query,
                max_results,
            )
            
            return results
            
        except Exception as e:
            self._log_error(f"Code search failed for '{query}'", e)
            return []
    
    def _sync_search_code(self, query: str, max_results: int) -> list:
        """同步搜索代码"""
        github = self._get_github()
        
        code_results = github.search_code(query=query)
        
        results = []
        for code in islice(code_results, max_results):
            try:
                results.append({
                    "name": code.name,
                    "path": code.path,
                    "url": code.html_url,
                    "repository": code.repository.full_name,
                    "sha": code.sha,
                })
            except Exception as e:
                logger.warning(f"[GitHub] Skip malformed code result: {e}")
        
        return results
    
    async def search_discussions(
        self,
        query: str,
        max_results: Optional[int] = None,
    ) -> List[SocialPost]:
        """
        搜索 GitHub Discussions (作为社交帖子)
        
        注意: GitHub GraphQL API 需要额外设置
        这里使用 Issues 作为替代
        """
        if max_results is None:
            max_results = self._github_settings.max_results
        
        logger.info(f"[GitHub] Searching issues/discussions: {query}")
        
        await self._wait_for_rate_limit()
        
        loop = asyncio.get_event_loop()
        
        try:
            results = await loop.run_in_executor(
                None,
                self._sync_search_issues,
                query,
                max_results,
            )
            
            posts = [self._convert_issue_to_post(issue) for issue in results]
            logger.info(f"[GitHub] Found {len(posts)} issues for '{query}'")
            
            return posts
            
        except Exception as e:
            self._log_error(f"Issue search failed for '{query}'", e)
            return []
    
    def _sync_search_issues(self, query: str, max_results: int) -> list:
        """同步搜索 Issues"""
        github = self._get_github()
        
        issues = github.search_issues(query=query)

        return list(islice(issues, max_results))
    
    async def get_details(self, repo_full_name: str) -> Optional[GitHubRepo]:
        """
        获取仓库详情
        
        Args:
            repo_full_name: 仓库完整名称 (owner/repo)
            
        Returns:
            仓库详情
        """
        loop = asyncio.get_event_loop()
        
        try:
            await self._wait_for_rate_limit()
            
            result = await loop.run_in_executor(
                None,
                self._sync_get_repo,
                repo_full_name,
            )
            
            if result:
                return self._convert_to_repo(result)
            return None
            
        except Exception as e:
            self._log_error(f"Failed to get repo {repo_full_name}", e)
            return None
    
    def _sync_get_repo(self, repo_full_name: str):
        """同步获取仓库"""
        github = self._get_github()
        return github.get_repo(repo_full_name)
    
    async def get_trending(
        self,
        language: Optional[str] = None,
        since: str = "weekly",  # daily, weekly, monthly
    ) -> List[GitHubRepo]:
        """
        获取 Trending 仓库
        
        注意: GitHub API 不直接支持 Trending
        这里使用搜索按 stars 排序的最近创建仓库作为替代
        """
        # 构建查询：最近创建且 stars 较高的仓库
        from datetime import timedelta
        
        if since == "daily":
            days = 1
        elif since == "weekly":
            days = 7
        else:
            days = 30
        
        date_threshold = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        query = f"created:>{date_threshold}"
        
        if language:
            query += f" language:{language}"
        
        return await self.search_repos(
            query=query,
            max_results=30,
            sort="stars",
        )
    
    def _convert_to_repo(self, repo) -> GitHubRepo:
        """将 GitHub Repository 转换为 GitHubRepo"""
        return GitHubRepo(
            id=str(repo.id),
            name=repo.name,
            full_name=repo.full_name,
            description=repo.description,
            owner=repo.owner.login,
            url=repo.html_url,
            stars=repo.stargazers_count,
            forks=repo.forks_count,
            watchers=repo.watchers_count,
            language=repo.language,
            topics=list(repo.get_topics()) if hasattr(repo, 'get_topics') else [],
            created_at=repo.created_at,
            updated_at=repo.updated_at,
            source=SourceType.GITHUB,
            extra={
                "default_branch": repo.default_branch,
                "open_issues": repo.open_issues_count,
                "license": repo.license.name if repo.license else None,
                "is_fork": repo.fork,
                "archived": repo.archived,
            }
        )
    
    def _convert_issue_to_post(self, issue) -> SocialPost:
        """将 GitHub Issue 转换为 SocialPost"""
        content = issue.title
        if issue.body:
            content += f"\n\n{issue.body[:500]}..."  # 截断长文本
        
        return SocialPost(
            id=str(issue.id),
            content=content,
            author=issue.user.login if issue.user else "unknown",
            author_url=issue.user.html_url if issue.user else None,
            created_at=issue.created_at,
            url=issue.html_url,
            likes=0,  # Issues 没有点赞
            reposts=0,
            comments=issue.comments,
            source=SourceType.GITHUB,
            extra={
                "state": issue.state,
                "labels": [label.name for label in issue.labels],
                "repository": issue.repository.full_name if issue.repository else None,
            }
        )
    
    async def close(self):
        """关闭连接"""
        if self._github:
            self._github.close()
            self._github = None
        await super().close()


# 便捷函数
async def search_github(
    query: str, 
    max_results: int = 30,
    language: Optional[str] = None,
) -> List[GitHubRepo]:
    """便捷函数：搜索 GitHub 仓库"""
    async with GitHubScraper() as scraper:
        return await scraper.search_repos(query, max_results, language)
