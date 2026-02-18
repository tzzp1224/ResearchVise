"""
Data Models / Schemas
定义统一的数据结构
"""
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """数据来源类型"""
    ARXIV = "arxiv"
    HUGGINGFACE = "huggingface"
    TWITTER = "twitter"
    REDDIT = "reddit"
    GITHUB = "github"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    STACKOVERFLOW = "stackoverflow"
    HACKERNEWS = "hackernews"


class Author(BaseModel):
    """作者信息"""
    name: str
    affiliation: Optional[str] = None
    url: Optional[str] = None


class Paper(BaseModel):
    """论文数据模型"""
    id: str = Field(..., description="论文唯一标识")
    title: str = Field(..., description="论文标题")
    abstract: str = Field(..., description="摘要")
    authors: List[Author] = Field(default_factory=list, description="作者列表")
    published_date: Optional[datetime] = Field(None, description="发布日期")
    updated_date: Optional[datetime] = Field(None, description="更新日期")
    categories: List[str] = Field(default_factory=list, description="分类标签")
    url: str = Field(..., description="论文链接")
    pdf_url: Optional[str] = Field(None, description="PDF链接")
    source: SourceType = Field(..., description="数据来源")
    citation_count: Optional[int] = Field(None, description="引用数")
    extra: Dict[str, Any] = Field(default_factory=dict, description="额外信息")


class Model(BaseModel):
    """模型数据模型 (Hugging Face)"""
    id: str = Field(..., description="模型ID")
    name: str = Field(..., description="模型名称")
    author: Optional[str] = Field(None, description="作者/组织")
    description: Optional[str] = Field(None, description="描述")
    tags: List[str] = Field(default_factory=list, description="标签")
    downloads: int = Field(default=0, description="下载量")
    likes: int = Field(default=0, description="点赞数")
    url: str = Field(..., description="模型链接")
    created_at: Optional[datetime] = Field(None, description="创建时间")
    updated_at: Optional[datetime] = Field(None, description="更新时间")
    source: SourceType = Field(default=SourceType.HUGGINGFACE)
    extra: Dict[str, Any] = Field(default_factory=dict)


class Dataset(BaseModel):
    """数据集模型 (Hugging Face)"""
    id: str = Field(..., description="数据集ID")
    name: str = Field(..., description="数据集名称")
    author: Optional[str] = Field(None, description="作者/组织")
    description: Optional[str] = Field(None, description="描述")
    tags: List[str] = Field(default_factory=list, description="标签")
    downloads: int = Field(default=0, description="下载量")
    url: str = Field(..., description="数据集链接")
    source: SourceType = Field(default=SourceType.HUGGINGFACE)
    extra: Dict[str, Any] = Field(default_factory=dict)


class SocialPost(BaseModel):
    """社交媒体帖子模型"""
    id: str = Field(..., description="帖子ID")
    content: str = Field(..., description="内容")
    author: str = Field(..., description="作者")
    author_url: Optional[str] = Field(None, description="作者主页")
    created_at: Optional[datetime] = Field(None, description="发布时间")
    url: str = Field(..., description="帖子链接")
    likes: int = Field(default=0, description="点赞数")
    reposts: int = Field(default=0, description="转发/分享数")
    comments: int = Field(default=0, description="评论数")
    source: SourceType = Field(..., description="数据来源")
    extra: Dict[str, Any] = Field(default_factory=dict)


class GitHubRepo(BaseModel):
    """GitHub 仓库模型"""
    id: str = Field(..., description="仓库ID")
    name: str = Field(..., description="仓库名称")
    full_name: str = Field(..., description="完整名称 (owner/repo)")
    description: Optional[str] = Field(None, description="描述")
    owner: str = Field(..., description="所有者")
    url: str = Field(..., description="仓库链接")
    stars: int = Field(default=0, description="Star数")
    forks: int = Field(default=0, description="Fork数")
    watchers: int = Field(default=0, description="Watch数")
    language: Optional[str] = Field(None, description="主要语言")
    topics: List[str] = Field(default_factory=list, description="话题标签")
    created_at: Optional[datetime] = Field(None, description="创建时间")
    updated_at: Optional[datetime] = Field(None, description="更新时间")
    source: SourceType = Field(default=SourceType.GITHUB)
    extra: Dict[str, Any] = Field(default_factory=dict)


class StackOverflowQuestion(BaseModel):
    """Stack Overflow 问答模型"""
    id: str = Field(..., description="问题ID")
    title: str = Field(..., description="问题标题")
    body: Optional[str] = Field(None, description="问题内容")
    tags: List[str] = Field(default_factory=list, description="标签")
    author: str = Field(..., description="提问者")
    author_reputation: int = Field(default=0, description="提问者声望")
    url: str = Field(..., description="问题链接")
    score: int = Field(default=0, description="得分")
    view_count: int = Field(default=0, description="浏览数")
    answer_count: int = Field(default=0, description="回答数")
    is_answered: bool = Field(default=False, description="是否已回答")
    accepted_answer_id: Optional[str] = Field(None, description="采纳答案ID")
    created_at: Optional[datetime] = Field(None, description="创建时间")
    last_activity_at: Optional[datetime] = Field(None, description="最后活动时间")
    source: SourceType = Field(default=SourceType.STACKOVERFLOW)
    extra: Dict[str, Any] = Field(default_factory=dict)


class HackerNewsItem(BaseModel):
    """Hacker News 帖子模型"""
    id: str = Field(..., description="帖子ID")
    title: str = Field(..., description="标题")
    url: Optional[str] = Field(None, description="外链URL")
    text: Optional[str] = Field(None, description="帖子内容 (Ask HN等)")
    author: str = Field(..., description="作者")
    points: int = Field(default=0, description="得分")
    comment_count: int = Field(default=0, description="评论数")
    hn_url: str = Field(..., description="HN讨论链接")
    item_type: str = Field(default="story", description="类型: story, comment, ask, show, job")
    created_at: Optional[datetime] = Field(None, description="创建时间")
    source: SourceType = Field(default=SourceType.HACKERNEWS)
    extra: Dict[str, Any] = Field(default_factory=dict)


class AggregatedResult(BaseModel):
    """聚合结果模型"""
    topic: str = Field(..., description="查询主题")
    query_time: datetime = Field(default_factory=datetime.now, description="查询时间")
    papers: List[Paper] = Field(default_factory=list, description="论文列表")
    models: List[Model] = Field(default_factory=list, description="模型列表")
    datasets: List[Dataset] = Field(default_factory=list, description="数据集列表")
    social_posts: List[SocialPost] = Field(default_factory=list, description="社交帖子")
    github_repos: List[GitHubRepo] = Field(default_factory=list, description="GitHub仓库")
    stackoverflow_questions: List[StackOverflowQuestion] = Field(default_factory=list, description="Stack Overflow问答")
    hackernews_items: List[HackerNewsItem] = Field(default_factory=list, description="Hacker News帖子")
    
    @property
    def total_count(self) -> int:
        """总结果数"""
        return (
            len(self.papers) + 
            len(self.models) + 
            len(self.datasets) + 
            len(self.social_posts) + 
            len(self.github_repos) +
            len(self.stackoverflow_questions) +
            len(self.hackernews_items)
        )
    
    def summary(self) -> Dict[str, int]:
        """获取各类型结果统计"""
        return {
            "papers": len(self.papers),
            "models": len(self.models),
            "datasets": len(self.datasets),
            "social_posts": len(self.social_posts),
            "github_repos": len(self.github_repos),
            "stackoverflow": len(self.stackoverflow_questions),
            "hackernews": len(self.hackernews_items),
            "total": self.total_count,
        }
