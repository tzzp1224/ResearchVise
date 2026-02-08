"""
Data Cleaner
数据清洗模块
"""
import re
import html
from typing import Optional, List, Union
from datetime import datetime

from models import Paper, SocialPost, Model, Dataset, GitHubRepo


class DataCleaner:
    """
    数据清洗器
    处理各种数据源的文本清洗
    """
    
    # 需要移除的模式
    URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
    MENTION_PATTERN = re.compile(r'@\w+')
    HASHTAG_PATTERN = re.compile(r'#\w+')
    EMAIL_PATTERN = re.compile(r'\S+@\S+\.\S+')
    LATEX_PATTERN = re.compile(r'\$[^$]+\$|\\\[.*?\\\]|\\\(.*?\\\)')
    MULTIPLE_SPACES = re.compile(r'\s+')
    MULTIPLE_NEWLINES = re.compile(r'\n{3,}')
    
    def __init__(
        self,
        remove_urls: bool = True,
        remove_mentions: bool = False,
        remove_hashtags: bool = False,
        remove_emails: bool = True,
        remove_latex: bool = False,
        lowercase: bool = False,
        max_length: Optional[int] = None,
    ):
        """
        初始化清洗器
        
        Args:
            remove_urls: 是否移除URL
            remove_mentions: 是否移除@提及
            remove_hashtags: 是否移除#标签
            remove_emails: 是否移除邮箱
            remove_latex: 是否移除LaTeX公式
            lowercase: 是否转小写
            max_length: 最大文本长度
        """
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.remove_emails = remove_emails
        self.remove_latex = remove_latex
        self.lowercase = lowercase
        self.max_length = max_length
    
    def clean(self, text: str) -> str:
        """
        清洗文本
        
        Args:
            text: 原始文本
            
        Returns:
            清洗后的文本
        """
        if not text:
            return ""
        
        # HTML 解码
        text = html.unescape(text)
        
        # 移除 URL
        if self.remove_urls:
            text = self.URL_PATTERN.sub(' ', text)
        
        # 移除 @ 提及
        if self.remove_mentions:
            text = self.MENTION_PATTERN.sub(' ', text)
        
        # 移除 # 标签
        if self.remove_hashtags:
            text = self.HASHTAG_PATTERN.sub(' ', text)
        
        # 移除邮箱
        if self.remove_emails:
            text = self.EMAIL_PATTERN.sub(' ', text)
        
        # 移除 LaTeX
        if self.remove_latex:
            text = self.LATEX_PATTERN.sub(' [FORMULA] ', text)
        
        # 转小写
        if self.lowercase:
            text = text.lower()
        
        # 规范化空白
        text = self.MULTIPLE_SPACES.sub(' ', text)
        text = self.MULTIPLE_NEWLINES.sub('\n\n', text)
        text = text.strip()
        
        # 截断
        if self.max_length and len(text) > self.max_length:
            text = text[:self.max_length] + "..."
        
        return text
    
    def clean_paper(self, paper: Paper) -> dict:
        """
        清洗论文数据，返回适合存储的格式
        
        Args:
            paper: Paper 对象
            
        Returns:
            清洗后的字典
        """
        # 论文保留 LaTeX 公式，但清洗其他
        cleaner = DataCleaner(
            remove_urls=False,  # 保留引用链接
            remove_latex=False,  # 保留公式
        )
        
        # 构建作者字符串
        authors_str = ", ".join([a.name for a in paper.authors[:5]])
        if len(paper.authors) > 5:
            authors_str += " et al."
        
        return {
            "id": paper.id,
            "title": cleaner.clean(paper.title),
            "abstract": cleaner.clean(paper.abstract),
            "authors": authors_str,
            "categories": ", ".join(paper.categories),
            "published_date": paper.published_date.isoformat() if paper.published_date else None,
            "url": paper.url,
            "pdf_url": paper.pdf_url,
            "source": paper.source.value,
            # 组合文本用于检索
            "content": f"Title: {paper.title}\n\nAuthors: {authors_str}\n\nAbstract: {paper.abstract}",
        }
    
    def clean_social_post(self, post: SocialPost) -> dict:
        """
        清洗社交帖子
        
        Args:
            post: SocialPost 对象
            
        Returns:
            清洗后的字典
        """
        # 社交帖子更激进地清洗
        cleaner = DataCleaner(
            remove_urls=True,
            remove_mentions=True,
            remove_hashtags=False,  # 保留标签作为关键词
        )
        
        return {
            "id": post.id,
            "content": cleaner.clean(post.content),
            "author": post.author,
            "created_at": post.created_at.isoformat() if post.created_at else None,
            "url": post.url,
            "likes": post.likes,
            "comments": post.comments,
            "source": post.source.value,
        }
    
    def clean_model(self, model: Model) -> dict:
        """
        清洗模型数据
        
        Args:
            model: Model 对象
            
        Returns:
            清洗后的字典
        """
        description = self.clean(model.description) if model.description else ""
        
        return {
            "id": model.id,
            "name": model.name,
            "author": model.author,
            "description": description,
            "tags": ", ".join(model.tags),
            "downloads": model.downloads,
            "likes": model.likes,
            "url": model.url,
            "source": model.source.value,
            "content": f"Model: {model.name}\n\nDescription: {description}\n\nTags: {', '.join(model.tags)}",
        }
    
    def clean_dataset(self, dataset: Dataset) -> dict:
        """
        清洗数据集
        """
        description = self.clean(dataset.description) if dataset.description else ""
        
        return {
            "id": dataset.id,
            "name": dataset.name,
            "author": dataset.author,
            "description": description,
            "tags": ", ".join(dataset.tags),
            "downloads": dataset.downloads,
            "url": dataset.url,
            "source": dataset.source.value,
            "content": f"Dataset: {dataset.name}\n\nDescription: {description}\n\nTags: {', '.join(dataset.tags)}",
        }
    
    def clean_github_repo(self, repo: GitHubRepo) -> dict:
        """
        清洗 GitHub 仓库
        """
        description = self.clean(repo.description) if repo.description else ""
        
        return {
            "id": repo.id,
            "name": repo.name,
            "full_name": repo.full_name,
            "description": description,
            "owner": repo.owner,
            "language": repo.language,
            "topics": ", ".join(repo.topics),
            "stars": repo.stars,
            "forks": repo.forks,
            "url": repo.url,
            "source": repo.source.value,
            "content": f"Repository: {repo.full_name}\n\nDescription: {description}\n\nLanguage: {repo.language}\n\nTopics: {', '.join(repo.topics)}",
        }


# 便捷函数
_default_cleaner = DataCleaner()


def clean_text(text: str, **kwargs) -> str:
    """便捷函数：清洗文本"""
    if kwargs:
        cleaner = DataCleaner(**kwargs)
        return cleaner.clean(text)
    return _default_cleaner.clean(text)


def clean_paper(paper: Paper) -> dict:
    """便捷函数：清洗论文"""
    return _default_cleaner.clean_paper(paper)


def clean_social_post(post: SocialPost) -> dict:
    """便捷函数：清洗社交帖子"""
    return _default_cleaner.clean_social_post(post)
