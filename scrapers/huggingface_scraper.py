"""
Hugging Face Scraper
从 Hugging Face Hub 抓取模型、数据集和论文
"""
import asyncio
from datetime import datetime
from typing import List, Optional, Literal
import logging
import inspect

from huggingface_hub import HfApi
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BaseScraper
from models import Paper, Model, Dataset, Author, SourceType
from config import get_settings


logger = logging.getLogger(__name__)


class HuggingFaceScraper(BaseScraper[Model]):
    """
    Hugging Face Hub 抓取器
    支持搜索模型、数据集和论文
    """
    
    def __init__(self):
        super().__init__()
        self._hf_settings = get_settings().huggingface
        self._api = HfApi(token=self._hf_settings.token)
    
    @property
    def source_type(self) -> SourceType:
        return SourceType.HUGGINGFACE
    
    @property
    def name(self) -> str:
        return "HuggingFace"
    
    def is_configured(self) -> bool:
        # HuggingFace Hub 可以不需要token访问公开资源
        return True
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def search(
        self, 
        query: str, 
        max_results: Optional[int] = None,
    ) -> List[Model]:
        """
        搜索 Hugging Face 模型
        
        Args:
            query: 搜索关键词
            max_results: 最大返回结果数
            
        Returns:
            模型列表
        """
        return await self.search_models(query, max_results)
    
    async def search_models(
        self,
        query: str,
        max_results: Optional[int] = None,
        task: Optional[str] = None,
        library: Optional[str] = None,
        sort: str = "downloads",
    ) -> List[Model]:
        """
        搜索模型
        
        Args:
            query: 搜索关键词
            max_results: 最大返回结果数
            task: 任务类型 (如 "text-generation", "image-classification")
            library: 框架 (如 "transformers", "pytorch")
            sort: 排序方式 (downloads, likes, lastModified)
            
        Returns:
            模型列表
        """
        if max_results is None:
            max_results = self._hf_settings.max_results
        
        logger.info(f"[HuggingFace] Searching models: {query}")
        
        loop = asyncio.get_event_loop()
        
        results = await loop.run_in_executor(
            None,
            self._sync_search_models,
            query,
            max_results,
            task,
            library,
            sort,
        )
        
        models = [self._convert_to_model(m) for m in results]
        self._log_search(query, len(models))
        
        return models
    
    def _sync_search_models(
        self,
        query: str,
        max_results: int,
        task: Optional[str],
        library: Optional[str],
        sort: str,
    ) -> list:
        """同步搜索模型"""
        kwargs = {
            "search": query,
            "limit": max_results,
            "sort": sort,
        }

        sig = inspect.signature(self._api.list_models)
        if "task" in sig.parameters and task:
            kwargs["task"] = task
        if "pipeline_tag" in sig.parameters and task:
            kwargs["pipeline_tag"] = task
        if "library" in sig.parameters and library:
            kwargs["library"] = library
        if "library_name" in sig.parameters and library:
            kwargs["library_name"] = library
        return list(self._api.list_models(**kwargs))
    
    async def search_datasets(
        self,
        query: str,
        max_results: Optional[int] = None,
        task: Optional[str] = None,
    ) -> List[Dataset]:
        """
        搜索数据集
        
        Args:
            query: 搜索关键词
            max_results: 最大返回结果数
            task: 任务类型
            
        Returns:
            数据集列表
        """
        if max_results is None:
            max_results = self._hf_settings.max_results
        
        logger.info(f"[HuggingFace] Searching datasets: {query}")
        
        loop = asyncio.get_event_loop()
        
        results = await loop.run_in_executor(
            None,
            self._sync_search_datasets,
            query,
            max_results,
            task,
        )
        
        datasets = [self._convert_to_dataset(d) for d in results]
        logger.info(f"[HuggingFace] Found {len(datasets)} datasets for '{query}'")
        
        return datasets
    
    def _sync_search_datasets(
        self,
        query: str,
        max_results: int,
        task: Optional[str],
    ) -> list:
        """同步搜索数据集"""
        kwargs = {
            "search": query,
            "limit": max_results,
            "sort": "downloads",
        }

        sig = inspect.signature(self._api.list_datasets)
        if "task_categories" in sig.parameters and task:
            kwargs["task_categories"] = task
        return list(self._api.list_datasets(**kwargs))
    
    async def search_papers(
        self,
        query: str,
        max_results: Optional[int] = None,
    ) -> List[Paper]:
        """
        搜索 Hugging Face Daily Papers
        
        注意: HuggingFace API 目前对论文搜索支持有限
        这里我们通过搜索关联论文的模型来间接获取论文信息
        """
        if max_results is None:
            max_results = self._hf_settings.max_results
        
        logger.info(f"[HuggingFace] Searching papers related to: {query}")
        
        # 通过 HF API 获取有论文链接的模型
        loop = asyncio.get_event_loop()
        
        results = await loop.run_in_executor(
            None,
            self._sync_search_papers,
            query,
            max_results,
        )
        
        papers = []
        seen_papers = set()
        
        for model in results:
            # 检查模型是否有关联论文
            if hasattr(model, 'cardData') and model.cardData:
                paper_url = model.cardData.get('arxiv') or model.cardData.get('paper')
                if paper_url and paper_url not in seen_papers:
                    seen_papers.add(paper_url)
                    papers.append(Paper(
                        id=paper_url,
                        title=f"Paper for {model.modelId}",
                        abstract="",  # 需要额外请求获取
                        authors=[],
                        url=paper_url,
                        source=SourceType.HUGGINGFACE,
                        extra={"related_model": model.modelId}
                    ))
        
        logger.info(f"[HuggingFace] Found {len(papers)} papers for '{query}'")
        return papers
    
    def _sync_search_papers(self, query: str, max_results: int) -> list:
        """同步搜索论文"""
        # 搜索可能有论文的模型
        kwargs = {
            "search": query,
            "limit": max_results * 2,  # 多搜一些因为不是所有模型都有论文
            "sort": "downloads",
        }
        sig = inspect.signature(self._api.list_models)
        if "direction" in sig.parameters:
            kwargs["direction"] = -1
        return list(self._api.list_models(**kwargs))
    
    async def get_details(self, model_id: str) -> Optional[Model]:
        """
        获取模型详情
        
        Args:
            model_id: 模型ID (如 "meta-llama/Llama-2-7b")
            
        Returns:
            模型详情
        """
        loop = asyncio.get_event_loop()
        
        try:
            result = await loop.run_in_executor(
                None,
                self._api.model_info,
                model_id,
            )
            return self._convert_to_model(result)
        except Exception as e:
            self._log_error(f"Failed to get model {model_id}", e)
            return None
    
    async def get_dataset_details(self, dataset_id: str) -> Optional[Dataset]:
        """
        获取数据集详情
        
        Args:
            dataset_id: 数据集ID
            
        Returns:
            数据集详情
        """
        loop = asyncio.get_event_loop()
        
        try:
            result = await loop.run_in_executor(
                None,
                self._api.dataset_info,
                dataset_id,
            )
            return self._convert_to_dataset(result)
        except Exception as e:
            self._log_error(f"Failed to get dataset {dataset_id}", e)
            return None
    
    def _convert_to_model(self, model_info) -> Model:
        """将 HuggingFace ModelInfo 转换为 Model"""
        # 提取作者/组织
        author = None
        if hasattr(model_info, 'author'):
            author = model_info.author
        elif '/' in model_info.modelId:
            author = model_info.modelId.split('/')[0]
        
        # 提取标签
        tags = list(model_info.tags) if hasattr(model_info, 'tags') and model_info.tags else []
        
        return Model(
            id=model_info.modelId,
            name=model_info.modelId.split('/')[-1] if '/' in model_info.modelId else model_info.modelId,
            author=author,
            description=getattr(model_info, 'description', None),
            tags=tags,
            downloads=getattr(model_info, 'downloads', 0) or 0,
            likes=getattr(model_info, 'likes', 0) or 0,
            url=f"https://huggingface.co/{model_info.modelId}",
            created_at=getattr(model_info, 'createdAt', None),
            updated_at=getattr(model_info, 'lastModified', None),
            source=SourceType.HUGGINGFACE,
            extra={
                "pipeline_tag": getattr(model_info, 'pipeline_tag', None),
                "library_name": getattr(model_info, 'library_name', None),
                "private": getattr(model_info, 'private', False),
            }
        )
    
    def _convert_to_dataset(self, dataset_info) -> Dataset:
        """将 HuggingFace DatasetInfo 转换为 Dataset"""
        # 提取作者/组织
        author = None
        if hasattr(dataset_info, 'author'):
            author = dataset_info.author
        elif '/' in dataset_info.id:
            author = dataset_info.id.split('/')[0]
        
        # 提取标签
        tags = list(dataset_info.tags) if hasattr(dataset_info, 'tags') and dataset_info.tags else []
        
        return Dataset(
            id=dataset_info.id,
            name=dataset_info.id.split('/')[-1] if '/' in dataset_info.id else dataset_info.id,
            author=author,
            description=getattr(dataset_info, 'description', None),
            tags=tags,
            downloads=getattr(dataset_info, 'downloads', 0) or 0,
            url=f"https://huggingface.co/datasets/{dataset_info.id}",
            source=SourceType.HUGGINGFACE,
            extra={
                "citation": getattr(dataset_info, 'citation', None),
            }
        )


# 便捷函数
async def search_hf_models(query: str, max_results: int = 30) -> List[Model]:
    """便捷函数：搜索 HuggingFace 模型"""
    async with HuggingFaceScraper() as scraper:
        return await scraper.search_models(query, max_results)


async def search_hf_datasets(query: str, max_results: int = 30) -> List[Dataset]:
    """便捷函数：搜索 HuggingFace 数据集"""
    async with HuggingFaceScraper() as scraper:
        return await scraper.search_datasets(query, max_results)
