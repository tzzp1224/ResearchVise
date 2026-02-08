"""
Tests for Scrapers
"""
import asyncio
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapers import (
    ArxivScraper,
    HuggingFaceScraper,
    TwitterScraper,
    RedditScraper,
    GitHubScraper,
)
from aggregator import DataAggregator
from models import SourceType


class TestArxivScraper:
    """ArXiv Scraper Tests"""
    
    @pytest.mark.asyncio
    async def test_search_basic(self):
        """Test basic ArXiv search"""
        async with ArxivScraper() as scraper:
            papers = await scraper.search("transformer", max_results=5)
            
            assert len(papers) > 0
            assert len(papers) <= 5
            
            paper = papers[0]
            assert paper.title
            assert paper.abstract
            assert paper.url
            assert paper.source == SourceType.ARXIV
    
    @pytest.mark.asyncio
    async def test_search_with_category(self):
        """Test ArXiv search with category filter"""
        async with ArxivScraper() as scraper:
            papers = await scraper.search(
                "attention mechanism",
                max_results=3,
                categories=["cs.AI", "cs.LG"]
            )
            
            assert len(papers) > 0
            for paper in papers:
                # Check that papers are in the expected categories
                assert any(cat.startswith("cs.") for cat in paper.categories)
    
    @pytest.mark.asyncio
    async def test_get_details(self):
        """Test getting paper details by ID"""
        async with ArxivScraper() as scraper:
            # "Attention Is All You Need" paper
            paper = await scraper.get_details("1706.03762")
            
            assert paper is not None
            assert "attention" in paper.title.lower()


class TestHuggingFaceScraper:
    """HuggingFace Scraper Tests"""
    
    @pytest.mark.asyncio
    async def test_search_models(self):
        """Test HuggingFace model search"""
        async with HuggingFaceScraper() as scraper:
            models = await scraper.search_models("llama", max_results=5)
            
            assert len(models) > 0
            assert len(models) <= 5
            
            model = models[0]
            assert model.id
            assert model.url
            assert model.source == SourceType.HUGGINGFACE
    
    @pytest.mark.asyncio
    async def test_search_datasets(self):
        """Test HuggingFace dataset search"""
        async with HuggingFaceScraper() as scraper:
            datasets = await scraper.search_datasets("sentiment", max_results=5)
            
            assert len(datasets) > 0
            
            dataset = datasets[0]
            assert dataset.id
            assert dataset.url


class TestGitHubScraper:
    """GitHub Scraper Tests"""
    
    @pytest.mark.asyncio
    async def test_search_repos(self):
        """Test GitHub repo search"""
        async with GitHubScraper() as scraper:
            repos = await scraper.search_repos("machine learning", max_results=5)
            
            assert len(repos) > 0
            
            repo = repos[0]
            assert repo.full_name
            assert repo.url
            assert repo.source == SourceType.GITHUB
    
    @pytest.mark.asyncio
    async def test_search_with_language(self):
        """Test GitHub search with language filter"""
        async with GitHubScraper() as scraper:
            repos = await scraper.search_repos(
                "deep learning",
                max_results=5,
                language="python"
            )
            
            assert len(repos) > 0
            # Most repos should be Python
            python_repos = [r for r in repos if r.language and r.language.lower() == "python"]
            assert len(python_repos) >= len(repos) // 2


class TestDataAggregator:
    """Data Aggregator Tests"""
    
    @pytest.mark.asyncio
    async def test_aggregate_basic(self):
        """Test basic aggregation (ArXiv and HuggingFace only)"""
        async with DataAggregator(
            enable_arxiv=True,
            enable_huggingface=True,
            enable_twitter=False,  # Requires API key
            enable_reddit=False,   # Requires API key
            enable_github=True,
        ) as aggregator:
            result = await aggregator.aggregate(
                topic="BERT",
                max_results_per_source=3,
                show_progress=False,
            )
            
            assert result.topic == "BERT"
            assert result.total_count > 0
            
            # Check summary
            summary = result.summary()
            assert "papers" in summary
            assert "models" in summary
            assert "total" in summary


# Quick test runner
async def run_quick_test():
    """Run a quick test without pytest"""
    print("🧪 Running quick tests...\n")
    
    # Test ArXiv
    print("1. Testing ArXiv...")
    async with ArxivScraper() as scraper:
        papers = await scraper.search("DeepSeek", max_results=3)
        print(f"   ✅ Found {len(papers)} papers")
        if papers:
            print(f"   📄 First: {papers[0].title[:60]}...")
    
    # Test HuggingFace
    print("\n2. Testing HuggingFace...")
    async with HuggingFaceScraper() as scraper:
        models = await scraper.search_models("DeepSeek", max_results=3)
        print(f"   ✅ Found {len(models)} models")
        if models:
            print(f"   🤗 First: {models[0].id}")
    
    # Test GitHub
    print("\n3. Testing GitHub...")
    async with GitHubScraper() as scraper:
        repos = await scraper.search_repos("DeepSeek", max_results=3)
        print(f"   ✅ Found {len(repos)} repos")
        if repos:
            print(f"   🐙 First: {repos[0].full_name} (⭐ {repos[0].stars})")
    
    print("\n✅ All quick tests passed!")


if __name__ == "__main__":
    # Run quick test without pytest
    asyncio.run(run_quick_test())
