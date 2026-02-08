"""
Demo Script - Quick demonstration of the Academic Research Agent
演示脚本 - 快速展示学术研究助手功能
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from scrapers import ArxivScraper, HuggingFaceScraper, GitHubScraper
from aggregator import DataAggregator


console = Console()


async def demo_individual_scrapers():
    """Demo individual scrapers"""
    topic = "DeepSeek"
    
    console.print(Panel.fit(
        f"[bold blue]🔍 Demonstrating Individual Scrapers[/bold blue]\n"
        f"Topic: [yellow]{topic}[/yellow]",
        border_style="blue"
    ))
    
    # ArXiv Demo
    console.print("\n[bold cyan]📄 ArXiv Scraper Demo[/bold cyan]")
    console.print("-" * 40)
    
    async with ArxivScraper() as scraper:
        papers = await scraper.search(topic, max_results=3)
        
        for i, paper in enumerate(papers, 1):
            console.print(f"\n{i}. [bold]{paper.title}[/bold]")
            console.print(f"   📅 Published: {paper.published_date.strftime('%Y-%m-%d') if paper.published_date else 'N/A'}")
            console.print(f"   🏷️ Categories: {', '.join(paper.categories[:3])}")
            console.print(f"   🔗 {paper.url}")
    
    # HuggingFace Demo
    console.print("\n\n[bold cyan]🤗 HuggingFace Scraper Demo[/bold cyan]")
    console.print("-" * 40)
    
    async with HuggingFaceScraper() as scraper:
        # Models
        console.print("\n[yellow]Models:[/yellow]")
        models = await scraper.search_models(topic, max_results=3)
        
        for i, model in enumerate(models, 1):
            console.print(f"  {i}. [bold]{model.id}[/bold]")
            console.print(f"     ⬇️ {model.downloads:,} downloads  ❤️ {model.likes} likes")
        
        # Datasets
        console.print("\n[yellow]Datasets:[/yellow]")
        datasets = await scraper.search_datasets(topic, max_results=3)
        
        for i, dataset in enumerate(datasets, 1):
            console.print(f"  {i}. [bold]{dataset.id}[/bold]")
            console.print(f"     ⬇️ {dataset.downloads:,} downloads")
    
    # GitHub Demo
    console.print("\n\n[bold cyan]🐙 GitHub Scraper Demo[/bold cyan]")
    console.print("-" * 40)
    
    async with GitHubScraper() as scraper:
        repos = await scraper.search_repos(topic, max_results=3)
        
        for i, repo in enumerate(repos, 1):
            console.print(f"\n{i}. [bold]{repo.full_name}[/bold]")
            if repo.description:
                console.print(f"   {repo.description[:80]}...")
            console.print(f"   ⭐ {repo.stars:,}  🍴 {repo.forks:,}  📝 {repo.language or 'N/A'}")


async def demo_aggregator():
    """Demo the aggregator"""
    topic = "Mixture of Experts"
    
    console.print("\n")
    console.print(Panel.fit(
        f"[bold blue]🔄 Demonstrating Data Aggregator[/bold blue]\n"
        f"Topic: [yellow]{topic}[/yellow]\n\n"
        f"[dim]This will search multiple sources simultaneously...[/dim]",
        border_style="blue"
    ))
    
    # Note: Twitter and Reddit require API keys, so we disable them for the demo
    async with DataAggregator(
        enable_arxiv=True,
        enable_huggingface=True,
        enable_twitter=False,  # Requires API key
        enable_reddit=False,   # Requires API key
        enable_github=True,
    ) as aggregator:
        
        result = await aggregator.aggregate(
            topic=topic,
            max_results_per_source=5,
            show_progress=True,
        )
    
    # Show a few highlights
    console.print("\n[bold green]✨ Highlights from aggregated results:[/bold green]")
    
    if result.papers:
        console.print(f"\n📄 Top Paper: [bold]{result.papers[0].title}[/bold]")
    
    if result.models:
        console.print(f"🤗 Top Model: [bold]{result.models[0].id}[/bold] ({result.models[0].downloads:,} downloads)")
    
    if result.github_repos:
        console.print(f"🐙 Top Repo: [bold]{result.github_repos[0].full_name}[/bold] (⭐ {result.github_repos[0].stars:,})")


async def main():
    """Main demo function"""
    console.print(Panel.fit(
        Markdown("""
# 🎓 Academic Research Agent - Demo

This demo shows the core data fetching capabilities:
1. **Individual Scrapers** - ArXiv, HuggingFace, GitHub
2. **Data Aggregator** - Unified search across all sources

> Note: Twitter and Reddit require API keys to be configured.
        """),
        border_style="blue"
    ))
    
    console.print("\n[dim]Press Enter to start...[/dim]")
    input()
    
    await demo_individual_scrapers()
    
    console.print("\n[dim]Press Enter to continue to aggregator demo...[/dim]")
    input()
    
    await demo_aggregator()
    
    console.print("\n" + "=" * 50)
    console.print("[bold green]✅ Demo completed![/bold green]")
    console.print("\nTo use with your own topic:")
    console.print("  [cyan]python main.py --topic \"Your Topic\"[/cyan]")
    console.print("\nTo configure social media APIs, edit:")
    console.print("  [cyan]config/.env[/cyan]")


if __name__ == "__main__":
    asyncio.run(main())
