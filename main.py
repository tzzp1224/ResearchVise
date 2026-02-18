"""
Academic Research Agent - Main Entry Point
"""
import asyncio
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.markdown import Markdown

from config import get_settings
from aggregator import DataAggregator
from models import AggregatedResult


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)
console = Console()


def print_banner():
    """Print welcome banner"""
    banner = """
# 🎓 Academic Research Agent

> 学术严谨与社区热度兼顾的智能研究助手

**Supported Sources:**
- 📄 ArXiv (Papers)
- 🤗 Hugging Face (Models & Datasets)  
- 🐦 Twitter/X (Discussions)
- 🔴 Reddit (Community)
- 🐙 GitHub (Code & Issues)
    """
    console.print(Panel(Markdown(banner), border_style="blue"))


def print_results(result: AggregatedResult):
    """Print detailed results"""
    
    # Top Papers
    if result.papers:
        console.print("\n📄 [bold cyan]Top Papers (ArXiv)[/bold cyan]")
        console.print("-" * 50)
        for i, paper in enumerate(result.papers[:5], 1):
            console.print(f"{i}. [bold]{paper.title}[/bold]")
            authors = ", ".join([a.name for a in paper.authors[:3]])
            if len(paper.authors) > 3:
                authors += " et al."
            console.print(f"   Authors: {authors}")
            console.print(f"   📅 {paper.published_date.strftime('%Y-%m-%d') if paper.published_date else 'N/A'}")
            console.print(f"   🔗 {paper.url}")
            console.print()
    
    # Top Models
    if result.models:
        console.print("\n🤗 [bold cyan]Top Models (Hugging Face)[/bold cyan]")
        console.print("-" * 50)
        for i, model in enumerate(result.models[:5], 1):
            console.print(f"{i}. [bold]{model.id}[/bold]")
            console.print(f"   ⬇️ Downloads: {model.downloads:,}  ❤️ Likes: {model.likes}")
            console.print(f"   🔗 {model.url}")
            console.print()
    
    # Top GitHub Repos
    if result.github_repos:
        console.print("\n🐙 [bold cyan]Top GitHub Repos[/bold cyan]")
        console.print("-" * 50)
        for i, repo in enumerate(result.github_repos[:5], 1):
            console.print(f"{i}. [bold]{repo.full_name}[/bold]")
            if repo.description:
                console.print(f"   {repo.description[:100]}...")
            console.print(f"   ⭐ {repo.stars:,}  🍴 {repo.forks:,}  📝 {repo.language or 'N/A'}")
            console.print(f"   🔗 {repo.url}")
            console.print()
    
    # Top Social Posts
    if result.social_posts:
        console.print("\n💬 [bold cyan]Top Social Discussions[/bold cyan]")
        console.print("-" * 50)
        
        # Sort by engagement
        sorted_posts = sorted(
            result.social_posts, 
            key=lambda x: x.likes + x.comments, 
            reverse=True
        )
        
        for i, post in enumerate(sorted_posts[:5], 1):
            source_emoji = {"twitter": "🐦", "reddit": "🔴", "github": "🐙"}.get(post.source.value, "💬")
            console.print(f"{i}. {source_emoji} [{post.source.value.upper()}] [bold]{post.author}[/bold]")
            content = post.content[:150].replace('\n', ' ')
            if len(post.content) > 150:
                content += "..."
            console.print(f"   {content}")
            console.print(f"   ❤️ {post.likes}  💬 {post.comments}")
            console.print(f"   🔗 {post.url}")
            console.print()
    
    # Top Stack Overflow Questions
    if result.stackoverflow_questions:
        console.print("\n📚 [bold cyan]Top Stack Overflow Questions[/bold cyan]")
        console.print("-" * 50)
        
        sorted_questions = sorted(
            result.stackoverflow_questions, 
            key=lambda x: x.score, 
            reverse=True
        )
        
        for i, q in enumerate(sorted_questions[:5], 1):
            answered = "✅" if q.is_answered else "❓"
            console.print(f"{i}. {answered} [bold]{q.title}[/bold]")
            console.print(f"   👤 {q.author} (Rep: {q.author_reputation:,})")
            console.print(f"   🏷️ {', '.join(q.tags[:5])}")
            console.print(f"   👍 {q.score}  👁️ {q.view_count:,}  💬 {q.answer_count}")
            console.print(f"   🔗 {q.url}")
            console.print()
    
    # Top Hacker News Items
    if result.hackernews_items:
        console.print("\n🔶 [bold cyan]Top Hacker News Discussions[/bold cyan]")
        console.print("-" * 50)
        
        sorted_items = sorted(
            result.hackernews_items, 
            key=lambda x: x.points, 
            reverse=True
        )
        
        for i, item in enumerate(sorted_items[:5], 1):
            console.print(f"{i}. [bold]{item.title}[/bold]")
            console.print(f"   👤 {item.author}  🔥 {item.points} points  💬 {item.comment_count}")
            if item.url:
                console.print(f"   🔗 {item.url}")
            console.print(f"   📰 {item.hn_url}")
            console.print()


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Academic Research Agent - Multi-source research aggregator"
    )
    parser.add_argument(
        "--topic", "-t",
        type=str,
        required=True,
        help="Research topic to search for"
    )
    parser.add_argument(
        "--max-results", "-n",
        type=int,
        default=30,
        help="Maximum results per source (default: 30)"
    )
    parser.add_argument(
        "--sort", "-s",
        type=str,
        choices=["relevance", "date", "updated"],
        default=None,  # None means use .env config
        help="Sort order for ArXiv: relevance, date, updated (default: from .env or relevance)"
    )
    parser.add_argument(
        "--no-arxiv",
        action="store_true",
        help="Disable ArXiv search"
    )
    parser.add_argument(
        "--no-huggingface",
        action="store_true",
        help="Disable Hugging Face search"
    )
    parser.add_argument(
        "--no-twitter",
        action="store_true",
        help="Disable Twitter search"
    )
    parser.add_argument(
        "--no-reddit",
        action="store_true",
        help="Disable Reddit search"
    )
    parser.add_argument(
        "--no-github",
        action="store_true",
        help="Disable GitHub search"
    )
    parser.add_argument(
        "--no-semantic-scholar",
        action="store_true",
        help="Disable Semantic Scholar search"
    )
    parser.add_argument(
        "--no-stackoverflow",
        action="store_true",
        help="Disable Stack Overflow search"
    )
    parser.add_argument(
        "--no-hackernews",
        action="store_true",
        help="Disable Hacker News search"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path (JSON format)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print_banner()
    
    # Create aggregator with specified options
    async with DataAggregator(
        enable_arxiv=not args.no_arxiv,
        enable_huggingface=not args.no_huggingface,
        enable_twitter=not args.no_twitter,
        enable_reddit=not args.no_reddit,
        enable_github=not args.no_github,
        enable_semantic_scholar=not args.no_semantic_scholar,
        enable_stackoverflow=not args.no_stackoverflow,
        enable_hackernews=not args.no_hackernews,
    ) as aggregator:
        
        # Map sort option to ArXiv sort_by
        # Priority: CLI arg > .env config > default (relevance)
        sort_map = {
            "relevance": "relevance",
            "date": "submittedDate",
            "updated": "lastUpdatedDate",
        }
        
        if args.sort:
            # CLI 指定了排序方式
            arxiv_sort = sort_map.get(args.sort, "relevance")
        else:
            # 使用 .env 配置，如果没有则默认 relevance
            settings = get_settings()
            env_sort = settings.arxiv.sort_by
            # 反向映射：submittedDate -> date
            reverse_map = {"submittedDate": "date", "lastUpdatedDate": "updated", "relevance": "relevance"}
            arxiv_sort = env_sort if env_sort in sort_map.values() else "relevance"
        
        # Run aggregation
        result = await aggregator.aggregate(
            topic=args.topic,
            max_results_per_source=args.max_results,
            arxiv_sort_by=arxiv_sort,
        )
        
        # Print results
        print_results(result)
        
        # Save to file if specified
        if args.output:
            import json
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result.model_dump(mode='json'), f, ensure_ascii=False, indent=2, default=str)
            
            console.print(f"\n✅ Results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
