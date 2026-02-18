"""
Phase 3 Demo: Intelligence Layer
演示 LLM + LangGraph + 多Agent协作
"""
import asyncio
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel

console = Console()


def check_environment():
    """检查环境配置"""
    console.print("\n[bold blue]Environment Check[/bold blue]\n")
    
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / "config" / ".env")
    
    # 检查 LLM API Key
    llm_keys = {
        "DEEPSEEK": os.getenv("LLM_DEEPSEEK_API_KEY"),
        "OPENAI": os.getenv("LLM_OPENAI_API_KEY"),
        "ANTHROPIC": os.getenv("LLM_ANTHROPIC_API_KEY"),
        "GEMINI": os.getenv("LLM_GEMINI_API_KEY"),
    }
    
    available_llms = []
    for name, key in llm_keys.items():
        if key and not key.startswith("your_"):
            console.print(f"  [green]OK[/green] {name}: {key[:10]}...{key[-4:]}")
            available_llms.append(name.lower())
        else:
            console.print(f"  [dim]--[/dim] {name}: Not configured")
    
    # 检查 Embedding API Key
    embed_key = os.getenv("SILICONFLOW_API_KEY")
    if embed_key and not embed_key.startswith("your_"):
        console.print(f"  [green]OK[/green] SILICONFLOW (Embedding): {embed_key[:10]}...")
    
    return available_llms


async def demo_llm():
    """演示 LLM 抽象层"""
    console.print("\n[bold cyan]1. LLM Abstraction Layer[/bold cyan]\n")
    
    from intelligence.llm import get_llm, Message
    
    # 获取 LLM (从 .env 配置)
    try:
        llm = get_llm()
        console.print(f"  Using: {llm}")
        
        # 简单对话
        messages = [
            Message.system("You are a helpful AI assistant. Keep responses brief."),
            Message.user("What is DeepSeek-V3 in one sentence?"),
        ]
        
        console.print("  Sending request...")
        response = await llm.acomplete(messages)
        
        console.print(f"  [green]Response:[/green] {response.content[:200]}...")
        console.print(f"  [dim]Tokens: {response.usage}[/dim]")
        
        return True
    except Exception as e:
        console.print(f"  [red]Error:[/red] {e}")
        return False


async def demo_search_agent():
    """演示搜索Agent"""
    console.print("\n[bold cyan]2. Search Agent (ReAct)[/bold cyan]\n")
    
    from intelligence.agents import SearchAgent
    
    agent = SearchAgent(max_iterations=3)
    
    # 生成搜索计划
    console.print("  Generating search plan for 'DeepSeek-V3'...")
    
    try:
        plan = await agent.plan_search("DeepSeek-V3")
        console.print(f"  [green]Search keywords:[/green] {plan}")
        return True
    except Exception as e:
        console.print(f"  [red]Error:[/red] {e}")
        return False


async def demo_analyst_agent():
    """演示分析Agent"""
    console.print("\n[bold cyan]3. Analyst Agent (RAG)[/bold cyan]\n")
    
    from intelligence.agents import AnalystAgent
    
    agent = AnalystAgent()
    
    # 模拟搜索结果
    mock_state = {
        "topic": "DeepSeek-V3",
        "search_results": [
            {
                "id": "arxiv_001",
                "source": "arxiv",
                "title": "DeepSeek-V3 Technical Report",
                "content": "DeepSeek-V3 is a 671B parameter MoE model with 37B activated parameters. It achieves GPT-4 level performance at 1/10 the training cost.",
                "metadata": {"published_date": "2024-12"},
            },
            {
                "id": "github_001",
                "source": "github",
                "title": "deepseek-ai/DeepSeek-V3",
                "content": "Official implementation of DeepSeek-V3. MIT License. 10k+ stars.",
                "metadata": {"stars": 10000},
            },
        ],
    }
    
    console.print("  Analyzing mock search results...")
    
    try:
        result = await agent.run(mock_state)
        
        facts = result.get("facts", [])
        console.print(f"  [green]Extracted {len(facts)} facts[/green]")
        
        for f in facts[:3]:
            console.print(f"    - [{f.get('category')}] {f.get('claim')[:80]}...")
        
        gaps = result.get("knowledge_gaps", [])
        if gaps:
            console.print(f"  [yellow]Knowledge gaps:[/yellow] {gaps[:2]}")
        
        return True
    except Exception as e:
        console.print(f"  [red]Error:[/red] {e}")
        return False


async def demo_content_agent():
    """演示内容生成Agent"""
    console.print("\n[bold cyan]4. Content Agent (Parallel Generation)[/bold cyan]\n")
    
    from intelligence.agents import ContentAgent
    
    agent = ContentAgent()
    
    # 模拟分析结果
    mock_state = {
        "topic": "DeepSeek-V3",
        "facts": [
            {
                "id": "fact_001",
                "claim": "DeepSeek-V3 is a 671B parameter MoE model with 37B activated parameters",
                "confidence": 0.95,
                "source_type": "paper",
                "category": "architecture",
            },
            {
                "id": "fact_002",
                "claim": "Training cost is estimated at $5.5M, 1/10 of GPT-4",
                "confidence": 0.9,
                "source_type": "paper",
                "category": "training",
            },
            {
                "id": "fact_003",
                "claim": "Achieves 87.1% on MMLU benchmark",
                "confidence": 0.95,
                "source_type": "paper",
                "category": "performance",
            },
        ],
    }
    
    console.print("  Generating content in parallel...")
    console.print("    - Timeline")
    console.print("    - One-Pager")
    console.print("    - Video Brief")
    
    try:
        result = await agent.run(mock_state)
        
        if result.get("timeline"):
            console.print(f"  [green]Timeline:[/green] {len(result['timeline'])} events")
        
        if result.get("one_pager"):
            op = result["one_pager"]
            console.print(f"  [green]One-Pager:[/green] {op.get('title', 'N/A')}")
            console.print(f"    Summary: {op.get('executive_summary', 'N/A')[:100]}...")
        
        if result.get("video_brief"):
            vb = result["video_brief"]
            console.print(f"  [green]Video Brief:[/green] {vb.get('title', 'N/A')}")
            console.print(f"    Duration: {vb.get('duration_estimate', 'N/A')}")
        
        return True
    except Exception as e:
        console.print(f"  [red]Error:[/red] {e}")
        import traceback
        traceback.print_exc()
        return False


async def demo_langgraph():
    """演示完整 LangGraph 流程"""
    console.print("\n[bold cyan]5. LangGraph Research Flow[/bold cyan]\n")
    
    try:
        from intelligence.graph import ResearchGraph
        
        console.print("  Creating research graph...")
        graph = ResearchGraph()
        
        console.print("  Graph structure:")
        console.print("    START -> planning -> searching -> analyzing")
        console.print("    analyzing -> [refine OR generating]")
        console.print("    refine -> [searching OR generating]")
        console.print("    generating -> completion -> END")
        
        console.print("\n  [yellow]Note:[/yellow] Full graph execution requires API calls.")
        console.print("  Use 'python -m intelligence.graph.research_graph' for full demo.")
        
        return True
    except Exception as e:
        console.print(f"  [red]Error:[/red] {e}")
        import traceback
        traceback.print_exc()
        return False


def show_architecture():
    """显示 Phase 3 架构"""
    console.print(Panel.fit(
        """
[bold]Phase 3 Architecture: Intelligence Layer[/bold]

+------------------+     +------------------+     +------------------+
|  Search Agent    |     |  Analyst Agent   |     |  Content Agent   |
|  (ReAct Loop)    | --> |  (RAG + Verify)  | --> |  (Parallel Gen)  |
+------------------+     +------------------+     +------------------+
        |                        |                        |
        v                        v                        v
+------------------+     +------------------+     +------------------+
| ArXiv, HuggingFace|    | Vector Search   |     | Timeline         |
| Twitter, Reddit  |     | Fact Extraction  |     | One-Pager        |
| GitHub           |     | Confidence Score |     | Video Brief      |
+------------------+     +------------------+     +------------------+

[bold]LangGraph Orchestration:[/bold]
- State Management: AgentState (topic, results, facts, outputs)
- Cyclic Workflows: Refine loop for insufficient information
- Parallel Execution: Content generation in parallel
        """,
        title="Phase 3 Demo",
        border_style="blue",
    ))


async def main():
    """主函数"""
    show_architecture()
    
    # 检查环境
    available_llms = check_environment()
    
    if not available_llms:
        console.print("\n[red]No LLM API key configured![/red]")
        console.print("Please set one of the following in config/.env:")
        console.print("  - LLM_DEEPSEEK_API_KEY (recommended)")
        console.print("  - LLM_OPENAI_API_KEY")
        console.print("  - LLM_ANTHROPIC_API_KEY")
        console.print("  - LLM_GEMINI_API_KEY")
        return
    
    # 运行演示
    console.print("\n" + "="*50)
    
    results = {}
    
    # 1. LLM 抽象层
    results["LLM"] = await demo_llm()
    
    # 2. Search Agent
    results["Search Agent"] = await demo_search_agent()
    
    # 3. Analyst Agent
    results["Analyst Agent"] = await demo_analyst_agent()
    
    # 4. Content Agent
    results["Content Agent"] = await demo_content_agent()
    
    # 5. LangGraph
    results["LangGraph"] = await demo_langgraph()
    
    # 总结
    console.print("\n" + "="*50)
    console.print("[bold]Summary[/bold]\n")
    
    for name, success in results.items():
        status = "[green]PASS[/green]" if success else "[red]FAIL[/red]"
        console.print(f"  {name}: {status}")
    
    console.print("\n[bold green]Phase 3 Demo Complete![/bold green]")
    
    # 显式关闭 Qdrant 客户端以避免退出时警告
    from intelligence.tools.rag_tools import _vector_store
    if _vector_store is not None:
        _vector_store.client.close()


if __name__ == "__main__":
    asyncio.run(main())
