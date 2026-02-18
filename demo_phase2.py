#!/usr/bin/env python
"""
Phase 2 Demo - Processing & Storage Pipeline
演示完整的处理+存储流程
"""
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def check_environment():
    """检查环境配置"""
    console.print("\n[bold blue]🔧 环境检查[/bold blue]\n")
    
    api_key = os.environ.get("SILICONFLOW_API_KEY")
    if api_key:
        console.print(f"  ✅ SILICONFLOW_API_KEY: {api_key[:10]}...{api_key[-4:]}")
        return True
    else:
        console.print("  ⚠️ SILICONFLOW_API_KEY 未设置")
        console.print("     请访问 https://siliconflow.cn/ 注册获取免费 API Key")
        console.print("     然后设置: set SILICONFLOW_API_KEY=your_key_here\n")
        return False


def demo_cleaner():
    """演示数据清洗"""
    console.print("\n[bold green]📝 1. 数据清洗 (DataCleaner)[/bold green]\n")
    
    from processing import clean_text
    
    # 模拟脏数据
    dirty_text = """
    Check out this paper: https://arxiv.org/abs/2401.12345
    @researcher mentioned this is &amp; amazing! #DeepLearning
    
    The transformer architecture...  uses    multiple attention heads.
    """
    
    console.print("[dim]原始文本:[/dim]")
    console.print(Panel(dirty_text, border_style="red"))
    
    # 清洗
    cleaned = clean_text(
        dirty_text,
        remove_urls=True,
        remove_mentions=True,
        remove_hashtags=True,
    )
    
    console.print("[dim]清洗后:[/dim]")
    console.print(Panel(cleaned, border_style="green"))


def demo_chunker():
    """演示文本分块"""
    console.print("\n[bold green]📦 2. 文本分块 (TextChunker)[/bold green]\n")
    
    from processing import TextChunker, ChunkingStrategy
    
    long_text = """
    Transformers have revolutionized natural language processing. 
    The key innovation is the self-attention mechanism, which allows 
    the model to weigh the importance of different parts of the input.
    
    Unlike RNNs, transformers process all positions in parallel.
    This makes them much faster to train on modern hardware.
    The architecture consists of an encoder and decoder, each with 
    multiple layers of attention and feed-forward networks.
    
    BERT, GPT, and T5 are famous transformer-based models.
    They have achieved state-of-the-art results on many NLP tasks.
    """
    
    chunker = TextChunker(
        strategy=ChunkingStrategy.RECURSIVE,
        chunk_size=200,
        chunk_overlap=30,
    )
    
    chunks = chunker.chunk(
        long_text,
        doc_id="demo_doc",
        metadata={"source": "demo"},
    )
    
    table = Table(title="分块结果", show_header=True)
    table.add_column("Chunk ID", style="cyan")
    table.add_column("Length", style="green")
    table.add_column("Preview", style="white")
    
    for chunk in chunks:
        preview = chunk.content[:50] + "..." if len(chunk.content) > 50 else chunk.content
        table.add_row(
            chunk.id,
            str(len(chunk.content)),
            preview.strip(),
        )
    
    console.print(table)


def demo_embedder(use_api: bool = True):
    """演示向量化"""
    console.print("\n[bold green]🔢 3. 向量化 (Embedder)[/bold green]\n")
    
    from processing import get_embedder
    
    if use_api:
        embedder = get_embedder("siliconflow")
        console.print(f"  使用: SiliconFlow BGE-M3 (维度: {embedder.dimension})")
    else:
        embedder = get_embedder("sentence_transformers")
        console.print(f"  使用: SentenceTransformers (维度: {embedder.dimension})")
    
    texts = [
        "机器学习是人工智能的一个子集",
        "深度学习使用多层神经网络",
        "Transformers are attention-based models",
    ]
    
    console.print("\n  [dim]输入文本:[/dim]")
    for i, text in enumerate(texts):
        console.print(f"    {i+1}. {text}")
    
    console.print("\n  [dim]计算向量...[/dim]")
    embeddings = embedder.embed(texts)
    
    console.print(f"\n  ✅ 生成向量: shape = {embeddings.shape}")
    console.print(f"     每个文本 → {embeddings.shape[1]} 维向量")
    
    # 计算相似度
    import numpy as np
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    console.print("\n  [dim]文本相似度:[/dim]")
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            console.print(f"    [{i+1}] vs [{j+1}]: {sim:.4f}")
    
    return embedder


def demo_vector_store(embedder):
    """演示向量存储"""
    console.print("\n[bold green]💾 4. 向量存储 (Qdrant)[/bold green]\n")
    
    from storage import QdrantVectorStore
    from processing import chunk_document
    
    # 模拟论文数据
    paper_abstract = """
    We introduce the Transformer, a new architecture for sequence transduction.
    Unlike recurrent models, the Transformer relies entirely on attention mechanisms.
    Our experiments show the model achieves state-of-the-art results on translation tasks.
    The Transformer is more parallelizable and requires significantly less time to train.
    """
    
    # 分块
    chunks = chunk_document(
        content=paper_abstract,
        doc_id="paper_001",
        doc_type="paper",
        metadata={"title": "Attention Is All You Need", "year": 2017},
        chunk_size=150,
    )
    
    # 计算向量
    texts = [c.content for c in chunks]
    embeddings = embedder.embed(texts)
    
    # 存储 (使用 Qdrant)
    store = QdrantVectorStore(
        collection_name="demo_papers",
        persist_directory=None,  # 内存模式
        dimension=embedder.dimension,
    )
    
    # 添加到存储
    store.add_with_embeddings(
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[c.metadata for c in chunks],
        ids=[c.id for c in chunks],
    )
    
    console.print(f"  ✅ 存储 {store.count()} 个文档块")
    
    # 演示元数据过滤 (Qdrant 特性)
    console.print("\n  [dim]Qdrant 支持元数据过滤:[/dim]")
    console.print("    - filter={\"year\": {\"$gte\": 2015}}")
    console.print("    - filter={\"topic\": \"LLM\"}")
    
    # 搜索
    query = "What is the main innovation?"
    console.print(f"\n  [dim]搜索: '{query}'[/dim]")
    
    query_embedding = embedder.embed(query)
    results = store.search_with_embedding(
        query_embedding=query_embedding[0].tolist(),
        top_k=2,
    )
    
    console.print("\n  [bold]搜索结果:[/bold]")
    for i, result in enumerate(results):
        console.print(f"    {i+1}. [Score: {result.score:.4f}]")
        console.print(f"       {result.content[:80]}...")
    
    # 清理
    store.clear()


def demo_cache():
    """演示缓存"""
    console.print("\n[bold green]💨 5. 缓存 (Cache)[/bold green]\n")
    
    from storage import MemoryCache
    
    # 内存缓存
    cache = MemoryCache(ttl=60)
    
    cache.set("query_result", {
        "query": "transformer attention",
        "results": ["paper1", "paper2"],
        "timestamp": datetime.now().isoformat(),
    })
    
    result = cache.get("query_result")
    console.print(f"  ✅ 缓存读写正常: {result}")


def main():
    """主函数"""
    console.print(Panel.fit(
        "[bold blue]Phase 2 Demo: Processing & Storage Pipeline[/bold blue]\n"
        "演示数据清洗、分块、向量化、存储的完整流程",
        border_style="blue",
    ))
    
    # 加载环境变量
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / "config" / ".env")
    
    # 1. 检查环境
    has_api_key = check_environment()
    
    # 2. 演示清洗
    demo_cleaner()
    
    # 3. 演示分块
    demo_chunker()
    
    # 4. 演示缓存 (不需要 API)
    demo_cache()
    
    if has_api_key:
        # 5. 演示向量化 (需要 API)
        embedder = demo_embedder(use_api=True)
        
        # 6. 演示向量存储 (需要 API)
        demo_vector_store(embedder)
        
        console.print("\n" + "="*50)
        console.print("[bold green]✅ Phase 2 完整演示完成！[/bold green]\n")
    else:
        console.print("\n" + "="*50)
        console.print("[yellow]⚠️ 部分演示跳过 (需要 API Key)[/yellow]")
        console.print("设置 SILICONFLOW_API_KEY 后可体验完整功能\n")


if __name__ == "__main__":
    main()
