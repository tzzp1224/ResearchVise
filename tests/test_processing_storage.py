"""
Tests for Processing and Storage modules
"""
import importlib.util
import os
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

HAS_SENTENCE_TRANSFORMERS = importlib.util.find_spec("sentence_transformers") is not None


class TestCleaner:
    """数据清洗测试"""
    
    def test_clean_text_basic(self):
        from processing import clean_text
        
        text = "Hello  World!   Multiple   spaces"
        cleaned = clean_text(text)
        assert "  " not in cleaned
    
    def test_clean_text_urls(self):
        from processing import clean_text
        
        text = "Check this out https://example.com and more"
        cleaned = clean_text(text, remove_urls=True)
        assert "https://" not in cleaned
    
    def test_clean_text_html(self):
        from processing import clean_text
        
        text = "Hello &amp; World &lt;tag&gt;"
        cleaned = clean_text(text)
        assert "&amp;" not in cleaned
        assert "& World" in cleaned


class TestChunker:
    """文本分块测试"""
    
    def test_chunk_basic(self):
        from processing import chunk_text
        
        text = "This is a test. " * 100
        chunks = chunk_text(text, chunk_size=200)
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.content) <= 250  # 允许一些余量
    
    def test_chunk_with_metadata(self):
        from processing import chunk_document
        
        content = "Abstract paragraph. " * 50
        chunks = chunk_document(
            content=content,
            doc_id="paper_001",
            doc_type="paper",
            metadata={"title": "Test Paper"},
        )
        
        assert len(chunks) > 0
        assert chunks[0].metadata["doc_type"] == "paper"
        assert chunks[0].metadata["title"] == "Test Paper"
    
    def test_chunk_strategies(self):
        from processing import TextChunker, ChunkingStrategy
        
        text = """
        First paragraph with some content.
        
        Second paragraph with more content.
        
        Third paragraph with even more content.
        """
        
        # 按段落分块
        chunker = TextChunker(
            strategy=ChunkingStrategy.PARAGRAPH,
            chunk_size=100,
        )
        chunks = chunker.chunk(text)
        
        assert len(chunks) >= 1


class TestCache:
    """缓存测试"""
    
    def test_memory_cache_basic(self):
        from storage import MemoryCache
        
        cache = MemoryCache()
        
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        cache.delete("key1")
        assert cache.get("key1") is None
    
    def test_memory_cache_ttl(self):
        from storage import MemoryCache
        import time
        
        cache = MemoryCache(ttl=1)  # 1秒过期
        
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        time.sleep(1.5)
        assert cache.get("key1") is None
    
    def test_disk_cache_basic(self):
        from storage import DiskCache
        import tempfile
        import shutil
        
        cache_dir = tempfile.mkdtemp()
        
        try:
            cache = DiskCache(cache_dir=cache_dir)
            
            cache.set("key1", {"data": [1, 2, 3]})
            result = cache.get("key1")
            
            assert result == {"data": [1, 2, 3]}
        finally:
            shutil.rmtree(cache_dir)


class TestVectorStore:
    """向量存储测试"""
    
    @pytest.mark.slow
    def test_qdrant_basic(self):
        from storage import QdrantVectorStore
        
        # 使用内存模式
        store = QdrantVectorStore(
            collection_name="test_collection",
            persist_directory=None,  # 内存模式
        )
        
        # 添加文档
        docs = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with many layers.",
            "Natural language processing handles text and speech.",
        ]
        
        ids = store.add(docs)
        assert len(ids) == 3
        assert store.count() == 3
        
        # 搜索
        results = store.search("What is deep learning?", top_k=2)
        assert len(results) <= 2
        assert results[0].score > 0
        
        # 清空
        store.clear()
    
    @pytest.mark.slow
    def test_qdrant_with_metadata(self):
        from storage import QdrantVectorStore

        store = QdrantVectorStore(
        )
        
        docs = ["Doc about AI", "Doc about ML"]
        metadatas = [{"type": "ai"}, {"type": "ml"}]
        
        store.add(docs, metadatas)
        
        # 带过滤条件搜索
        results = store.search(
            "artificial intelligence",
            top_k=5,
            filter={"type": "ai"},
        )
        
        assert len(results) >= 1
        assert results[0].metadata.get("type") == "ai"
        
        store.clear()


class TestEmbedder:
    """Embedder 测试"""
    
    @pytest.mark.slow
    @pytest.mark.skipif(
        not HAS_SENTENCE_TRANSFORMERS,
        reason="需要 sentence_transformers 依赖",
    )
    def test_sentence_transformer_embedder(self):
        from processing import SentenceTransformerEmbedder
        
        embedder = SentenceTransformerEmbedder(
            model_name="all-MiniLM-L6-v2"
        )
        
        # 单个文本
        embedding = embedder.embed("Hello world")
        assert embedding.shape == (1, 384)
        
        # 多个文本
        embeddings = embedder.embed(["Hello", "World"])
        assert embeddings.shape == (2, 384)
    
    @pytest.mark.slow
    @pytest.mark.skipif(
        not HAS_SENTENCE_TRANSFORMERS,
        reason="需要 sentence_transformers 依赖",
    )
    def test_embedder_factory(self):
        from processing import get_embedder
        
        embedder = get_embedder(provider="sentence_transformers")
        assert embedder.dimension > 0
    
    def test_siliconflow_embedder_without_key(self):
        """测试 SiliconFlow Embedder (无 API Key 应抛出异常)"""
        import os
        from processing import SiliconFlowEmbedder
        from utils.exceptions import EmbeddingError
        
        # 保存原值
        original_key = os.environ.get("SILICONFLOW_API_KEY")
        
        try:
            # 清除环境变量
            if "SILICONFLOW_API_KEY" in os.environ:
                del os.environ["SILICONFLOW_API_KEY"]
            
            embedder = SiliconFlowEmbedder(api_key=None)
            
            # 调用应抛出异常
            try:
                embedder.embed("test")
                assert False, "Should have raised EmbeddingError"
            except EmbeddingError:
                pass  # 预期行为
        finally:
            # 恢复原值
            if original_key:
                os.environ["SILICONFLOW_API_KEY"] = original_key
    
    @pytest.mark.skipif(
        not os.environ.get("SILICONFLOW_API_KEY"),
        reason="需要 SILICONFLOW_API_KEY 环境变量"
    )
    def test_siliconflow_embedder_with_key(self):
        """测试 SiliconFlow BGE-M3 Embedder (需要 API Key)"""
        from processing import SiliconFlowEmbedder
        
        embedder = SiliconFlowEmbedder()
        
        # 单个文本
        embedding = embedder.embed("机器学习与深度学习")
        assert embedding.shape == (1, 1024)
        
        # 多个文本
        embeddings = embedder.embed(["Hello", "World", "你好"])
        assert embeddings.shape == (3, 1024)


# 集成测试
class TestIntegration:
    """集成测试"""
    
    @pytest.mark.slow
    def test_full_pipeline(self):
        """测试完整的处理流程"""
        from processing import DataCleaner, TextChunker, get_embedder
        from storage import QdrantVectorStore
        from models import Paper, Author, SourceType
        from datetime import datetime
        
        # 1. 创建测试数据
        paper = Paper(
            id="test_001",
            title="Attention Is All You Need",
            abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
            authors=[Author(name="Vaswani et al.")],
            published_date=datetime(2017, 6, 12),
            categories=["cs.CL", "cs.LG"],
            url="https://arxiv.org/abs/1706.03762",
            source=SourceType.ARXIV,
        )
        
        # 2. 清洗数据
        cleaner = DataCleaner()
        cleaned = cleaner.clean_paper(paper)
        
        assert cleaned["title"] == paper.title
        assert "content" in cleaned
        
        # 3. 分块
        chunker = TextChunker(chunk_size=200)
        chunks = chunker.chunk(
            cleaned["content"],
            metadata={"paper_id": paper.id, "type": "paper"},
            doc_id=paper.id,
        )
        
        assert len(chunks) >= 1
        
        # 4. 存储到向量数据库
        store = QdrantVectorStore(collection_name="test_integration")
        store.add_chunks(chunks)
        
        # 5. 搜索
        results = store.search("attention mechanism", top_k=2)
        
        assert len(results) > 0
        assert "attention" in results[0].content.lower()
        
        # 清理
        store.clear()
