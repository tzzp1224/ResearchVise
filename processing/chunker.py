"""
Text Chunker
文本分块模块
"""
from enum import Enum
from typing import List, Optional
from dataclasses import dataclass, field
import re
import hashlib


class ChunkingStrategy(str, Enum):
    """分块策略"""
    FIXED_SIZE = "fixed_size"           # 固定大小
    SENTENCE = "sentence"               # 按句子
    PARAGRAPH = "paragraph"             # 按段落
    SEMANTIC = "semantic"               # 语义分块 (需要模型)
    RECURSIVE = "recursive"             # 递归分块


@dataclass
class DocumentChunk:
    """文档块"""
    id: str                             # 块ID
    content: str                        # 块内容
    metadata: dict = field(default_factory=dict)  # 元数据
    index: int = 0                      # 在原文档中的索引
    start_char: int = 0                 # 起始字符位置
    end_char: int = 0                   # 结束字符位置
    
    @property
    def token_count(self) -> int:
        """估算 token 数量 (粗略估计)"""
        # 英文约 4 字符 = 1 token，中文约 1.5 字符 = 1 token
        # 这里使用简单估计
        return len(self.content) // 3
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "index": self.index,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


class TextChunker:
    """
    文本分块器
    支持多种分块策略
    """
    
    # 句子分隔符
    SENTENCE_ENDINGS = re.compile(r'(?<=[.!?。！？])\s+')
    # 段落分隔符
    PARAGRAPH_SEP = re.compile(r'\n\s*\n')
    
    def __init__(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
        separators: Optional[List[str]] = None,
    ):
        """
        初始化分块器
        
        Args:
            strategy: 分块策略
            chunk_size: 目标块大小 (字符数)
            chunk_overlap: 块之间的重叠 (字符数)
            min_chunk_size: 最小块大小
            separators: 自定义分隔符列表
        """
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.separators = separators or ["\n\n", "\n", ". ", "。", " ", ""]
    
    def chunk(
        self,
        text: str,
        metadata: Optional[dict] = None,
        doc_id: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """
        将文本分块
        
        Args:
            text: 要分块的文本
            metadata: 附加到每个块的元数据
            doc_id: 文档ID (用于生成块ID)
            
        Returns:
            文档块列表
        """
        if not text or not text.strip():
            return []
        
        metadata = metadata or {}
        doc_id = doc_id or self._generate_doc_id(text)
        
        # 根据策略选择分块方法
        if self.strategy == ChunkingStrategy.FIXED_SIZE:
            chunks = self._chunk_fixed_size(text)
        elif self.strategy == ChunkingStrategy.SENTENCE:
            chunks = self._chunk_by_sentence(text)
        elif self.strategy == ChunkingStrategy.PARAGRAPH:
            chunks = self._chunk_by_paragraph(text)
        elif self.strategy == ChunkingStrategy.RECURSIVE:
            chunks = self._chunk_recursive(text, self.separators)
        else:
            chunks = self._chunk_recursive(text, self.separators)
        
        # 转换为 DocumentChunk 对象
        result = []
        for i, (content, start, end) in enumerate(chunks):
            chunk_id = f"{doc_id}_{i}"
            result.append(DocumentChunk(
                id=chunk_id,
                content=content,
                metadata={**metadata, "doc_id": doc_id},
                index=i,
                start_char=start,
                end_char=end,
            ))
        
        return result
    
    def _chunk_fixed_size(self, text: str) -> List[tuple]:
        """固定大小分块"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # 确保不会在单词中间切断
            if end < len(text):
                # 向后找到最近的空格
                space_pos = text.rfind(' ', start, end)
                if space_pos > start + self.min_chunk_size:
                    end = space_pos
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append((chunk_text, start, end))
            
            start = end - self.chunk_overlap
            if start < 0:
                start = 0
        
        return chunks
    
    def _chunk_by_sentence(self, text: str) -> List[tuple]:
        """按句子分块"""
        sentences = self.SENTENCE_ENDINGS.split(text)
        
        chunks = []
        current_chunk = ""
        current_start = 0
        char_pos = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                    current_start = char_pos
            else:
                if current_chunk:
                    chunks.append((current_chunk, current_start, char_pos))
                current_chunk = sentence
                current_start = char_pos
            
            char_pos += len(sentence) + 1
        
        if current_chunk:
            chunks.append((current_chunk, current_start, len(text)))
        
        return chunks
    
    def _chunk_by_paragraph(self, text: str) -> List[tuple]:
        """按段落分块"""
        paragraphs = self.PARAGRAPH_SEP.split(text)
        
        chunks = []
        current_chunk = ""
        current_start = 0
        char_pos = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(current_chunk) + len(para) <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                    current_start = char_pos
            else:
                if current_chunk:
                    chunks.append((current_chunk, current_start, char_pos))
                
                # 如果段落本身超过 chunk_size，需要进一步分割
                if len(para) > self.chunk_size:
                    sub_chunks = self._chunk_fixed_size(para)
                    for sub_content, sub_start, sub_end in sub_chunks:
                        chunks.append((sub_content, char_pos + sub_start, char_pos + sub_end))
                    current_chunk = ""
                    current_start = char_pos + len(para)
                else:
                    current_chunk = para
                    current_start = char_pos
            
            char_pos += len(para) + 2  # +2 for \n\n
        
        if current_chunk:
            chunks.append((current_chunk, current_start, len(text)))
        
        return chunks
    
    def _chunk_recursive(
        self,
        text: str,
        separators: List[str],
        start_offset: int = 0,
    ) -> List[tuple]:
        """
        递归分块 (类似 LangChain 的 RecursiveCharacterTextSplitter)
        """
        if len(text) <= self.chunk_size:
            return [(text.strip(), start_offset, start_offset + len(text))] if text.strip() else []
        
        # 找到第一个可用的分隔符
        separator = ""
        for sep in separators:
            if sep in text:
                separator = sep
                break
        
        if not separator:
            # 没有分隔符，强制分割
            return self._chunk_fixed_size(text)
        
        # 按分隔符分割
        splits = text.split(separator)
        
        chunks = []
        current_chunk = ""
        current_start = start_offset
        char_pos = 0
        
        for split in splits:
            split_with_sep = split + separator if separator else split
            
            if len(current_chunk) + len(split_with_sep) <= self.chunk_size:
                current_chunk += split_with_sep
            else:
                if current_chunk:
                    chunk_text = current_chunk.strip()
                    if chunk_text:
                        chunks.append((chunk_text, current_start, start_offset + char_pos))
                
                # 如果单个分割超过 chunk_size，递归处理
                if len(split_with_sep) > self.chunk_size:
                    next_separators = separators[separators.index(separator) + 1:] if separator in separators else separators[1:]
                    sub_chunks = self._chunk_recursive(
                        split_with_sep,
                        next_separators,
                        start_offset + char_pos,
                    )
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                    current_start = start_offset + char_pos + len(split_with_sep)
                else:
                    current_chunk = split_with_sep
                    current_start = start_offset + char_pos
            
            char_pos += len(split_with_sep)
        
        if current_chunk.strip():
            chunks.append((current_chunk.strip(), current_start, start_offset + len(text)))
        
        return chunks
    
    def _generate_doc_id(self, text: str) -> str:
        """生成文档ID"""
        return hashlib.md5(text[:1000].encode()).hexdigest()[:12]


# 便捷函数
_default_chunker = TextChunker()


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    metadata: Optional[dict] = None,
) -> List[DocumentChunk]:
    """
    便捷函数：分块文本
    
    Args:
        text: 要分块的文本
        chunk_size: 块大小
        chunk_overlap: 重叠大小
        metadata: 元数据
        
    Returns:
        文档块列表
    """
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk(text, metadata)


def chunk_document(
    content: str,
    doc_id: str,
    doc_type: str,
    metadata: Optional[dict] = None,
    chunk_size: int = 512,
) -> List[DocumentChunk]:
    """
    便捷函数：分块文档
    
    Args:
        content: 文档内容
        doc_id: 文档ID
        doc_type: 文档类型 (paper, model, post, repo)
        metadata: 额外元数据
        chunk_size: 块大小
        
    Returns:
        文档块列表
    """
    base_metadata = {
        "doc_id": doc_id,
        "doc_type": doc_type,
    }
    if metadata:
        base_metadata.update(metadata)
    
    chunker = TextChunker(chunk_size=chunk_size)
    return chunker.chunk(content, base_metadata, doc_id)
