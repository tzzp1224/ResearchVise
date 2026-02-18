# Academic Research Agent 📚🔬

> 一个"学术严谨"与"社区热度"兼顾的智能研究助手

## 🎯 核心定位

针对各个学术领域信息爆炸痛点，打造智能研究助手。它不仅能回答问题，还能主动整理知识脉络，将晦涩的论文转化为多模态的易读内容。

## ✨ 核心功能

### 1. 全网情报聚合
用户输入 Topic（如 "DeepSeek"），Agent 自动拉取：
- **ArXiv**: 最新学术论文
- **Semantic Scholar**: 论文引用关系 + 深度学术搜索
- **Hugging Face**: 模型/论文/数据集
- **Stack Overflow**: 技术问答 + 最佳实践
- **Hacker News**: 极客社区热议
- **Social Media**: 社区讨论/舆情（Twitter/X, Reddit, GitHub等）

### 2. 深度 RAG 问答
基于多源信息，支持用户自由提问

### 3. 多模态输出
- **Video Brief**: 模仿 NotebookLLM，生成技术内容的视频/音频解读
- **Timeline Mapper**: 自动梳理该技术的演进时间轴，知识树
- **One-Pager**: 生成"一页纸"备忘录（含核心结论、参数、资源链接）
- **质量守护**: 自动清洗无效资源链接，并补全 Video Brief 的 `duration_sec` / `visual_prompt`，确保可直接用于文生视频流程

## 🏗️ 项目架构

项目采用分层架构设计，共 5 个阶段：

| 阶段 | 层级 | 状态 | 描述 |
|------|------|------|------|
| Phase 1 | 数据层 | ✅ 完成 | 多源数据抓取 (ArXiv, HuggingFace, Social) |
| Phase 2 | 知识层 | ✅ 完成 | 数据处理 + 向量存储 |
| Phase 3 | 智能层 | ✅ 完成 | LLM + RAG + LangGraph Agent 编排 |
| Phase 4 | 输出层 | ✅ 完成 | Timeline / One-Pager / Video Brief + Slidev 成片视频（含旁白音轨） |
| Phase 5 | 接口层 | 🔲 待开发 | CLI, Web UI, API |

## 📁 项目结构

```
AcademicResearchAgent/
├── config/                      # 📦 配置管理模块
│   ├── __init__.py              #    模块导出
│   ├── settings.py              #    Pydantic 配置类定义
│   ├── .env                     #    环境变量 (API Keys等)
│   └── .env.example             #    环境变量示例模板
│
├── models/                      # 📦 数据模型模块
│   ├── __init__.py              #    模块导出
│   └── schemas.py               #    Pydantic 数据模型 (Paper, Model, SocialPost等)
│
├── scrapers/                    # 📦 数据抓取模块 (Phase 1)
│   ├── __init__.py              #    模块导出
│   ├── base.py                  #    BaseScraper 抽象基类 + 限速器
│   ├── arxiv_scraper.py         #    ArXiv 论文抓取器
│   ├── huggingface_scraper.py   #    HuggingFace 模型/数据集抓取器
│   ├── semantic_scholar_scraper.py  # Semantic Scholar 学术搜索
│   ├── stackoverflow_scraper.py     # Stack Overflow 技术问答
│   ├── hackernews_scraper.py        # Hacker News 极客社区
│   └── social/                  #    社交媒体子模块
│       ├── __init__.py          #        模块导出
│       ├── twitter_scraper.py   #        Twitter/X 抓取器
│       ├── reddit_scraper.py    #        Reddit 抓取器
│       └── github_scraper.py    #        GitHub 仓库抓取器
│
├── aggregator/                  # 📦 数据聚合模块 (Phase 1)
│   ├── __init__.py              #    模块导出
│   └── data_aggregator.py       #    多源并行聚合器
│
├── processing/                  # 📦 数据处理模块 (Phase 2)
│   ├── __init__.py              #    模块导出
│   ├── cleaner.py               #    文本清洗 (URL/HTML/特殊字符处理)
│   ├── chunker.py               #    文本分块 (固定/句子/段落/递归策略)
│   └── embedder.py              #    向量化 (SiliconFlow/Jina/OpenAI等)
│
├── storage/                     # 📦 存储模块 (Phase 2)
│   ├── __init__.py              #    模块导出
│   ├── vector_store.py          #    向量数据库 (Qdrant)
│   └── cache.py                 #    缓存 (内存缓存/磁盘缓存)
│
├── intelligence/                # 📦 智能模块 (Phase 3)
│   ├── __init__.py              #    模块导出
│   ├── llm/                     #    LLM 抽象层
│   │   ├── base.py              #        BaseLLM 抽象基类
│   │   ├── openai_llm.py        #        OpenAI (GPT-4o)
│   │   ├── anthropic_llm.py     #        Anthropic (Claude)
│   │   ├── deepseek_llm.py      #        DeepSeek (V3/R1)
│   │   ├── gemini_llm.py        #        Google Gemini
│   │   └── factory.py           #        get_llm() 工厂函数
│   ├── state/                   #    状态定义
│   │   └── agent_state.py       #        AgentState TypedDict
│   ├── agents/                  #    多 Agent 系统
│   │   ├── search_agent.py      #        搜索情报员 (ReAct)
│   │   ├── analyst_agent.py     #        学术分析师 (RAG)
│   │   └── content_agent.py     #        内容官 (并行生成)
│   ├── tools/                   #    Agent 工具
│   │   ├── search_tools.py      #        搜索工具封装
│   │   └── rag_tools.py         #        RAG 工具封装
│   └── graph/                   #    LangGraph 编排
│       └── research_graph.py    #        研究工作流图
│
├── outputs/                     # 📦 输出模块 (Phase 4)
│   ├── __init__.py              #    模块导出
│   ├── models.py                #    输出数据结构 (Timeline/OnePager/VideoBrief)
│   ├── renderers.py             #    渲染 (Markdown/Mermaid)
│   └── exporter.py              #    导出 (md/json/manifest)
│
├── utils/                       # 📦 工具模块
│   ├── __init__.py              #    模块导出
│   ├── logger.py                #    Rich 美化日志
│   └── exceptions.py            #    自定义异常类层次
│
├── tests/                       # 📦 测试模块
│   ├── __init__.py              #    模块导出
│   ├── test_scrapers.py         #    Phase 1 抓取器测试
│   └── test_processing_storage.py  # Phase 2 处理存储测试
│
├── main.py                      # 🚀 Phase 1 主入口 (数据抓取CLI)
├── demo.py                      # 🎮 Phase 1 演示脚本
├── demo_phase2.py               # 🎮 Phase 2 演示脚本
├── demo_phase3.py               # 🎮 Phase 3 演示脚本 (LLM + Agent)
├── demo_phase4.py               # 🎮 Phase 4 演示脚本 (Render + Export)
├── requirements.txt             # 📋 依赖清单
└── README.md                    # 📖 项目文档
```

## 📝 模块详解

### config/ - 配置管理

| 文件 | 类/函数 | 作用 |
|------|---------|------|
| `settings.py` | `Settings` | 主配置类，聚合所有子配置 |
| | `EmbeddingSettings` | Embedding 服务配置 (provider, model_name, api_keys) |
| | `StorageSettings` | 存储配置 (vector_db_path, cache_path) |
| | `LLMSettings` | LLM 配置 (provider, temperature, api_keys) |
| | `get_settings()` | 获取全局配置单例 |
| | `get_embedding_settings()` | 获取 Embedding 配置 |
| | `get_llm_settings()` | 获取 LLM 配置 |

### scrapers/ - 数据抓取

| 文件 | 类 | 作用 |
|------|-----|------|
| `base.py` | `BaseScraper` | 抽象基类，定义 `search()` 接口 |
| | `RateLimitedScraper` | 带限速的抽象基类 |
| `arxiv_scraper.py` | `ArxivScraper` | ArXiv 论文搜索，返回 `Paper` 对象 |
| `huggingface_scraper.py` | `HuggingFaceScraper` | HuggingFace 模型/数据集搜索 |
| `semantic_scholar_scraper.py` | `SemanticScholarScraper` | 学术论文 + 引用关系 (1 req/s 免费) |
| `stackoverflow_scraper.py` | `StackOverflowScraper` | 技术问答 (按投票排序) |
| `hackernews_scraper.py` | `HackerNewsScraper` | Algolia 搜索 + Firebase 热门 |
| `social/twitter_scraper.py` | `TwitterScraper` | Twitter/X 搜索 (需 Bearer Token) |
| `social/reddit_scraper.py` | `RedditScraper` | Reddit 帖子搜索 |
| `social/github_scraper.py` | `GitHubScraper` | GitHub 仓库搜索 |

### processing/ - 数据处理

| 文件 | 类/函数 | 作用 |
|------|---------|------|
| `cleaner.py` | `DataCleaner` | 文本清洗器 |
| | `clean_text()` | 清洗单段文本 (去URL/HTML/空白等) |
| | `clean_paper()` | 清洗论文数据 |
| `chunker.py` | `TextChunker` | 文本分块器 |
| | `ChunkingStrategy` | 分块策略枚举 (FIXED_SIZE/SENTENCE/PARAGRAPH/RECURSIVE) |
| | `DocumentChunk` | 分块结果数据类 |
| | `chunk_text()` / `chunk_document()` | 便捷函数 |
| `embedder.py` | `SiliconFlowEmbedder` | SiliconFlow BGE-M3 (免费，推荐) |
| | `JinaEmbedder` | Jina Embeddings API |
| | `OpenAIEmbedder` | OpenAI Embeddings API |
| | `SentenceTransformerEmbedder` | 本地 SentenceTransformers |
| | `get_embedder()` | 工厂函数，自动读取 .env 配置 |

### storage/ - 存储

| 文件 | 类 | 作用 |
|------|-----|------|
| `vector_store.py` | `QdrantVectorStore` | Qdrant 向量存储 (支持元数据过滤、持久化) |
| | `SearchResult` | 搜索结果数据类 |
| | `get_vector_store()` | 工厂函数，自动读取 .env 配置 |
| `cache.py` | `MemoryCache` | 内存缓存 (带 TTL) |
| | `DiskCache` | 磁盘缓存 (JSON 持久化) |

### intelligence/ - 智能层 (Phase 3)

| 文件 | 类/函数 | 作用 |
|------|---------|------|
| `llm/base.py` | `BaseLLM` | LLM 抽象基类 |
| | `Message` | 对话消息类 |
| | `LLMResponse` | LLM 响应类 |
| `llm/openai_llm.py` | `OpenAILLM` | OpenAI GPT-4o 实现 |
| `llm/anthropic_llm.py` | `AnthropicLLM` | Claude 3.5 Sonnet 实现 |
| `llm/deepseek_llm.py` | `DeepSeekLLM` | DeepSeek V3/R1 实现 |
| `llm/gemini_llm.py` | `GeminiLLM` | Gemini 2.0 实现 |
| `llm/factory.py` | `get_llm()` | 工厂函数，自动读取 .env 配置 |
| `agents/search_agent.py` | `SearchAgent` | 搜索情报员 (ReAct 驱动) |
| `agents/analyst_agent.py` | `AnalystAgent` | 学术分析师 (RAG + 事实验证) |
| `agents/content_agent.py` | `ContentAgent` | 内容官 (并行生成 Timeline/One-Pager/Video) |
| `graph/research_graph.py` | `ResearchGraph` | LangGraph 研究工作流 |
| | `create_research_graph()` | 创建工作流图 |
| | `run_research()` | 执行研究任务 |

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp config/.env.example config/.env
# 编辑 .env 文件，填入 API Keys

# 3. 运行 Phase 1 演示 (数据抓取)
python demo.py

# 4. 运行 Phase 2 演示 (处理+存储)
python demo_phase2.py

# 5. 运行 Phase 3 演示 (LLM + Agent)
python demo_phase3.py

# 6. 运行 Phase 4 真实端到端 (关键词 -> 深度输出)
python demo_phase4.py --topic "FlashAttention" -n 3 --sources "arxiv,huggingface,github"

# 7. 安装 ffmpeg（Slidev 视频编码依赖）
# macOS:
brew install ffmpeg

# 8. 安装免费高质量 TTS（推荐）
pip install edge-tts

# 9. 运行 Phase 4 + 视频产物（Slidev 讲解视频 + 自动旁白音轨）
python demo_phase4.py --topic "FlashAttention" -n 3 --sources "arxiv,github" --generate-video --tts-provider auto --tts-speed 1.2 --narration-model deepseek-chat

# 10. Manus 端到端测试样例（多源 + 深度输出 + 成片视频）
python demo_phase4.py --topic "manus" -n 5 --sources "arxiv,huggingface,github,semantic_scholar,stackoverflow,hackernews" --generate-video

# 11. 命令行搜索 (Phase 1)
python main.py --topic "DeepSeek" -n 20

# 12. 运行 Phase 5 Web UI（三栏布局 + SSE 文档流）
python demo_phase5.py --host 127.0.0.1 --port 8765
# 浏览器打开 http://127.0.0.1:8765
```

### Phase 5 UI 布局说明

- 左栏（文档输出）：`one_pager.md / timeline.md / report.md / video_brief.md` 通过 SSE 分块流式显示
- 中栏（Agent + Chat）：研究任务配置、执行流日志、基于 KB 的聊天问答
- 右栏（视频输出）：视频生成进度、视频播放器、产物快捷链接

## 🔧 配置说明

所有配置集中在 `config/.env`，修改后自动生效：

### Embedding 配置 (核心)

```env
# 切换 Embedding 供应商
EMBEDDING_PROVIDER=siliconflow   # 可选: siliconflow, jina, openai, sentence_transformers

# 切换模型 (可选，不填用默认)
EMBEDDING_MODEL_NAME=BAAI/bge-m3

# API Keys (按需填写)
SILICONFLOW_API_KEY=your_key    # 免费注册: https://siliconflow.cn/
JINA_API_KEY=your_key           # 免费额度: https://jina.ai/
OPENAI_API_KEY=your_key         # 付费
```

**支持的 Embedding 供应商：**

| Provider | 环境变量 | 默认模型 | 维度 | 费用 |
|----------|----------|----------|------|------|
| `siliconflow` | `SILICONFLOW_API_KEY` | BAAI/bge-m3 | 1024 | **免费** |
| `jina` | `JINA_API_KEY` | jina-embeddings-v3 | 1024 | 免费额度 |
| `openai` | `OPENAI_API_KEY` | text-embedding-3-small | 1536 | 付费 |
| `sentence_transformers` | 无需 | all-MiniLM-L6-v2 | 384 | 本地运行 |

### 数据源配置

```env
# ArXiv (无需 API Key)
ARXIV_MAX_RESULTS=50

# Hugging Face
HUGGINGFACE_TOKEN=hf_xxx

# Twitter/X (需申请开发者账号)
TWITTER_BEARER_TOKEN=xxx

# Reddit
REDDIT_CLIENT_ID=xxx
REDDIT_CLIENT_SECRET=xxx

# GitHub
GITHUB_TOKEN=xxx
```

### 存储配置 (向量数据库)

```env
# 向量数据库选择
STORAGE_VECTOR_DB_PROVIDER=qdrant
STORAGE_VECTOR_DB_PATH=./data/qdrant_db
STORAGE_CACHE_PATH=./data/cache
STORAGE_CACHE_TTL=3600   # 缓存过期时间(秒)

# Qdrant Cloud (可选，用于远程部署)
# STORAGE_QDRANT_URL=https://xxx.qdrant.io
# STORAGE_QDRANT_API_KEY=your_qdrant_api_key
```

**支持的向量数据库：**

| Provider | 特性 | 适用场景 |
|----------|------|----------|
| `qdrant` | 元数据过滤、持久化、并发安全、可内嵌运行 | 生产环境 |

**Qdrant 元数据过滤示例：**

```python
# 搜索 2023 年以后关于 LLM 的论文
results = store.search(
    "transformer architecture",
    filter={"year": {"$gte": 2023}, "topic": "LLM"}
)
```

### LLM 配置 (多服务商支持)

```env
# LLM 服务商选择: openai, anthropic, deepseek, gemini
LLM_PROVIDER=deepseek

# LLM 通用设置
LLM_MODEL_NAME=          # 可选，不填则使用默认模型
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=4096

# OpenAI (GPT-4o)
LLM_OPENAI_API_KEY=sk-xxx

# Anthropic (Claude 3.5 Sonnet)
LLM_ANTHROPIC_API_KEY=sk-ant-xxx

# DeepSeek (V3/R1 - 推荐，性价比高)
LLM_DEEPSEEK_API_KEY=sk-xxx

# Google Gemini (Gemini 2.0)
LLM_GEMINI_API_KEY=AIza-xxx
```

### 视频生成配置 (Phase 4)

说明:
- 当前仅保留 Slidev 单路线：将 `video_brief + one_pager + facts` 生成 Slidev 幻灯片，并自动生成 TTS 旁白音轨，最终使用 ffmpeg 合成 mp4。
- 默认不生成视频：只有显式传 `--generate-video` 才会进入视频流程。
- 即使视频生成失败，文档产物（report/one-pager/timeline/video-brief）仍会完整导出。
- 可通过 `--slides-target-duration-sec` 与 `--slides-fps` 控制讲解视频时长和帧率。
- 默认启用旁白：可用 `--disable-narration` 输出静音视频；可用 `--tts-provider` / `--tts-voice` / `--tts-speed` 控制音色与语速。
- 每页旁白先由小模型重写（默认 `deepseek-chat`，可用 `--narration-model` 调整），避免逐字念 PPT。
- 翻页时长由每页真实音轨时长决定，不再按固定秒数硬切页，避免长时间静默空窗。
- TTS provider 自动回退顺序：`edge-tts` -> `say`（macOS）-> `espeak`（Linux）。
- 首次运行会自动在 `data/.slidev_runtime` 安装 `@slidev/cli + @slidev/theme-default + playwright-chromium`。

推荐环境变量（可选）：

```env
# Edge-TTS (回退优先级 1)
# 推荐 voice: zh-CN-YunxiNeural / en-US-GuyNeural
# 可用 --tts-voice 覆盖

# 旁白脚本重写（可选）
LLM_DEEPSEEK_API_KEY=sk-xxx
VIDEO_NARRATION_DEEPSEEK_MODEL=deepseek-chat
```

**支持的 LLM 服务商：**

| Provider | 默认模型 | 特性 | 推荐场景 |
|----------|----------|------|----------|
| `deepseek` | deepseek-chat | 性价比高、中文友好 | 🌟 日常研究 |
| `openai` | gpt-4o | 能力强、生态完善 | 复杂分析 |
| `anthropic` | claude-3-5-sonnet | 长上下文、安全对齐 | 长文档处理 |
| `gemini` | gemini-2.0-flash | 多模态、速度快 | 快速原型 |

**使用示例：**

```python
from intelligence import get_llm, Message

# 自动读取 .env 配置
llm = get_llm()

# 或指定服务商
llm = get_llm(provider="anthropic")

# 调用 LLM
response = await llm.acomplete([
    Message(role="user", content="什么是 Transformer 架构?")
])
print(response.content)
```

## 💡 使用示例

### 示例 1: 搜索并聚合多源数据

```python
import asyncio
from aggregator import DataAggregator

async def main():
    aggregator = DataAggregator()
    result = await aggregator.aggregate("Transformer", max_results=20)
    
    print(f"论文: {len(result.papers)}")
    print(f"模型: {len(result.models)}")
    print(f"数据集: {len(result.datasets)}")
    print(f"社交帖子: {len(result.social_posts)}")

asyncio.run(main())
```

### 示例 2: 处理并向量化文档

```python
from processing import clean_text, chunk_document, get_embedder
from storage import QdrantVectorStore

# 1. 清洗
text = clean_text(raw_text, remove_urls=True)

# 2. 分块
chunks = chunk_document(text, doc_id="paper_001", chunk_size=200)

# 3. 向量化 (自动从 .env 读取配置)
embedder = get_embedder()
embeddings = embedder.embed([c.content for c in chunks])

# 4. 存储 (使用 Qdrant)
store = QdrantVectorStore(collection_name="papers", dimension=embedder.dimension)
store.add_with_embeddings(
    documents=[c.content for c in chunks],
    embeddings=embeddings.tolist(),
    metadatas=[c.metadata for c in chunks],
    ids=[c.id for c in chunks],
)

# 5. 搜索 (支持元数据过滤!)
results = store.search("attention mechanism", top_k=5)
for r in results:
    print(f"[{r.score:.3f}] {r.content[:100]}...")

# 6. 带元数据过滤的搜索 (Qdrant 特性)
results = store.search(
    "LLM architectures",
    top_k=5,
    filter={"year": {"$gte": 2023}}  # 只搜索2023年以后的论文
)
```

### 示例 3: 切换 Embedding 模型

只需修改 `config/.env`：

```env
# 方式1: 使用 SiliconFlow BGE-M3 (免费)
EMBEDDING_PROVIDER=siliconflow
SILICONFLOW_API_KEY=sk-xxx

# 方式2: 使用 Jina
EMBEDDING_PROVIDER=jina
JINA_API_KEY=jina_xxx

# 方式3: 使用本地模型 (无需API)
EMBEDDING_PROVIDER=sentence_transformers
```

代码无需修改，`get_embedder()` 会自动读取配置。

### 示例 4: 使用 LangGraph 研究流程

```python
import asyncio
from intelligence import ResearchGraph
from outputs import export_research_outputs, render_one_pager_markdown

async def main():
    # 创建研究图
    graph = ResearchGraph()
    
    # 运行完整研究流程
    result = await graph.run("Transformer 注意力机制的最新进展")
    
    # 渲染输出（Phase 4）
    print(render_one_pager_markdown(result.get("one_pager"), default_title="One-Pager"))

    # 导出到目录（生成 Markdown + JSON + manifest）
    export_research_outputs(
        "./data/outputs/transformer_demo",
        topic=result["topic"],
        timeline=result.get("timeline"),
        one_pager=result.get("one_pager"),
        video_brief=result.get("video_brief"),
    )

asyncio.run(main())
```

### 示例 5: 使用单独的 Agent

```python
import asyncio
from intelligence import SearchAgent, AnalystAgent, ContentAgent
from outputs import render_research_report_markdown

async def main():
    topic = "Mamba architecture"
    
    # 1. 搜索 Agent (ReAct 模式)
    search_agent = SearchAgent(max_iterations=3)
    search_results = await search_agent.search(topic)
    print(f"找到 {len(search_results)} 条结果")
    
    # 2. 分析 Agent (RAG 模式)
    analyst = AnalystAgent()
    analysis = await analyst.analyze(topic, search_results)
    facts = analysis.get("facts", [])
    print(f"提取 {len(facts)} 条事实")
    
    # 3. 内容 Agent (并行生成)
    content_agent = ContentAgent()
    outputs = await content_agent.generate(topic, facts)
    
    report_md = render_research_report_markdown(
        topic,
        timeline=outputs.get("timeline"),
        one_pager=outputs.get("one_pager"),
        video_brief=outputs.get("video_brief"),
    )
    print(report_md)

asyncio.run(main())
```

### 示例 6: 关键词驱动的端到端流程（推荐）

```python
import asyncio
from intelligence import run_research_end_to_end

async def main():
    result = await run_research_end_to_end(
        topic="FlashAttention",
        max_results_per_source=3,
        show_progress=False,
        generate_video=True,
        enable_knowledge_indexing=False,
        aggregator_kwargs={
            "enable_arxiv": True,
            "enable_huggingface": True,
            "enable_github": True,
            "enable_twitter": False,
            "enable_reddit": False,
            "enable_semantic_scholar": False,
            "enable_stackoverflow": False,
            "enable_hackernews": False,
        },
    )

    print(result["output_dir"])
    print(result["depth_assessment"])
    print(result["video_artifact"])

asyncio.run(main())
```

## 🧪 测试

```bash
# 运行 Phase 1 测试 (数据抓取)
python -m pytest tests/test_scrapers.py -v

# 运行 Phase 2 测试 (处理+存储)
python tests/test_processing_storage.py

# 运行 Phase 2 演示
python demo_phase2.py

# 运行 Phase 4 输出与视频测试
python -m pytest tests/test_outputs.py tests/test_video_generator.py tests/test_end_to_end_pipeline.py tests/test_github_scraper_unit.py -q

# 仅运行 Manus 端到端用例
python -m pytest tests/test_end_to_end_pipeline.py -k manus -q
```

## 🗺️ 开发路线图

- [x] **Phase 1**: 数据抓取层 (ArXiv, HuggingFace, Twitter, Reddit, GitHub)
- [x] **Phase 2**: 处理存储层 (Cleaner, Chunker, Embedder, VectorStore, Cache)
- [x] **Phase 3**: 智能层 (LLM 抽象, 多 Agent 协作, LangGraph 编排)
- [x] **Phase 4**: 输出层 (Timeline, One-Pager, Video Brief, 带旁白音轨视频)
- [ ] **Phase 5**: 接口层 (CLI 增强, Web UI, REST API)

## 📝 License

MIT License

MIT License
