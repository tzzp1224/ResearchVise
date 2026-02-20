# AcademicResearchAgent 当前工作流总览（v2）

本文档目标：让你在最短时间内看懂这个项目“从请求到产物”的完整端到端链路，以及每一步是怎么做的、在哪个文件里做的。

## 1. 项目在做什么（先看结论）

这个项目是一个 **研究内容生产流水线**：

1. 拉取多源数据（GitHub/HuggingFace/HackerNews/RSS/Web）。
2. 清洗、去重、聚类、打分、选 Top picks。
3. 生成结构化事实 `facts`、口播脚本 `script`、分镜 `storyboard`、渲染提示 `prompt_bundle`。
4. 产出图文摘要 `onepager`、音频 `audio`、字幕 `srt`、视频 `mp4`、打包 `zip`。
5. 用校验器做质量门禁，判断本次 run 是否可交付。

核心路径都在：

- `main.py`
- `pipeline_v2/runtime.py`
- `pipeline_v2/{normalize.py, scoring.py, script_generator.py, storyboard_generator.py, prompt_compiler.py, report_export.py, sanitize.py}`
- `render/manager.py`
- `scripts/validate_artifacts_v2.py`

## 2. 系统组件和职责

### 2.1 入口层（CLI / API）

- CLI：`main.py`
  - 关键命令：`run-once`、`ondemand`、`worker-run-next`、`worker-render-next`、`daily-subscribe`、`daily-tick`、`status`、`cancel`
- API：`webapp/v2_app.py`
  - 关键接口：`/api/v2/runs/ondemand`、`/api/v2/workers/runs/next`、`/api/v2/workers/render/next`

### 2.2 编排层（队列 + 状态）

- `orchestrator/service.py`: run 生命周期编排（排队、取消、事件、订阅触发）
- `orchestrator/queue.py`: 内存 FIFO 队列（去重 enqueue）
- `orchestrator/store.py`: 内存状态存储（请求、状态、artifact、事件、idempotency）

注意：当前是 **InMemory** 实现，重启进程后状态不持久化。

### 2.3 执行层（Run Worker / Render Worker）

- Run Worker：`RunPipelineRuntime.run_next()` -> `_execute_run()`（`pipeline_v2/runtime.py`）
- Render Worker：`RunPipelineRuntime.process_next_render()` -> `RenderManager.process_next()`（`render/manager.py`）

## 3. 端到端流程（E2E）

下面是一次完整 run 的真实执行顺序。

### Step 1: 接收请求并入队

入口可来自 CLI 或 API，最终都构造 `RunRequest` 并调用 `orchestrator.enqueue_run(...)`。

关键字段：

- `mode`: `ondemand` / `daily`
- `topic`, `time_window`, `tz`
- `budget`: `top_k`, `data_mode`, `duration_sec`, `max_retries`, `style_profile`, etc.
- `output_targets`: 例如 `web,mp4`

### Step 2: Run Worker 取任务并初始化上下文

`runtime.run_next()` 从队列取任务后进入 `_execute_run()`：

1. 创建本次输出目录：`data/outputs/v2_runs/<run_id>/`
2. 解析数据模式：`live` / `smoke`（`pipeline_v2/data_mode.py`）
3. 记录 `run_started` 事件

### Step 3: 数据采集（connectors）

`_collect_raw_items()`（`pipeline_v2/runtime.py`）调用 `sources/connectors.py`：

- 主题搜索（live + 有 topic 时）：
  - `fetch_github_topic_search`
  - `fetch_huggingface_search`
  - `fetch_hackernews_search`
- 基础趋势源（Tier A）：
  - `fetch_github_trending`
  - `fetch_huggingface_trending`
  - `fetch_hackernews_top`
- 增强源：
  - `fetch_github_releases`（从 repo 衍生 release）
  - 可选 `fetch_rss_feed`、`fetch_web_article`（受 budget 控制）

采集会记录 connector 调用耗时、状态、条数，最终写入 `run_context.json` 的 `connector_stats` 和 `extraction_stats`。

### Step 4: 规范化与清洗

`normalize(raw_item)`（`pipeline_v2/normalize.py`）对每条数据做：

1. 文本清洗：`sanitize_markdown()`（去 HTML、badge、噪声链接）
2. URL 标准化：`canonicalize_url()`
3. citation 抽取 + evidence-only 过滤
4. 质量信号计算：
   - `content_density`
   - `has_quickstart`
   - `has_results_or_bench`
   - `update_recency_days`
   - `evidence_links_quality`
5. 生成 `NormalizedItem`

### Step 5: 去重、聚类、合并

`pipeline_v2/dedup_cluster.py`：

1. `dedup_exact()`：按 hash 精确去重
2. `embed()`：本地哈希向量（确定性 embedding）
3. `cluster()`：余弦相似聚类
4. `merge_cluster()`：生成 `CanonicalItem`（合并 citation/source）

### Step 6: 排序与门禁（最关键）

`rank_items()`（`pipeline_v2/scoring.py`）执行多分打分：

- `novelty`
- `talkability`
- `credibility`
- `visual_assets`
- `relevance`（topic 相关度）

并执行硬门禁/惩罚：

- relevance 阈值门禁（live 默认从 0.55 起）
- body/citation/link 质量门禁
- 无证据、过旧内容惩罚

`runtime.py` 额外做两件事：

1. **自适应阈值放宽**：不够 `top_k` 时按 `0.55 -> ... -> 0.35` 逐步放宽。
2. **来源多样性选择**：`_select_diverse_top()` 先保证 source 多样性，再补齐。

最终写入 `run_context.json/ranking_stats`（包括阈值、放宽步数、候选不足、top_relevance_scores、drop_reason_samples）。

### Step 7: 生成 facts -> script -> storyboard

1. `build_facts()`（`pipeline_v2/script_generator.py`）
   - 产出：`hook/what_it_is/why_now/how_it_works/proof/proof_links/metrics`
   - 输出文件：`facts.json`
2. `generate_script(...)`
   - 基于 facts 生成时间轴脚本行（含 references）
   - 输出文件：`script.json`
3. `script_to_storyboard(...)`（`pipeline_v2/storyboard_generator.py`）
   - 生成 5-8 个镜头，overlay 文本压缩，校验 + auto-fix
   - 输出文件：`storyboard.json`

### Step 8: 素材清单与本地资产物化

`runtime.py`：

1. `_build_materials_manifest()` 生成 `materials.json`（source_urls、screenshot_plan、citations、quality_metrics）
2. `_materialize_assets()` 将截图计划转为本地 `assets/*`（下载失败则生成本地 SVG 卡片）
3. `_attach_reference_assets()` 将本地 asset 和 evidence URL 绑定回镜头

### Step 9: prompt 编译与内容产物导出

1. `compile_storyboard()`（`pipeline_v2/prompt_compiler.py`）
   - 输出：`prompt_bundle.json`
   - 策略：`references` 只保留 evidence URL，`render_assets` 放本地素材
2. `generate_onepager()`（`pipeline_v2/report_export.py`）
   - 输出：`onepager.md`
   - 包含 Top Picks、Compact Brief、Evidence 区域
3. `generate_thumbnail()`
   - 输出：`thumbnail_*.svg`
4. 音频字幕：
   - `tts_generate()` -> `tts_narration.wav`
   - `align_subtitles()` -> `captions.srt`
   - `mix_bgm()` -> 混音输出

### Step 10: 渲染与视频产出

如果目标包含 `mp4`，会 `enqueue_render()` 并由 Render Worker 处理：

1. 逐镜头调用渲染 adapter（默认 `SeedanceAdapter`）
2. 失败镜头重试（受 `max_retries` 限制）
3. 全失败或拼接失败时 fallback render（仍产出 mp4）
4. `stitch_shots()` 产出最终视频：
   - 首选：`rendered_final.mp4`
   - 兜底：`fallback_render.mp4`

### Step 11: 打包、通知、状态完成

`runtime.py`：

1. `export_package()` 打 zip
2. 记录 artifact（含 mp4、zip、script、facts、storyboard、onepager、audio、srt）
3. 触发通知钩子：`notify_user` / `post_to_web` / `send_email`
4. run 状态标记 completed

## 4. 目录与产物地图

### 4.1 Run 目录（核心真相源）

`data/outputs/v2_runs/<run_id>/` 典型文件：

- `run_context.json`：数据模式、connector/extraction 统计、ranking_stats
- `facts.json`
- `script.json`
- `storyboard.json`
- `prompt_bundle.json`
- `materials.json`
- `onepager.md`
- `thumbnail_*.svg`
- `tts_narration.wav` / `tts_narration_with_bgm.wav`
- `captions.srt`
- `*_package.zip`
- `notifications.jsonl`
- `assets/*`（本地素材）

### 4.2 Render 目录

`data/outputs/render_jobs/<render_job_id>/` 典型文件：

- `render_status.json`
- `shot_*.mp4`（逐镜头）
- `rendered_final.mp4` 或 `fallback_render.mp4`

## 5. 质量门禁（validator）

`scripts/validate_artifacts_v2.py` 会检查：

1. 必要文件存在性（script/facts/storyboard/prompt/materials/onepager/mp4）
2. 脚本结构和长度、无 blocklist、无 HTML 噪声
3. onepager 一致性（TopPicksCount / RequestedTopK / relevance / compact bullets）
4. evidence 去重质量
5. facts 是否包含 why_now + proof
6. storyboard 镜头数与 overlay 安全（<=42 字符，避免硬截断）
7. mp4 可探测、时长 >= 10s、帧变化正常
8. live 模式特殊门禁（不得混入 smoke token/fixture id/placeholder repo）
9. URL 合法性检查（`urls_valid`）

`run-once` 命令会在同进程跑完 run + render 后自动调用这个 validator，并返回 `validator_ok`。

## 6. 最快上手路径（建议顺序）

### 6.1 先跑一遍完整链路

```bash
python main.py run-once --mode live --topic "AI agent" --time_window today --tz Asia/Singapore --targets web,mp4 --top-k 3
```

如果只想稳定本地自测，用 smoke：

```bash
python scripts/e2e_smoke_v2.py --out-dir /tmp/ara_v2_smoke
```

### 6.2 再看本次 run 的关键文件

按这个顺序看最省时间：

1. `run_context.json`
2. `onepager.md`
3. `facts.json`
4. `script.json`
5. `storyboard.json`
6. `materials.json`

### 6.3 再回头读代码（推荐阅读顺序）

1. `main.py`
2. `pipeline_v2/runtime.py`
3. `pipeline_v2/normalize.py`
4. `pipeline_v2/scoring.py`
5. `pipeline_v2/script_generator.py`
6. `pipeline_v2/storyboard_generator.py`
7. `pipeline_v2/prompt_compiler.py`
8. `render/manager.py`
9. `scripts/validate_artifacts_v2.py`

## 7. 你现在可以怎么用这套流程

1. 当作“主题研究自动化生产线”：给 topic，直接拿 onepager + mp4。
2. 当作“可解释排序引擎”：重点看 `run_context.json/ranking_stats`。
3. 当作“内容生成中台”：把 `facts/script/storyboard/prompt` 作为可插拔中间层接入别的渲染器或发布系统。

