# AcademicResearchAgent v2 状态说明（实装审计版）

## Changelog (Last Updated: 2026-02-18)
### Commit: Acceptance Gates & Placeholder Cleanup (Commit 6)
- 本次目标：
  - 完成自动化验收门禁（artifact 质量、视频有效性、占位逻辑清理）。
  - 保证 smoke 数据可覆盖 `Top picks >= 3` 验收项。
- 实际改动：
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/scripts/validate_artifacts_v2.py`：
    - 新增门禁：`script_structure_ok`、`onepager_top_picks_ge_3`、`onepager_domain_rows_ge_3`。
    - 新增 MP4 门禁：`duration>=10s`、三帧差异校验（避免静态/彩条类输出）。
    - 新增 `render_status_seedance_flag_present` 校验（读取 `render_status.json`）。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/render/manager.py`：
    - 渲染目录持久化 `render_status.json`。
    - 在无 `drawtext` 环境下启用位图文本卡片渲染路径，避免退化为静态占位合成。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/scripts/e2e_smoke_v2.py`：
    - smoke connectors 扩展到 >=3 条候选，确保 onepager 门禁可验证。
  - 新增 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/scripts/check_no_placeholders_v2.py`：
    - 扫描旧占位标记（legacy filler/testsrc/placeholder mp4 marker）并以 0/1 退出码返回。
  - 新增 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/tests/v2/test_validate_artifacts_v2.py`：
    - 覆盖 smoke + validator 一体化验收。
- 新增/删除文件：
  - 新增：`scripts/check_no_placeholders_v2.py`, `tests/v2/test_validate_artifacts_v2.py`
  - 修改：`scripts/validate_artifacts_v2.py`, `scripts/e2e_smoke_v2.py`, `render/manager.py`, `README.md`
- 如何验证：
  - `pytest -q tests/v2`
  - `python scripts/e2e_smoke_v2.py --out-dir /tmp/ara_v2_accept > /tmp/ara_v2_accept/result.json`
  - `python scripts/validate_artifacts_v2.py --run-dir /tmp/ara_v2_accept/runs/<run_id> --render-dir /tmp/ara_v2_accept/render_jobs/<render_job_id>`
  - `python scripts/check_no_placeholders_v2.py --root /Users/dexter/Documents/Dexter_Work/AcademicResearchAgent`
- 已知风险与回滚：
  - 风险：位图文本卡片是无 `drawtext` 环境的兼容路径，视觉可读性优先于高级动效。
  - 回滚：`git revert <this_commit_sha>`。

### Commit: Seedance Optional Real HTTP Adapter (Commit 5)
- 本次目标：
  - 将 Seedance 从“仅注入回调”升级为“可选真实 HTTP 接入（默认关闭）”。
  - 覆盖成功/鉴权失败/超时路径，并确保失败自动回退 fallback render。
- 实际改动：
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/render/adapters/seedance.py`：
    - 新增 env 配置读取：`SEEDANCE_ENABLED/SEEDANCE_BASE_URL/SEEDANCE_API_KEY/SEEDANCE_REGION/SEEDANCE_TIMEOUT_S`。
    - 内置最小 HTTP client（`httpx`），支持 base64 输出落盘和 URL 下载落盘。
    - 规范错误分类：鉴权失败、配额超限、超时、请求失败。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/core/contracts.py` 与 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/render/manager.py`：
    - `RenderStatus` 增加 `seedance_used` 字段。
    - 渲染流程记录 Seedance 是否真正产出镜头。
  - 新增 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/tests/v2/test_seedance_adapter.py`：
    - 覆盖 disabled/mock-success/http-success/auth-failure/timeout。
- 新增/删除文件：
  - 新增：`tests/v2/test_seedance_adapter.py`
  - 修改：`render/adapters/seedance.py`, `render/manager.py`, `core/contracts.py`
- 如何验证：
  - `pytest -q tests/v2/test_seedance_adapter.py tests/v2/test_render_manager.py tests/v2/test_contracts.py tests/v2/test_runtime_integration.py`
  - `SEEDANCE_ENABLED=0 python scripts/e2e_smoke_v2.py --out-dir /tmp/ara_v2_smoke_c5 > /tmp/ara_v2_smoke_c5/result.json`
- 已知风险与回滚：
  - 风险：真实 API 的请求/响应字段在不同版本可能变化，必要时需适配具体 provider schema。
  - 回滚：`git revert <this_commit_sha>`。

### Commit: Minimum-Usable Audio & Subtitles (Commit 4)
- 本次目标：
  - 将 `audio/subtitles` 从占位实现升级为最低可用的可交付链路。
- 实际改动：
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/render/audio_subtitles.py`：
    - `tts_generate` 改为本地确定性可听音轨生成（非静默占位）。
    - `align_subtitles` 新增单调时间轴修正，确保 SRT 时间递增。
    - `mix_bgm` 改为“仅在提供且可读取 BGM 时混音”，否则保持主音轨并返回原路径。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/tests/v2/test_audio_subtitles.py`：
    - 新增时间轴异常场景与 BGM 缺失场景测试。
- 新增/删除文件：
  - 无新增文件。
  - 修改：`render/audio_subtitles.py`, `tests/v2/test_audio_subtitles.py`。
- 如何验证：
  - `pytest -q tests/v2/test_audio_subtitles.py tests/v2/test_runtime_integration.py tests/v2/test_e2e_smoke_command.py`
  - `python scripts/e2e_smoke_v2.py --out-dir /tmp/ara_v2_smoke_c4 > /tmp/ara_v2_smoke_c4/result.json`
  - `python scripts/validate_artifacts_v2.py --run-dir /tmp/ara_v2_smoke_c4/runs/<run_id> --render-dir /tmp/ara_v2_smoke_c4/render_jobs/<render_job_id>`
- 已知风险与回滚：
  - 风险：当前本地音轨仍是规则合成音，音色自然度有限；外部 TTS 接入作为后续增强。
  - 回滚：`git revert <this_commit_sha>`。

### Commit: Motion Graphics Fallback Renderer (Commit 3)
- 本次目标：
  - 将 `fallback_render/stitch_shots` 从占位合成升级为可发布预览的工程化视频输出。
  - 禁用默认 `testsrc/colorbars` 路径。
- 实际改动：
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/render/manager.py`：
    - 新增 storyboard-driven motion graphics 渲染（开场卡 + shot 卡 + 结尾 CTA 卡）。
    - `stitch_shots` 在拼接失败时优先回退到 `fallback_render(board)`，不再走占位字节拼接。
    - `fallback_render` 优先使用文本动效渲染，不再默认 `testsrc`。
    - 保留原子写与 `ffprobe` 校验，渲染状态继续落盘 `valid_mp4/probe_error`。
- 新增/删除文件：
  - 无新增文件。
  - 修改：`render/manager.py`。
- 如何验证：
  - `pytest -q tests/v2/test_render_manager.py tests/v2/test_runtime_integration.py tests/v2/test_e2e_smoke_command.py`
  - `python scripts/e2e_smoke_v2.py --out-dir /tmp/ara_v2_smoke_c3 > /tmp/ara_v2_smoke_c3/result.json`
  - `python scripts/validate_artifacts_v2.py --run-dir /tmp/ara_v2_smoke_c3/runs/<run_id> --render-dir /tmp/ara_v2_smoke_c3/render_jobs/<render_job_id>`
- 已知风险与回滚：
  - 风险：当前动效渲染仍以文字卡片为主，素材截图/图层模板将在后续 commit 增强。
  - 回滚：`git revert <this_commit_sha>`。

### Commit: Content Realization & Quality Metrics (Commit 2)
- 本次目标：
  - 将 `script/onepager/materials` 从占位描述升级为可审阅内容。
  - 落盘抓取质量指标，并写入排序解释 reasons。
- 实际改动：
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/script_generator.py`：
    - 脚本结构固定为 `hook(前3秒)/main_thesis/3个要点/cta`，并为每段输出时间轴与 section。
    - 移除旧的 `evidence-backed detail` 占位文案生成路径。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/normalize.py`：
    - 新增并落盘质量指标：`body_len/citation_count/published_recency/link_count`。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/scoring.py`：
    - 在 `RankedItem.reasons` 中新增质量指标解释字段。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/report_export.py`：
    - `onepager.md` 每条 Top pick 增加来源域名、两段摘要、citation 或 `无引用`、质量指标、排序原因。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/runtime.py`：
    - `materials.json` 增加 `screenshot_plan/icon_keyword_suggestions/broll_categories/quality_metrics`。
- 新增/删除文件：
  - 无新增文件。
  - 修改：`pipeline_v2/normalize.py`, `pipeline_v2/scoring.py`, `pipeline_v2/script_generator.py`, `pipeline_v2/report_export.py`, `pipeline_v2/runtime.py`, 对应测试文件。
- 如何验证：
  - `pytest -q tests/v2/test_normalize.py tests/v2/test_scoring.py tests/v2/test_script_storyboard_prompt.py tests/v2/test_report_export_notification.py tests/v2/test_runtime_integration.py`
  - `python scripts/e2e_smoke_v2.py --out-dir /tmp/ara_v2_smoke_c2 > /tmp/ara_v2_smoke_c2/result.json`
  - `python scripts/validate_artifacts_v2.py --run-dir /tmp/ara_v2_smoke_c2/runs/<run_id> --render-dir /tmp/ara_v2_smoke_c2/render_jobs/<render_job_id>`
- 已知风险与回滚：
  - 风险：当真实抓取正文极短时，脚本会退化到规则化句式，后续需结合 LLM/模板库进一步提升文案自然度。
  - 回滚：`git revert <this_commit_sha>`。

### Commit: Audit & De-placeholder Entry Cleanup (Commit 1)
- 本次目标：
  - 记录当前可复现实测证据（smoke 与点播链路）。
  - 移除一个直接制造占位镜头的输出入口。
  - 增加 `validate_artifacts_v2.py` 验收脚本骨架。
- 实际改动：
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/scripts/e2e_smoke_v2.py`：
    - 删除 `SmokeAdapter` 文本镜头产出逻辑，改为默认渲染路径（Seedance 不可用则走 fallback）。
  - 新增 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/scripts/validate_artifacts_v2.py`：
    - 校验 `script/onepager/storyboard/materials/mp4` 是否存在；
    - blocklist（placeholder/dummy/lorem/todo/testsrc/colorbars）扫描；
    - `ffprobe`/`ftyp` 基础视频校验；
    - 输出 JSON 报告并返回 0/1 退出码。
- 新增/删除文件：
  - 新增：`scripts/validate_artifacts_v2.py`
  - 修改：`scripts/e2e_smoke_v2.py`, `README.md`
- 如何验证：
  - `python scripts/e2e_smoke_v2.py --out-dir /tmp/ara_v2_smoke > /tmp/ara_v2_smoke/result.json`
  - `python scripts/validate_artifacts_v2.py --run-dir /tmp/ara_v2_smoke/runs/<run_id> --render-dir /tmp/ara_v2_smoke/render_jobs/<render_job_id>`
  - `pytest -q tests/v2/test_e2e_smoke_command.py`
- 已知风险与回滚：
  - 风险：当前 `validate_artifacts_v2.py` 仍是基础门禁，帧差与更严格质量规则后续 commit 增强。
  - 回滚：`git revert <this_commit_sha>` 可恢复 smoke adapter 与当前门禁脚本变更。

本文档说明当前仓库的真实实现状态、可运行命令、产物路径、Seedance 接入现状、限制与下一步。

## 1) 当前真实入口与数据流

### 1.1 触发入口
- CLI 入口：`main.py`
  - 点播：`ondemand`
  - 订阅：`daily-subscribe`
  - 定时触发：`daily-tick`
  - Worker：`worker-run-next` / `worker-render-next`
- API 入口：`webapp/v2_app.py`
  - 点播：`POST /api/v2/runs/ondemand`
  - 订阅：`POST /api/v2/runs/daily/schedule`
  - 定时触发：`POST /api/v2/runs/daily/tick`
  - Worker：`POST /api/v2/workers/runs/next` / `POST /api/v2/workers/render/next`
- 队列与调度：
  - Run 队列：`orchestrator/queue.py` `InMemoryRunQueue`
  - Orchestrator：`orchestrator/service.py` `RunOrchestrator`
  - 状态存储：`orchestrator/store.py` `InMemoryRunStore`
- 日常 08:00 机制：
  - 由 `RunOrchestrator.trigger_due_daily_runs()` 计算本地时区是否到点并入队。
  - 本项目未内置 OS 级 cron 守护，需外部调度器定时调用 `daily-tick` 或对应 API。

### 1.2 核心结构定义
- `RunRequest` / `RunStatus` / `RenderStatus` / `Artifact`：
  - 定义在 `core/contracts.py`
- `RenderJob`（渲染内部任务对象）：
  - 定义在 `render/manager.py`

### 1.3 E2E 调用链（真实代码）

```text
CLI/API
  -> RunOrchestrator.enqueue_run()                         (orchestrator/service.py)
  -> InMemoryRunQueue.enqueue()                            (orchestrator/queue.py)
  -> worker-run-next / POST /workers/runs/next
  -> RunPipelineRuntime.run_next()                         (pipeline_v2/runtime.py)
      -> _collect_raw_items() -> sources.connectors.*      (sources/connectors.py)
      -> normalize()                                        (pipeline_v2/normalize.py)
      -> dedup_exact/embed/cluster/merge_cluster()          (pipeline_v2/dedup_cluster.py)
      -> rank_items()                                       (pipeline_v2/scoring.py)
      -> generate_script()                                  (pipeline_v2/script_generator.py)
      -> script_to_storyboard()/validate/auto_fix           (pipeline_v2/storyboard_generator.py)
      -> compile_storyboard()                               (pipeline_v2/prompt_compiler.py)
      -> enqueue_render()                                   (render/manager.py)
      -> tts_generate/align_subtitles/mix_bgm              (render/audio_subtitles.py)
      -> generate_onepager/thumbnail/export_package         (pipeline_v2/report_export.py)
  -> worker-render-next / POST /workers/render/next
  -> RenderManager.process_next()                           (render/manager.py)
      -> stitch_shots()/fallback_render()
      -> validate_mp4() -> RenderStatus.valid_mp4/probe_error
```

## 2) PRD 模块对照（代码证据）

| 模块 | 状态 | 代码证据 | 备注 |
|---|---|---|---|
| A Orchestrator | ✅ | `orchestrator/service.py` `schedule_daily_digest/enqueue_run/get_run_status/cancel_run/trigger_due_daily_runs` | 支持点播+订阅入队 |
| B Source Connectors | ✅ | `sources/connectors.py` `fetch_github_trending/fetch_github_releases/fetch_huggingface_trending/fetch_hackernews_top/fetch_rss_feed/fetch_web_article` | Tier A/B 全部函数存在 |
| C Normalization | ✅ | `pipeline_v2/normalize.py` `normalize/extract_citations/content_hash` | 含 tier/credibility/citation_count |
| D Dedup & Clustering | ✅ | `pipeline_v2/dedup_cluster.py` `dedup_exact/embed/cluster/merge_cluster` | 本地哈希 embedding，非外部模型 |
| E Scoring & Ranking | ✅ | `pipeline_v2/scoring.py` `score_* / rank_items` | 可解释 reasons，Tier B Top3 门控 |
| F Script Generator | ✅ | `pipeline_v2/script_generator.py` `generate_script/generate_variants` | 时码脚本已实现 |
| G Storyboard Generator | ✅ | `pipeline_v2/storyboard_generator.py` `script_to_storyboard/validate_storyboard/auto_fix_storyboard` | 约束 5-8 镜头 |
| H Prompt Compiler | ✅ | `pipeline_v2/prompt_compiler.py` `compile_shot_prompt/compile_storyboard/consistency_pack` | 输出 PromptSpec |
| I Render Manager | 🟡 | `render/manager.py` `enqueue_render/process_next/retry_failed_shots/fallback_render/stitch_shots` | 任务编排完整；Seedance 真实调用见第 4 节 |
| J Audio/Subtitles | ✅ | `render/audio_subtitles.py` `tts_generate/align_subtitles/mix_bgm` | 本地可听音轨 + 单调 SRT + 可选 BGM 混音 |
| K Report & Export | ✅ | `pipeline_v2/report_export.py` `generate_onepager/generate_thumbnail/export_package` | 可产出 onepager/svg/zip |
| L Notification | 🟡 | `pipeline_v2/notification.py` `notify_user/post_to_web/send_email` | 当前为本地 JSONL 记录，不是真实外发 |

## 3) MP4 可播放性说明（已修复）

历史问题根因：
- 旧逻辑在 `render/manager.py` 中把 `rendered_final.mp4` / `fallback_render.mp4` 直接写成文本占位字节，不是 MP4 容器。

当前修复后：
- 优先使用 `ffmpeg` 进行拼接或合成，输出真实 MP4（H.264 + AAC）。
- 输出写入采用原子写（`tmp -> os.replace`）。
- 渲染结束后执行 `ffprobe` 校验，结果写入：
  - `RenderStatus.valid_mp4`
  - `RenderStatus.probe_error`

## 4) Seedance 接入现状（结论）

结论：`🟡 已内置可选真实 HTTP 接入；默认关闭（SEEDANCE_ENABLED=0）`

代码证据：
- Adapter 边界：`render/adapters/base.py` `BaseRendererAdapter`
- Seedance 适配器：`render/adapters/seedance.py` `SeedanceAdapter`
  - 支持 env 自动读取：`SEEDANCE_ENABLED/SEEDANCE_BASE_URL/SEEDANCE_API_KEY/SEEDANCE_REGION/SEEDANCE_TIMEOUT_S`
  - 内置 `httpx` 最小 client，请求 `POST /v1/renders/shots`
  - 支持响应 base64 输出或 output_url 下载并落盘
  - 可继续注入 `client` 回调覆盖默认行为（测试/私有网关）

### 如何开启真实调用（当前版本）
默认关闭：`SEEDANCE_ENABLED=0`（走 fallback motion render）

开启真实调用：
- `SEEDANCE_ENABLED=1`
- `SEEDANCE_BASE_URL=https://api.seedance.example`
- `SEEDANCE_API_KEY=...`
- `SEEDANCE_REGION=us`
- `SEEDANCE_TIMEOUT_S=45`

运行后可在 `render_status` 查看：
- `seedance_used=true/false`
- `valid_mp4=true/false`
- `probe_error`

风险提示：
- 成本风险：镜头级调用会快速累积费用，务必设置 `max_total_cost` 和 `max_retries`。
- 时延风险：外部接口超时会触发重试和 fallback。
- 合规风险：需自行接入内容安全/审核策略。

## 5) 多源抓取现状与质量

### 5.1 Connector 列表
- GitHub Trending：`sources/connectors.py` `fetch_github_trending`
- GitHub Releases：`sources/connectors.py` `fetch_github_releases`
- HuggingFace：`sources/connectors.py` `fetch_huggingface_trending`
- HackerNews：`sources/connectors.py` `fetch_hackernews_top`
- RSS：`sources/connectors.py` `fetch_rss_feed`
- WebArticle：`sources/connectors.py` `fetch_web_article`

### 5.2 是否复用旧抓取代码
- 已复用旧抓取器：
  - `scrapers/social/github_scraper.py`
  - `scrapers/huggingface_scraper.py`
  - `scrapers/hackernews_scraper.py`
- 新增统一封装层：
  - `sources/connectors.py`（统一输出 `RawItem`）

### 5.3 抽取质量现状
- 正文抽取：
  - `fetch_web_article` 使用正则 + 去 HTML 标签，鲁棒性中等（复杂网页可能退化）。
- 引用抽取：
  - `normalize.extract_citations` 会从 metadata、markdown link、正文 URL 提取并去重。
- 去重/聚类：
  - `dedup_exact` + `embed/cluster/merge_cluster` 已具备。

## 6) 最小可运行命令

### 6.1 点播跑一次（CLI）
```bash
python main.py ondemand --user-id u1 --topic "MCP deployment" --time-window 24h --tz America/Los_Angeles --targets web,mp4
python main.py worker-run-next
python main.py worker-render-next
python main.py status --run-id <run_id>
```

注意：当前 `main.py` 的状态存储为进程内内存，多条独立 `python main.py ...` 命令不会共享队列状态。
要稳定复现整条链路，优先使用：
- 同进程脚本（`scripts/e2e_smoke_v2.py`）
- 或启动一个常驻 API 进程后通过 HTTP 顺序调用 enqueue/worker 接口

### 6.2 daily 模拟跑一次（CLI）
```bash
python main.py daily-subscribe --user-id u1 --run-at 08:00 --tz America/Los_Angeles --top-k 3
python main.py daily-tick --now-utc 2026-02-18T16:10:00+00:00
python main.py worker-run-next
python main.py worker-render-next
```

## 7) 产物类型与路径示例

典型目录：`data/outputs/v2_runs/<run_id>/`
- `script.json`
- `storyboard.json`
- `prompt_bundle.json`
- `materials.json`
- `onepager.md`
- `thumbnail_*.svg`
- `tts_narration.wav`
- `tts_narration_with_bgm.wav`
- `captions.srt`
- `<run_id>_package.zip`

渲染目录：`data/outputs/render_jobs/<render_job_id>/`
- `rendered_final.mp4` 或 `fallback_render.mp4`

## 8) 自检命令（含 MP4 校验）

```bash
python scripts/e2e_smoke_v2.py --out-dir /tmp/ara_v2_smoke > /tmp/ara_v2_smoke/result.json
```

```bash
python - <<'PY'
import json, pathlib, subprocess
p = pathlib.Path('/tmp/ara_v2_smoke/result.json')
d = json.loads(p.read_text())
mp4 = next(a['path'] for a in d['artifacts'] if a['type']=='mp4')
print('run_id=', d['run_id'])
print('mp4=', mp4)
print('valid_mp4=', d['render_status'].get('valid_mp4'))
subprocess.run(['ffprobe','-hide_banner','-v','error','-show_format','-show_streams',mp4], check=False)
PY
```

```bash
python - <<'PY'
import json, pathlib, subprocess
p = pathlib.Path('/tmp/ara_v2_smoke/result.json')
d = json.loads(p.read_text())
run_dir = next(e['message'].split('output_dir=',1)[1] for e in d['events'] if e['event']=='run_started')
render_dir = str(pathlib.Path(d['render_status']['output_path']).parent)
subprocess.run([
  'python', 'scripts/validate_artifacts_v2.py',
  '--run-dir', run_dir,
  '--render-dir', render_dir
], check=False)
PY
```

```bash
python scripts/check_no_placeholders_v2.py --root /Users/dexter/Documents/Dexter_Work/AcademicResearchAgent
```

## 9) 已知限制与下一步

已知限制：
- 当前测试中的 renderer 仍以 mock 为主；线上接入时需验证真实 Seedance 返回 schema 与下载链路。
- Notification 仍是本地日志，不是实际 Webhook/SMTP 投递。
- 队列与状态存储目前是内存实现，进程重启后丢失。

下一步建议：
1. 接入持久化队列（Redis/RQ/Celery）与 DB 状态表。
2. 增加 Seedance API 契约回归测试（响应字段变更报警 + 下载链路探针）。
3. 提升音频链路到自然语音 TTS（当前为本地规则合成音轨）。
4. 增加抓取质量指标（正文长度、引用密度、来源新鲜度）并纳入排序权重。
