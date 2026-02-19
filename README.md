# AcademicResearchAgent v2 状态说明（实装审计版）

## Changelog (Last Updated: 2026-02-19)
### Commit: No-Candidate Crash Fix + Shortage Rescue (New)
- 本次目标：
  - 修复 live 场景下 `no candidates survived evidence audit` 直接导致 run 失败的问题。
  - 保持硬相关不变量不变（`relevance>0`、`hard_match_pass`、`body_len>0`）前提下，避免“全 reject 即崩溃”。
- 关键改动：
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/runtime.py`
    - 新增 `shortage rescue`：当最大 phase 后 `picks` 为空时，从 hard-eligible 候选中选择 1 条并标记 `shortage_fallback_after_max_phase`（downgrade），继续生成产物。
    - 回填 `attempts/quality_snapshot/selection_outcome`，避免 onepager 继续显示 `TopPicksMinRelevance=0` 之类失真值。
    - 新增 HF metadata-only 深抓取触发条件：`extraction_method=*_metadata` 且正文过短也会尝试补抓。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/tests/v2/test_runtime_integration.py`
    - 新增“all reject 不崩溃”回归测试，确保 run 仍输出 artifacts，并在质量原因中标注 shortage fallback。
- 如何验证：
  - `pytest -q tests/v2`
  - `python main.py run-once --mode live --topic "AI agent" --time_window today --tz Asia/Singapore --targets web --top-k 3`
  - 预期：命令不再因 `no candidates survived evidence audit` 失败；会产出 diagnosis/facts/script/onepager（可能 `<top_k`，并附 shortage 原因）。
- 已知限制与回滚：
  - 当当天信号极弱时，仍可能只产出 1 条 downgrade 兜底（不会凑满）。
  - 回滚：`git revert <this_commit_sha>`。

### Commit: Hard Invariants + Source-Coverage Shortage Policy (New)
- 本次目标：
  - 修复 `topic="AI agent"` 下扩检后仍混入非 agent 条目（如 Qwen-VL/ColBERT）的问题。
  - 把“不相关/空正文候选”从 `ranking/audit` 层上升为 `SelectionController` 的硬不变量，任何阶段不可绕过。
  - 在 source coverage 无法达标时，执行“宁缺毋滥”策略：允许 `<top_k` 并强制解释 `Why not more?`。
- 根因回顾（本次落地修复）：
  - relevance 语义污染：`source_query` 曾参与 topic 语义匹配，导致检索词命中可“借道”通过 hard gate。
  - 选择层兜底过宽：最后阶段 downgrade fallback 未统一执行 hard relevance/body gate，出现 `relevance=0/body_len=0` 入选。
  - HF metadata-only 场景：正文为空时未强制补抓 model card，导致弱证据候选被 downgrade 填充。
- 关键策略改动：
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/scoring.py`
    - relevance/hard gate 文本不再包含 `source_query`，避免“查询词借道命中”。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/retrieval_controller.py`
    - 新增 controller 统一硬过滤：`relevance<=0` / `topic_hard_match_pass=false` / `body_len<=0` 一律不可入选。
    - downgrade fallback 仅在满足 `min_relevance_for_selection` 且硬过滤通过的候选上执行。
    - source diversity 成为硬约束：当 `source_coverage < min_source_coverage` 时不再盲目补满 top_k，返回 shortage。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/evidence_auditor.py`
    - 新增硬拒绝：`topic_relevance_zero`、`topic_hard_gate_fail`、`body_len_zero`。
    - 保持 machine_action（`action/reason_code/human_reason`）供 controller/diagnosis 追踪。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/runtime.py`
    - controller 评估改为使用 `relevance_eligible`，并传入当次 `relevance_threshold`。
    - quality trigger 新增 `source_coverage_lt_2` 可观测触发原因。
    - HF deep fetch 增强：metadata-only/短正文候选优先抓 `raw/README.md`，回填后重走 normalize/scoring/audit。
    - diagnosis attempt 新增 `source_coverage`、`deep_fetch_details`（method/before/after/accepted/error）。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/topic_profile.py`
    - AI agent 话题增强 HF 排除词：`vision-language/embedding/retriever/reranker/colbert` 等。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/report_export.py`
    - onepager header 新增 `SelectedSourceCoverage`，配合 `Why not more?` 解释不足而非凑数。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/scripts/validate_artifacts_v2.py`
    - 新增 `top_picks_hard_relevance_invariants` 门禁（Top picks 禁止 `relevance<=0/hard_gate_fail/body_len<=0`）。
    - 候选不足策略升级：`candidate_shortage_explained`，live 下允许 `<top_k` 但必须有 `Why not more?` 解释。
- 测试更新：
  - `tests/v2/test_scoring.py`：新增“source_query 不能绕过 hard gate”回归。
  - `tests/v2/test_quality_trigger_expansion.py`：新增“controller 永不选入 relevance=0/body_len=0/hard_gate_fail”回归。
  - `tests/v2/test_runtime_integration.py`：
    - 新增“单源不可扩展时返回 shortage + Why not more”回归。
    - 新增“HuggingFace metadata-only 触发 deep fetch 并补正文”回归。
  - `tests/v2/test_evidence_auditor.py`：新增硬拒绝规则回归。
  - `tests/v2/test_validate_artifacts_v2.py`：新增 shortage 解释门禁回归。
- 如何验证：
  - `pytest -q tests/v2`
  - `OUT="/tmp/ara_v2_live_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$OUT"; python main.py run-once --mode live --topic "AI agent" --time_window today --tz Asia/Singapore --targets web,mp4 --top-k 3 > "$OUT/result.json"`
  - `python scripts/validate_artifacts_v2.py --run-dir <run_dir> --render-dir <render_dir>`
- 已知限制与未来扩展：
  - 当数据窗口天然稀疏且可用源单一时，Top picks 可能小于 `requested_top_k`（这是设计上的“诚实输出”）。
  - LLM Planner/Auditor 仍可通过 `budget.use_llm_planner` / `budget.use_llm_auditor` 挂载，默认 deterministic。
  - 回滚：`git revert <this_commit_sha>`。

### Commit: Close-the-loop Selection + Audit-driven Expansion (New)
- 本次目标：
  - 把 `planner + auditor` 从“仅产出报告”改为“闭环控制器”，保证 Top picks 优先由 `PASS` 候选构成，质量不足时自动扩检。
  - 修复 `AI agent` 主题下相关性失真（泛词打高分、CV/CLIP 误入）和 facts 文本质量问题（截断词、HN 元信息污染）。
- 核心策略变更：
  - 新增 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/retrieval_controller.py`：
    - `SelectionController` 执行 `pass-first -> expansion -> downgrade fallback(仅最后阶段)`。
    - 质量触发器：`pass_count < top_k` / `pass_ratio < 0.3` / `top_picks_min_evidence_quality < 2` / `bucket_coverage < 2` / 多样性不足。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/runtime.py`：
    - `_execute_run` 接入 controller 闭环，attempt 级别落盘 `phase/window/query_set/candidate_count/pass/downgrade/reject/next_phase_reason`。
    - `retrieval_diagnosis.json` 与 onepager header 同步输出：`hard_match_terms_used/hard_match_pass_count/top_picks_min_relevance/top_picks_hard_match_count/quality_triggered_expansion`。
    - Top picks 仅在最后阶段才允许 downgrade 兜底，并把 downgrade 原因写入上下文。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/evidence_auditor.py`：
    - 新增 `machine_action`（`action/reason_code/human_reason`）供 controller 直接消费。
    - HN 低互动（`points<5 && comments<2`）默认 reject；AI agent 话题下 HF 的 `clip/vit/diffusion` 类默认强 reject。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/scoring.py`：
    - 保留 hard gate；新增“泛词降权”与 `1.0` 上限约束：仅在“高价值词命中 + 实质内容信号”同时满足时允许 `relevance=1.0`。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/script_generator.py`：
    - facts 抽取按句子边界截取，避免半词截断。
    - 过滤 `Points: x | Comments: y` 等 HN 元信息行，`how_it_works` 保证为完整可读句。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/report_export.py`：
    - onepager Top picks 对 downgrade 条目显式标注 `(降级: reason)`。
    - 当高质量候选不足时新增 `Why not more?` 段落，解释扩检后仍不足的原因。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/scripts/validate_artifacts_v2.py`：
    - 新增门禁：`top_picks_not_all_downgrade`、`retrieval_quality_triggered_expansion_recorded`、`facts_no_truncated_words`。
- 阈值配置（可覆盖）：
  - 环境变量：`ARA_V2_MIN_PASS_RATIO`、`ARA_V2_MIN_EVIDENCE_QUALITY`、`ARA_V2_MIN_BUCKET_COVERAGE`、`ARA_V2_MIN_SOURCE_COVERAGE`。
  - 或 `RunRequest.budget` 对应键：`min_pass_ratio/min_evidence_quality/min_bucket_coverage/min_source_coverage`。
- 如何验证：
  - `pytest -q tests/v2`
  - `OUT="/tmp/ara_v2_live_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$OUT"; python main.py run-once --mode live --topic "AI agent" --time_window today --tz Asia/Singapore --targets web,mp4 --top-k 3 > "$OUT/result.json"`
  - `python scripts/validate_artifacts_v2.py --run-dir <run_dir> --render-dir <render_dir>`
- 已知限制与未来扩展：
  - 限制：在极稀疏数据窗口下，严格 pass-first 可能导致 Top picks 数量下降（但会显式解释原因而非静默凑数）。
  - 限制：deterministic 规则对长尾新术语适应较慢，需要持续补词典。
  - 扩展：可启用 `budget.use_llm_planner=true` / `budget.use_llm_auditor=true` 走 LLM 插槽（当前默认关闭并回退 deterministic）。
  - 回滚：`git revert <this_commit_sha>`。

### Commit: Topic Hard Gate + Bucket Coverage + Quality Trigger Expansion (New)
- 本次目标：
  - 修复 `topic="AI agent"` 场景的硬语义偏离，阻断 `CLIP/ViT/vision-only model card` 误入 Top picks。
  - 引入子主题 bucket 覆盖与质量触发扩容，避免“凑满 top-k 但很水”。
  - 增强诊断可观测：在 diagnosis 与 onepager header 直接显示 hard-match/quality-trigger 关键指标。
- 实际改动：
  - 新增 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/topic_profile.py`：
    - 提供 `TopicProfile.for_topic(topic)`，内置 AI agent 专用 `hard_include_any`、`soft_boost/soft_penalty`、`source_filters` 与四类 bucket（Frameworks / Protocols & tool use / Ops & runtime / Evaluation）。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/scoring.py`：
    - 增加 hard semantic gate：AI agent 话题下，未命中 `hard_include_any` 时 `relevance=0`。
    - 增加 soft penalty：`clip/vit/diffusion/image classification` 等会显著降分。
    - reasons 新增 `topic.hard_match/topic.hard_terms/topic.hard_gate_fail`，便于解释排序。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/planner.py`：
    - planner 升级为 `ResearchPlanner`（deterministic）并预留 `LLMPlanner` 插槽（默认关闭）。
    - AI agent 生成 bucket 查询计划，按 source 输出 bucket queries。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/runtime.py`：
    - 接入 `TopicProfile` + planner source-aware queries。
    - attempt 诊断新增：`hard_match_terms_used/hard_match_pass_count/top_picks_min_relevance/top_picks_hard_match_count/quality_triggered_expansion`。
    - TopK 选择新增 bucket 多样性软约束（优先覆盖 >=2 buckets）。
    - 新增质量触发扩容：即使 top-k 数量已满，若 `min_relevance<0.75` 或 `hard_match_count<top_k` 或 `bucket_coverage<2`，继续进入下一 recall phase。
    - 预留 `LLMEvidenceAuditor` 插槽（默认关闭）。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/report_export.py`：
    - onepager header 新增 `HardMatchTermsUsed/HardMatchPassCount/TopPicksMinRelevance/TopPicksHardMatchCount/BucketCoverage/QualityTriggeredExpansion`。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/scripts/validate_artifacts_v2.py`：
    - 新增门禁：`retrieval_attempt_quality_fields_present`、`onepager_relevance_summary_fields_present`。
  - 测试更新：
    - 新增 `tests/v2/test_topic_profile.py`。
    - 新增 `tests/v2/test_quality_trigger_expansion.py`。
    - 更新 `tests/v2/test_runtime_integration.py`、`tests/v2/test_planner.py`、`tests/v2/test_validate_artifacts_v2.py`。
- LLM 插槽（默认 off）：
  - `budget.use_llm_planner=true` 可切换到 `LLMPlanner`（当前为空实现，回落 deterministic）。
  - `budget.use_llm_auditor=true` 可切换到 `LLMEvidenceAuditor`（当前为空实现，回落 deterministic）。
- 如何验证：
  - `pytest -q tests/v2`
  - `python scripts/e2e_smoke_v2.py --out-dir /tmp/ara_v2_smoke_planner_auditor > /tmp/ara_v2_smoke_planner_auditor/result.json`
  - live：`OUT=\"/tmp/ara_v2_live_$(date +%Y%m%d_%H%M%S)\"; mkdir -p \"$OUT\"; python main.py run-once --mode live --topic \"AI agent\" --time_window today --tz Asia/Singapore --targets web,mp4 --top-k 3 > \"$OUT/result.json\"`
  - validator：`python scripts/validate_artifacts_v2.py --run-dir <run_dir> --render-dir <render_dir>`
- 人工验收标准：
  - Top picks 每条命中 `hard_include_any`（onepager header + reasons 可见 hard-match 线索）。
  - bucket 覆盖 `>=2`。
  - 不再出现 `CLIP/VIT/vision-only model card` 作为 `AI agent` Top pick。
- 已知风险与回滚：
  - 风险：硬门禁与质量扩容会提高候选不足概率，但能显著降低主题偏离。
  - 风险：bucket 软约束会改变同分候选顺序，历史回放 Top picks 可能变化。
  - 回滚：`git revert <this_commit_sha>`。

### Commit: Planner + Evidence Auditor + Verifiable Facts (New)
- 本次目标：
  - 引入 deterministic `ResearchPlanner`，把 topic 扩展从“静态 expanded”升级为“可观测检索计划”。
  - 引入 deterministic `EvidenceAuditor`，对证据弱/重复/单域自述候选做 reject/downgrade 并替补。
  - 重构 facts 抽取为“section 优先 + 句子打分”，减少口号句，提升 WHAT/HOW 可验证性。
- 实际改动：
  - 新增 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/planner.py`：
    - 输出 `RetrievalPlan`（queries/source_weights/source_limits/time_window_policy/must_include_terms/must_exclude_terms）。
  - 新增 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/evidence_auditor.py`：
    - 输出 `evidence_audit.json`（verdict/reasons/evidence urls/duplicate ratio）。
    - 对 top picks 执行 pass/downgrade/reject，并在 reject 时尝试替补候选。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/runtime.py`：
    - 接入 planner 生成 phase queries；plan 落盘到 `retrieval_diagnosis.json`。
    - 召回尝试记录新增 `queries/must_include_terms/must_exclude_terms/deep_fetch_applied`。
    - 新增轻量 deep extraction（HN/Web body 过短时二次抓取外链）并记录诊断。
    - ranking 后接入 evidence auditor，落盘 `evidence_audit.json` 与 `run_context.evidence_audit_path`。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/sources/connectors.py`：
    - topic-search connector 支持 planner 注入的 `queries/must_include_terms/must_exclude_terms`。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/script_generator.py`：
    - 新增 markdown section 解析与句子打分；优先 Features/Quickstart/Usage/How it works/Examples。
    - `what_it_is/how_it_works/proof` 改为功能点优先，抑制宣言句/营销句。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/report_export.py`：
    - onepager header 新增 `EvidenceAuditPath`。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/scripts/validate_artifacts_v2.py`：
    - 新增门禁：`evidence_audit_parse_ok`、`top_picks_all_pass_or_downgrade_reason_present`、`citations_not_mostly_duplicate`、`onepager_evidence_audit_path_present`。
  - 测试更新：
    - 新增 `tests/v2/test_planner.py`、`tests/v2/test_evidence_auditor.py`。
    - 更新 `tests/v2/test_runtime_integration.py`、`tests/v2/test_script_storyboard_prompt.py`、`tests/v2/test_validate_artifacts_v2.py`。
- 如何验证：
  - `pytest -q tests/v2`
  - `python scripts/e2e_smoke_v2.py --out-dir /tmp/ara_v2_smoke_planner_auditor > /tmp/ara_v2_smoke_planner_auditor/result.json`
  - live：`OUT="/tmp/ara_v2_live_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$OUT"; python main.py run-once --mode live --topic "AI agent" --time_window today --tz Asia/Singapore --targets web,mp4 --top-k 3 > "$OUT/result.json"`
  - validator：`python scripts/validate_artifacts_v2.py --run-dir <run_dir> --render-dir <render_dir>`
- 已知风险与回滚：
  - 风险：planner 引入 include/exclude 约束后，窄主题在弱网络环境可能更容易出现候选不足。
  - 风险：auditor reject 规则变严后，live 场景 `top_k` 可能下降（但可解释性更高）。
  - 回滚：`git revert <this_commit_sha>`。

## Gap Assessment（2026-02-19 自审）
- 漏斗解释（为什么用户仍感知“不相关/太少”）：
  - `raw_items -> candidates -> filtered_by_relevance -> top_picks` 漏斗在 live 下会快速收缩；`retrieval_diagnosis.json` 可见每个 attempt 的 `raw_items/candidate_count/top_picks_count`。
  - 常见掉队原因集中在 `drop_reason`: `low_relevance`、`low_density`、`no_evidence_links`、`too_old`，且在 `query` 不够细时会叠加触发。
- facts/script 核心问题：
  - 旧版 `facts.what_it_is/how_it_works` 仍会抽到宣言/观点句，功能点密度不足，导致“可读但不可证”。
  - onepager 的模板复用较重，Top picks 间观感趋同，用户主观上会认为“像在凑数”。
- 质量门禁缺口：
  - validator 通过仅表示“结构完整”，不等于“产品可用”。
  - 旧门禁缺少：facts 可验证性、证据多样性、引用片段重复率、子主题覆盖度的强约束。

## 优先级计划（P0/P1）
- P0：
  - `ResearchPlanner`：topic 拆解、子主题召回、可观测 plan 落盘，解决粗粒度 query。
  - `EvidenceAuditor`：证据质量/重复/单域自述审计，reject/downgrade 并替补。
  - facts 抽取重构：section 优先 + 句子打分，保证 WHAT/HOW/PROOF 可验证。
  - validator 升级：evidence audit 相关门禁 + citations 重复门禁。
- P1：
  - deep extraction 扩展：对 HN 外链短正文候选做低成本二次深抽取并重排。
  - 子主题覆盖门禁：在 onepager 中显式检查 top picks 是否覆盖 planner 核心子主题。
  - onepager 去模板同质化：在保持结构化的前提下提高每条 pick 的差异化表达。

### Commit: Topic-First Retrieval + Recall Expansion Diagnosis (New)
- 本次目标：
  - 把 live 召回从“榜单优先”升级为“topic-first 优先”，并在候选不足时自动扩容召回范围。
  - 为候选不足问题提供可观测诊断（attempt/phase/drop_reason），避免只能靠 onepager 肉眼排查。
- 实际改动：
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/sources/connectors.py`：
    - 新增 `fetch_github_topic_search` / `fetch_huggingface_search` / `fetch_hackernews_search`。
    - 新增 topic 词扩展与时间窗解析（`today -> 3d`）能力，用于 recall expansion。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/runtime.py`：
    - 新增 `RecallProfile`、多阶段召回尝试（`base -> limit_x2 -> window_3d -> window_7d -> query_expanded`）。
    - 每次尝试记录 connector 调用统计并落盘 `retrieval_diagnosis.json`。
    - `run_context.retrieval` 新增扩容步骤与诊断路径，`ranking_stats` 新增 `selected_recall_phase/recall_attempt_count`。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/report_export.py`：
    - onepager header 新增 `RecallPhase` 与 `DiagnosisPath` 字段。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/scripts/validate_artifacts_v2.py`：
    - 新增诊断门禁：`retrieval_diagnosis_parse_ok`、`retrieval_diagnosis_attempts_present`、`retrieval_expansion_recorded`。
    - 新增 onepager 诊断字段门禁：`onepager_diagnosis_path_present`。
  - 修改测试：
    - `tests/v2/test_runtime_integration.py` 新增“候选不足触发扩容补齐并写诊断”的集成测试。
    - `tests/v2/test_validate_artifacts_v2.py` 新增诊断门禁断言。
- 新增/删除文件：
  - 无新增文件。
  - 修改：`core/contracts.py`, `sources/connectors.py`, `pipeline_v2/runtime.py`, `pipeline_v2/report_export.py`, `scripts/validate_artifacts_v2.py`, `tests/v2/test_runtime_integration.py`, `tests/v2/test_validate_artifacts_v2.py`, `README.md`
- 如何验证：
  - `pytest -q tests/v2`
  - `python scripts/e2e_smoke_v2.py --out-dir /tmp/ara_v2_smoke_recall > /tmp/ara_v2_smoke_recall/result.json`
  - `python - <<'PY'\nimport json,pathlib,subprocess\np=pathlib.Path('/tmp/ara_v2_smoke_recall/result.json')\nd=json.loads(p.read_text())\nrun_dir=next(pathlib.Path(a['path']).parent for a in d['artifacts'] if a['type']=='script')\nrender_dir=next(pathlib.Path(a['path']).parent for a in d['artifacts'] if a['type']=='mp4')\nsubprocess.run(['python','scripts/validate_artifacts_v2.py','--run-dir',str(run_dir),'--render-dir',str(render_dir)],check=True)\nprint('diagnosis=',run_dir/'retrieval_diagnosis.json')\nPY`
  - live（需网络）：`OUT=\"/tmp/ara_v2_live_$(date +%Y%m%d_%H%M%S)\"; mkdir -p \"$OUT\"; python main.py run-once --mode live --topic \"AI agent\" --time_window today --tz Asia/Singapore --targets web,mp4 --top-k 3 > \"$OUT/result.json\"`
- 已知风险与回滚：
  - 风险：网络受限时 live topic-search connector 会全部失败，run 会报 `no live items fetched from connectors`（不会自动回退 smoke）。
  - 风险：召回扩容增加了 connector 调用次数，外部 API 限流概率上升。
  - 回滚：`git revert <this_commit_sha>`。

### Commit: P1 Facts Cleanup + Local Asset-Driven Shots (New)
- 本次目标：
  - 消除 facts/script/storyboard 中的 URL 碎片与 quote 残留，提升口播与分镜可读性。
  - 打通本地素材引用链路，让前置镜头稳定使用 `run_dir/assets/*` 而不是仅在线 URL。
- 实际改动：
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/script_generator.py`：
    - 新增 URL/path fragment 清洗规则（例如 `com/s/...`、markdown quote 前缀 `>`）。
    - `build_facts` 增加 `metrics` 且 `proof_links` 去重；`how_it_works/proof` 过滤 URL 残片样式句子。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/runtime.py`：
    - `_attach_reference_assets` 调整为优先注入本地素材（前 3 个镜头强制带本地 asset），同时保留有效 evidence URL。
  - 测试更新：
    - `tests/v2/test_script_storyboard_prompt.py` 新增 facts 去碎片回归。
    - `tests/v2/test_runtime_integration.py` 要求至少 2 个镜头引用本地 assets。
- 新增/删除文件：
  - 无新增文件。
  - 修改：`pipeline_v2/{script_generator.py,runtime.py}`, `tests/v2/{test_script_storyboard_prompt.py,test_runtime_integration.py}`, `README.md`
- 如何验证：
  - `pytest -q tests/v2`
  - `OUT="/tmp/ara_v2_live_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$OUT"; python main.py run-once --mode live --topic "AI agent" --time_window today --tz Asia/Singapore --targets web,mp4 --top-k 3 > "$OUT/result.json"`
  - `python scripts/validate_artifacts_v2.py --run-dir <run_dir> --render-dir <render_dir>`
- 已知风险与回滚：
  - 风险：更激进的清洗可能删除少量边界信息（但可读性会显著提升）。
  - 风险：本地素材优先会增加镜头视觉一致性，可能减少部分在线证据链接的直接可见度。
  - 回滚：`git revert <this_commit_sha>`。

### Commit: P0 Onepager Gate + Evidence Scope + Link Hygiene (New)
- 本次目标：
  - 修复 live 场景下 Top picks 数量门禁“假通过”问题。
  - 修复 onepager Evidence 噪声灌入与低质量链接（localhost/apikey/api endpoint）污染。
- 实际改动：
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/scripts/validate_artifacts_v2.py`：
    - live 门禁改为 `top_picks_count >= requested_top_k`（从 `run_context.ranking_stats` 读取）。
    - 新增 onepager 一致性检查：header 的 `RequestedTopK/TopPicksCount` 必须与 run_context 和实际 heading 数一致。
    - 新增 `candidate_shortage:*` 明确错误，避免“全绿但产品不可交付”。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/report_export.py`：
    - onepager header 新增 `RequestedTopK`，并保持与 ranking_stats 对齐。
    - Evidence 继续按 Top picks 聚合并去重，不再使用候选池全集刷屏。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/sanitize.py`：
    - `classify_link` 增强：过滤 `*.localhost`/`127.0.0.1`/`0.0.0.0`、`aistudio.../apikey`、典型 API endpoint。
    - `is_allowed_citation_url` 只允许 evidence 链接进入 script/onepager/prompt 引用。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/normalize.py`：
    - citations 全链路使用 canonical URL + evidence-only 过滤。
  - 测试更新：
    - `tests/v2/test_sanitize.py`、`tests/v2/test_normalize.py`、`tests/v2/test_validate_artifacts_v2.py` 增加 localhost/tooling/shortage 回归。
- 新增/删除文件：
  - 无新增文件。
  - 修改：`pipeline_v2/{sanitize.py,normalize.py,report_export.py}`, `scripts/validate_artifacts_v2.py`, `tests/v2/{test_sanitize.py,test_normalize.py,test_validate_artifacts_v2.py}`, `README.md`
- 如何验证：
  - `pytest -q tests/v2`
  - `OUT="/tmp/ara_v2_live_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$OUT"; python main.py run-once --mode live --topic "AI agent" --time_window today --tz Asia/Singapore --targets web,mp4 --top-k 3 > "$OUT/result.json"`
  - `python scripts/validate_artifacts_v2.py --run-dir <run_dir> --render-dir <render_dir>`
- 已知风险与回滚：
  - 风险：live 数据稀疏时会显式失败（candidate_shortage），不再“看起来通过”。
  - 风险：更严格链接过滤会减少可用 citations 数量。
  - 回滚：`git revert <this_commit_sha>`。

### Commit: Canonical URLs + Evidence Dedup + Adaptive Relevance + Better Facts (New)
- 本次目标：
  - 解决 live 产物中 URL 尾部噪声、tooling 链接误入 citations、Evidence 重复刷屏，以及 Top picks 在高阈值下过少的问题。
- 实际改动：
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/sanitize.py`：
    - 新增 `canonicalize_url` 与 `classify_link`（`evidence/tooling/asset/blocked/invalid`）。
    - URL 统一去尾噪声、scheme 归一为 `https`、query 白名单保留；`bun.sh/api.*` 等 tooling 链接默认不可用于 citation/reference。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/normalize.py`：
    - `extract_citations` 全链路改为 canonical URL + evidence-only 过滤。
    - metadata 新增 `evidence_links/tooling_links/asset_links`，为 facts 与 onepager 提供可解释证据输入。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/runtime.py`：
    - relevance gate 支持自适应放宽：`0.55 -> ... -> 0.35`（每步 `0.05`），并落盘 `topic_relevance_threshold_used/relaxation_steps/requested_top_k/diversity_sources`。
    - Top picks 采用 source 多样性软约束选取，避免同源挤占。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/report_export.py`：
    - onepager header 新增 `TopicRelevanceThresholdUsed/RelevanceRelaxationSteps`。
    - Evidence 改为按 item 聚合去重（每 item 上限 4，全局上限 20），并输出“候选不足”提示。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/script_generator.py`：
    - facts 增加 `metrics`，`what_it_is` 去指标拼接，`why_now` 优先 recency/热度信号，`proof` 至少 2 条可验证事实。
    - script evidence 与 line references 全部 evidence-only，禁止 tooling 链接回流。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/prompt_compiler.py` 与 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/render/manager.py`：
    - prompt `references` 仅保留 evidence URL；本地/asset 素材转入 `seedance_params.render_assets`，渲染侧继续可用。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/scripts/validate_artifacts_v2.py`：
    - 新增 `evidence_dedup_ok`（按 item evidence 分组去重 + 全局重复上限）。
    - `topic_relevance_ok` 改为读取 `topic_relevance_threshold_used`。
- 新增/删除文件：
  - 无新增文件。
  - 修改：`pipeline_v2/{sanitize.py,normalize.py,runtime.py,script_generator.py,report_export.py,prompt_compiler.py}`, `render/manager.py`, `scripts/validate_artifacts_v2.py`, `tests/v2/{test_sanitize.py,test_normalize.py,test_prompt_compile.py,test_script_storyboard_prompt.py,test_runtime_integration.py,test_report_export_notification.py,test_validate_artifacts_v2.py}`, `README.md`
- 如何验证：
  - `pytest -q tests/v2`
  - `OUT="/tmp/ara_v2_live_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$OUT"; python main.py run-once --mode live --topic "AI agent" --time_window today --tz Asia/Singapore --targets web,mp4 --top-k 3 > "$OUT/result.json"`
  - `python scripts/validate_artifacts_v2.py --run-dir <run_dir> --render-dir <render_dir>`
- 已知风险与回滚：
  - 风险：外网不可达时，live 抓取会直接失败（不会回退为 smoke）。
  - 风险：自适应阈值会改变边界样本排序，可能导致 Top picks 组成变化。
  - 回滚：`git revert <this_commit_sha>`。

### Commit: Relevance Hard Gate + URL Validity + Local Asset References (New)
- 本次目标：
  - 强制 topic relevance 硬门禁，不再把低相关候选补齐进 Top picks。
  - 修复 URL 尾引号/畸形格式问题并加入产物级合法性校验。
  - 让分镜 overlay 不再硬截断，并把 `materials` 中截图计划落地到本地 `assets/`。
- 实际改动：
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/runtime.py`：
    - live 模式只允许 `relevance_score >= threshold` 的候选进入 Top picks；不足 `top-k` 时允许少于 `top-k`。
    - `ranking_stats` 新增 `candidate_count/top_picks_count/filtered_by_relevance/drop_reason_samples`。
    - 新增 assets 物化流程：将 screenshot plan URL 转为 `run_dir/assets/*` 本地文件，并回填到 storyboard `reference_assets`。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/sanitize.py` 与 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/normalize.py`：
    - 新增 `normalize_url/is_valid_http_url`，统一用于 citations/links/reference_assets。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/script_generator.py`：
    - facts 去噪，移除 `Stars/Forks/LastPush` 拼接句与 markdown 残片，proof 优先绑定有效 citation。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/storyboard_generator.py` 与 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/render/manager.py`：
    - 新增 `overlay_compact` 安全压缩（<=42 chars）与图片卡片优先渲染路径（有本地 asset 时优先图文卡）。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/scripts/validate_artifacts_v2.py`：
    - 新增 `urls_valid`、`storyboard_overlay_safe`；live/smoke 分别执行不同 Top picks 最小数量门禁。
  - 修改测试：
    - `tests/v2/test_sanitize.py`, `tests/v2/test_script_storyboard_prompt.py`, `tests/v2/test_runtime_integration.py`, `tests/v2/test_validate_artifacts_v2.py`, `tests/v2/test_v2_webapp_api.py`。
- 新增/删除文件：
  - 无新增文件。
  - 修改：`pipeline_v2/{runtime.py,sanitize.py,normalize.py,script_generator.py,storyboard_generator.py,report_export.py,__init__.py}`, `render/manager.py`, `scripts/validate_artifacts_v2.py`, `tests/v2/{test_sanitize.py,test_script_storyboard_prompt.py,test_runtime_integration.py,test_validate_artifacts_v2.py,test_v2_webapp_api.py}`, `README.md`
- 如何验证：
  - `pytest -q tests/v2`
  - `python scripts/e2e_smoke_v2.py --out-dir /tmp/ara_v2_smoke_check > /tmp/ara_v2_smoke_check/result.json`
  - `python scripts/validate_artifacts_v2.py --run-dir /tmp/ara_v2_smoke_check/runs/<run_id> --render-dir /tmp/ara_v2_smoke_check/render_jobs/<render_job_id>`
  - live 网络可用时：`python main.py run-once --mode live --topic "AI agent" --time_window today --tz Asia/Singapore --targets web,mp4 --top-k 3`
- 已知风险与回滚：
  - 风险：live 网络不可达时会因“无可用抓取内容”直接失败（设计上不回退 smoke）。
  - 风险：relevance 硬门禁可能导致 live 场景 Top picks 少于 `top-k`（换取主题一致性）。
  - 回滚：`git revert <this_commit_sha>`。

### Commit: URL Parse Guard for Validator Crash (Hotfix)
- 本次目标：
  - 修复 `run-once` 内置 validator 因畸形 URL 崩溃的问题（`Invalid IPv6 URL`）。
- 实际改动：
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/sanitize.py`：
    - `_hostname()` 对 `urlparse` 的 `ValueError` 做容错，解析失败返回空 host 并判定为不合规 URL。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/tests/v2/test_sanitize.py`：
    - 新增 `https://[invalid` 回归用例，确保不抛异常且返回 `False`。
- 新增/删除文件：
  - 无新增文件。
  - 修改：`pipeline_v2/sanitize.py`, `tests/v2/test_sanitize.py`, `README.md`
- 如何验证：
  - `pytest -q tests/v2/test_sanitize.py tests/v2/test_validate_artifacts_v2.py`
  - `python main.py run-once --mode smoke --topic \"AI agent\" --time_window today --tz Asia/Singapore --targets web,mp4 --top-k 3`
- 已知风险与回滚：
  - 风险：畸形 URL 会被静默过滤，不再进入 citations（预期行为）。
  - 回滚：`git revert <this_commit_sha>`。

### Commit: Compact Onepager Compiler & Live Quality Gates (New)
- 本次目标：
  - 将 onepager 从“报告式长句”压缩为“可发布短句”（短、狠、可口播）。
  - 给 live 模式新增验收门禁：compact bullets / facts 完整性 / update signal。
- 实际改动：
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/report_export.py`：
    - Top pick 文案改为 `Compact Brief`（固定结构：`WHAT/WHY NOW/HOW/PROOF/CTA`）。
    - 约束：每个 pick 最多 6 条 bullet，单条 <= 90 bytes（UTF-8）。
    - 新增 `Why Ranked` 短行：`relevance + 更新信号/证据链/Quickstart`。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/scripts/validate_artifacts_v2.py`：
    - live 新门禁：`onepager_bullets_compact_ok`、`facts_has_why_now_and_proof`、`ranked_items_have_update_signal`。
    - 失败报错包含 pick 序号与具体原因（缺少标签/超长等）。
  - 修改测试：
    - `tests/v2/test_report_export_notification.py` 验证 compact bullets 结构与长度。
    - `tests/v2/test_validate_artifacts_v2.py` 验证新增门禁存在且通过。
- 新增/删除文件：
  - 无新增文件。
  - 修改：`pipeline_v2/report_export.py`, `scripts/validate_artifacts_v2.py`, `tests/v2/test_report_export_notification.py`, `tests/v2/test_validate_artifacts_v2.py`, `tests/v2/test_runtime_integration.py`, `README.md`
- 如何验证：
  - `pytest -q tests/v2`
  - `OUT=\"/tmp/ara_v2_live_$(date +%Y%m%d_%H%M%S)\"; mkdir -p \"$OUT\"; python main.py run-once --mode live --topic \"AI agent\" --time_window today --tz Asia/Singapore --targets web,mp4 --top-k 3 > \"$OUT/result.json\"`
  - `python - <<'PY'\nimport json, pathlib, subprocess\np=pathlib.Path('$OUT/result.json')\nd=json.loads(p.read_text())\nrun_dir=d.get('run_dir'); render_dir=d.get('render_dir')\nprint(subprocess.run(['python','scripts/validate_artifacts_v2.py','--run-dir',run_dir,'--render-dir',render_dir],capture_output=True,text=True).stdout)\nPY`
- 已知风险与回滚：
  - 风险：bullet 压缩会牺牲部分上下文细节（换来更强发布可读性）。
  - 回滚：`git revert <this_commit_sha>`。

### Commit: Deep Extraction Signals & Ranking Stability (New)
- 本次目标：
  - 把 live 候选的“正文密度/更新时间/证据质量”做成结构化信号。
  - 在 relevance 硬门禁后引入可解释 quality boost，减少跑题与空壳内容。
- 实际改动：
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/normalize.py`：
    - 新增 `quality_signals`：`content_density/has_quickstart/has_results_or_bench/has_images_non_badge/publish_or_update_time/update_recency_days/evidence_links_quality`。
    - 更新时间统一产出 `publish_or_update_time`（ISO）并按 `Asia/Singapore` 计算 `update_recency_days`。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/scoring.py`：
    - relevance 继续硬门禁（live 默认阈值 0.55）。
    - relevance 通过后引入 `quality_signal_boost`（density/quickstart/results/evidence/recency）。
    - reasons 增加 quality signals 与 `penalty.no_evidence_links/penalty.too_old`。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/runtime.py`：
    - `run_context.ranking_stats` 新增：`top_quality_signals/top_why_ranked/drop_reason_samples`。
  - 修改测试：
    - `tests/v2/test_normalize.py` 新增 github-like 质量信号抽取断言。
    - `tests/v2/test_scoring.py` 新增“高热度但低相关不得进 Top picks”硬门禁回归。
    - `tests/v2/test_runtime_integration.py` 断言 `drop_reason_samples/top_quality_signals` 已落盘。
- 新增/删除文件：
  - 无新增文件。
  - 修改：`pipeline_v2/normalize.py`, `pipeline_v2/scoring.py`, `pipeline_v2/runtime.py`, `tests/v2/test_normalize.py`, `tests/v2/test_scoring.py`, `tests/v2/test_runtime_integration.py`, `README.md`
- 如何验证：
  - `pytest -q tests/v2`
  - `python scripts/e2e_smoke_v2.py --out-dir /tmp/ara_v2_quality_signals > /tmp/ara_v2_quality_signals/result.json`
- 已知风险与回滚：
  - 风险：quality boost 可能在边界样本上改变排序顺序（预期行为，已保留 relevance 硬门禁）。
  - 回滚：`git revert <this_commit_sha>`。

### Commit: Sanitize + Relevance Gate + Facts-Driven Script (New)
- 本次目标：
  - 消除 live 产物中的 README HTML/badge/赞助链接污染。
  - 增加 topic relevance 硬门禁，防止选题跑偏。
  - 用 `facts.json` 驱动 script/storyboard/prompt，避免 raw snippet 拼接。
- 实际改动：
  - 新增 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/sanitize.py`：
    - 清洗 HTML/徽章/推广链接/图片直链，提供 citation URL denylist 过滤。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/normalize.py`：
    - 全链路改用清洗后的 `clean_text` 做正文与质量指标。
    - `extract_citations` 新增 denylist 过滤（屏蔽 `img.shields.io`/`buymeacoffee`/`github sponsors`/图片直链）。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/scoring.py`：
    - 新增 `score_relevance(topic)` 与 `relevance_threshold` 门禁。
    - Top picks 需通过 `body/evidence/relevance` 三重 gate。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/script_generator.py`：
    - 新增 `build_facts`，输出结构化事实（what/why/how/proof/links）。
    - `generate_script` 改为 facts 驱动并输出 `facts` 字段，行级 references 落盘。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/storyboard_generator.py`：
    - 将 script 行级 `references` 映射到 `shot.reference_assets`。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/runtime.py`：
    - ranking 传入 `topic`；新增 `facts.json` 产物；`run_context.json` 落盘 `topic/ranking_stats`。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/scripts/validate_artifacts_v2.py`：
    - 新增门禁：`script/onepager/storyboard/prompt_bundle` 不含 HTML token。
    - 新增门禁：citation denylist 检查、topic relevance 检查。
  - 修改 contracts/exports/tests：
    - `ArtifactType` 新增 `facts`，`RankedItem` 新增 `relevance_score`。
    - 新增 `tests/v2/test_sanitize.py`，并更新 normalize/scoring/script/runtime/validator 测试。
- 新增/删除文件：
  - 新增：`pipeline_v2/sanitize.py`, `tests/v2/test_sanitize.py`
  - 修改：`core/contracts.py`, `pipeline_v2/{__init__.py,normalize.py,scoring.py,script_generator.py,storyboard_generator.py,runtime.py,report_export.py,prompt_compiler.py}`, `scripts/validate_artifacts_v2.py`, `tests/v2/{test_normalize.py,test_scoring.py,test_script_storyboard_prompt.py,test_runtime_integration.py,test_validate_artifacts_v2.py}`
- 如何验证：
  - `pytest -q tests/v2`
  - `python scripts/e2e_smoke_v2.py --out-dir /tmp/ara_v2_quality_check > /tmp/ara_v2_quality_check/result.json`
  - `python scripts/validate_artifacts_v2.py --run-dir /tmp/ara_v2_quality_check/runs/<run_id> --render-dir /tmp/ara_v2_quality_check/render_jobs/<render_job_id>`
- 已知风险与回滚：
  - 风险：`topic relevance` 阈值偏高时可能减少 Top picks 数量（尤其是 narrow topic + 弱内容源场景）。
  - 回滚：`git revert <this_commit_sha>`。

### Commit: Run-Once Live Workflow & Live-Smoke Validator Guard (New)
- 本次目标：
  - 把“最小可运行链路”统一到单命令 `run-once --mode live`，降低操作复杂度。
  - 补齐 live 模式下 smoke 污染回归测试，防止 fixture 内容误判为真实数据。
- 实际改动：
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/README.md`：
    - `6) 最小可运行命令` 改为优先展示 `run-once --mode live ...`。
    - `8) 自检命令` 改为读取 `run-once` 输出的 `run_dir/render_dir` 并执行 validator。
    - `5.3 抽取质量现状` 补充 extraction 元数据说明。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/tests/v2/test_validate_artifacts_v2.py`：
    - 新增 live 模式拒绝 smoke token 的回归用例（强制 FAIL）。
- 新增/删除文件：
  - 无新增文件。
  - 修改：`README.md`, `tests/v2/test_validate_artifacts_v2.py`
- 如何验证：
  - `pytest -q tests/v2/test_validate_artifacts_v2.py tests/v2/test_runtime_integration.py`
  - `pytest -q tests/v2`
- 已知风险与回滚：
  - 风险：若本机网络不可达，`run-once --mode live` 可能因抓取失败导致 validator 不通过。
  - 回滚：`git revert <this_commit_sha>`。

### Commit: Ranking Gates & Evidence-Rich Onepager (New)
- 本次目标：
  - 把 `ranking` 改成真实质量信号驱动，避免短空壳内容进入 Top picks。
  - 让 onepager 每条 Top pick 输出事实 bullets + 原文引用 bullets。
- 实际改动：
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/scoring.py`：
    - 新增 `body_len` 门槛（默认 `<300` 延后，不进 Top picks，除非短公告且有证据）。
    - 新增 `citation_count==0` 与 `published_recency is None` 降权。
    - reasons 增加 gate/penalty 解释字段。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/normalize.py`：
    - credibility 从常量规则升级为 source+tier+证据密度动态计算。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/report_export.py`：
    - onepager 每条 Top pick 增加 `Facts`（2-4条）与 `Citations`（1-2条）。
  - 修改测试：
    - `tests/v2/test_scoring.py` 新增 `body_len gate` 用例。
    - `tests/v2/test_report_export_notification.py` 断言 `Facts/Citations` 分块输出。
- 新增/删除文件：
  - 无新增文件。
  - 修改：`pipeline_v2/scoring.py`, `pipeline_v2/normalize.py`, `pipeline_v2/report_export.py`, 对应测试文件。
- 如何验证：
  - `pytest -q tests/v2/test_scoring.py tests/v2/test_report_export_notification.py tests/v2/test_runtime_integration.py tests/v2/test_normalize.py`
  - `pytest -q tests/v2`
- 已知风险与回滚：
  - 风险：在弱网络场景下可用内容不足时，更多候选会被 gate 延后，可能导致可选 Top picks 数量下降。
  - 回滚：`git revert <this_commit_sha>`。

### Commit: Live Connectors Enrichment (New)
- 本次目标：
  - 让 `live` 模式抓取结果从“标题级摘要”升级为“可写文案的正文级内容”。
  - 对每条来源记录抽取方法与失败原因，供后续质量门禁使用。
- 实际改动：
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/sources/connectors.py`：
    - GitHub：仓库抓取补充 README 文本 + stars/forks/last_push，release 抓取 release notes 全文与发布时间。
    - HuggingFace：模型/数据集补充 card markdown 抓取（`README.md`），写入 `last_modified/extraction_method`。
    - HackerNews：补充 top comments 摘要并合并进 body。
    - RSS/WebArticle：新增正文抽取链（`trafilatura -> readability -> newspaper3k -> fallback`），落盘 `extraction_method/extraction_error/extraction_failed`。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/tests/v2/test_connectors.py`：
    - 增加 RSS 文章页抽取分支与 extraction 元数据断言。
- 新增/删除文件：
  - 无新增文件。
  - 修改：`sources/connectors.py`, `tests/v2/test_connectors.py`, `README.md`
- 如何验证：
  - `pytest -q tests/v2/test_connectors.py tests/v2/test_runtime_integration.py tests/v2/test_e2e_smoke_command.py`
  - `pytest -q tests/v2`
- 已知风险与回滚：
  - 风险：不同环境可能缺少 `trafilatura/readability/newspaper3k`，会自动降级到 fallback 抽取，质量会受限。
  - 回滚：`git revert <this_commit_sha>`。

### Commit: Live/Smoke Mode Isolation & Run Context (New)
- 本次目标：
  - 将 `live` 与 `smoke` 运行模式解耦，避免 smoke fixture 污染线上输出。
  - 在产物中落盘并展示 `DataMode/ConnectorStats/ExtractionStats`。
- 实际改动：
  - 新增 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/data_mode.py`：
    - 提供 `resolve_data_mode/should_allow_smoke`，默认 `live`，`smoke` 需显式请求。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/runtime.py`：
    - 执行时解析 data mode，写入 `run_context.json`。
    - `materials.json` 与 `onepager.md` 注入 run context（connector/extraction 统计）。
    - `live` 模式下无抓取结果时直接失败，不再静默回退 synthetic。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/pipeline_v2/report_export.py`：
    - onepager 顶部新增 `DataMode/ConnectorStats/ExtractionStats`。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/scripts/validate_artifacts_v2.py`：
    - 增加 `data_mode` 读取与 live fixture 检查（`-smoke`、HN fixture id、`org/repo` 占位 URL）。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/main.py`：
    - `ondemand` 新增 `--mode live|smoke`。
    - 新增同进程命令 `run-once`（执行 run+render+validator 并输出结果）。
  - 修改 `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/scripts/e2e_smoke_v2.py`：
    - 显式注入 `data_mode=smoke`。
- 新增/删除文件：
  - 新增：`pipeline_v2/data_mode.py`
  - 修改：`main.py`, `pipeline_v2/runtime.py`, `pipeline_v2/report_export.py`, `scripts/validate_artifacts_v2.py`, `scripts/e2e_smoke_v2.py`, `tests/v2/test_runtime_integration.py`, `tests/v2/test_data_mode.py`
- 如何验证：
  - `pytest -q tests/v2/test_data_mode.py tests/v2/test_runtime_integration.py tests/v2/test_validate_artifacts_v2.py tests/v2/test_e2e_smoke_command.py`
  - `python scripts/e2e_smoke_v2.py --out-dir /tmp/ara_v2_smoke_mode_check > /tmp/ara_v2_smoke_mode_check/result.json`
- 已知风险与回滚：
  - 风险：`live` 模式在网络不可达时将直接失败，便于暴露抓取问题但会降低离线容错。
  - 回滚：`git revert <this_commit_sha>`。

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
  - `fetch_web_article` 与 `fetch_rss_feed` 使用 `trafilatura -> readability -> newspaper3k -> fallback` 级联抽取。
  - 每条记录写入：`extraction_method/extraction_error/extraction_failed`。
- 引用抽取：
  - `normalize.extract_citations` 会从 metadata、markdown link、正文 URL 提取并去重。
- 去重/聚类：
  - `dedup_exact` + `embed/cluster/merge_cluster` 已具备。

## 6) 最小可运行命令

### 6.1 一条命令跑今日 live 热点（推荐）
```bash
python main.py run-once --mode live --topic "AI agent" --time_window today --tz Asia/Singapore --targets web,mp4 --top-k 3
```

预期输出：
- JSON 包含 `run_id/render_job_id/run_dir/render_dir/validator_ok`
- `validator_ok=true` 时代表本地验收通过
- `onepager.md` 顶部包含 `DataMode: live`

### 6.2 分进程 worker 方式（兼容保留）
```bash
python main.py ondemand --mode live --user-id u1 --topic "AI agent" --time-window today --tz Asia/Singapore --targets web,mp4
python main.py worker-run-next
python main.py worker-render-next
python main.py status --run-id <run_id>
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
python main.py run-once --mode live --topic "AI agent" --time_window today --tz Asia/Singapore --targets web,mp4 --top-k 3 > /tmp/ara_v2_live_run.json
```

```bash
python - <<'PY'
import json, pathlib, subprocess
p = pathlib.Path('/tmp/ara_v2_live_run.json')
d = json.loads(p.read_text())
print('run_id=', d.get('run_id'))
print('validator_ok=', d.get('validator_ok'))
print('run_dir=', d.get('run_dir'))
print('render_dir=', d.get('render_dir'))
PY
```

```bash
python - <<'PY'
import json, pathlib, subprocess
p = pathlib.Path('/tmp/ara_v2_live_run.json')
d = json.loads(p.read_text())
run_dir = d['run_dir']
render_dir = d['render_dir']
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
