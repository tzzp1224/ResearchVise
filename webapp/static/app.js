const state = {
  runId: null,
  source: null,
  documents: {},
  activeDoc: null,
  lastHeartbeatLogTs: 0,
  imageZoom: 1,
  inputMode: "topic",
  videoTimelineLocked: false,
  cacheCandidates: [],
  cacheRequestSeq: 0,
  chatSessionId: "",
  chatTopic: "",
};

const DOC_ORDER = ["knowledge_tree_md", "one_pager_md", "timeline_md", "report_md"];

const el = {
  runIndicator: document.getElementById("runIndicator"),
  statusPill: document.getElementById("statusPill"),
  docStreamState: document.getElementById("docStreamState"),
  videoState: document.getElementById("videoState"),
  researchForm: document.getElementById("researchForm"),
  runButton: document.getElementById("runButton"),
  inputModeRadios: Array.from(document.querySelectorAll("input[name='inputMode']")),
  inputHint: document.getElementById("inputHint"),
  topicInput: document.getElementById("topicInput"),
  arxivUrlInput: document.getElementById("arxivUrlInput"),
  pdfFileInput: document.getElementById("pdfFileInput"),
  maxResultsInput: document.getElementById("maxResultsInput"),
  iterationsInput: document.getElementById("iterationsInput"),
  toolTimeoutInput: document.getElementById("toolTimeoutInput"),
  reactThoughtTimeoutInput: document.getElementById("reactThoughtTimeoutInput"),
  timeBudgetInput: document.getElementById("timeBudgetInput"),
  generateVideoInput: document.getElementById("generateVideoInput"),
  allowCacheHitInput: document.getElementById("allowCacheHitInput"),
  ttsProviderInput: document.getElementById("ttsProviderInput"),
  ttsSpeedInput: document.getElementById("ttsSpeedInput"),
  docTabs: document.getElementById("docTabs"),
  docMeta: document.getElementById("docMeta"),
  docViewer: document.getElementById("docViewer"),
  activityLog: document.getElementById("activityLog"),
  videoLog: document.getElementById("videoLog"),
  videoPlayer: document.getElementById("videoPlayer"),
  videoMeta: document.getElementById("videoMeta"),
  videoBriefPanel: document.getElementById("videoBriefPanel"),
  artifactLinks: document.getElementById("artifactLinks"),
  professionalSources: document.getElementById("professionalSources"),
  communitySources: document.getElementById("communitySources"),
  chatForm: document.getElementById("chatForm"),
  chatInput: document.getElementById("chatInput"),
  chatButton: document.getElementById("chatButton"),
  chatMessages: document.getElementById("chatMessages"),
  imageModal: document.getElementById("imageModal"),
  imageModalBackdrop: document.getElementById("imageModalBackdrop"),
  imageModalTitle: document.getElementById("imageModalTitle"),
  imageModalImg: document.getElementById("imageModalImg"),
  imageModalClose: document.getElementById("imageModalClose"),
  imageZoomIn: document.getElementById("imageZoomIn"),
  imageZoomOut: document.getElementById("imageZoomOut"),
  imageZoomRange: document.getElementById("imageZoomRange"),
  mainLayout: document.getElementById("mainLayout"),
  panelDocs: document.getElementById("panelDocs"),
  panelChat: document.getElementById("panelChat"),
  panelVideo: document.getElementById("panelVideo"),
  leftResizer: document.getElementById("leftResizer"),
  rightResizer: document.getElementById("rightResizer"),
  cacheSuggest: document.getElementById("cacheSuggest"),
  cacheSuggestList: document.getElementById("cacheSuggestList"),
  cacheRefreshBtn: document.getElementById("cacheRefreshBtn"),
};

function timestamp() {
  return new Date().toLocaleTimeString();
}

function artifactUrl(path) {
  return `/api/artifacts/file?path=${encodeURIComponent(path)}`;
}

function appendLog(container, text) {
  const li = document.createElement("li");
  li.textContent = `[${timestamp()}] ${text}`;
  container.prepend(li);
}

function pushActivity(text) {
  appendLog(el.activityLog, text);
}

function pushVideoLog(text) {
  appendLog(el.videoLog, text);
}

function shortError(value) {
  const text = String(value || "").replace(/\s+/g, " ").trim();
  if (!text) {
    return "unknown";
  }
  return text.length > 120 ? `${text.slice(0, 117)}...` : text;
}

function setStatus(text) {
  el.statusPill.textContent = text;
  el.runIndicator.textContent = text;
}

function resetRunView() {
  state.documents = {};
  state.activeDoc = null;
  state.videoTimelineLocked = false;
  state.lastHeartbeatLogTs = 0;
  state.chatSessionId = "";
  state.chatTopic = "";
  el.docTabs.innerHTML = "";
  el.docMeta.textContent = "文档流初始化中...";
  el.docViewer.textContent = "";
  el.activityLog.innerHTML = "";
  el.videoLog.innerHTML = "";
  el.artifactLinks.innerHTML = "";
  el.videoPlayer.removeAttribute("src");
  el.videoPlayer.load();
  el.videoMeta.textContent = "等待视频产物...";
  el.videoBriefPanel.textContent = "等待 video brief 文档流...";
  el.professionalSources.textContent = "待生成";
  el.communitySources.textContent = "待生成";
  el.docStreamState.textContent = "SSE: connected";
  el.videoState.textContent = "Pending";
}

function formatDateTime(value) {
  const text = String(value || "").trim();
  if (!text) {
    return "";
  }
  const d = new Date(text);
  if (Number.isNaN(d.getTime())) {
    return text;
  }
  return d.toLocaleString();
}

function normalizeDocLabel(name) {
  return {
    knowledge_tree_md: "Knowledge Tree",
    one_pager_md: "One Pager",
    timeline_md: "Timeline",
    report_md: "Report",
    video_brief_md: "Video Brief",
  }[name] || name;
}

function renderDocTabs() {
  const names = Object.keys(state.documents)
    .filter((name) => DOC_ORDER.includes(name))
    .sort((a, b) => {
    const ai = DOC_ORDER.indexOf(a);
    const bi = DOC_ORDER.indexOf(b);
    return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi);
  });

  el.docTabs.innerHTML = "";
  if (!names.length) {
    if (state.activeDoc && !DOC_ORDER.includes(state.activeDoc)) {
      state.activeDoc = null;
    }
    return;
  }
  if (!state.activeDoc || !names.includes(state.activeDoc)) {
    state.activeDoc = names[0];
  }
  names.forEach((name) => {
    const doc = state.documents[name];
    const btn = document.createElement("button");
    btn.className = `doc-tab${state.activeDoc === name ? " active" : ""}${doc.done ? " done" : ""}`;
    btn.type = "button";
    btn.textContent = normalizeDocLabel(name);
    btn.onclick = () => {
      state.activeDoc = name;
      renderDocTabs();
      renderDocViewer();
    };
    el.docTabs.appendChild(btn);
  });
}

function renderDocViewer() {
  if (!state.activeDoc || !state.documents[state.activeDoc]) {
    el.docViewer.textContent = "暂无文档内容";
    el.docMeta.textContent = "请等待 SSE 文档流...";
    return;
  }

  const doc = state.documents[state.activeDoc];
  renderMarkdownInto(el.docViewer, doc.content || "", ".doc-viewer .mermaid");
  const status = doc.done ? "完成" : "流式输出中";
  const path = doc.path ? ` · ${doc.path}` : "";
  el.docMeta.textContent = `${normalizeDocLabel(state.activeDoc)} · ${status}${path}`;
}

function escapeHtml(text) {
  return String(text || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function renderMarkdownInto(container, markdownText, mermaidSelector) {
  if (!container) {
    return;
  }
  const raw = String(markdownText || "");
  if (!window.marked || typeof window.marked.parse !== "function") {
    container.textContent = raw;
    return;
  }

  let html = window.marked.parse(raw);
  if (!html) {
    html = `<pre>${escapeHtml(raw)}</pre>`;
  }
  container.innerHTML = html;

  const mermaidCodeBlocks = container.querySelectorAll("pre code.language-mermaid");
  mermaidCodeBlocks.forEach((codeNode) => {
    const wrapper = codeNode.closest("pre");
    const graph = document.createElement("div");
    graph.className = "mermaid";
    graph.textContent = codeNode.textContent || "";
    if (wrapper && wrapper.parentNode) {
      wrapper.parentNode.replaceChild(graph, wrapper);
    }
  });

  if (window.mermaid && typeof window.mermaid.run === "function") {
    try {
      window.mermaid.run({
        querySelector: mermaidSelector,
      });
    } catch (error) {
      pushActivity(`Mermaid 渲染失败: ${shortError(error)}`);
    }
  }
}

function applyImageZoom(value) {
  const zoom = Math.max(0.5, Math.min(3, Number(value) || 1));
  state.imageZoom = zoom;
  if (el.imageModalImg) {
    el.imageModalImg.style.transform = `scale(${zoom})`;
  }
  if (el.imageZoomRange) {
    el.imageZoomRange.value = String(zoom);
  }
}

function openImageModal(src, title) {
  if (!src || !el.imageModal || !el.imageModalImg) {
    return;
  }
  el.imageModalImg.src = src;
  el.imageModalTitle.textContent = title || "Image Preview";
  el.imageModal.hidden = false;
  document.body.style.overflow = "hidden";
  applyImageZoom(1);
}

function closeImageModal() {
  if (!el.imageModal || !el.imageModalImg) {
    return;
  }
  el.imageModal.hidden = true;
  el.imageModalImg.removeAttribute("src");
  document.body.style.overflow = "";
}

function bindImagePreviewFor(container) {
  if (!container) {
    return;
  }
  container.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof Element)) {
      return;
    }

    const imageNode = target.closest("img");
    if (imageNode instanceof HTMLImageElement) {
      event.preventDefault();
      event.stopPropagation();
      openImageModal(imageNode.currentSrc || imageNode.src, imageNode.alt || "Image Preview");
      return;
    }

    const svgNode = target.closest(".mermaid svg");
    if (svgNode instanceof SVGElement) {
      event.preventDefault();
      event.stopPropagation();
      try {
        const svgText = new XMLSerializer().serializeToString(svgNode);
        const dataUrl = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svgText)}`;
        openImageModal(dataUrl, "Mermaid Diagram");
      } catch (error) {
        pushActivity(`Mermaid 放大失败: ${shortError(error)}`);
      }
    }
  });
}

function bindDocumentImagePreview() {
  bindImagePreviewFor(el.docViewer);
}

function bindImageModalControls() {
  if (!el.imageModal) {
    return;
  }
  el.imageModalClose?.addEventListener("click", closeImageModal);
  el.imageModalBackdrop?.addEventListener("click", closeImageModal);
  el.imageZoomRange?.addEventListener("input", (event) => {
    const value = event && event.target && "value" in event.target ? event.target.value : state.imageZoom;
    applyImageZoom(value);
  });
  el.imageZoomIn?.addEventListener("click", () => {
    applyImageZoom(state.imageZoom + 0.1);
  });
  el.imageZoomOut?.addEventListener("click", () => {
    applyImageZoom(state.imageZoom - 0.1);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !el.imageModal.hidden) {
      closeImageModal();
    }
  });
}

function parseSecondsFromDuration(text) {
  const value = String(text || "").trim().toLowerCase();
  if (!value) {
    return 0;
  }
  const clock = value.match(/^(\d{1,2}):(\d{2})(?::(\d{2}))?$/);
  if (clock) {
    const h = clock[3] ? Number(clock[1]) : 0;
    const m = clock[3] ? Number(clock[2]) : Number(clock[1]);
    const s = clock[3] ? Number(clock[3]) : Number(clock[2]);
    return Math.max(0, h * 3600 + m * 60 + s);
  }
  const numberMatch = value.match(/(\d+(?:\.\d+)?)/);
  if (!numberMatch) {
    return 0;
  }
  const num = Number(numberMatch[1]);
  if (!Number.isFinite(num) || num <= 0) {
    return 0;
  }
  if (value.includes("min")) {
    return Math.round(num * 60);
  }
  return Math.round(num);
}

function formatTime(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0));
  const m = String(Math.floor(total / 60)).padStart(2, "0");
  const s = String(total % 60).padStart(2, "0");
  return `${m}:${s}`;
}

function parseVideoBriefSegments(markdown) {
  const lines = String(markdown || "").split(/\r?\n/);
  const segments = [];
  let current = null;
  for (const line of lines) {
    const heading = line.match(/^###\s+(?:\d+\s*[).:]?\s*)?(.+)$/);
    if (heading) {
      if (current) {
        segments.push(current);
      }
      current = { title: heading[1].trim(), durationSec: 0, startSec: null, summary: "" };
      continue;
    }
    if (!current) {
      continue;
    }
    const durationMatch = line.match(/^-+\s*(?:Duration|时长)\s*:\s*(.+)$/i);
    if (durationMatch) {
      current.durationSec = parseSecondsFromDuration(durationMatch[1]);
      continue;
    }
    const startMatch = line.match(/^-+\s*(?:Start|开始时间)\s*:\s*(.+)$/i);
    if (startMatch) {
      const startSec = parseSecondsFromDuration(startMatch[1]);
      current.startSec = Number.isFinite(startSec) ? startSec : null;
      continue;
    }
    const plain = String(line || "").trim();
    if (plain && !plain.startsWith("-") && !current.summary) {
      current.summary = plain.replace(/^>\s*/, "");
    }
  }
  if (current) {
    segments.push(current);
  }

  let cursor = 0;
  return segments.map((item) => {
    const start = item.startSec != null ? item.startSec : cursor;
    const duration = item.durationSec > 0 ? item.durationSec : 30;
    cursor = start + duration;
    return { ...item, startSec: start };
  });
}

function renderVideoTimeline(items, emptyText) {
  if (!el.videoBriefPanel) {
    return;
  }
  if (!Array.isArray(items) || !items.length) {
    el.videoBriefPanel.textContent = emptyText;
    return;
  }

  const rows = items
    .map(
      (item) =>
        `<div class="brief-row">
          <button type="button" class="brief-jump" data-seconds="${item.startSec || 0}">${formatTime(item.startSec || 0)}</button>
          <div>
            <div class="brief-title">${escapeHtml(item.title || "Untitled")}</div>
            <div class="brief-main">${escapeHtml(item.summary || "")}</div>
          </div>
        </div>`,
    )
    .join("");
  el.videoBriefPanel.innerHTML = rows;
}

function renderVideoBriefPanel(markdown) {
  const text = String(markdown || "").trim();
  if (!text) {
    renderVideoTimeline([], "等待 video brief 文档流...");
    return;
  }
  const segments = parseVideoBriefSegments(text);
  if (!segments.length) {
    el.videoBriefPanel.textContent = text.slice(0, 4000);
    return;
  }
  renderVideoTimeline(segments, "等待 video brief 文档流...");
}

function parseSlidePlanSlides(plan) {
  const slides = Array.isArray(plan && plan.slides) ? plan.slides : [];
  let cursor = 0;
  return slides.map((slide, index) => {
    const duration = Math.max(1, Number(slide.duration_sec || 0) || 18);
    const title = String(slide.title || `Slide ${index + 1}`).trim();
    const bullets = Array.isArray(slide.bullets) ? slide.bullets : [];
    const primary = String(bullets[0] || slide.notes || "").replace(/^\[[^\]]+\]\s*/, "").trim();
    const item = {
      title: `Slide ${index + 1}: ${title}`,
      summary: primary || "主要内容见该页讲解。",
      startSec: cursor,
    };
    cursor += duration;
    return item;
  });
}

function parseNarrationScriptSlides(markdown) {
  const lines = String(markdown || "").split(/\r?\n/);
  const slides = [];
  let current = null;
  for (const line of lines) {
    const heading = line.match(/^##\s+Slide\s+(\d+):\s+(.+?)\s+\((\d+)s\)\s*$/i);
    if (heading) {
      if (current) {
        slides.push(current);
      }
      current = {
        index: Number(heading[1]),
        title: `Slide ${heading[1]}: ${heading[2].trim()}`,
        durationSec: Number(heading[3]),
        summary: "",
      };
      continue;
    }
    if (!current) {
      continue;
    }
    const plain = String(line || "").trim();
    if (plain && !current.summary) {
      current.summary = plain;
    }
  }
  if (current) {
    slides.push(current);
  }

  let cursor = 0;
  return slides.map((slide) => {
    const item = {
      title: slide.title,
      summary: slide.summary || "主要内容见该页讲解。",
      startSec: cursor,
    };
    cursor += Math.max(1, Number(slide.durationSec || 0) || 18);
    return item;
  });
}

async function fetchArtifactJson(path) {
  if (!path) {
    return null;
  }
  try {
    const response = await fetch(artifactUrl(path));
    if (!response.ok) {
      return null;
    }
    return await response.json();
  } catch {
    return null;
  }
}

async function fetchArtifactText(path) {
  if (!path) {
    return "";
  }
  try {
    const response = await fetch(artifactUrl(path));
    if (!response.ok) {
      return "";
    }
    return await response.text();
  } catch {
    return "";
  }
}

async function renderSlideTimelineFromArtifacts(videoArtifact) {
  if (!videoArtifact || !el.videoBriefPanel) {
    return false;
  }

  const metadata = await fetchArtifactJson(videoArtifact.metadata_path);
  if (metadata && metadata.flow && metadata.flow.plan_json) {
    const plan = await fetchArtifactJson(metadata.flow.plan_json);
    const slides = parseSlidePlanSlides(plan || {});
    if (slides.length) {
      state.videoTimelineLocked = true;
      renderVideoTimeline(slides, "等待视频页级时间轴...");
      return true;
    }
  }

  const narrationScriptPath =
    (videoArtifact && videoArtifact.narration_script_path) ||
    (metadata && metadata.flow && metadata.flow.narration_script_path) ||
    "";
  if (narrationScriptPath) {
    const script = await fetchArtifactText(narrationScriptPath);
    const slides = parseNarrationScriptSlides(script);
    if (slides.length) {
      state.videoTimelineLocked = true;
      renderVideoTimeline(slides, "等待视频页级时间轴...");
      return true;
    }
  }

  return false;
}

function renderSourceGroups(groups) {
  const professional = (groups && groups.professional && groups.professional.breakdown) || {};
  const community = (groups && groups.community && groups.community.breakdown) || {};
  const render = (obj) => {
    const entries = Object.entries(obj);
    if (!entries.length) {
      return "无";
    }
    return entries.map(([k, v]) => `${k}: ${v}`).join(" · ");
  };
  if (el.professionalSources) {
    el.professionalSources.textContent = render(professional);
  }
  if (el.communitySources) {
    el.communitySources.textContent = render(community);
  }
}

function bindVideoBriefJumps() {
  if (!el.videoBriefPanel || !el.videoPlayer) {
    return;
  }
  el.videoBriefPanel.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof Element)) {
      return;
    }
    const jump = target.closest(".brief-jump");
    if (!(jump instanceof HTMLButtonElement)) {
      return;
    }
    const seconds = Number(jump.dataset.seconds || "0");
    if (!Number.isFinite(seconds) || seconds < 0) {
      return;
    }
    try {
      el.videoPlayer.currentTime = seconds;
      void el.videoPlayer.play().catch(() => {});
    } catch {
      // no-op
    }
  });
}

function applyInputMode(mode) {
  state.inputMode = mode || "topic";
  const isTopic = state.inputMode === "topic";
  const isArxiv = state.inputMode === "arxiv_url";
  const isPdf = state.inputMode === "pdf_upload";

  if (el.topicInput) {
    el.topicInput.disabled = !isTopic;
    el.topicInput.required = isTopic;
  }
  if (el.arxivUrlInput) {
    el.arxivUrlInput.disabled = !isArxiv;
    el.arxivUrlInput.required = isArxiv;
  }
  if (el.pdfFileInput) {
    el.pdfFileInput.disabled = !isPdf;
    el.pdfFileInput.required = isPdf;
    if (!isPdf) {
      el.pdfFileInput.value = "";
    }
  }
  if (el.inputHint) {
    if (isTopic) {
      el.inputHint.textContent = "关键词模式：输入研究主题后开始。";
    } else if (isArxiv) {
      el.inputHint.textContent = "arXiv 模式：只输入 arXiv 链接，系统将按论文内容扩展检索。";
    } else {
      el.inputHint.textContent = "PDF 模式：上传论文 PDF，系统将先解析文档再执行多源研究。";
    }
  }
  if (!isTopic && el.cacheSuggest) {
    el.cacheSuggest.hidden = true;
  }
}

function renderCacheSuggestions(candidates) {
  if (!el.cacheSuggest || !el.cacheSuggestList) {
    return;
  }
  const rows = Array.isArray(candidates) ? candidates : [];
  state.cacheCandidates = rows;
  if (!rows.length || state.inputMode !== "topic") {
    el.cacheSuggestList.innerHTML = "";
    el.cacheSuggest.hidden = true;
    return;
  }

  el.cacheSuggestList.innerHTML = rows
    .map((item) => {
      const score = Number(item.score || 0);
      const scoreText = Number.isFinite(score) ? score.toFixed(3) : "-";
      const overlap = Number(item.topic_overlap || 0);
      const overlapText = Number.isFinite(overlap) ? overlap.toFixed(2) : "-";
      const createdAt = formatDateTime(item.created_at);
      const summary = String(item.summary || "").trim();
      const stats = [
        createdAt ? `创建: ${createdAt}` : "",
        `相关度: ${scoreText}`,
        `主题重合: ${overlapText}`,
        `结果数: ${Number(item.search_results_count || 0)}`,
        `事实: ${Number(item.facts_count || 0)}`,
        item.has_video ? "含视频" : "无视频",
      ]
        .filter(Boolean)
        .join(" · ");
      return `
        <article class="cache-item">
          <div class="cache-item-head">
            <div class="cache-item-topic">${escapeHtml(item.topic || "Cached Topic")}</div>
            <button
              type="button"
              class="cache-open-btn"
              data-snapshot-path="${escapeHtml(item.snapshot_path || "")}"
            >
              直接打开
            </button>
          </div>
          <div class="cache-item-meta">${escapeHtml(stats)}</div>
          <div class="cache-item-summary">${escapeHtml(summary || "该缓存包含可直接浏览的文档与视频。")}</div>
        </article>
      `;
    })
    .join("");
  el.cacheSuggest.hidden = false;
}

async function fetchCacheSuggestions(topic, { force = false } = {}) {
  const query = String(topic || "").trim();
  if (state.inputMode !== "topic" || query.length < 2) {
    renderCacheSuggestions([]);
    return;
  }
  const seq = ++state.cacheRequestSeq;
  if (el.cacheRefreshBtn) {
    el.cacheRefreshBtn.disabled = true;
  }
  try {
    const response = await fetch(
      `/api/cache/suggestions?topic=${encodeURIComponent(query)}&top_k=4${force ? "&score_threshold=0.30" : ""}`,
    );
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const payload = await response.json();
    if (seq !== state.cacheRequestSeq) {
      return;
    }
    renderCacheSuggestions(payload.candidates || []);
    if ((payload.candidates || []).length) {
      pushActivity(`找到 ${payload.candidates.length} 条相似缓存，可直接打开。`);
    }
  } catch (error) {
    if (seq === state.cacheRequestSeq) {
      renderCacheSuggestions([]);
    }
    pushActivity(`缓存候选加载失败: ${shortError(error)}`);
  } finally {
    if (el.cacheRefreshBtn) {
      el.cacheRefreshBtn.disabled = false;
    }
  }
}

let cacheSuggestTimer = null;
function scheduleCacheSuggestions() {
  if (cacheSuggestTimer) {
    clearTimeout(cacheSuggestTimer);
  }
  cacheSuggestTimer = setTimeout(() => {
    cacheSuggestTimer = null;
    fetchCacheSuggestions(el.topicInput ? el.topicInput.value : "");
  }, 280);
}

async function hydrateDocumentsFromWrittenFiles(writtenFiles) {
  const files = writtenFiles && typeof writtenFiles === "object" ? writtenFiles : {};
  const orderedKeys = [
    "knowledge_tree_md",
    "one_pager_md",
    "timeline_md",
    "report_md",
    "video_brief_md",
  ];
  for (const key of orderedKeys) {
    const path = String(files[key] || "").trim();
    if (!path) {
      continue;
    }
    addOrUpdateDocument(key, { path, content: "", done: false });
    const content = await fetchArtifactText(path);
    addOrUpdateDocument(key, { path, content, done: true });
  }
}

async function openCachedResult(snapshotPath) {
  const path = String(snapshotPath || "").trim();
  if (!path) {
    pushActivity("缓存打开失败: 缺少 snapshot path");
    return;
  }

  closeEventSource();
  setStatus("Loading Cache");
  el.runButton.disabled = true;
  resetRunView();

  try {
    const response = await fetch("/api/cache/open", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ snapshot_path: path }),
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const payload = await response.json();
    const result = payload && payload.result ? payload.result : {};
    if (!result || typeof result !== "object") {
      throw new Error("invalid cache result payload");
    }
    await hydrateDocumentsFromWrittenFiles(result.written_files || {});
    handleResult(result);
    setStatus("Cache Loaded");
    el.docStreamState.textContent = "CACHE: loaded";
    pushActivity(`已直接打开缓存内容: ${result.topic || "unknown topic"}`);
  } catch (error) {
    setStatus("Failed");
    pushActivity(`打开缓存失败: ${shortError(error)}`);
  } finally {
    el.runButton.disabled = false;
  }
}

function addOrUpdateDocument(name, partial) {
  if (!state.documents[name]) {
    state.documents[name] = { content: "", done: false, path: "" };
  }
  state.documents[name] = { ...state.documents[name], ...partial };
  if (name === "video_brief_md") {
    if (!state.videoTimelineLocked) {
      renderVideoBriefPanel(state.documents[name].content || "");
    }
  }
  if (!state.activeDoc) {
    state.activeDoc = name;
  }
  renderDocTabs();
  if (state.activeDoc === name && DOC_ORDER.includes(name)) {
    renderDocViewer();
  }
}

function addArtifactLink(label, path) {
  if (!path) {
    return;
  }
  const link = document.createElement("a");
  link.href = artifactUrl(path);
  link.target = "_blank";
  link.rel = "noreferrer";
  link.textContent = label;
  el.artifactLinks.appendChild(link);
}

function handleResult(result) {
  pushActivity(`完成：results=${result.search_results_count}, facts=${result.facts_count}, cache_hit=${result.cache_hit}`);
  state.chatSessionId = String(result.session_id || "").trim();
  state.chatTopic = String(result.topic || result.input_topic || "").trim();
  renderSourceGroups(result.source_groups || {});

  const written = result.written_files || {};
  Object.entries(written).forEach(([name, path]) => {
    addArtifactLink(name, path);
  });

  const videoArtifact = result.video_artifact || {};
  if (videoArtifact.output_path) {
    el.videoPlayer.src = artifactUrl(videoArtifact.output_path);
    el.videoMeta.textContent = videoArtifact.output_path;
    el.videoState.textContent = "Ready";
    void renderSlideTimelineFromArtifacts(videoArtifact);
  }

  if (result.video_error) {
    el.videoState.textContent = "Failed";
    pushVideoLog(`视频失败: ${result.video_error}`);
  }

  const recommendations = result.quality_recommendations || [];
  if (recommendations.length) {
    recommendations.slice(0, 3).forEach((item) => {
      pushActivity(`Critic: ${item}`);
    });
  }
}

function closeEventSource() {
  if (state.source) {
    state.source.close();
    state.source = null;
  }
}

function registerListener(source, name, handler) {
  source.addEventListener(name, (event) => {
    let payload = {};
    try {
      payload = JSON.parse(event.data || "{}");
    } catch {
      payload = {};
    }
    handler(payload);
  });
}

function openEventStream(runId) {
  closeEventSource();

  const source = new EventSource(`/api/research/${encodeURIComponent(runId)}/events`);
  state.source = source;

  registerListener(source, "connected", () => {
    setStatus("Running");
    pushActivity(`SSE connected (${runId})`);
  });
  registerListener(source, "heartbeat", (p) => {
    const stage = String(p.last_event || "working");
    setStatus(`Running · ${stage} · ${p.events || 0} events`);
    const now = Date.now();
    if (now - state.lastHeartbeatLogTs >= 12000) {
      state.lastHeartbeatLogTs = now;
      pushActivity(`运行中: stage=${stage}, elapsed=${Math.round(Number(p.elapsed_sec || 0))}s`);
    }
  });

  registerListener(source, "research_started", (p) => {
    state.chatTopic = String(p.input_topic || "").trim() || state.chatTopic;
    pushActivity(`Research: topic=${p.input_topic || ""}`);
  });
  registerListener(source, "input_ingestion_started", (p) => {
    pushActivity(`Input ingest: ${p.mode || "unknown"} started`);
  });
  registerListener(source, "input_ingestion_note", (p) => {
    pushActivity(`Input ingest: ${p.note || "..."}`);
  });
  registerListener(source, "input_ingestion_completed", (p) => {
    state.chatTopic = String(p.derived_topic || "").trim() || state.chatTopic;
    pushActivity(`Input ingest done: topic=${p.derived_topic || ""}, seed=${p.seed_count || 0}`);
  });
  registerListener(source, "planner_started", () => pushActivity("Planner: 分析主题与重写查询"));
  registerListener(source, "planner_completed", (p) => pushActivity(`Planner: ${p.planned_topic || "done"}`));
  registerListener(source, "planner_failed", (p) => pushActivity(`Planner failed: ${shortError(p.error)}`));
  registerListener(source, "cache_hit", (p) => pushActivity(`Cache hit: score=${Number(p.cache_score || 0).toFixed(4)}`));
  registerListener(source, "cache_miss", () => pushActivity("Cache miss: 执行 ReAct 搜索"));
  registerListener(source, "search_started", () => pushActivity("Search: 开始多源检索"));
  registerListener(source, "search_failed", (p) => {
    pushActivity(`Search failed: ${shortError(p.error)}`);
  });
  registerListener(source, "search_agent_started", (p) => {
    pushActivity(
      `ReAct: 规划 ${Array.isArray(p.plan) ? p.plan.length : 0} 个维度，最多 ${p.max_iterations || "?"} 轮`,
    );
  });
  registerListener(source, "search_iteration_started", (p) => {
    const coverage = p.coverage || {};
    pushActivity(
      `ReAct ${p.iteration || "?"}/${p.max_iterations || "?"} · results=${coverage.result_count || 0}`,
    );
  });
  registerListener(source, "search_tool_call", (p) => {
    pushActivity(`Tool: ${p.tool} → ${p.query || ""}`);
  });
  registerListener(source, "search_tool_result", (p) => {
    const breakdown = p.source_breakdown || {};
    const sourceText = Object.keys(breakdown).length
      ? ` [${Object.entries(breakdown).map(([k, v]) => `${k}:${v}`).join(", ")}]`
      : "";
    pushActivity(`Tool done: ${p.tool} = ${p.result_count || 0}${sourceText}`);
  });
  registerListener(source, "search_tool_error", (p) => {
    pushActivity(`Tool error: ${p.tool} · ${shortError(p.error)}`);
  });
  registerListener(source, "search_thought_timeout", (p) => {
    pushActivity(`ReAct thought timeout: ${p.timeout_sec || "?"}s`);
  });
  registerListener(source, "search_thought_error", (p) => {
    pushActivity(`ReAct thought error: ${shortError(p.error)}`);
  });
  registerListener(source, "search_tool_disabled", (p) => {
    pushActivity(`Tool disabled: ${p.tool} (${p.reason || "cooldown"})`);
  });
  registerListener(source, "search_bootstrap_attempt", (p) => {
    pushActivity(`Bootstrap: ${p.tool} → ${p.query || ""}`);
  });
  registerListener(source, "search_bootstrap_completed", (p) => {
    const coverage = p.coverage || {};
    pushActivity(
      `Bootstrap done: results=${coverage.result_count || 0}, sources=${(coverage.source_coverage || []).length || 0}`,
    );
  });
  registerListener(source, "search_fallback_attempt", (p) => {
    pushActivity(`Fallback: ${p.tool} → ${p.query || ""}`);
  });
  registerListener(source, "search_topup_started", (p) => {
    const coverage = p.coverage || {};
    pushActivity(
      `Top-up: results=${coverage.result_count || 0}, sources=${(coverage.source_coverage || []).length || 0}`,
    );
  });
  registerListener(source, "search_topup_completed", (p) => {
    pushActivity(`Top-up done: +${p.added_results || 0} results`);
  });
  registerListener(source, "search_enrichment_started", (p) => {
    const sources = Array.isArray(p.enrich_sources) ? p.enrich_sources.join(", ") : "";
    pushActivity(`Enrichment: 补搜来源 ${sources || "(auto)"}`);
  });
  registerListener(source, "search_enrichment_completed", (p) => {
    pushActivity(`Enrichment done: +${p.added_results || 0} results`);
  });
  registerListener(source, "search_filter_completed", (p) => {
    const signals = Array.isArray(p.signals) && p.signals.length ? ` (${p.signals.join(", ")})` : "";
    pushActivity(`Filter: removed=${p.removed || 0}, kept=${p.kept || 0}${signals}`);
  });
  registerListener(source, "deep_enrichment_started", (p) => {
    pushActivity(
      `Deep enrichment: candidates=${p.candidate_count || 0}, per_source=${p.max_items_per_source || 0}`,
    );
  });
  registerListener(source, "deep_enrichment_completed", (p) => {
    const summary = p.summary || {};
    const sourceText = summary.sources
      ? Object.entries(summary.sources)
          .map(([k, v]) => `${k}:${v}`)
          .join(", ")
      : "";
    pushActivity(
      `Deep enrichment done: ${summary.enriched || 0}/${summary.attempted || 0}${sourceText ? ` [${sourceText}]` : ""}`,
    );
  });
  registerListener(source, "deep_enrichment_failed", (p) => {
    pushActivity(`Deep enrichment failed: ${shortError(p.error)}`);
  });
  registerListener(source, "seed_input_merged", (p) => {
    pushActivity(`Seed merged: +${p.added || 0}, total=${p.merged_total || 0}`);
  });
  registerListener(source, "search_coverage_update", (p) => {
    const coverage = p.coverage || {};
    const missing = Array.isArray(coverage.missing_dimensions) ? coverage.missing_dimensions.length : 0;
    pushActivity(
      `Coverage: src=${(coverage.source_coverage || []).length || 0}, missing_dim=${missing}`,
    );
  });
  registerListener(source, "search_agent_completed", (p) => {
    pushActivity(`ReAct done: ${p.result_count || 0} results`);
  });
  registerListener(source, "search_agent_terminated", (p) => {
    pushActivity(`ReAct terminated: ${p.reason || "stopped"}`);
  });
  registerListener(source, "search_completed", (p) => {
    const total = (p.aggregated_summary || {}).total || 0;
    pushActivity(`Search done: total=${total}`);
  });
  registerListener(source, "search_fallback_started", () => {
    pushActivity("Aggregator fallback: running...");
  });
  registerListener(source, "search_fallback_completed", (p) => {
    const total = (p.aggregated_summary || {}).total || 0;
    pushActivity(`Aggregator fallback done: total=${total}`);
  });
  registerListener(source, "analysis_started", () => pushActivity("Analyst: 提取事实"));
  registerListener(source, "analysis_completed", (p) => pushActivity(`Analyst done: facts=${p.facts_count || 0}`));
  registerListener(source, "analysis_failed", (p) => pushActivity(`Analyst failed: ${shortError(p.error)}`));
  registerListener(source, "analysis_fallback_applied", (p) => {
    pushActivity(`Analyst fallback applied: facts=${p.facts_count || 0}`);
  });
  registerListener(source, "content_generation_started", () => pushActivity("Content: 生成多源产物"));
  registerListener(source, "content_generation_failed", (p) => pushActivity(`Content failed: ${shortError(p.error)}`));
  registerListener(source, "critic_completed", (p) => {
    pushActivity(`Critic: pass=${Boolean(p.quality_gate_pass)}`);
  });
  registerListener(source, "critic_failed", (p) => {
    pushActivity(`Critic failed: ${shortError(p.error)}`);
  });

  registerListener(source, "documents_exported", () => {
    el.docStreamState.textContent = "SSE: streaming docs";
    pushActivity("Documents: exported, streaming markdown...");
  });

  registerListener(source, "document_start", (p) => {
    addOrUpdateDocument(p.name, { path: p.path, content: "", done: false });
    pushActivity(`Document start: ${normalizeDocLabel(p.name)}`);
  });

  registerListener(source, "document_chunk", (p) => {
    const current = state.documents[p.name] || { content: "", done: false, path: "" };
    addOrUpdateDocument(p.name, { ...current, content: `${current.content}${p.chunk || ""}` });
  });

  registerListener(source, "document_done", (p) => {
    const current = state.documents[p.name] || { content: "", done: false, path: "" };
    addOrUpdateDocument(p.name, { ...current, done: true, path: p.path || current.path });
    pushActivity(`Document done: ${normalizeDocLabel(p.name)}`);
  });

  registerListener(source, "video_generation_started", () => {
    el.videoState.textContent = "Generating";
    pushVideoLog("开始生成视频");
  });

  registerListener(source, "video_progress", (p) => {
    pushVideoLog(p.message || "video step");
  });

  registerListener(source, "video_generation_completed", (p) => {
    const artifact = p.video_artifact || {};
    if (artifact.output_path) {
      el.videoPlayer.src = artifactUrl(artifact.output_path);
      el.videoMeta.textContent = artifact.output_path;
      el.videoState.textContent = "Ready";
      addArtifactLink("video.mp4", artifact.output_path);
      void renderSlideTimelineFromArtifacts(artifact);
    }
    pushVideoLog("视频生成完成");
  });

  registerListener(source, "video_generation_failed", (p) => {
    el.videoState.textContent = "Failed";
    pushVideoLog(`视频失败: ${p.error || "unknown"}`);
  });

  registerListener(source, "video_ready", (p) => {
    if (p.output_path) {
      el.videoPlayer.src = artifactUrl(p.output_path);
      el.videoMeta.textContent = p.output_path;
      el.videoState.textContent = "Ready";
      addArtifactLink("video.mp4", p.output_path);
    }
    if (p.metadata_path || p.narration_script_path) {
      void renderSlideTimelineFromArtifacts(p);
    }
  });

  registerListener(source, "result", (p) => handleResult(p));

  registerListener(source, "job_failed", (p) => {
    setStatus("Failed");
    pushActivity(`任务失败: ${p.error || "unknown"}`);
  });

  registerListener(source, "job_completed", () => {
    setStatus("Completed");
    el.docStreamState.textContent = "SSE: done";
  });

  registerListener(source, "stream_end", (p) => {
    if (p.status === "completed") {
      setStatus("Completed");
      el.docStreamState.textContent = "SSE: done";
    } else if (p.status === "failed") {
      setStatus("Failed");
      el.docStreamState.textContent = "SSE: error";
    }
    closeEventSource();
    el.runButton.disabled = false;
  });

  source.onerror = () => {
    pushActivity("SSE 连接中断");
  };
}

async function startResearch(event) {
  event.preventDefault();
  const mode = state.inputMode || "topic";
  const topic = (el.topicInput && el.topicInput.value.trim()) || "";
  const arxivUrl = (el.arxivUrlInput && el.arxivUrlInput.value.trim()) || "";
  const pdfFile = el.pdfFileInput && el.pdfFileInput.files && el.pdfFileInput.files[0];

  const sources = Array.from(document.querySelectorAll(".sources-fieldset input[type='checkbox']"))
    .filter((node) => node.checked)
    .map((node) => node.value);

  const configPayload = {
    max_results: Number(el.maxResultsInput.value || 8),
    search_max_iterations: Number(el.iterationsInput.value || 2),
    search_tool_timeout_sec: Number(el.toolTimeoutInput.value || 6),
    react_thought_timeout_sec: Number(el.reactThoughtTimeoutInput.value || 5),
    search_time_budget_sec: Number(el.timeBudgetInput.value || 45),
    sources,
    generate_video: el.generateVideoInput.checked,
    tts_provider: el.ttsProviderInput.value,
    tts_speed: Number(el.ttsSpeedInput.value || 1.25),
    allow_cache_hit: Boolean(el.allowCacheHitInput && el.allowCacheHitInput.checked),
    enable_critic_gate: false,
  };
  let endpoint = "/api/research";
  let fetchOptions = {};

  if (mode === "topic") {
    if (!topic) {
      pushActivity("请输入研究关键词");
      return;
    }
    fetchOptions = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...configPayload, topic }),
    };
  } else if (mode === "arxiv_url") {
    if (!arxivUrl) {
      pushActivity("请输入 arXiv 链接");
      return;
    }
    fetchOptions = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...configPayload, arxiv_url: arxivUrl }),
    };
  } else if (mode === "pdf_upload") {
    if (!pdfFile) {
      pushActivity("请选择 PDF 文件");
      return;
    }
    endpoint = "/api/research/upload";
    const form = new FormData();
    form.append("file", pdfFile);
    form.append("config_json", JSON.stringify(configPayload));
    fetchOptions = {
      method: "POST",
      body: form,
    };
  } else {
    pushActivity(`未知输入模式: ${mode}`);
    return;
  }

  setStatus("Starting");
  el.runButton.disabled = true;
  resetRunView();
  closeEventSource();

  try {
    const response = await fetch(endpoint, fetchOptions);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    state.runId = data.run_id;
    pushActivity(`任务创建: ${state.runId}`);
    renderCacheSuggestions([]);
    openEventStream(state.runId);
  } catch (error) {
    setStatus("Failed");
    el.runButton.disabled = false;
    pushActivity(`启动失败: ${error.message || error}`);
  }
}

function appendChatMessage(role, text, citations = []) {
  const box = document.createElement("div");
  box.className = `message ${role}`;

  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = role === "user" ? "你" : "Agent";

  const body = document.createElement("div");
  body.textContent = text;

  box.appendChild(meta);
  box.appendChild(body);

  if (role === "assistant" && citations.length) {
    const cite = document.createElement("div");
    cite.className = "meta";
    cite.style.marginTop = "6px";
    cite.textContent = `Citations: ${citations.slice(0, 3).map((c) => c.source || "src").join(", ")}`;
    box.appendChild(cite);
  }

  el.chatMessages.appendChild(box);
  el.chatMessages.scrollTop = el.chatMessages.scrollHeight;
}

async function sendChat(event) {
  event.preventDefault();

  const question = el.chatInput.value.trim();
  if (!question) {
    return;
  }

  appendChatMessage("user", question);
  el.chatInput.value = "";
  el.chatButton.disabled = true;

  try {
    const topic = String(state.chatTopic || (el.topicInput && el.topicInput.value) || "").trim();
    const sessionId = String(state.chatSessionId || "").trim();
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        topic: topic || undefined,
        session_id: sessionId || undefined,
        top_k: 6,
        use_hybrid: true,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    if (data && data.session_id) {
      state.chatSessionId = String(data.session_id).trim();
    }
    appendChatMessage("assistant", data.answer || "(empty)", data.citations || []);
  } catch (error) {
    appendChatMessage("assistant", `请求失败: ${error.message || error}`);
  } finally {
    el.chatButton.disabled = false;
  }
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function getColumnWidths() {
  const style = getComputedStyle(document.documentElement);
  const left = Number.parseFloat(style.getPropertyValue("--left-col-width")) || 31;
  const center = Number.parseFloat(style.getPropertyValue("--center-col-width")) || 39;
  const right = Number.parseFloat(style.getPropertyValue("--right-col-width")) || 30;
  return { left, center, right };
}

function setColumnWidths({ left, center, right }, persist = true) {
  const total = left + center + right;
  if (!Number.isFinite(total) || total <= 0) {
    return;
  }
  const nextLeft = (left / total) * 100;
  const nextCenter = (center / total) * 100;
  const nextRight = (right / total) * 100;
  document.documentElement.style.setProperty("--left-col-width", `${nextLeft.toFixed(3)}`);
  document.documentElement.style.setProperty("--center-col-width", `${nextCenter.toFixed(3)}`);
  document.documentElement.style.setProperty("--right-col-width", `${nextRight.toFixed(3)}`);
  if (persist) {
    try {
      localStorage.setItem(
        "scholarstudio:layout",
        JSON.stringify({ left: nextLeft, center: nextCenter, right: nextRight }),
      );
    } catch {
      // ignore storage errors
    }
  }
}

function restoreColumnWidths() {
  try {
    const raw = localStorage.getItem("scholarstudio:layout");
    if (!raw) {
      return;
    }
    const data = JSON.parse(raw);
    if (!data || typeof data !== "object") {
      return;
    }
    const left = Number(data.left);
    const center = Number(data.center);
    const right = Number(data.right);
    if ([left, center, right].every((n) => Number.isFinite(n) && n > 0)) {
      setColumnWidths({ left, center, right }, false);
    }
  } catch {
    // ignore
  }
}

function bindResizer(resizer, side) {
  if (!resizer || !el.mainLayout) {
    return;
  }
  const minPanel = 18;
  const maxPanel = 64;

  resizer.addEventListener("mousedown", (event) => {
    event.preventDefault();
    const containerRect = el.mainLayout.getBoundingClientRect();
    const startX = event.clientX;
    const start = getColumnWidths();
    document.body.classList.add("resizing-panels");

    function onMove(moveEvent) {
      const deltaPx = moveEvent.clientX - startX;
      const deltaPct = (deltaPx / containerRect.width) * 100;
      if (side === "left") {
        let left = clamp(start.left + deltaPct, minPanel, maxPanel);
        let center = clamp(start.center - deltaPct, minPanel, maxPanel);
        if (left + center > 100 - minPanel) {
          center = 100 - minPanel - left;
        }
        setColumnWidths({ left, center, right: start.right }, false);
      } else {
        let center = clamp(start.center + deltaPct, minPanel, maxPanel);
        let right = clamp(start.right - deltaPct, minPanel, maxPanel);
        if (center + right > 100 - minPanel) {
          right = 100 - minPanel - center;
        }
        setColumnWidths({ left: start.left, center, right }, false);
      }
    }

    function onUp() {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      document.body.classList.remove("resizing-panels");
      setColumnWidths(getColumnWidths(), true);
    }

    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  });
}

function initResizableColumns() {
  if (window.matchMedia("(max-width: 1200px)").matches) {
    return;
  }
  restoreColumnWidths();
  bindResizer(el.leftResizer, "left");
  bindResizer(el.rightResizer, "right");
}

function init() {
  if (window.mermaid && typeof window.mermaid.initialize === "function") {
    window.mermaid.initialize({
      startOnLoad: false,
      securityLevel: "loose",
      theme: "base",
      flowchart: {
        useMaxWidth: true,
        htmlLabels: false,
        nodeSpacing: 55,
        rankSpacing: 72,
        wrap: true,
      },
      themeVariables: {
        primaryTextColor: "#1f2d36",
        textColor: "#1f2d36",
        lineColor: "#506877",
        fontSize: "15px",
      },
    });
  }
  bindDocumentImagePreview();
  bindImageModalControls();
  bindVideoBriefJumps();
  initResizableColumns();
  el.researchForm.addEventListener("submit", startResearch);
  el.chatForm.addEventListener("submit", sendChat);
  if (el.topicInput) {
    el.topicInput.addEventListener("input", scheduleCacheSuggestions);
    el.topicInput.addEventListener("blur", () => fetchCacheSuggestions(el.topicInput.value, { force: true }));
  }
  if (el.cacheRefreshBtn) {
    el.cacheRefreshBtn.addEventListener("click", () => fetchCacheSuggestions(el.topicInput ? el.topicInput.value : "", { force: true }));
  }
  if (el.cacheSuggestList) {
    el.cacheSuggestList.addEventListener("click", (event) => {
      const target = event.target;
      if (!(target instanceof Element)) {
        return;
      }
      const button = target.closest(".cache-open-btn");
      if (!(button instanceof HTMLButtonElement)) {
        return;
      }
      const snapshotPath = button.dataset.snapshotPath || "";
      void openCachedResult(snapshotPath);
    });
  }
  el.inputModeRadios.forEach((radio) => {
    radio.addEventListener("change", (event) => {
      const target = event.target;
      if (target && "value" in target) {
        applyInputMode(target.value);
        if (target.value === "topic") {
          scheduleCacheSuggestions();
        } else {
          renderCacheSuggestions([]);
        }
      }
    });
  });
  el.topicInput.value = "MCP production deployment";
  applyInputMode("topic");
  scheduleCacheSuggestions();
}

init();
