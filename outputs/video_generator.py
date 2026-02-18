"""
Video Generator
基于 Slidev 的技术讲解视频生成（单一路线）。

Pipeline:
1) video_brief + one_pager + facts -> 结构化 slide 计划
2) 生成 Slidev markdown（支持公式/流程图/图像）
3) Slidev export PNG
4) 基于 slide 内容生成旁白脚本并合成音轨
5) ffmpeg 合成带音轨 MP4
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import inspect
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .models import OnePager, VideoBrief
from .video_narration import NarrationPipeline, NarrationSpec


logger = logging.getLogger(__name__)

_RUNTIME_DEPENDENCIES = {
    "@slidev/cli": "52.11.5",
    "@slidev/theme-default": "0.25.0",
    "playwright-chromium": "1.58.2",
}


class VideoGenerationError(RuntimeError):
    """视频生成失败"""


def _compact_text(value: str, max_len: int = 400) -> str:
    text = re.sub(r"\s+", " ", (value or "")).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _sanitize_text(value: Any) -> str:
    text = str(value or "")
    text = text.replace("\u200b", " ").replace("\u00a0", " ")
    text = re.sub(r"[\U00010000-\U0010ffff]", " ", text)
    text = re.sub(r"\{\{[^{}]*\}\}", " ", text)
    text = re.sub(r"\[\[[^\[\]]*\]\]", " ", text)
    text = re.sub(r"<\s*placeholder\s*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" \t\n\r-·•")

    lowered = text.lower()
    if lowered in {"", "n/a", "na", "none", "null", "...", "tbd", "todo"}:
        return ""
    if text in {"（请补充）", "（待补充）", "待补充"}:
        return ""
    return text


def _parse_duration_estimate_to_seconds(value: str) -> Optional[int]:
    text = str(value or "").strip().lower()
    if not text:
        return None

    numbers = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]
    if not numbers:
        return None

    number = float(sum(numbers) / len(numbers))
    has_minute = any(token in text for token in ["min", "minute", "分钟", "分"])
    has_second = any(token in text for token in ["sec", "second", "秒"])

    if has_second and not has_minute:
        return max(60, int(number))
    if has_minute:
        return max(120, int(number * 60))
    if number <= 20:
        return max(120, int(number * 60))
    return max(60, int(number))


def _extract_numeric(value: str) -> Optional[float]:
    text = str(value or "").strip().lower()
    if not text:
        return None
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if not matches:
        return None
    number = float(matches[0])
    if "%" in text:
        number = abs(number)
    return number


def _contains_cjk(value: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(value or "")))


def _short_label(text: str, *, cjk_max: int = 10, latin_max: int = 20) -> str:
    cleaned = _sanitize_text(text)
    if not cleaned:
        return ""

    cut_points = [":", "：", "。", "，", ",", ";", "；", "(", "（"]
    for marker in cut_points:
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[0].strip()
            break

    # Keep decimal versions like "2.5", but still cut full-stop sentence tails.
    parts = re.split(r"(?<!\d)\.(?!\d)", cleaned, maxsplit=1)
    if parts:
        cleaned = parts[0].strip()

    limit = cjk_max if _contains_cjk(cleaned) else latin_max
    if len(cleaned) <= limit:
        return cleaned
    wrapped = _wrap_text_units(cleaned, max_len=limit)
    if wrapped:
        return " / ".join(wrapped[:2])
    return cleaned


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value or "").strip("_").lower()
    return cleaned or f"run_{int(time.time())}"


def _split_sentences(text: str) -> List[str]:
    cleaned = _sanitize_text(text)
    if not cleaned:
        return []
    parts = re.split(r"[。！？!?；;\n]+", cleaned)
    return [_sanitize_text(part) for part in parts if _sanitize_text(part)]


def _split_clauses(text: str) -> List[str]:
    cleaned = _sanitize_text(text)
    if not cleaned:
        return []
    parts = re.split(r"[，,、：:]+", cleaned)
    return [_sanitize_text(part) for part in parts if _sanitize_text(part)]


def _wrap_text_units(text: str, max_len: int) -> List[str]:
    cleaned = _sanitize_text(text)
    if not cleaned:
        return []
    if len(cleaned) <= max_len:
        return [cleaned]

    clauses = _split_clauses(cleaned)
    if clauses and len(clauses) > 1:
        units: List[str] = []
        for clause in clauses:
            if len(clause) <= max_len:
                units.append(clause)
            else:
                units.extend(_wrap_text_units(clause, max_len))
        return units

    if " " in cleaned:
        words = cleaned.split()
        lines: List[str] = []
        current: List[str] = []
        for word in words:
            candidate = " ".join(current + [word]).strip()
            if not candidate:
                continue
            if len(candidate) <= max_len:
                current.append(word)
            else:
                if current:
                    lines.append(" ".join(current))
                current = [word]
        if current:
            lines.append(" ".join(current))
        return [_sanitize_text(line) for line in lines if _sanitize_text(line)]

    return []


def _normalize_key(text: str) -> str:
    return re.sub(r"\s+", "", _sanitize_text(text).lower())


def _dedupe_texts(items: Sequence[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        cleaned = _sanitize_text(item)
        if not cleaned:
            continue
        key = _normalize_key(cleaned)
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def _contains_ellipsis(text: str) -> bool:
    return "..." in text or "…" in text


_TECH_KEYWORDS = [
    "api",
    "retrieval",
    "rag",
    "benchmark",
    "latency",
    "throughput",
    "precision",
    "recall",
    "architecture",
    "pipeline",
    "deployment",
    "cost",
    "token",
    "model",
    "inference",
    "evaluation",
    "指标",
    "检索",
    "推理",
    "模型",
    "架构",
    "部署",
    "吞吐",
    "延迟",
    "成本",
    "公式",
    "算法",
    "数据集",
    "实验",
    "准确率",
    "召回",
    "一致性",
    "并发",
    "工程",
    "上下文",
    "视频生成",
]

_NON_TECH_PATTERNS = [
    "想象一下",
    "听起来像科幻",
    "改变世界",
    "革命",
    "愿景",
    "未来已来",
    "这就是像",
    "总而言之",
]

_WEAK_PREFIXES = ("但", "然后", "并且", "此外", "其中", "而且", "以及", "同时", "另外")
_GENERIC_SINGLE_BULLETS = {
    "architecture",
    "performance",
    "deployment",
    "comparison",
    "limitation",
    "risk",
    "metrics",
    "overview",
    "summary",
}

_CATEGORY_TO_THEME = {
    "architecture": "architecture",
    "mechanism": "architecture",
    "deployment": "implementation",
    "performance": "metrics",
    "benchmark": "metrics",
    "comparison": "comparison",
    "limitation": "risk",
    "training": "implementation",
    "community": "risk",
}

_THEME_KEYWORDS: Dict[str, List[str]] = {
    "overview": ["overview", "summary", "目标", "范围", "结论", "背景", "问题定义", "hook"],
    "architecture": [
        "architecture",
        "pipeline",
        "workflow",
        "routing",
        "context",
        "module",
        "protocol",
        "架构",
        "机制",
        "链路",
        "检索",
        "推理",
        "组件",
        "协议",
    ],
    "metrics": [
        "benchmark",
        "latency",
        "throughput",
        "precision",
        "recall",
        "cost",
        "指标",
        "性能",
        "吞吐",
        "延迟",
        "准确率",
        "召回",
        "成本",
        "oom",
        "qps",
    ],
    "comparison": [
        "compare",
        "versus",
        "vs",
        "trade-off",
        "alternative",
        "baseline",
        "对比",
        "取舍",
        "替代",
        "基线",
    ],
    "implementation": [
        "deploy",
        "production",
        "runtime",
        "infra",
        "monitor",
        "rollback",
        "工程",
        "落地",
        "部署",
        "监控",
        "灰度",
        "回滚",
        "实现",
    ],
    "risk": [
        "risk",
        "limitation",
        "failure",
        "drift",
        "attack",
        "constraint",
        "风险",
        "局限",
        "失败",
        "攻击面",
        "约束",
    ],
    "next_step": ["next", "action", "roadmap", "todo", "建议", "下一步", "行动", "验证"],
}

_THEME_TITLES = {
    "overview": "研究范围与核心结论",
    "architecture": "机制与架构拆解",
    "metrics": "指标与实验结果",
    "comparison": "方案对比与取舍",
    "implementation": "工程落地与运维策略",
    "risk": "风险与失效边界",
    "next_step": "下一步验证计划",
}

_THEME_NOTES = {
    "overview": "先定义研究范围，再进入细节。",
    "architecture": "强调组件关系和执行机制。",
    "metrics": "说明指标口径与实验边界。",
    "comparison": "对比收益与代价，避免单边结论。",
    "implementation": "聚焦可执行配置、监控和回滚。",
    "risk": "风险要对应缓解动作，而不是只列问题。",
    "next_step": "给出立即可执行的验证任务。",
}

_THEME_VISUAL_KIND = {
    "overview": "diagram",
    "architecture": "diagram",
    "metrics": "chart",
    "comparison": "diagram",
    "implementation": "diagram",
    "risk": "none",
    "next_step": "none",
}

_THEME_PRIORITY = ["overview", "architecture", "metrics", "comparison", "implementation", "risk", "next_step"]


def _has_unbalanced_brackets(text: str) -> bool:
    pairs = [("(", ")"), ("（", "）"), ("[", "]"), ("【", "】")]
    for left, right in pairs:
        if text.count(left) != text.count(right):
            return True
    return False


def _looks_like_meta_directive(text: str) -> bool:
    cleaned = _sanitize_text(text).lower()
    if not cleaned:
        return False
    return bool(
        re.match(
            r"^(layout|theme|title|mdc|katex|fonts|class|background|transition|lineNumbers)\s*:",
            cleaned,
        )
    )


def _escape_html(text: str) -> str:
    cleaned = _sanitize_text(text)
    return (
        cleaned.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def build_video_prompt(
    *,
    topic: str,
    video_brief: Optional[Dict[str, Any]] = None,
    one_pager: Optional[Dict[str, Any]] = None,
    facts: Optional[List[Dict[str, Any]]] = None,
    search_results: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """构建视频讲解脚本提示词。"""
    vb = VideoBrief.from_dict(video_brief or {}, default_title=f"{topic} Video Brief")
    op = OnePager.from_dict(one_pager or {}, default_title=f"{topic} One-Pager")
    facts = facts or []
    search_results = list(search_results or [])

    fact_lines: List[str] = []
    for fact in facts[:12]:
        claim = _compact_text(_sanitize_text(fact.get("claim", "")), max_len=220)
        category = _sanitize_text(fact.get("category", "other")) or "other"
        confidence = float(fact.get("confidence", 0.0))
        if claim:
            fact_lines.append(f"- [{category}] {claim} (confidence={confidence:.2f})")

    segment_lines: List[str] = []
    for seg in vb.segments[:6]:
        title = _compact_text(_sanitize_text(seg.get("title", "")), max_len=70)
        content = _compact_text(_sanitize_text(seg.get("content", "")), max_len=220)
        duration_sec = seg.get("duration_sec", "")
        if title and content:
            segment_lines.append(f"- Segment: {title}; duration={duration_sec}s; content={content}")

    key_findings = "\n".join(
        [f"- {_compact_text(_sanitize_text(item), max_len=180)}" for item in op.key_findings[:6] if _sanitize_text(item)]
    ) or "- N/A"

    source_priority = {
        "arxiv": 0,
        "semantic_scholar": 1,
        "openreview": 2,
        "github": 3,
        "huggingface": 4,
        "stackoverflow": 5,
        "hackernews": 6,
        "reddit": 7,
        "twitter": 8,
    }
    evidence_lines: List[str] = []
    for item in sorted(
        [row for row in search_results if isinstance(row, dict)],
        key=lambda row: source_priority.get(_sanitize_text(row.get("source", "")).lower(), 99),
    )[:16]:
        source = _sanitize_text(item.get("source", "")) or "unknown"
        rid = _sanitize_text(item.get("id", ""))
        title = _compact_text(_sanitize_text(item.get("title", "")), max_len=90)
        content = _compact_text(_sanitize_text(item.get("content", "")), max_len=170)
        snippet = content
        if snippet:
            parts = re.split(r"[。.!?；;\n]+", snippet)
            snippet = _compact_text(_sanitize_text(parts[0] if parts else snippet), max_len=130)
        if not title and not snippet:
            continue
        rid_text = f"{rid} | " if rid else ""
        evidence_lines.append(f"- [{source}] {rid_text}{title}: {snippet}")
        if len(evidence_lines) >= 8:
            break

    prompt = f"""
Create a technical explainer slide video about "{topic}".

Narrative hook:
{_compact_text(_sanitize_text(vb.hook), max_len=260)}

Target audience: {_compact_text(_sanitize_text(vb.target_audience or "ML engineers and researchers"), max_len=120)}
Visual style: {_compact_text(_sanitize_text(vb.visual_style or "clean technical presentation"), max_len=160)}

Core findings:
{key_findings}

Verified facts:
{os.linesep.join(fact_lines) if fact_lines else "- N/A"}

Retrieved evidence:
{os.linesep.join(evidence_lines) if evidence_lines else "- N/A"}

Slide plan:
{os.linesep.join(segment_lines) if segment_lines else "- Prefer evidence-backed sections only; if evidence is weak, include explicit evidence-gap section."}

Constraints:
- Keep high factual density and explicit technical details.
- Explain mechanism, benchmark context, engineering trade-offs, and deployment guidance.
- Avoid hype and unsupported claims.
- If evidence is insufficient, state uncertainty and list concrete follow-up validation tasks.
""".strip()

    return prompt


@dataclass
class VideoArtifact:
    provider: str
    prompt: str
    output_path: Path
    metadata_path: Path


class BaseVideoGenerator:
    provider = "base"

    async def generate(
        self,
        *,
        topic: str,
        out_dir: Path,
        video_brief: Optional[Dict[str, Any]] = None,
        one_pager: Optional[Dict[str, Any]] = None,
        facts: Optional[List[Dict[str, Any]]] = None,
        search_results: Optional[List[Dict[str, Any]]] = None,
    ) -> VideoArtifact:
        raise NotImplementedError


@dataclass
class SlideSpec:
    title: str
    bullets: List[str] = field(default_factory=list)
    duration_sec: int = 30
    notes: str = ""
    visual_kind: str = "diagram"  # diagram / chart / image / none
    visual_payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidencePoint:
    text: str
    theme: str
    source: str
    score: float = 1.0
    title_hint: str = ""


@dataclass
class SectionDraft:
    theme: str
    title: str
    bullets: List[str]
    notes: str = ""
    visual_kind: str = "diagram"
    visual_payload: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    backup_points: List[str] = field(default_factory=list)


class SlidevVideoGenerator(BaseVideoGenerator):
    """
    单一路线视频生成器：Slidev + ffmpeg
    """

    provider = "slidev"

    def __init__(
        self,
        *,
        target_duration_sec: int = 180,
        min_duration_sec: int = 150,
        max_duration_sec: int = 180,
        fps: int = 24,
        width: int = 1920,
        height: int = 1080,
        runtime_dir: Optional[Path] = None,
        slidev_timeout_ms: int = 180000,
        slidev_wait_ms: int = 400,
        enable_narration: bool = True,
        tts_provider: str = "auto",
        tts_voice: Optional[str] = None,
        tts_speed: float = 1.25,
        narration_model: str = "deepseek-chat",
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.target_duration_sec = int(target_duration_sec)
        self.max_duration_sec = max(int(max_duration_sec), self.target_duration_sec)
        self.min_duration_sec = min(int(min_duration_sec), self.max_duration_sec)
        self.fps = int(fps)
        self.width = int(width)
        self.height = int(height)
        self.runtime_dir = Path(runtime_dir) if runtime_dir else Path("data") / ".slidev_runtime"
        self.slidev_timeout_ms = int(slidev_timeout_ms)
        self.slidev_wait_ms = int(slidev_wait_ms)
        self.enable_narration = bool(enable_narration)
        self.tts_provider = _sanitize_text(tts_provider).lower().replace("-", "_") or "auto"
        self.tts_voice = _sanitize_text(tts_voice or "")
        self.tts_speed = float(max(0.8, min(1.5, tts_speed)))
        self.narration_model = _sanitize_text(narration_model) or "deepseek-chat"
        self._progress_callback = progress_callback
        self._narration_pipeline = NarrationPipeline(
            tts_provider=self.tts_provider,
            tts_voice=self.tts_voice,
            tts_speed=self.tts_speed,
            narration_model=self.narration_model,
            run_subprocess=self._run_subprocess,
            sanitize_text=_sanitize_text,
            contains_cjk=_contains_cjk,
            split_sentences=_split_sentences,
        )
        self._contract_slide_titles: List[str] = []

    def _emit_progress(self, message: str) -> None:
        logger.info("[SlidevVideo] %s", message)
        if self._progress_callback:
            try:
                self._progress_callback(message)
            except Exception:
                logger.debug("Progress callback failed", exc_info=True)

    def _run_subprocess(
        self,
        cmd: List[str],
        *,
        context: str,
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        process = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if process.returncode != 0:
            combined = "\n".join(
                [
                    (process.stdout or "").strip(),
                    (process.stderr or "").strip(),
                ]
            ).strip()
            tail = "\n".join(combined.splitlines()[-20:])
            raise VideoGenerationError(f"{context} failed: {tail or 'unknown error'}")

    def _narration(self) -> NarrationPipeline:
        self._narration_pipeline.update_runtime(
            tts_provider=self.tts_provider,
            tts_voice=self.tts_voice,
            tts_speed=self.tts_speed,
            narration_model=self.narration_model,
        )
        return self._narration_pipeline

    def _resolve_target_duration(self, brief_duration: str, slide_count: int) -> int:
        candidate = _parse_duration_estimate_to_seconds(brief_duration)
        if candidate is None:
            candidate = self.target_duration_sec
        duration = max(self.min_duration_sec, min(self.max_duration_sec, int(candidate)))
        min_by_slide_count = slide_count * 12
        if duration < min_by_slide_count:
            duration = min(min_by_slide_count, self.max_duration_sec)
        return duration

    def _is_technical_line(self, text: str) -> bool:
        cleaned = _sanitize_text(text)
        if not cleaned:
            return False
        lowered = cleaned.lower()
        if any(pattern in lowered for pattern in _NON_TECH_PATTERNS):
            return False
        if any(keyword in lowered for keyword in _TECH_KEYWORDS):
            return True
        if re.search(r"\d", cleaned):
            return True
        return False

    def _technical_score(self, text: str) -> int:
        cleaned = _sanitize_text(text)
        if not cleaned:
            return 0
        lowered = cleaned.lower()
        score = 0
        for keyword in _TECH_KEYWORDS:
            if keyword in lowered:
                score += 2
        if re.search(r"\d", cleaned):
            score += 3
        if "：" in cleaned or ":" in cleaned:
            score += 1
        if "->" in cleaned or "→" in cleaned:
            score += 2
        return score

    def _compress_sentence(self, text: str, max_len: int) -> str:
        cleaned = _sanitize_text(text)
        if not cleaned:
            return ""
        if len(cleaned) <= max_len and not _has_unbalanced_brackets(cleaned):
            return cleaned

        no_paren = re.sub(r"（[^（）]{1,32}）", "", cleaned)
        no_paren = re.sub(r"\([^()]{1,32}\)", "", no_paren)
        no_paren = _sanitize_text(no_paren)
        if no_paren and len(no_paren) <= max_len and not _has_unbalanced_brackets(no_paren):
            return no_paren

        major_clauses = re.split(r"[；;:：]+", no_paren or cleaned)
        major_clauses = [_sanitize_text(clause) for clause in major_clauses if _sanitize_text(clause)]
        if major_clauses:
            ranked = sorted(major_clauses, key=lambda item: self._technical_score(item), reverse=True)
            for clause in ranked:
                if len(clause) <= max_len and not _has_unbalanced_brackets(clause):
                    return clause

        return ""

    def _fit_bullets(
        self,
        texts: Sequence[str],
        *,
        max_bullets: int = 4,
        max_len: int = 72,
        technical_only: bool = True,
    ) -> List[str]:
        def _accept_unit(unit: str) -> bool:
            cleaned = _sanitize_text(unit)
            if not cleaned:
                return False
            if _contains_ellipsis(cleaned):
                return False
            if _has_unbalanced_brackets(cleaned):
                return False
            if _looks_like_meta_directive(cleaned):
                return False
            if re.fullmatch(r"[（(]?\s*(置信度|confidence)[^）)]*[）)]?", cleaned, flags=re.IGNORECASE):
                return False
            if len(cleaned) < 6:
                return False
            if any(cleaned.startswith(prefix) for prefix in _WEAK_PREFIXES) and len(cleaned) < 14:
                return False
            return True

        candidates: List[str] = []
        for raw in texts:
            for sentence in _split_sentences(raw):
                unit = self._compress_sentence(sentence, max_len=max_len)
                if not _accept_unit(unit):
                    continue
                if technical_only and not self._is_technical_line(unit):
                    continue
                candidates.append(unit)

        deduped = _dedupe_texts(candidates)
        if max_bullets > 0:
            return deduped[:max_bullets]
        return deduped

    def _select_points(
        self,
        pool: Sequence[str],
        *,
        include_keywords: Sequence[str] = (),
        exclude_keywords: Sequence[str] = (),
        max_bullets: int = 4,
        max_len: int = 72,
        technical_only: bool = True,
    ) -> List[str]:
        include_tokens = [token.lower() for token in include_keywords if token]
        exclude_tokens = [token.lower() for token in exclude_keywords if token]

        filtered: List[str] = []
        for item in pool:
            cleaned = _sanitize_text(item)
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if include_tokens and not any(token in lowered for token in include_tokens):
                continue
            if exclude_tokens and any(token in lowered for token in exclude_tokens):
                continue
            filtered.append(cleaned)

        points = self._fit_bullets(
            filtered,
            max_bullets=max_bullets,
            max_len=max_len,
            technical_only=technical_only,
        )
        return points

    def _normalize_bullet_text(self, text: str) -> str:
        cleaned = _sanitize_text(text)
        if not cleaned:
            return ""
        lowered = cleaned.lower()
        if lowered in _GENERIC_SINGLE_BULLETS:
            return ""
        max_len = 72 if _contains_cjk(cleaned) else 120
        compressed = self._compress_sentence(cleaned, max_len=max_len)
        if not compressed:
            return ""
        return compressed

    def _fact_to_bullet(self, fact: Dict[str, Any]) -> str:
        category = _sanitize_text(fact.get("category", "other")) or "other"
        claim = _sanitize_text(fact.get("claim", ""))
        if not claim:
            return ""
        claim_line = self._compress_sentence(claim, max_len=72)
        if not claim_line:
            return ""
        prefix = f"[{category}] " if category and category != "other" else ""
        return f"{prefix}{claim_line}"

    def _build_metrics_chart_payload(self, metrics: Dict[str, str]) -> Dict[str, Any]:
        labels: List[str] = []
        values: List[float] = []
        for key, value in list(metrics.items())[:6]:
            number = _extract_numeric(str(value))
            if number is None:
                continue
            labels.append(_short_label(key, cjk_max=12, latin_max=24))
            values.append(float(abs(number)))
        if not labels:
            return {}
        max_val = max(values) if values else 1.0
        normalized = [v / max_val if max_val > 0 else 0.0 for v in values]
        return {"labels": labels, "values": values, "normalized": normalized}

    def _build_diagram_payload(self, title: str, bullets: List[str]) -> Dict[str, Any]:
        def _diagram_label(text: str) -> str:
            cleaned = _sanitize_text(text)
            if not cleaned:
                return ""
            cleaned = re.sub(r"^\[[^\]]+\]\s*", "", cleaned).strip()
            if "：" in cleaned:
                head = _sanitize_text(cleaned.split("：", 1)[0])
                if head and len(head) <= 18:
                    return head
            if ":" in cleaned:
                head = _sanitize_text(cleaned.split(":", 1)[0])
                if head and len(head) <= 22:
                    return head

            keyword_map = [
                ("检索", "检索质量"),
                ("api", "API调用"),
                ("上下文", "上下文建模"),
                ("一致性", "一致性约束"),
                ("延迟", "延迟控制"),
                ("吞吐", "吞吐能力"),
                ("成本", "成本边界"),
                ("部署", "部署策略"),
                ("评估", "评估指标"),
                ("实验", "实验验证"),
                ("风险", "风险控制"),
                ("模型", "模型能力"),
                ("视频", "视频生成"),
            ]
            lowered = cleaned.lower()
            for key, label in keyword_map:
                if key in lowered:
                    return label

            limit = 18 if _contains_cjk(cleaned) else 24
            parts = _wrap_text_units(cleaned, max_len=limit)
            if parts:
                return parts[0]
            clauses = _split_clauses(cleaned)
            for clause in clauses:
                if len(clause) <= limit and clause:
                    return clause
            return "关键要点"

        tags = [_diagram_label(item) for item in bullets[:4] if _sanitize_text(item)]
        tags = [item for item in tags if item]
        if not tags:
            tags = ["Inputs", "Reasoning", "Outputs"]
        return {"title": _diagram_label(title), "tags": tags[:4]}

    def _bullet_uniqueness_key(self, text: str) -> str:
        cleaned = _sanitize_text(text)
        cleaned = re.sub(r"^\[[^\]]+\]\s*", "", cleaned)
        cleaned = cleaned.lower()
        cleaned = re.sub(r"(第\s*\d+\s*点|point\s*\d+)\s*[:：]?", "", cleaned)
        cleaned = re.sub(r"\d+(?:\.\d+)?%?", "", cleaned)
        cleaned = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", cleaned)
        return cleaned

    def _theme_scaffold_bullets(self, *, theme: str, title: str, topic: str) -> List[str]:
        topic_text = _sanitize_text(topic) or "该主题"
        title_text = _sanitize_text(title) or _THEME_TITLES.get(theme, "关键板块")
        templates = {
            "architecture": [
                f"{title_text}：拆解核心组件、数据流和关键约束，明确能力边界。",
                f"{topic_text} 的架构选择需结合上下文长度、推理吞吐和内存成本综合评估。",
            ],
            "metrics": [
                f"{title_text}：统一指标口径（准确率、延迟、吞吐、成本）并标注测试条件。",
                "对照基线报告绝对值与相对提升，避免只汇报单一最优结果。",
            ],
            "comparison": [
                f"{title_text}：明确与主流替代方案的收益、代价和适用场景差异。",
                "比较时需同时给出能力上限、工程复杂度和稳定性风险。",
            ],
            "implementation": [
                f"{title_text}：定义上线流程、依赖组件、监控阈值与回滚条件。",
                "落地前应先做灰度验证并记录可复现实验脚本与配置。",
            ],
            "risk": [
                f"{title_text}：列出失效边界，并为每项风险绑定可执行缓解动作。",
                "对高风险路径建立告警与熔断机制，避免局部故障扩散。",
            ],
        }
        return templates.get(theme, [f"{title_text}：补充可验证证据，收敛关键技术结论。"])

    def _build_image_payload(self, one_pager: OnePager) -> Dict[str, Any]:
        for resource in list(one_pager.resources or [])[:10]:
            title = _sanitize_text(resource.get("title", ""))
            for key in ("image_path", "path", "file", "local_path"):
                raw = _sanitize_text(resource.get(key, ""))
                if not raw:
                    continue
                path = Path(raw).expanduser()
                if path.exists() and path.is_file():
                    return {"image_path": str(path), "caption": title or path.name}
        return {}

    def _infer_theme(self, text: str, *, category: str = "", source: str = "") -> str:
        category_key = _sanitize_text(category).lower().replace("-", "_")
        if category_key in _CATEGORY_TO_THEME:
            return _CATEGORY_TO_THEME[category_key]

        cleaned = _sanitize_text(text).lower()
        best_theme = ""
        best_score = 0
        for theme, keywords in _THEME_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                if keyword and keyword in cleaned:
                    score += 1
            if score > best_score:
                best_score = score
                best_theme = theme
        if best_theme:
            return best_theme

        if source.startswith("video_segment"):
            return "architecture"
        if source.startswith("fact"):
            return "overview"
        return "overview"

    def _pick_section_title(self, *, topic: str, theme: str, hints: Sequence[str]) -> str:
        for raw in hints:
            candidate = _sanitize_text(raw)
            if not candidate:
                continue
            lowered = candidate.lower()
            if re.fullmatch(r"(segment|section)\s*\d+", lowered):
                continue
            if len(candidate) >= 4:
                return _short_label(candidate, cjk_max=16, latin_max=34)

        default_title = _THEME_TITLES.get(theme, "关键发现")
        topic_text = _sanitize_text(topic)
        if theme == "overview" and topic_text:
            return _short_label(f"{topic_text}：{default_title}", cjk_max=18, latin_max=40)
        return default_title

    def _collect_evidence_points(
        self,
        *,
        topic: str,
        video_brief: VideoBrief,
        one_pager: OnePager,
        facts: List[Dict[str, Any]],
        search_results: Optional[List[Dict[str, Any]]] = None,
    ) -> List[EvidencePoint]:
        points: List[EvidencePoint] = []

        def _add(
            text: str,
            *,
            source: str,
            score: float,
            theme: Optional[str] = None,
            category: str = "",
            title_hint: str = "",
        ) -> None:
            cleaned = _sanitize_text(text)
            if not cleaned:
                return

            normalized = self._normalize_bullet_text(cleaned)
            if not normalized:
                normalized = self._compress_sentence(cleaned, max_len=120) or cleaned
            if not normalized:
                return

            resolved_theme = theme or self._infer_theme(normalized, category=category, source=source)
            hint = _sanitize_text(title_hint)
            points.append(
                EvidencePoint(
                    text=normalized,
                    theme=resolved_theme,
                    source=source,
                    score=float(max(0.1, score)),
                    title_hint=hint,
                )
            )

        _add(
            one_pager.executive_summary or video_brief.hook or f"研究主题：{_sanitize_text(topic)}",
            source="one_pager.summary",
            score=1.8,
            theme="overview",
            title_hint=topic,
        )
        _add(video_brief.conclusion, source="video_brief.conclusion", score=1.4, theme="next_step")
        _add(video_brief.call_to_action, source="video_brief.cta", score=1.6, theme="next_step")

        for item in list(one_pager.key_findings or [])[:10]:
            _add(item, source="one_pager.key_findings", score=1.6)

        for key, value in list(one_pager.metrics.items())[:10]:
            key_clean = _sanitize_text(key)
            val_clean = _sanitize_text(value)
            if not key_clean or not val_clean:
                continue
            _add(
                f"{key_clean}: {val_clean}",
                source="one_pager.metrics",
                score=1.9,
                theme="metrics",
                title_hint="关键指标",
            )

        for item in list(one_pager.technical_deep_dive or [])[:10]:
            _add(item, source="one_pager.deep_dive", score=1.7, theme="architecture")
        for item in list(one_pager.implementation_notes or [])[:10]:
            _add(item, source="one_pager.impl", score=1.7, theme="implementation")
        for item in list(one_pager.risks_and_mitigations or [])[:10]:
            _add(item, source="one_pager.risk", score=1.6, theme="risk")
        for item in list(one_pager.weaknesses or [])[:8]:
            _add(item, source="one_pager.weakness", score=1.4, theme="risk")
        for item in list(one_pager.strengths or [])[:8]:
            _add(item, source="one_pager.strength", score=1.3)

        for segment in list(video_brief.segments or [])[:10]:
            seg_title = _sanitize_text(segment.get("title", ""))
            seg_content = _sanitize_text(segment.get("content", ""))
            seg_theme = self._infer_theme(
                f"{seg_title} {seg_content}",
                source="video_segment.content",
            )
            _add(
                seg_content,
                source="video_segment.content",
                score=1.55,
                theme=seg_theme,
                title_hint=seg_title,
            )
            for talking in list(segment.get("talking_points") or [])[:6]:
                _add(
                    talking,
                    source="video_segment.talking_point",
                    score=1.45,
                    theme=seg_theme,
                    title_hint=seg_title,
                )
            _add(seg_title, source="video_segment.title", score=1.2, theme=seg_theme, title_hint=seg_title)

        for fact in sorted(list(facts or []), key=lambda item: float(item.get("confidence", 0.0)), reverse=True):
            category = _sanitize_text(fact.get("category", ""))
            confidence = float(fact.get("confidence", 0.0) or 0.0)
            bullet = self._fact_to_bullet(fact) or _sanitize_text(fact.get("claim", ""))
            _add(
                bullet,
                source=f"fact.{category or 'other'}",
                score=1.2 + max(0.0, min(1.0, confidence)),
                category=category,
            )

        source_priority = {
            "arxiv": 0,
            "semantic_scholar": 1,
            "openreview": 2,
            "github": 3,
            "huggingface": 4,
            "stackoverflow": 5,
            "hackernews": 6,
            "reddit": 7,
            "twitter": 8,
        }
        ranked_results = sorted(
            [item for item in list(search_results or []) if isinstance(item, dict)],
            key=lambda item: (
                source_priority.get(_sanitize_text(item.get("source", "")).lower(), 99),
                -len(_sanitize_text(item.get("content", ""))),
            ),
        )
        for item in ranked_results[:20]:
            source = _sanitize_text(item.get("source", "")).lower() or "unknown"
            title = _sanitize_text(item.get("title", ""))
            content = _sanitize_text(item.get("content", ""))
            if not title and not content:
                continue
            parts = _split_sentences(content)
            sentence = parts[0] if parts else content
            metadata = dict(item.get("metadata", {}) or {})
            metrics: List[str] = []
            for key in ("citation_count", "stars", "downloads", "score", "points", "published_date", "year"):
                value = _sanitize_text(metadata.get(key, ""))
                if value:
                    metrics.append(f"{key}={value}")
                if len(metrics) >= 3:
                    break
            metrics_text = f" ({', '.join(metrics)})" if metrics else ""
            evidence_text = _compact_text(
                f"[{source}] {title}{metrics_text}: {sentence}",
                max_len=220,
            )
            if not evidence_text:
                continue
            _add(
                evidence_text,
                source=f"search_result.{source}",
                score=1.25 if source in {"arxiv", "semantic_scholar", "openreview", "github", "huggingface"} else 1.05,
                title_hint=title,
            )

        deduped: Dict[str, EvidencePoint] = {}
        for point in points:
            key = _normalize_key(point.text)
            if not key:
                continue
            existing = deduped.get(key)
            if existing is None:
                deduped[key] = point
                continue
            if point.score > existing.score:
                if not point.title_hint and existing.title_hint:
                    point.title_hint = existing.title_hint
                deduped[key] = point
            elif not existing.title_hint and point.title_hint:
                existing.title_hint = point.title_hint
                deduped[key] = existing

        return sorted(deduped.values(), key=lambda item: item.score, reverse=True)

    def _build_metrics_chart_from_points(
        self,
        *,
        one_pager: OnePager,
        points: Sequence[EvidencePoint],
    ) -> Dict[str, Any]:
        payload = self._build_metrics_chart_payload(one_pager.metrics)
        if payload:
            return payload

        derived: Dict[str, str] = {}
        for point in points:
            text = _sanitize_text(point.text)
            if ":" not in text:
                continue
            key, value = text.split(":", 1)
            key = _sanitize_text(key)
            value = _sanitize_text(value)
            if not key or not value:
                continue
            if _extract_numeric(value) is None:
                continue
            if key not in derived:
                derived[key] = value
            if len(derived) >= 6:
                break
        return self._build_metrics_chart_payload(derived)

    def _bucket_points_by_theme(self, points: Sequence[EvidencePoint]) -> Dict[str, List[EvidencePoint]]:
        buckets: Dict[str, List[EvidencePoint]] = {}
        for point in points:
            buckets.setdefault(point.theme, []).append(point)
        return buckets

    def _ordered_themes(self, buckets: Dict[str, List[EvidencePoint]], point_count: int) -> List[str]:
        theme_scores = {theme: sum(item.score for item in items) for theme, items in buckets.items()}
        priority_index = {name: idx for idx, name in enumerate(_THEME_PRIORITY)}

        ordered_themes: List[str] = []
        if "overview" in buckets:
            ordered_themes.append("overview")

        rest = [theme for theme in buckets.keys() if theme != "overview"]
        rest.sort(key=lambda theme: (-theme_scores.get(theme, 0.0), priority_index.get(theme, 99), theme))
        ordered_themes.extend(rest)

        for tail in ("risk", "next_step"):
            if tail in ordered_themes:
                ordered_themes.remove(tail)
                ordered_themes.append(tail)

        target_sections = max(2, min(8, int(math.ceil(point_count / 5.0))))
        return ordered_themes[:target_sections]

    def _resolve_visual_for_theme(
        self,
        *,
        theme: str,
        title: str,
        bullets: Sequence[str],
        metric_payload: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        visual_kind = _THEME_VISUAL_KIND.get(theme, "diagram")
        visual_payload: Dict[str, Any] = {}
        if visual_kind == "chart":
            visual_payload = dict(metric_payload)
            if not visual_payload:
                visual_kind = "diagram"
        if visual_kind == "diagram":
            visual_payload = self._build_diagram_payload(title, list(bullets))
        return visual_kind, visual_payload

    def _build_primary_section_draft(
        self,
        *,
        theme: str,
        topic: str,
        bucket: List[EvidencePoint],
        metric_payload: Dict[str, Any],
    ) -> Optional[SectionDraft]:
        raw_pool = [item.text for item in bucket]
        bullets = self._fit_bullets(
            raw_pool,
            max_bullets=5,
            max_len=72,
            technical_only=theme not in {"overview", "next_step"},
        )
        if len(bullets) < 2:
            bullets = self._fit_bullets(
                raw_pool,
                max_bullets=5,
                max_len=80,
                technical_only=False,
            )
        if not bullets:
            return None

        title_hints = [item.title_hint for item in bucket if _sanitize_text(item.title_hint)]
        title = self._pick_section_title(topic=topic, theme=theme, hints=title_hints)
        visual_kind, visual_payload = self._resolve_visual_for_theme(
            theme=theme,
            title=title,
            bullets=bullets,
            metric_payload=metric_payload,
        )

        return SectionDraft(
            theme=theme,
            title=title,
            bullets=bullets[:5],
            notes=_THEME_NOTES.get(theme, ""),
            visual_kind=visual_kind,
            visual_payload=visual_payload,
            weight=1.0 + min(0.45, 0.07 * len(bucket)) + (0.15 if visual_kind == "chart" else 0.0),
            backup_points=raw_pool[:12],
        )

    def _inject_metric_chart_draft_if_missing(
        self,
        *,
        drafts: List[SectionDraft],
        buckets: Dict[str, List[EvidencePoint]],
        topic: str,
        one_pager: OnePager,
        metric_payload: Dict[str, Any],
    ) -> None:
        has_chart = any(item.visual_kind == "chart" for item in drafts)
        if not metric_payload or has_chart:
            return

        metric_pool = [item.text for item in buckets.get("metrics", [])]
        if not metric_pool:
            metric_pool = [
                f"{_sanitize_text(key)}: {_sanitize_text(value)}"
                for key, value in list(one_pager.metrics.items())[:8]
                if _sanitize_text(key) and _sanitize_text(value)
            ]
        metric_bullets = self._fit_bullets(metric_pool, max_bullets=5, max_len=72, technical_only=False)
        if not metric_bullets:
            return

        metric_draft = SectionDraft(
            theme="metrics",
            title=self._pick_section_title(topic=topic, theme="metrics", hints=["关键指标"]),
            bullets=metric_bullets[:5],
            notes=_THEME_NOTES.get("metrics", ""),
            visual_kind="chart",
            visual_payload=dict(metric_payload),
            weight=1.2,
            backup_points=metric_pool[:12],
        )
        drafts.insert(min(2, len(drafts)), metric_draft)

    def _append_missing_theme_drafts(
        self,
        *,
        drafts: List[SectionDraft],
        buckets: Dict[str, List[EvidencePoint]],
        topic: str,
        metric_payload: Dict[str, Any],
    ) -> None:
        existing_themes = {item.theme for item in drafts}
        for theme in ("architecture", "metrics", "comparison", "implementation", "risk"):
            if theme in existing_themes:
                continue
            bucket = sorted(buckets.get(theme, []), key=lambda item: item.score, reverse=True)
            if not bucket:
                continue

            raw_pool = [item.text for item in bucket]
            relaxed_len = 96 if theme in {"comparison", "implementation", "risk"} else 84
            bullets = self._fit_bullets(
                raw_pool,
                max_bullets=5,
                max_len=relaxed_len,
                technical_only=False,
            )
            title_hints = [item.title_hint for item in bucket if _sanitize_text(item.title_hint)]
            title = self._pick_section_title(topic=topic, theme=theme, hints=title_hints)
            if not bullets:
                bullets = self._theme_scaffold_bullets(theme=theme, title=title, topic=topic)

            visual_kind, visual_payload = self._resolve_visual_for_theme(
                theme=theme,
                title=title,
                bullets=bullets,
                metric_payload=metric_payload,
            )
            drafts.append(
                SectionDraft(
                    theme=theme,
                    title=title,
                    bullets=bullets[:5],
                    notes=_THEME_NOTES.get(theme, ""),
                    visual_kind=visual_kind,
                    visual_payload=visual_payload,
                    weight=0.95,
                    backup_points=raw_pool[:12],
                )
            )
            existing_themes.add(theme)

    def _draft_sections(
        self,
        *,
        topic: str,
        points: List[EvidencePoint],
        one_pager: OnePager,
    ) -> List[SectionDraft]:
        if not points:
            return []

        buckets = self._bucket_points_by_theme(points)
        selected_themes = self._ordered_themes(buckets, len(points))
        metric_payload = self._build_metrics_chart_from_points(one_pager=one_pager, points=points)

        drafts: List[SectionDraft] = []
        for theme in selected_themes:
            bucket = sorted(buckets.get(theme, []), key=lambda item: item.score, reverse=True)
            if not bucket:
                continue
            draft = self._build_primary_section_draft(
                theme=theme,
                topic=topic,
                bucket=bucket,
                metric_payload=metric_payload,
            )
            if draft:
                drafts.append(draft)

        self._inject_metric_chart_draft_if_missing(
            drafts=drafts,
            buckets=buckets,
            topic=topic,
            one_pager=one_pager,
            metric_payload=metric_payload,
        )
        self._append_missing_theme_drafts(
            drafts=drafts,
            buckets=buckets,
            topic=topic,
            metric_payload=metric_payload,
        )
        return drafts[:10]

    def _ensure_minimum_section_drafts(self, drafts: List[SectionDraft], topic: str) -> None:
        if drafts:
            return

        topic_text = _sanitize_text(topic) or "当前主题"
        bullets = [
            f"{topic_text} 当前证据不足，无法稳定支撑完整视频结构。",
            "优先补齐论文、代码实现与线上观测三类证据链后再扩展脚本。",
        ]
        drafts.append(
            SectionDraft(
                theme="next_step",
                title="证据覆盖与缺口",
                bullets=bullets,
                notes="证据薄弱时只保留补证路线，避免编造细节。",
                visual_kind="none",
                visual_payload={},
                weight=0.88,
                backup_points=list(bullets),
            )
        )

    def _append_image_section_draft(self, drafts: List[SectionDraft], one_pager: OnePager) -> None:
        image_payload = self._build_image_payload(one_pager)
        if not image_payload:
            return

        drafts.append(
            SectionDraft(
                theme="evidence_image",
                title="图像证据",
                bullets=[
                    "该页展示补充证据图像，用于辅助理解关键结论。",
                    f"图像来源：{_sanitize_text(image_payload.get('caption', '未知来源'))}",
                ],
                notes="图像可与对应实验指标交叉核对。",
                visual_kind="image",
                visual_payload=image_payload,
                weight=0.85,
                backup_points=[],
            )
        )

    def _append_unique_bullets(
        self,
        *,
        target: List[str],
        candidates: Sequence[str],
        local_keys: set[str],
        global_keys: set[str],
        max_items: int = 5,
    ) -> None:
        for raw in candidates:
            if len(target) >= max_items:
                break
            bullet = self._normalize_bullet_text(raw)
            if not bullet:
                continue
            local_key = self._bullet_uniqueness_key(bullet)
            global_key = re.sub(r"\s+", " ", bullet).strip().lower()
            if not local_key or local_key in local_keys or global_key in global_keys:
                continue
            local_keys.add(local_key)
            global_keys.add(global_key)
            target.append(bullet)

    def _normalize_drafts_to_slides(
        self,
        *,
        drafts: Sequence[SectionDraft],
        topic: str,
    ) -> Tuple[List[SlideSpec], List[float]]:
        normalized: List[SlideSpec] = []
        normalized_weights: List[float] = []
        global_bullet_texts: set[str] = set()
        seen_slide_signatures: set[str] = set()

        for draft in drafts:
            title = _sanitize_text(draft.title)
            notes = _sanitize_text(draft.notes)
            if not title:
                continue

            local_keys: set[str] = set()
            unique_bullets: List[str] = []
            self._append_unique_bullets(
                target=unique_bullets,
                candidates=draft.bullets,
                local_keys=local_keys,
                global_keys=global_bullet_texts,
                max_items=5,
            )

            if len(unique_bullets) < 2:
                self._append_unique_bullets(
                    target=unique_bullets,
                    candidates=draft.backup_points,
                    local_keys=local_keys,
                    global_keys=global_bullet_texts,
                    max_items=5,
                )

            if len(unique_bullets) < 2:
                self._append_unique_bullets(
                    target=unique_bullets,
                    candidates=self._theme_scaffold_bullets(theme=draft.theme, title=title, topic=topic),
                    local_keys=local_keys,
                    global_keys=global_bullet_texts,
                    max_items=5,
                )

            if len(unique_bullets) < 2:
                gap_line = self._normalize_bullet_text(f"该板块证据不足：{title}，建议补充更多可验证来源。")
                if gap_line:
                    key = self._bullet_uniqueness_key(gap_line)
                    global_key = re.sub(r"\s+", " ", gap_line).strip().lower()
                    if key and key not in local_keys and global_key not in global_bullet_texts:
                        unique_bullets.append(gap_line)
                        local_keys.add(key)
                        global_bullet_texts.add(global_key)

            if len(unique_bullets) < 2:
                continue

            slide_signature = "|".join(
                [title, draft.visual_kind] + [self._bullet_uniqueness_key(item) for item in unique_bullets[:3]]
            )
            if slide_signature in seen_slide_signatures:
                continue
            seen_slide_signatures.add(slide_signature)

            visual_payload = dict(draft.visual_payload or {})
            if draft.visual_kind == "diagram" and not visual_payload:
                visual_payload = self._build_diagram_payload(title, unique_bullets)

            normalized.append(
                SlideSpec(
                    title=title,
                    bullets=unique_bullets[:5],
                    duration_sec=30,
                    notes=notes,
                    visual_kind=draft.visual_kind,
                    visual_payload=visual_payload,
                )
            )
            normalized_weights.append(float(max(0.6, draft.weight)))

        return normalized, normalized_weights

    def _ensure_minimum_slide_count(
        self,
        *,
        normalized: List[SlideSpec],
        normalized_weights: List[float],
        topic: str,
        one_pager: OnePager,
    ) -> None:
        if normalized:
            return

        topic_text = _sanitize_text(topic) or "当前主题"
        normalized.append(
            SlideSpec(
                title="证据缺口与后续动作",
                bullets=[
                    f"{topic_text} 当前证据不足，无法安全生成完整技术讲解。",
                    "建议补充可验证论文、代码实现、线上指标后重试。",
                ],
                duration_sec=30,
                notes="避免在证据薄弱时输出模板化或推测性内容。",
                visual_kind="none",
                visual_payload={},
            )
        )
        normalized_weights.append(0.85)

    def _compute_slide_durations(
        self,
        *,
        slide_count: int,
        duration_estimate: str,
        normalized_weights: Sequence[float],
    ) -> List[int]:
        target_total = self._resolve_target_duration(duration_estimate, slide_count)
        weight_sum = sum(normalized_weights) if normalized_weights else float(slide_count)
        durations = [max(12, int(target_total * w / max(0.1, weight_sum))) for w in normalized_weights]
        diff = target_total - sum(durations)
        index = 0
        while diff != 0 and durations:
            pos = index % len(durations)
            if diff > 0:
                durations[pos] += 1
                diff -= 1
            elif durations[pos] > 12:
                durations[pos] -= 1
                diff += 1
            index += 1
            if index > 10000:
                break
        return durations

    def _apply_slide_durations(self, slides: List[SlideSpec], durations: Sequence[int]) -> None:
        for slide, duration in zip(slides, durations):
            slide.duration_sec = int(duration)
            if slide.visual_kind == "diagram" and not slide.visual_payload:
                slide.visual_payload = self._build_diagram_payload(slide.title, slide.bullets)

    def _build_slide_specs(
        self,
        *,
        topic: str,
        video_brief: VideoBrief,
        one_pager: OnePager,
        facts: List[Dict[str, Any]],
        search_results: Optional[List[Dict[str, Any]]] = None,
    ) -> List[SlideSpec]:
        points = self._collect_evidence_points(
            topic=topic,
            video_brief=video_brief,
            one_pager=one_pager,
            facts=facts,
            search_results=search_results,
        )
        drafts = self._draft_sections(topic=topic, points=points, one_pager=one_pager)
        self._ensure_minimum_section_drafts(drafts, topic)
        self._append_image_section_draft(drafts, one_pager)

        normalized, normalized_weights = self._normalize_drafts_to_slides(drafts=drafts, topic=topic)
        self._ensure_minimum_slide_count(
            normalized=normalized,
            normalized_weights=normalized_weights,
            topic=topic,
            one_pager=one_pager,
        )

        normalized = normalized[:10]
        normalized_weights = normalized_weights[: len(normalized)]
        if not normalized:
            return []

        durations = self._compute_slide_durations(
            slide_count=len(normalized),
            duration_estimate=video_brief.duration_estimate,
            normalized_weights=normalized_weights,
        )
        self._apply_slide_durations(normalized, durations)
        return normalized

    def _write_slide_plan_markdown(self, slides: List[SlideSpec], out_dir: Path) -> Path:
        lines: List[str] = ["# PPT Video Slide Plan", ""]
        total = sum(max(1, int(item.duration_sec)) for item in slides)
        lines.append(f"- slide_count: {len(slides)}")
        lines.append(f"- estimated_duration_sec: {total}")
        lines.append("")

        for idx, slide in enumerate(slides, start=1):
            lines.append(f"## {idx}. {slide.title} ({slide.duration_sec}s)")
            for bullet in slide.bullets[:6]:
                lines.append(f"- {bullet}")
            if slide.notes:
                lines.append(f"- Notes: {slide.notes}")
            lines.append("")

        path = out_dir / "video_slides_plan.md"
        path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return path

    def _write_slide_plan_json(self, slides: List[SlideSpec], out_dir: Path) -> Path:
        payload = {
            "provider": self.provider,
            "slide_count": len(slides),
            "total_duration_sec": sum(max(1, int(item.duration_sec)) for item in slides),
            "slides": [
                {
                    "title": item.title,
                    "bullets": list(item.bullets),
                    "duration_sec": int(item.duration_sec),
                    "notes": item.notes,
                    "visual_kind": item.visual_kind,
                }
                for item in slides
            ],
        }
        path = out_dir / "video_slides_plan.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def _write_script_contract(self, out_dir: Path) -> Path:
        titles = [_sanitize_text(item) for item in list(self._contract_slide_titles or []) if _sanitize_text(item)]
        framework = [f"{idx}) {title}" for idx, title in enumerate(titles, start=1)]
        if not framework:
            framework = ["1) 研究范围与核心结论", "2) 关键机制与指标", "3) 风险与下一步验证"]

        lines: List[str] = [
            "# Slide Script Contract",
            "",
            "## Framework",
            *[f"- {item}" for item in framework],
            "",
            "## Content Constraints",
            "- 每页最多 5 条正文要点。",
            "- 单条要点不允许硬截断（禁止 `...` / `…`）。",
            "- 单条要点必须是完整技术语义，不允许残缺括号或破碎短语。",
            "- 优先保留含 `API/检索/架构/指标/延迟/吞吐/成本/部署/实验` 的技术内容。",
            "- 当某板块证据不足时，允许显式标注“证据不足”，禁止编造细节。",
        ]
        path = out_dir / "video_script_contract.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def _ensure_runtime_environment(self) -> Path:
        if shutil.which("npm") is None or shutil.which("npx") is None:
            raise VideoGenerationError("Node.js/npm is required for Slidev mode. Please install Node.js 20+.")

        runtime_dir = self.runtime_dir
        runtime_dir.mkdir(parents=True, exist_ok=True)
        package_path = runtime_dir / "package.json"
        desired_package = {
            "name": "academic-research-agent-slidev-runtime",
            "private": True,
            "dependencies": dict(_RUNTIME_DEPENDENCIES),
        }

        needs_install = False
        if package_path.exists():
            try:
                current = json.loads(package_path.read_text(encoding="utf-8"))
            except Exception:
                current = {}
            if current != desired_package:
                needs_install = True
        else:
            needs_install = True

        if needs_install:
            package_path.write_text(
                json.dumps(desired_package, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        node_modules = runtime_dir / "node_modules"
        if needs_install or not node_modules.exists():
            self._emit_progress("Bootstrapping Slidev runtime (npm install)...")
            self._run_subprocess(
                ["npm", "install", "--no-audit", "--no-fund"],
                context="npm install",
                cwd=runtime_dir,
            )
        return runtime_dir

    def _render_chart_html(self, payload: Dict[str, Any]) -> str:
        labels = list(payload.get("labels") or [])
        values = list(payload.get("values") or [])
        normalized = list(payload.get("normalized") or [])
        if not labels:
            return "<div class='visual-card'><div class='visual-title'>指标图表</div><p>暂无可用指标。</p></div>"

        rows: List[str] = []
        count = min(5, len(labels), len(values), len(normalized))
        for idx in range(count):
            label = _escape_html(labels[idx])
            value = f"{float(values[idx]):.2f}"
            pct = max(8, min(100, int(float(normalized[idx]) * 100)))
            rows.append(
                "<div class='metric-row'>"
                f"<div class='metric-label'>{label}</div>"
                "<div class='metric-bar-bg'>"
                f"<div class='metric-bar' style='width: {pct}%;'></div>"
                "</div>"
                f"<div class='metric-value'>{value}</div>"
                "</div>"
            )

        return (
            "<div class='visual-card'>"
            "<div class='visual-title'>指标图表</div>"
            + "".join(rows)
            + "</div>"
        )

    def _render_diagram_panel(self, payload: Dict[str, Any]) -> str:
        title = _escape_html(payload.get("title", "System Diagram")) or "System Diagram"
        tags = [_escape_html(item) for item in (payload.get("tags") or []) if _sanitize_text(item)]
        if len(tags) < 3:
            tags = ["Inputs", "Reasoning", "Outputs"]

        parts: List[str] = [
            "<div class='visual-card'>",
            "<div class='visual-title'>流程图</div>",
            "<div class='flow-diagram'>",
            f"<div class='flow-node'>{title}</div>",
            "<div class='flow-arrow'>↓</div>",
            f"<div class='flow-node'>{tags[0]}</div>",
            "<div class='flow-arrow'>↓</div>",
            f"<div class='flow-node'>{tags[1]}</div>",
            "<div class='flow-arrow'>↓</div>",
            f"<div class='flow-node'>{tags[2]}</div>",
        ]
        if len(tags) >= 4:
            parts.append(f"<div class='flow-branch'>补充：{tags[3]}</div>")
        parts.extend(["</div>", "</div>"])
        return "\n".join(parts)

    def _render_image_panel(self, payload: Dict[str, Any]) -> str:
        asset_rel = _sanitize_text(payload.get("asset_rel", ""))
        caption = _escape_html(payload.get("caption", "图像证据"))
        if asset_rel:
            return (
                "<div class='visual-card'>\n"
                "<div class='visual-title'>图像证据</div>\n\n"
                f"<img class='evidence-image' src='{_escape_html(asset_rel)}' alt='{caption}' />\n"
                f"<div class='image-caption'>{caption}</div>\n"
                "</div>"
            )
        return (
            "<div class='visual-card'>\n"
            "<div class='visual-title'>图像证据</div>\n"
            "<p>未提供可用图片资源（可在 one_pager.resources.image_path 提供本地路径）。</p>\n"
            "</div>"
        )

    def _render_none_panel(self) -> str:
        return (
            "<div class='visual-card'>\n"
            "<div class='visual-title'>摘要</div>\n"
            "<p>本页聚焦关键结论与可执行建议。</p>\n"
            "</div>"
        )

    def _render_visual_panel(self, slide: SlideSpec) -> str:
        if slide.visual_kind == "chart":
            return self._render_chart_html(slide.visual_payload)
        if slide.visual_kind == "image":
            return self._render_image_panel(slide.visual_payload)
        if slide.visual_kind == "none":
            return self._render_none_panel()
        return self._render_diagram_panel(slide.visual_payload)

    def _render_left_panel(self, slide: SlideSpec) -> str:
        parts: List[str] = [
            "<section class='deck-main'>",
            "<ul class='deck-bullets'>",
        ]
        for bullet in slide.bullets:
            parts.append(f"<li>{_escape_html(bullet)}</li>")
        parts.append("</ul>")
        if slide.notes:
            parts.append(f"<div class='speaker-notes'>Notes: {_escape_html(slide.notes)}</div>")
        parts.append("</section>")
        return "\n".join(parts)

    def _slidev_styles(self) -> str:
        return """
:root {
  --deck-blue: #1f5fd1;
  --deck-blue-light: #dbeafe;
  --deck-bg: #eef3ff;
  --deck-card-bg: #f8fbff;
  --deck-border: #c7d9f8;
  --deck-text: #163663;
}
.slidev-layout {
  background: linear-gradient(180deg, #f7f9ff 0%, var(--deck-bg) 100%);
  color: var(--deck-text);
}
.slidev-layout h1 {
  color: #ffffff;
  background: linear-gradient(90deg, #1e58c8, #2f74ef);
  margin: -1.1rem -2.4rem 0.8rem -2.4rem;
  padding: 0.65rem 1.4rem;
  border-radius: 0 0 12px 12px;
  font-size: 1.32rem;
  line-height: 1.22;
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
  overflow: hidden;
  min-height: 3.15rem;
}
.deck-grid {
  display: grid;
  grid-template-columns: 1.28fr 1fr;
  gap: 0.9rem;
  align-items: start;
}
.deck-main {
  border: 1px solid var(--deck-border);
  background: var(--deck-card-bg);
  border-radius: 14px;
  min-height: 430px;
  padding: 0.85rem 1rem 0.75rem 1rem;
}
.deck-bullets {
  margin: 0;
  padding-left: 1.1rem;
  line-height: 1.4;
  font-size: 0.95rem;
}
.deck-bullets li {
  margin-bottom: 0.35rem;
  overflow-wrap: anywhere;
}
.deck-side {
  min-height: 430px;
}
.speaker-notes {
  margin-top: 0.8rem;
  color: #5c76a8;
  font-size: 0.82rem;
}
.visual-card {
  border: 1px solid var(--deck-border);
  background: var(--deck-card-bg);
  border-radius: 14px;
  padding: 0.8rem 0.9rem;
  min-height: 420px;
  font-size: 0.9rem;
  overflow: hidden;
}
.visual-title {
  font-weight: 700;
  margin-bottom: 0.5rem;
  color: #1b4382;
}
.metric-row {
  display: grid;
  grid-template-columns: 40% 42% 18%;
  gap: 0.4rem;
  align-items: start;
  margin-bottom: 0.45rem;
}
.metric-label {
  font-size: 0.78rem;
  line-height: 1.2;
  overflow-wrap: anywhere;
  white-space: normal;
}
.metric-bar-bg {
  width: 100%;
  background: #d5e3fb;
  border-radius: 999px;
  height: 0.55rem;
}
.metric-bar {
  background: #4d8ef5;
  height: 100%;
  border-radius: 999px;
}
.metric-value {
  font-variant-numeric: tabular-nums;
  text-align: right;
  padding-top: 0.05rem;
}
.flow-diagram {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  gap: 0.35rem;
}
.flow-node {
  border: 1px solid #9eb8eb;
  background: #eaf2ff;
  border-radius: 10px;
  padding: 0.46rem 0.55rem;
  font-size: 0.88rem;
  text-align: center;
  overflow-wrap: anywhere;
}
.flow-arrow {
  text-align: center;
  color: #5075b8;
  font-weight: 700;
}
.flow-branch {
  margin-top: 0.25rem;
  border-radius: 9px;
  background: #e3f4eb;
  border: 1px solid #95d2ac;
  padding: 0.4rem 0.5rem;
  font-size: 0.84rem;
  overflow-wrap: anywhere;
}
.evidence-image {
  display: block;
  width: 100%;
  max-height: 320px;
  object-fit: contain;
  border-radius: 10px;
  border: 1px solid #bfd3f7;
  background: #ffffff;
  margin-top: 0.2rem;
}
.image-caption {
  margin-top: 0.4rem;
  font-size: 0.8rem;
  color: #5070a5;
}
"""

    def _copy_image_assets(self, slides: List[SlideSpec], build_dir: Path) -> None:
        assets_dir = build_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        for idx, slide in enumerate(slides, start=1):
            if slide.visual_kind != "image":
                continue
            raw = _sanitize_text(slide.visual_payload.get("image_path", ""))
            if not raw:
                continue
            source = Path(raw).expanduser()
            if not source.exists() or not source.is_file():
                continue
            safe_name = f"image_{idx:02d}{source.suffix.lower() or '.png'}"
            target = assets_dir / safe_name
            shutil.copy2(source, target)
            slide.visual_payload["asset_rel"] = f"./assets/{safe_name}"

    def _write_slidev_source(self, *, topic: str, slides: List[SlideSpec], build_dir: Path) -> Path:
        build_dir.mkdir(parents=True, exist_ok=True)
        self._copy_image_assets(slides, build_dir)

        style_path = build_dir / "style.css"
        style_path.write_text(self._slidev_styles().strip() + "\n", encoding="utf-8")

        lines: List[str] = [
            "---",
            "theme: default",
            "layout: default",
            f'title: "{_sanitize_text(topic)}"',
            "mdc: true",
            "katex: true",
            "lineNumbers: false",
            "fonts:",
            "  sans: \"'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', 'Noto Sans CJK SC', sans-serif\"",
            "---",
            "",
        ]

        for idx, slide in enumerate(slides):
            if idx > 0:
                lines.extend(["---", ""])
            lines.append(f"# {_sanitize_text(slide.title)}")
            lines.append("")

            lines.append("<div class='deck-grid'>")
            lines.append(self._render_left_panel(slide))
            lines.append("<aside class='deck-side'>")
            lines.append(self._render_visual_panel(slide))
            lines.append("</aside>")
            lines.append("</div>")
            lines.append("")

        entry_path = build_dir / "slides.md"
        entry_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return entry_path

    def _export_slides_with_slidev(self, *, runtime_dir: Path, entry_path: Path, slides_dir: Path) -> None:
        slides_dir.mkdir(parents=True, exist_ok=True)
        for old in slides_dir.glob("*.png"):
            old.unlink(missing_ok=True)

        runtime_dir = runtime_dir.resolve()
        entry_path = entry_path.resolve()
        slides_dir = slides_dir.resolve()

        env = dict(os.environ)
        env["CI"] = "1"

        cmd = [
            "npx",
            "slidev",
            "export",
            str(entry_path),
            "--format",
            "png",
            "--output",
            str(slides_dir),
            "--with-clicks=false",
            "--per-slide",
            "--timeout",
            str(self.slidev_timeout_ms),
            "--wait",
            str(self.slidev_wait_ms),
            "--wait-until",
            "networkidle",
        ]
        self._run_subprocess(
            cmd,
            context="slidev export",
            cwd=runtime_dir,
            env=env,
        )

    def _sorted_exported_images(self, slides_dir: Path) -> List[Path]:
        def _key(path: Path) -> Tuple[int, str]:
            match = re.search(r"(\d+)$", path.stem)
            number = int(match.group(1)) if match else 10**9
            return number, path.name

        images = sorted(slides_dir.glob("*.png"), key=_key)
        return images

    def _render_narration_segments(
        self,
        *,
        slides: List[SlideSpec],
        out_dir: Path,
        ffmpeg_bin: str,
    ) -> Dict[str, Any]:
        try:
            return self._narration().render_narration_segments(
                slides=slides,
                out_dir=out_dir,
                ffmpeg_bin=ffmpeg_bin,
            )
        except RuntimeError as e:
            raise VideoGenerationError(str(e)) from e

    def _concat_audio_segments(self, audio_paths: List[Path], output_path: Path, ffmpeg_bin: str) -> None:
        try:
            self._narration().concat_audio_segments(audio_paths, output_path, ffmpeg_bin)
        except RuntimeError as e:
            raise VideoGenerationError(str(e)) from e

    def _render_video_segments(
        self,
        image_paths: List[Path],
        slides: List[SlideSpec],
        segments_dir: Path,
        ffmpeg_bin: str,
    ) -> List[Path]:
        segments_dir.mkdir(parents=True, exist_ok=True)
        for old in segments_dir.glob("segment_*.mp4"):
            old.unlink(missing_ok=True)

        segment_paths: List[Path] = []
        for idx, (image_path, slide) in enumerate(zip(image_paths, slides), start=1):
            duration = max(1, int(slide.duration_sec))
            segment_path = segments_dir / f"segment_{idx:03d}.mp4"
            cmd = [
                ffmpeg_bin,
                "-y",
                "-loop",
                "1",
                "-framerate",
                str(self.fps),
                "-t",
                str(duration),
                "-i",
                str(image_path),
                "-vf",
                f"scale={self.width}:{self.height},format=yuv420p",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                str(segment_path),
            ]
            self._run_subprocess(cmd, context=f"ffmpeg segment render (slide {idx})")
            segment_paths.append(segment_path)
        return segment_paths

    def _concat_segments(
        self,
        segment_paths: List[Path],
        output_path: Path,
        ffmpeg_bin: str,
        narration_audio_path: Optional[Path] = None,
    ) -> None:
        concat_path = output_path.parent / "video_segments_concat.txt"

        def _escape(path: str) -> str:
            return path.replace("'", "'\\''")

        concat_lines = [f"file '{_escape(str(path.resolve()))}'" for path in segment_paths]
        concat_path.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")

        video_only_path = output_path
        if narration_audio_path:
            video_only_path = output_path.parent / f"{output_path.stem}.video_only.mp4"

        cmd = [
            ffmpeg_bin,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-movflags",
            "+faststart",
            str(video_only_path),
        ]
        self._run_subprocess(cmd, context="ffmpeg concat")

        if not narration_audio_path:
            return

        mux_cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(video_only_path),
            "-i",
            str(narration_audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-af",
            "apad",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        self._run_subprocess(mux_cmd, context="ffmpeg mux narration")
        video_only_path.unlink(missing_ok=True)

    def _require_ffmpeg(self) -> str:
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            raise VideoGenerationError("ffmpeg is required for Slidev mode. Please install ffmpeg first.")
        return ffmpeg_bin

    def _build_plan_assets(
        self,
        *,
        topic: str,
        out_dir: Path,
        video_brief: VideoBrief,
        one_pager: OnePager,
        facts: List[Dict[str, Any]],
        search_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[SlideSpec], Path, Path, Path]:
        slide_kwargs: Dict[str, Any] = {
            "topic": topic,
            "video_brief": video_brief,
            "one_pager": one_pager,
            "facts": facts,
        }
        try:
            signature = inspect.signature(self._build_slide_specs)
            if "search_results" in signature.parameters:
                slide_kwargs["search_results"] = search_results
        except Exception:
            slide_kwargs["search_results"] = search_results

        slides = self._build_slide_specs(**slide_kwargs)
        if not slides:
            raise VideoGenerationError("No slide content available to render video.")

        self._contract_slide_titles = [slide.title for slide in slides]
        plan_md = self._write_slide_plan_markdown(slides, out_dir)
        plan_json = self._write_slide_plan_json(slides, out_dir)
        contract_md = self._write_script_contract(out_dir)
        self._emit_progress(f"Slide plan ready: {len(slides)} slides.")
        return slides, plan_md, plan_json, contract_md

    def _prepare_slidev_build(
        self,
        *,
        topic: str,
        out_dir: Path,
        slides: List[SlideSpec],
    ) -> Tuple[Path, Path, Path]:
        runtime_dir = self._ensure_runtime_environment()
        build_id = f"{_slugify(out_dir.name)}_{int(time.time())}"
        build_dir = runtime_dir / "builds" / build_id
        build_dir.mkdir(parents=True, exist_ok=True)

        self._emit_progress("Rendering Slidev source...")
        entry_path = self._write_slidev_source(topic=topic, slides=slides, build_dir=build_dir)

        source_snapshot = out_dir / "video_slidev_source"
        if source_snapshot.exists():
            shutil.rmtree(source_snapshot)
        shutil.copytree(build_dir, source_snapshot)
        return runtime_dir, entry_path, source_snapshot

    def _export_slide_images(
        self,
        *,
        runtime_dir: Path,
        entry_path: Path,
        out_dir: Path,
        slides: List[SlideSpec],
    ) -> Tuple[Path, List[Path], List[SlideSpec]]:
        slides_dir = out_dir / "video_slides"
        self._emit_progress("Exporting slides via Slidev...")
        self._export_slides_with_slidev(
            runtime_dir=runtime_dir,
            entry_path=entry_path,
            slides_dir=slides_dir,
        )

        image_paths = self._sorted_exported_images(slides_dir)
        if not image_paths:
            raise VideoGenerationError("Slidev export produced no slide images.")

        if len(image_paths) != len(slides):
            logger.warning(
                "Slide count mismatch after export: specs=%s, images=%s",
                len(slides),
                len(image_paths),
            )
            effective = min(len(image_paths), len(slides))
            image_paths = image_paths[:effective]
            slides = slides[:effective]
            if not image_paths:
                raise VideoGenerationError("Slidev export output mismatch left no usable slides.")

        return slides_dir, image_paths, slides

    def _apply_narration_durations(self, slides: List[SlideSpec], narration_durations: Sequence[float]) -> None:
        min_slide_duration = 20 if len(slides) <= 4 else 12
        for idx, duration in enumerate(narration_durations):
            if idx >= len(slides):
                break
            slides[idx].duration_sec = max(min_slide_duration, int(math.ceil(float(duration) + 0.12)))

    def _maybe_render_narration(
        self,
        *,
        slides: List[SlideSpec],
        out_dir: Path,
        ffmpeg_bin: str,
        plan_md: Path,
        plan_json: Path,
    ) -> Tuple[Path, Path, Optional[Path], Optional[Path], Optional[Path], List[str], List[NarrationSpec]]:
        if not self.enable_narration:
            return plan_md, plan_json, None, None, None, [], []

        self._emit_progress("Generating narration audio track...")
        narration_meta = self._render_narration_segments(
            slides=slides,
            out_dir=out_dir,
            ffmpeg_bin=ffmpeg_bin,
        )
        narration_script_path = narration_meta["script_path"]
        narration_dir = narration_meta["narration_dir"]
        tts_providers = list(narration_meta.get("providers_used") or [])
        narration_specs = list(narration_meta.get("narration_specs") or [])
        self._apply_narration_durations(slides, list(narration_meta.get("segment_durations") or []))

        # Rewrite plan files with narration-aligned durations to keep timestamps consistent.
        plan_md = self._write_slide_plan_markdown(slides, out_dir)
        plan_json = self._write_slide_plan_json(slides, out_dir)
        narration_audio_path = out_dir / "video_narration.m4a"
        self._emit_progress("Concatenating narration audio...")
        self._concat_audio_segments(
            audio_paths=list(narration_meta["segment_paths"]),
            output_path=narration_audio_path,
            ffmpeg_bin=ffmpeg_bin,
        )
        return (
            plan_md,
            plan_json,
            narration_audio_path,
            narration_script_path,
            narration_dir,
            tts_providers,
            narration_specs,
        )

    def _build_slide_timeline(self, slides: Sequence[SlideSpec]) -> List[Dict[str, Any]]:
        timeline: List[Dict[str, Any]] = []
        cursor_sec = 0
        for idx, slide in enumerate(slides, start=1):
            duration = max(1, int(slide.duration_sec))
            timeline.append(
                {
                    "slide_index": idx,
                    "title": slide.title,
                    "start_sec": cursor_sec,
                    "duration_sec": duration,
                }
            )
            cursor_sec += duration
        return timeline

    def _build_flow_metadata(
        self,
        *,
        slides: Sequence[SlideSpec],
        slides_dir: Path,
        segments_dir: Path,
        plan_md: Path,
        plan_json: Path,
        contract_md: Path,
        source_snapshot: Path,
        runtime_dir: Path,
        entry_path: Path,
        narration_audio_path: Optional[Path],
        narration_script_path: Optional[Path],
        narration_dir: Optional[Path],
        tts_providers: Sequence[str],
        narration_specs: Sequence[NarrationSpec],
    ) -> Dict[str, Any]:
        flow: Dict[str, Any] = {
            "slide_count": len(slides),
            "estimated_duration_sec": int(sum(max(1, int(item.duration_sec)) for item in slides)),
            "slides_dir": str(slides_dir),
            "segments_dir": str(segments_dir),
            "plan_md": str(plan_md),
            "plan_json": str(plan_json),
            "contract_md": str(contract_md),
            "source_snapshot": str(source_snapshot),
            "runtime_dir": str(runtime_dir),
            "entry_path": str(entry_path),
            "fps": self.fps,
            "resolution": f"{self.width}x{self.height}",
            "slide_timeline": self._build_slide_timeline(slides),
        }
        if narration_audio_path:
            flow["narration_audio_path"] = str(narration_audio_path)
        if narration_script_path:
            flow["narration_script_path"] = str(narration_script_path)
        if narration_dir:
            flow["narration_segments_dir"] = str(narration_dir)
        if tts_providers:
            flow["tts_providers"] = list(tts_providers)
        if narration_specs:
            flow["narration_slide_count"] = len(narration_specs)
        return flow

    def _generate_video_sync(
        self,
        *,
        topic: str,
        out_dir: Path,
        video_brief: VideoBrief,
        one_pager: OnePager,
        facts: List[Dict[str, Any]],
        search_results: Optional[List[Dict[str, Any]]] = None,
        output_path: Path,
    ) -> Dict[str, Any]:
        ffmpeg_bin = self._require_ffmpeg()
        slides, plan_md, plan_json, contract_md = self._build_plan_assets(
            topic=topic,
            out_dir=out_dir,
            video_brief=video_brief,
            one_pager=one_pager,
            facts=facts,
            search_results=search_results,
        )
        runtime_dir, entry_path, source_snapshot = self._prepare_slidev_build(
            topic=topic,
            out_dir=out_dir,
            slides=slides,
        )
        slides_dir, image_paths, slides = self._export_slide_images(
            runtime_dir=runtime_dir,
            entry_path=entry_path,
            out_dir=out_dir,
            slides=slides,
        )
        (
            plan_md,
            plan_json,
            narration_audio_path,
            narration_script_path,
            narration_dir,
            tts_providers,
            narration_specs,
        ) = self._maybe_render_narration(
            slides=slides,
            out_dir=out_dir,
            ffmpeg_bin=ffmpeg_bin,
            plan_md=plan_md,
            plan_json=plan_json,
        )

        segments_dir = out_dir / "video_segments"
        self._emit_progress("Rendering per-slide video segments...")
        segment_paths = self._render_video_segments(
            image_paths=image_paths,
            slides=slides,
            segments_dir=segments_dir,
            ffmpeg_bin=ffmpeg_bin,
        )

        self._emit_progress("Concatenating final video...")
        self._concat_segments(
            segment_paths=segment_paths,
            output_path=output_path,
            ffmpeg_bin=ffmpeg_bin,
            narration_audio_path=narration_audio_path,
        )

        return self._build_flow_metadata(
            slides=slides,
            slides_dir=slides_dir,
            segments_dir=segments_dir,
            plan_md=plan_md,
            plan_json=plan_json,
            contract_md=contract_md,
            source_snapshot=source_snapshot,
            runtime_dir=runtime_dir,
            entry_path=entry_path,
            narration_audio_path=narration_audio_path,
            narration_script_path=narration_script_path,
            narration_dir=narration_dir,
            tts_providers=tts_providers,
            narration_specs=narration_specs,
        )

    async def generate(
        self,
        *,
        topic: str,
        out_dir: Path,
        video_brief: Optional[Dict[str, Any]] = None,
        one_pager: Optional[Dict[str, Any]] = None,
        facts: Optional[List[Dict[str, Any]]] = None,
        search_results: Optional[List[Dict[str, Any]]] = None,
    ) -> VideoArtifact:
        out_dir.mkdir(parents=True, exist_ok=True)
        prompt = build_video_prompt(
            topic=topic,
            video_brief=video_brief,
            one_pager=one_pager,
            facts=facts,
            search_results=search_results,
        )
        vb = VideoBrief.from_dict(video_brief or {}, default_title=f"{topic} Video Brief")
        op = OnePager.from_dict(one_pager or {}, default_title=f"{topic} One-Pager")
        normalized_facts = list(facts or [])
        normalized_results = list(search_results or [])

        output_path = out_dir / "video_brief.mp4"
        metadata_path = out_dir / "video_generation_metadata.json"

        sync_kwargs: Dict[str, Any] = {
            "topic": topic,
            "out_dir": out_dir,
            "video_brief": vb,
            "one_pager": op,
            "facts": normalized_facts,
            "output_path": output_path,
        }
        try:
            signature = inspect.signature(self._generate_video_sync)
            if "search_results" in signature.parameters:
                sync_kwargs["search_results"] = normalized_results
        except Exception:
            sync_kwargs["search_results"] = normalized_results

        flow_meta = await asyncio.to_thread(
            self._generate_video_sync,
            **sync_kwargs,
        )

        if not output_path.exists() or output_path.stat().st_size <= 0:
            raise VideoGenerationError("Slidev video rendering completed but output video is missing.")

        metadata_path.write_text(
            json.dumps(
                {
                    "provider": self.provider,
                    "topic": topic,
                    "flow": flow_meta,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        return VideoArtifact(
            provider=self.provider,
            prompt=prompt,
            output_path=output_path,
            metadata_path=metadata_path,
        )


def create_video_generator(provider: str = "slidev", **kwargs: Any) -> BaseVideoGenerator:
    """创建视频生成器。当前仅支持 Slidev 单路线。"""
    normalized = provider.strip().lower()
    if normalized not in {"slidev", "slides", "ppt", "ppt_video", "local", ""}:
        raise ValueError("Only slidev provider is supported now. Use provider='slidev'.")
    return SlidevVideoGenerator(**kwargs)
