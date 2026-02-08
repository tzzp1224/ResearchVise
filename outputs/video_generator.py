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


def _chunk(items: Sequence[str], size: int) -> List[List[str]]:
    cleaned = [item for item in (_sanitize_text(x) for x in items) if item]
    if size <= 0:
        return [cleaned] if cleaned else []
    return [cleaned[i : i + size] for i in range(0, len(cleaned), size)]


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

    cut_points = [":", "：", "。", ".", "，", ",", ";", "；", "(", "（"]
    for marker in cut_points:
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[0].strip()
            break

    limit = cjk_max if _contains_cjk(cleaned) else latin_max
    if len(cleaned) <= limit:
        return cleaned
    wrapped = _wrap_text_units(cleaned, max_len=limit)
    if wrapped:
        return " / ".join(wrapped[:2])
    return cleaned


def _wrap_for_preview(text: str, max_len: int = 120) -> str:
    cleaned = _sanitize_text(text)
    return _compact_text(cleaned, max_len=max_len) if cleaned else ""


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
) -> str:
    """构建视频讲解脚本提示词。"""
    vb = VideoBrief.from_dict(video_brief or {}, default_title=f"{topic} Video Brief")
    op = OnePager.from_dict(one_pager or {}, default_title=f"{topic} One-Pager")
    facts = facts or []

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

Slide plan:
{os.linesep.join(segment_lines) if segment_lines else "- Follow a 3-part technical structure with architecture, benchmarks, and deployment trade-offs."}

Constraints:
- Keep high factual density and explicit technical details.
- Explain mechanism, benchmark context, engineering trade-offs, and deployment guidance.
- Avoid hype and unsupported claims.
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
    ) -> VideoArtifact:
        raise NotImplementedError


@dataclass
class SlideSpec:
    title: str
    bullets: List[str] = field(default_factory=list)
    duration_sec: int = 30
    notes: str = ""
    visual_kind: str = "diagram"  # diagram / chart / formula / image / none
    visual_payload: Dict[str, Any] = field(default_factory=dict)


class SlidevVideoGenerator(BaseVideoGenerator):
    """
    单一路线视频生成器：Slidev + ffmpeg
    """

    provider = "slidev"

    def __init__(
        self,
        *,
        target_duration_sec: int = 420,
        min_duration_sec: int = 360,
        max_duration_sec: int = 600,
        fps: int = 24,
        width: int = 1920,
        height: int = 1080,
        runtime_dir: Optional[Path] = None,
        slidev_timeout_ms: int = 180000,
        slidev_wait_ms: int = 400,
        enable_narration: bool = True,
        tts_provider: str = "auto",
        tts_voice: Optional[str] = None,
        tts_speed: float = 1.2,
        narration_model: str = "deepseek-chat",
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.target_duration_sec = int(target_duration_sec)
        self.min_duration_sec = int(min_duration_sec)
        self.max_duration_sec = int(max_duration_sec)
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
        min_by_slide_count = slide_count * 22
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

    def _metric_symbol(self, metric_key: str, *, index: int = 0) -> str:
        lowered = _sanitize_text(metric_key).lower()
        symbol_map = [
            (("latency", "delay", "p95", "p99", "延迟", "时延"), "LAT"),
            (("throughput", "qps", "rps", "吞吐", "并发"), "THR"),
            (("cost", "price", "费用", "成本"), "COST"),
            (("precision", "准确", "em", "f1"), "ACC"),
            (("recall", "召回"), "REC"),
            (("coverage", "覆盖"), "COV"),
            (("error", "风险", "fail", "oom"), "ERR"),
            (("memory", "显存", "内存"), "MEM"),
        ]
        for keywords, symbol in symbol_map:
            if any(token in lowered for token in keywords):
                return symbol

        tokens = re.findall(r"[A-Za-z]+", lowered)
        if tokens:
            candidate = tokens[0][:4].upper()
            return candidate or f"M_{index + 1}"
        return f"M_{index + 1}"

    def _topic_symbol(self, topic: str) -> str:
        tokens = re.findall(r"[A-Za-z]+", _sanitize_text(topic))
        if tokens:
            return tokens[0][:4].upper()
        return "SYS"

    def _bullet_uniqueness_key(self, text: str) -> str:
        cleaned = _sanitize_text(text)
        cleaned = re.sub(r"^\[[^\]]+\]\s*", "", cleaned)
        cleaned = cleaned.lower()
        cleaned = re.sub(r"(第\s*\d+\s*点|point\s*\d+)\s*[:：]?", "", cleaned)
        cleaned = re.sub(r"\d+(?:\.\d+)?%?", "", cleaned)
        cleaned = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", cleaned)
        return cleaned

    def _build_formula_payload(
        self,
        *,
        topic: str,
        metrics: Dict[str, str],
        facts: Sequence[Dict[str, Any]],
        key_findings: Sequence[str],
    ) -> Dict[str, Any]:
        topic_symbol = self._topic_symbol(topic)
        metric_entries: List[Tuple[str, str, Optional[float]]] = []
        used_symbols: Dict[str, int] = {}
        for idx, (key, value) in enumerate(list(metrics.items())[:8]):
            symbol = self._metric_symbol(str(key), index=idx)
            dup_count = used_symbols.get(symbol, 0)
            used_symbols[symbol] = dup_count + 1
            if dup_count > 0:
                symbol = f"{symbol}_{dup_count + 1}"
            metric_entries.append((symbol, _sanitize_text(str(key)), _extract_numeric(str(value))))

        positives: List[str] = []
        negatives: List[str] = []
        for symbol, name, _ in metric_entries:
            lowered = name.lower()
            if any(token in lowered for token in ["latency", "delay", "cost", "error", "risk", "延迟", "成本", "风险", "失效"]):
                negatives.append(symbol)
            else:
                positives.append(symbol)

        signed_terms: List[str] = []
        for idx, symbol in enumerate(positives[:2], start=1):
            signed_terms.append(f"+ w_{{{idx}}} \\cdot {symbol}")
        for idx, symbol in enumerate(negatives[:2], start=1):
            signed_terms.append(f"- \\lambda_{{{idx}}} \\cdot {symbol}")
        if not signed_terms:
            signed_terms = [
                r"+ w_{1} \cdot Quality",
                r"+ w_{2} \cdot Coverage",
                r"- \lambda_{1} \cdot Latency",
                r"- \lambda_{2} \cdot Cost",
            ]

        objective = f"Score_{{{topic_symbol}}} = " + " ".join(signed_terms).lstrip("+ ").strip()

        primary_symbol = metric_entries[0][0] if metric_entries else "M"
        normalized_metric = (
            f"{primary_symbol}_{{norm}} = "
            f"\\frac{{{primary_symbol} - {primary_symbol}_{{min}}}}{{{primary_symbol}_{{max}} - {primary_symbol}_{{min}} + \\varepsilon}}"
        )

        fact_count = max(1, min(12, len(list(facts))))
        evidence = (
            f"Evidence_{{{topic_symbol}}} = "
            f"\\frac{{\\sum_{{i=1}}^{{{fact_count}}} c_i \\cdot w_i}}{{\\sum_{{i=1}}^{{{fact_count}}} w_i}}"
        )

        domain_text = " ".join(
            [
                _sanitize_text(topic),
                " ".join([_sanitize_text(item) for item in key_findings[:4]]),
                " ".join([_sanitize_text(str((fact or {}).get("claim", ""))) for fact in list(facts)[:4]]),
            ]
        ).lower()
        if any(token in domain_text for token in ["retrieval", "rag", "检索", "召回"]):
            domain_formula = r"Recall@k = \frac{Relevant@k}{Relevant\_Total}"
        elif any(token in domain_text for token in ["video", "world model", "时序", "一致性"]):
            domain_formula = r"TemporalConsistency_t = \frac{1}{N}\sum_{i=1}^{N}\mathbb{1}(state_i^{t} \approx state_i^{t-1})"
        elif positives and negatives:
            domain_formula = f"Efficiency = \\frac{{{positives[0]}}}{{{negatives[0]} + \\varepsilon}}"
        else:
            domain_formula = r"Throughput = \frac{Requests}{Second}"

        formulas = [objective, normalized_metric, evidence, domain_formula]
        return {"formulas": formulas}

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

    def _build_slide_specs(
        self,
        *,
        topic: str,
        video_brief: VideoBrief,
        one_pager: OnePager,
        facts: List[Dict[str, Any]],
    ) -> List[SlideSpec]:
        facts_sorted = sorted(list(facts or []), key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
        fact_bullets = [self._fact_to_bullet(item) for item in facts_sorted]
        fact_bullets = [item for item in fact_bullets if item]

        segment_pool: List[str] = []
        for segment in video_brief.segments[:8]:
            segment_pool.append(_sanitize_text(segment.get("title", "")))
            segment_pool.append(_sanitize_text(segment.get("content", "")))
            segment_pool.extend([_sanitize_text(item) for item in list(segment.get("talking_points") or [])[:4]])

        metric_lines = [
            f"{_sanitize_text(key)}: {_sanitize_text(value)}"
            for key, value in list(one_pager.metrics.items())[:8]
            if _sanitize_text(key) and _sanitize_text(value)
        ]
        deep_dive_pool = list(one_pager.technical_deep_dive or [])
        impl_pool = list(one_pager.implementation_notes or [])
        risk_pool = list(one_pager.risks_and_mitigations or []) + list(one_pager.weaknesses or [])
        strength_pool = list(one_pager.strengths or []) + list(one_pager.key_findings or [])

        global_pool = _dedupe_texts(
            [
                one_pager.executive_summary,
                video_brief.conclusion,
                *segment_pool,
                *deep_dive_pool,
                *impl_pool,
                *strength_pool,
                *risk_pool,
                *fact_bullets,
            ]
        )

        fallback_templates: Dict[str, List[str]] = {
            "问题定义与学习目标": [
                "输入约束：主题、检索范围、时间窗口与证据类型必须显式给定。",
                "输出目标：机制解释、指标口径、工程决策三类结果齐备。",
                "质量要求：仅保留可验证技术事实，过滤叙事性表述。",
                "验收标准：每页至少包含一条可执行工程建议。",
            ],
            "系统边界与输入输出": [
                "输入侧包含论文、代码仓库、技术文档与实验日志。",
                "处理中间层拆分为检索、推理、生成与评估四段。",
                "输出侧固定为结构化要点、指标表与可复现结论。",
                "边界定义避免把未经验证信息混入核心链路。",
            ],
            "核心架构：检索-推理-生成链路": [
                "检索层负责召回高相关证据并附带来源与置信信息。",
                "推理层在证据约束下完成机制归纳与冲突消解。",
                "生成层按固定模板输出讲解页，保证结构一致性。",
                "评估层基于准确率、延迟与成本做闭环优化。",
            ],
            "机制一：API上下文检索": [
                "先基于主题构建查询，再生成 API 候选集合。",
                "候选通过相关性与可用性双指标进行重排序。",
                "命中证据写入上下文窗口并保留来源锚点。",
                "检索失败场景必须回退到可解释的默认策略。",
            ],
            "机制二：世界建模与视频生成": [
                "模型需要同时保持空间一致性与时序一致性。",
                "生成控制信号来自结构化上下文而非自由采样。",
                "复杂场景重点关注动作连续性与因果合理性。",
                "输出质量通过可控性、稳定性与事实性联合评估。",
            ],
            "关键公式与计算框架": [
                "目标函数显式权衡质量、覆盖、延迟与成本四个维度。",
                "指标需要归一化后再进入加权评分，避免量纲污染。",
                "证据分数应按来源可信度进行加权平均。",
                "公式服务于决策解释，必须与实验结论联合使用。",
            ],
            "实验设置与指标口径": [
                "评估必须同时覆盖质量、覆盖率、延迟与单位成本。",
                "Precision@k 用于衡量检索命中质量。",
                "Throughput 用于衡量并发扩展能力。",
                "所有指标都需声明数据口径与统计区间。",
            ],
            "结果与证据解读": [
                "结果解读区分“观察事实”与“外推结论”。",
                "高置信证据优先进入主结论链路。",
                "冲突证据必须显式标注并给出解释路径。",
                "结论仅在已验证边界内成立。",
            ],
            "工程实现与模块拆解": [
                "模块划分建议采用检索服务、编排服务与渲染服务。",
                "状态管理采用幂等任务与可重试队列。",
                "缓存层区分热数据与冷数据并设置失效策略。",
                "故障隔离要求单模块异常不影响整条流水线。",
            ],
            "部署、并发与成本": [
                "部署目标是稳定吞吐与可预测尾延迟。",
                "并发策略采用队列限流与批处理调度组合。",
                "成本监控关注每请求算力时长与存储开销。",
                "上线前必须完成容量压测与降级预案演练。",
            ],
            "风险与失败模式": [
                "主要风险包括证据噪声、检索漂移与生成偏差。",
                "缓解策略包含数据回放、告警阈值与人工抽检。",
                "高风险结果需触发二次验证而非直接发布。",
                "失败案例应沉淀为回归测试样本库。",
            ],
            "结论与下一步验证": [
                "下一步优先补齐缺失指标并建立自动评测基线。",
                "对关键机制执行 A/B 验证并跟踪回归曲线。",
                "以工程成本为约束持续优化质量-延迟平衡点。",
                "发布前完成可复现实验与文档化交付。",
            ],
        }

        def _resolve_template(title: str) -> List[str]:
            if title in fallback_templates:
                return fallback_templates[title]
            for key, value in fallback_templates.items():
                if title in key or key in title:
                    return value
            return []

        def fallback(points: Sequence[str], title: str) -> List[str]:
            items = self._fit_bullets(points, max_bullets=4, max_len=72, technical_only=True)
            if items:
                return items
            template = _resolve_template(title)
            if template:
                return template[:4]
            return [f"{title}：技术证据不足，建议补充可复现实验与指标后重试。"]

        def diagram_payload(title: str, tags: Sequence[str]) -> Dict[str, Any]:
            cleaned = [_sanitize_text(tag) for tag in tags if _sanitize_text(tag)]
            return {"title": title, "tags": cleaned[:4]}

        def slide_points(
            *,
            include: Sequence[str],
            exclude: Sequence[str] = (),
            pools: Sequence[Sequence[str]],
            max_bullets: int = 4,
        ) -> List[str]:
            collected: List[str] = []
            for pool in pools:
                collected.extend(
                    self._select_points(
                        pool,
                        include_keywords=include,
                        exclude_keywords=exclude,
                        max_bullets=max_bullets,
                        max_len=72,
                    )
                )
            deduped = _dedupe_texts(collected)
            return deduped[:max_bullets]

        slides: List[SlideSpec] = []

        intro_points = fallback(
            slide_points(
                include=["目标", "问题", "输入", "输出", "指标", "评估", "技术", "约束"],
                pools=[deep_dive_pool, strength_pool, segment_pool, global_pool],
                max_bullets=4,
            )
            + [f"任务范围：{_sanitize_text(topic)}"],
            "问题定义与学习目标",
        )
        slides.append(
            SlideSpec(
                title="问题定义与学习目标",
                bullets=intro_points[:4],
                notes="定义问题、输入和可验证输出。",
                visual_kind="diagram",
                visual_payload=diagram_payload("问题定义", ["问题输入", "目标输出", "评估指标", "学习路径"]),
            )
        )

        boundary_points = slide_points(
            include=["输入", "输出", "接口", "api", "pipeline", "流程", "模块", "source"],
            pools=[segment_pool, impl_pool, deep_dive_pool, global_pool],
            max_bullets=4,
        )
        slides.append(
            SlideSpec(
                title="系统边界与输入输出",
                bullets=fallback(boundary_points or segment_pool, "系统边界与输入输出")[:4],
                notes="先给系统边界，再进入机制层。",
                visual_kind="diagram",
                visual_payload=diagram_payload("系统边界", ["输入来源", "上下文检索", "结构化输出", "生成约束"]),
            )
        )

        architecture_points = slide_points(
            include=["架构", "pipeline", "流程", "检索", "推理", "生成", "context", "模型"],
            pools=[deep_dive_pool, segment_pool, impl_pool, global_pool],
            max_bullets=4,
        )
        slides.append(
            SlideSpec(
                title="核心架构：检索-推理-生成链路",
                bullets=fallback(architecture_points, "核心架构")[:4],
                notes="把组件关系讲清楚，避免抽象空话。",
                visual_kind="diagram",
                visual_payload=diagram_payload("核心架构", ["事实检索", "上下文推理", "内容生成", "质量评估"]),
            )
        )

        retrieval_points = slide_points(
            include=["api", "检索", "context", "rag", "索引", "precision", "top"],
            pools=[deep_dive_pool, fact_bullets, segment_pool, global_pool],
            max_bullets=4,
        )
        slides.append(
            SlideSpec(
                title="机制一：API上下文检索",
                bullets=fallback(retrieval_points, "机制一")[:4],
                notes="强调检索口径、命中质量和误差来源。",
                visual_kind="diagram",
                visual_payload=diagram_payload("机制一", ["Query构建", "API候选", "命中排序", "一致性约束"]),
            )
        )

        generation_points = slide_points(
            include=["视频", "生成", "世界", "空间", "时序", "一致性", "sora", "model"],
            pools=[segment_pool, deep_dive_pool, fact_bullets, global_pool],
            max_bullets=4,
        )
        slides.append(
            SlideSpec(
                title="机制二：世界建模与视频生成",
                bullets=fallback(generation_points, "机制二")[:4],
                notes="把建模能力与局限并列说明。",
                visual_kind="diagram",
                visual_payload=diagram_payload("机制二", ["空间建模", "时序一致性", "动作推理", "可控生成"]),
            )
        )

        slides.append(
            SlideSpec(
                title="关键公式与计算框架",
                bullets=[
                    "统一目标函数用于平衡质量、覆盖、延迟与成本。",
                    "Precision@k 评估检索质量，Throughput 评估系统扩展能力。",
                    "公式用于解释工程决策，不用于替代实验验证。",
                ],
                notes="先解释变量含义，再给数值口径。",
                visual_kind="formula",
                visual_payload=self._build_formula_payload(
                    topic=topic,
                    metrics=one_pager.metrics,
                    facts=facts_sorted,
                    key_findings=one_pager.key_findings,
                ),
            )
        )

        metric_chart = self._build_metrics_chart_payload(one_pager.metrics)
        metric_points = fallback(metric_lines + fact_bullets, "实验设置与指标口径")
        slides.append(
            SlideSpec(
                title="实验设置与指标口径",
                bullets=metric_points[:4],
                notes="每个指标都要说明测量方式和口径。",
                visual_kind="chart",
                visual_payload=metric_chart,
            )
        )

        evidence_points = fallback(fact_bullets + strength_pool, "结果与证据解读")
        slides.append(
            SlideSpec(
                title="结果与证据解读",
                bullets=evidence_points[:5],
                notes="区分“观察到的结果”和“可外推结论”。",
                visual_kind="diagram",
                visual_payload=diagram_payload("结果解读", ["实验结果", "证据锚点", "适用边界", "外推风险"]),
            )
        )

        implementation_points = slide_points(
            include=["模块", "服务", "缓存", "调度", "接口", "部署", "pipeline", "engine", "索引"],
            pools=[impl_pool, deep_dive_pool, global_pool],
            max_bullets=4,
        )
        slides.append(
            SlideSpec(
                title="工程实现与模块拆解",
                bullets=fallback(implementation_points + impl_pool, "工程实现与模块拆解")[:4],
                notes="强调模块边界、状态管理和依赖关系。",
                visual_kind="diagram",
                visual_payload=diagram_payload("工程实现", ["模块划分", "状态管理", "缓存策略", "故障隔离"]),
            )
        )

        scale_points = slide_points(
            include=["延迟", "吞吐", "成本", "并发", "部署", "资源", "gpu", "cpu", "扩展"],
            pools=[metric_lines, impl_pool, risk_pool, fact_bullets, global_pool],
            max_bullets=4,
        )
        slides.append(
            SlideSpec(
                title="部署、并发与成本",
                bullets=fallback(scale_points + metric_lines, "部署、并发与成本")[:4],
                notes="同时说明性能上限与资源预算。",
                visual_kind="chart",
                visual_payload=metric_chart,
            )
        )

        risk_points = fallback(risk_pool + one_pager.weaknesses + fact_bullets, "风险与失败模式")
        slides.append(
            SlideSpec(
                title="风险与失败模式",
                bullets=risk_points[:4],
                notes="给出可执行缓解策略，不只列问题。",
                visual_kind="none",
            )
        )

        closing_points = fallback(
            slide_points(
                include=["下一步", "实验", "评估", "部署", "指标", "成本", "延迟", "并发", "验证", "监控"],
                pools=[impl_pool, risk_pool, metric_lines, fact_bullets, global_pool],
                max_bullets=4,
            )
            + [
                "下一步：补齐缺失指标并做可复现实验。",
                "下一步：构建自动化评估基线，跟踪回归。",
            ],
            "结论与下一步验证",
        )
        slides.append(
            SlideSpec(
                title="结论与下一步验证",
                bullets=closing_points[:4],
                notes="收束结论并给出立即可执行动作。",
                visual_kind="none",
            )
        )

        image_payload = self._build_image_payload(one_pager)
        if image_payload:
            slides.append(
                SlideSpec(
                    title="附录：图像证据",
                    bullets=[
                        "图像证据用于支撑核心结论，不替代实验指标。",
                        f"图像来源：{_sanitize_text(image_payload.get('caption', '未知来源'))}",
                    ],
                    notes="图像只作为辅助证据。",
                    visual_kind="image",
                    visual_payload=image_payload,
                )
            )

        normalized: List[SlideSpec] = []
        global_bullet_keys: set[str] = set()
        seen_slide_signatures: set[str] = set()

        def _append_unique_bullets(
            target: List[str],
            candidates: Sequence[str],
            *,
            local_keys: set[str],
            max_items: int = 5,
        ) -> None:
            for raw in candidates:
                if len(target) >= max_items:
                    break
                bullet = self._normalize_bullet_text(raw)
                if not bullet:
                    continue
                key = self._bullet_uniqueness_key(bullet)
                if not key or key in local_keys or key in global_bullet_keys:
                    continue
                local_keys.add(key)
                global_bullet_keys.add(key)
                target.append(bullet)

        for slide in slides:
            title = _sanitize_text(slide.title)
            notes = _sanitize_text(slide.notes)
            if not title:
                continue

            local_keys: set[str] = set()
            unique_bullets: List[str] = []
            _append_unique_bullets(
                unique_bullets,
                slide.bullets,
                local_keys=local_keys,
                max_items=5,
            )

            if len(unique_bullets) < 2:
                supplement_pool: List[str] = []
                supplement_pool.extend(_resolve_template(title))
                supplement_pool.extend(global_pool)
                supplement_pool.extend(fact_bullets)
                supplement_pool.extend(metric_lines)
                _append_unique_bullets(
                    unique_bullets,
                    supplement_pool,
                    local_keys=local_keys,
                    max_items=5,
                )

            if len(unique_bullets) < 2:
                continue

            slide_signature = "|".join(
                [title, slide.visual_kind]
                + [self._bullet_uniqueness_key(item) for item in unique_bullets[:3]]
            )
            if slide_signature in seen_slide_signatures:
                continue
            seen_slide_signatures.add(slide_signature)

            normalized.append(
                SlideSpec(
                    title=title,
                    bullets=unique_bullets[:5],
                    duration_sec=slide.duration_sec,
                    notes=notes,
                    visual_kind=slide.visual_kind,
                    visual_payload=dict(slide.visual_payload),
                )
            )

        normalized = normalized[:14]
        target_total = self._resolve_target_duration(video_brief.duration_estimate, len(normalized))

        weights: List[float] = []
        for slide in normalized:
            base = 1.0
            if slide.visual_kind == "chart":
                base = 1.2
            elif slide.visual_kind == "formula":
                base = 1.15
            elif "机制" in slide.title or "工程" in slide.title:
                base = 1.15
            weights.append(base)

        weight_sum = sum(weights) if weights else 1.0
        durations = [max(22, int(target_total * w / weight_sum)) for w in weights]
        diff = target_total - sum(durations)
        index = 0
        while diff != 0 and durations:
            pos = index % len(durations)
            if diff > 0:
                durations[pos] += 1
                diff -= 1
            elif durations[pos] > 22:
                durations[pos] -= 1
                diff += 1
            index += 1
            if index > 10000:
                break

        for slide, duration in zip(normalized, durations):
            slide.duration_sec = int(duration)
            if slide.visual_kind == "diagram" and not slide.visual_payload:
                slide.visual_payload = self._build_diagram_payload(slide.title, slide.bullets)
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
        framework = [
            "1) 问题定义与学习目标",
            "2) 系统边界与输入输出",
            "3) 核心架构：检索-推理-生成链路",
            "4) 机制一：API上下文检索",
            "5) 机制二：世界建模与视频生成",
            "6) 关键公式与计算框架",
            "7) 实验设置与指标口径",
            "8) 结果与证据解读",
            "9) 工程实现与模块拆解",
            "10) 部署、并发与成本",
            "11) 风险与失败模式",
            "12) 结论与下一步验证",
        ]

        lines: List[str] = [
            "# Slide Script Contract",
            "",
            "## Framework",
            *[f"- {item}" for item in framework],
            "",
            "## Content Constraints",
            "- 每页最多 4 条正文要点（公式页除外）。",
            "- 单条要点不允许硬截断（禁止 `...` / `…`）。",
            "- 单条要点必须是完整技术语义，不允许残缺括号或破碎短语。",
            "- 优先保留含 `API/检索/架构/指标/延迟/吞吐/成本/部署/实验` 的技术内容。",
            "- 非技术叙述（科普修辞、愿景口号）默认过滤。",
            "",
            "## Formula Constraints",
            "- 公式页启用 KaTeX（`katex: true`）。",
            "- 关键公式以块级数学语法输出，避免被当作普通文本。",
            "",
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

    def _render_formula_panel(self, payload: Dict[str, Any]) -> str:
        formulas = [item for item in (payload.get("formulas") or []) if _sanitize_text(item)]
        if not formulas:
            formulas = [r"Utility = \alpha Q + \beta C - \gamma L - \delta Cost"]

        blocks: List[str] = ["<div class='visual-card'>", "<div class='visual-title'>关键公式</div>"]
        blocks.append(f"<div class='formula-main'>{_escape_html(formulas[0])}</div>")
        for formula in formulas[1:4]:
            blocks.append(f"<div class='formula-line'>{_escape_html(formula)}</div>")
        blocks.append("</div>")
        return "\n".join(blocks)

    def _render_formula_markdown_lines(self, payload: Dict[str, Any]) -> List[str]:
        formulas = [item for item in (payload.get("formulas") or []) if _sanitize_text(item)]
        if not formulas:
            formulas = [r"Utility = \alpha \cdot Q + \beta \cdot C - \gamma \cdot L - \delta \cdot Cost"]

        lines: List[str] = ["## 关键公式", ""]
        for formula in formulas[:4]:
            lines.append("$$")
            lines.append(formula)
            lines.append("$$")
            lines.append("")
        return lines

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
        if slide.visual_kind == "formula":
            return self._render_formula_panel(slide.visual_payload)
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
.formula-main {
  font-family: "SFMono-Regular", Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  font-size: 0.82rem;
  line-height: 1.35;
  background: #edf3ff;
  border: 1px solid #c4d6f7;
  border-radius: 10px;
  padding: 0.5rem 0.55rem;
  margin-bottom: 0.45rem;
  overflow-wrap: anywhere;
}
.formula-line {
  font-family: "SFMono-Regular", Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  font-size: 0.78rem;
  line-height: 1.3;
  color: #28497e;
  margin-bottom: 0.35rem;
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
            if slide.visual_kind == "formula":
                lines.append("## 技术要点")
                lines.append("")
                for bullet in slide.bullets:
                    lines.append(f"- {_sanitize_text(bullet)}")
                if slide.notes:
                    lines.append("")
                    lines.append(f"<div class='speaker-notes'>Notes: {_escape_html(slide.notes)}</div>")
                lines.append("")
                lines.extend(self._render_formula_markdown_lines(slide.visual_payload))
                continue

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

    def _generate_video_sync(
        self,
        *,
        topic: str,
        out_dir: Path,
        video_brief: VideoBrief,
        one_pager: OnePager,
        facts: List[Dict[str, Any]],
        output_path: Path,
    ) -> Dict[str, Any]:
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            raise VideoGenerationError("ffmpeg is required for Slidev mode. Please install ffmpeg first.")

        slides = self._build_slide_specs(
            topic=topic,
            video_brief=video_brief,
            one_pager=one_pager,
            facts=facts,
        )
        if not slides:
            raise VideoGenerationError("No slide content available to render video.")

        plan_md = self._write_slide_plan_markdown(slides, out_dir)
        plan_json = self._write_slide_plan_json(slides, out_dir)
        contract_md = self._write_script_contract(out_dir)
        self._emit_progress(f"Slide plan ready: {len(slides)} slides.")

        runtime_dir = self._ensure_runtime_environment()
        build_id = f"{_slugify(out_dir.name)}_{int(time.time())}"
        build_dir = runtime_dir / "builds" / build_id
        build_dir.mkdir(parents=True, exist_ok=True)

        self._emit_progress("Rendering Slidev source...")
        entry_path = self._write_slidev_source(topic=topic, slides=slides, build_dir=build_dir)

        # 保存一份到输出目录，便于调试复现
        source_snapshot = out_dir / "video_slidev_source"
        if source_snapshot.exists():
            shutil.rmtree(source_snapshot)
        shutil.copytree(build_dir, source_snapshot)

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

        narration_audio_path: Optional[Path] = None
        narration_script_path: Optional[Path] = None
        narration_dir: Optional[Path] = None
        tts_providers: List[str] = []
        narration_specs: List[NarrationSpec] = []
        if self.enable_narration:
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
            narration_durations = list(narration_meta.get("segment_durations") or [])
            for idx, duration in enumerate(narration_durations):
                if idx >= len(slides):
                    break
                slides[idx].duration_sec = max(1, int(math.ceil(float(duration) + 0.12)))
            narration_audio_path = out_dir / "video_narration.m4a"
            self._emit_progress("Concatenating narration audio...")
            self._concat_audio_segments(
                audio_paths=list(narration_meta["segment_paths"]),
                output_path=narration_audio_path,
                ffmpeg_bin=ffmpeg_bin,
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

        total_duration = int(sum(max(1, int(item.duration_sec)) for item in slides))
        flow: Dict[str, Any] = {
            "slide_count": len(slides),
            "estimated_duration_sec": total_duration,
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
        }
        if narration_audio_path:
            flow["narration_audio_path"] = str(narration_audio_path)
        if narration_script_path:
            flow["narration_script_path"] = str(narration_script_path)
        if narration_dir:
            flow["narration_segments_dir"] = str(narration_dir)
        if tts_providers:
            flow["tts_providers"] = tts_providers
        if narration_specs:
            flow["narration_slide_count"] = len(narration_specs)
        return flow

    async def generate(
        self,
        *,
        topic: str,
        out_dir: Path,
        video_brief: Optional[Dict[str, Any]] = None,
        one_pager: Optional[Dict[str, Any]] = None,
        facts: Optional[List[Dict[str, Any]]] = None,
    ) -> VideoArtifact:
        out_dir.mkdir(parents=True, exist_ok=True)
        prompt = build_video_prompt(
            topic=topic,
            video_brief=video_brief,
            one_pager=one_pager,
            facts=facts,
        )
        vb = VideoBrief.from_dict(video_brief or {}, default_title=f"{topic} Video Brief")
        op = OnePager.from_dict(one_pager or {}, default_title=f"{topic} One-Pager")
        normalized_facts = list(facts or [])

        output_path = out_dir / "video_brief.mp4"
        metadata_path = out_dir / "video_generation_metadata.json"

        flow_meta = await asyncio.to_thread(
            self._generate_video_sync,
            topic=topic,
            out_dir=out_dir,
            video_brief=vb,
            one_pager=op,
            facts=normalized_facts,
            output_path=output_path,
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
