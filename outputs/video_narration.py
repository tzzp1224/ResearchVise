"""
Narration pipeline for slide videos.

This module owns:
- per-slide narration script building/rewrite
- TTS provider fallback and synthesis
- audio normalization and concatenation
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Callable, Dict, List, Optional


logger = logging.getLogger(__name__)


@dataclass
class NarrationSpec:
    slide_index: int
    slide_title: str
    duration_sec: int
    text: str


def _normalize_tts_provider_name(sanitize_text: Callable[[Any], str], value: Any) -> str:
    normalized = sanitize_text(value).lower().replace("-", "_")
    aliases = {
        "": "",
        "edge": "edge_tts",
        "edgetts": "edge_tts",
        "edge_tts": "edge_tts",
    }
    return aliases.get(normalized, normalized)


class NarrationPipeline:
    """Stateful narration/TTS helper used by SlidevVideoGenerator."""

    def __init__(
        self,
        *,
        tts_provider: str,
        tts_voice: Optional[str],
        tts_speed: float,
        narration_model: str,
        run_subprocess: Callable[..., None],
        sanitize_text: Callable[[Any], str],
        contains_cjk: Callable[[str], bool],
        split_sentences: Callable[[str], List[str]],
    ) -> None:
        self._run_subprocess = run_subprocess
        self._sanitize_text = sanitize_text
        self._contains_cjk = contains_cjk
        self._split_sentences = split_sentences

        self.tts_provider = "auto"
        self.tts_voice = ""
        self.tts_speed = 1.25
        self.narration_model = "deepseek-chat"
        self._tts_disabled_providers: set[str] = set()
        self.update_runtime(
            tts_provider=tts_provider,
            tts_voice=tts_voice,
            tts_speed=tts_speed,
            narration_model=narration_model,
        )

    def update_runtime(
        self,
        *,
        tts_provider: str,
        tts_voice: Optional[str],
        tts_speed: float,
        narration_model: str,
    ) -> None:
        self.tts_provider = _normalize_tts_provider_name(self._sanitize_text, tts_provider) or "auto"
        self.tts_voice = self._sanitize_text(tts_voice or "")
        self.tts_speed = float(max(0.8, min(1.5, tts_speed)))
        self.narration_model = self._sanitize_text(narration_model) or "deepseek-chat"

    def build_narration_text_for_slide(self, slide: Any, *, index: int) -> str:
        title = self._sanitize_text(getattr(slide, "title", "")) or f"Slide {index}"
        bullets = [
            self._sanitize_text(re.sub(r"^\[[^\]]+\]\s*", "", item))
            for item in (getattr(slide, "bullets", []) or [])
            if self._sanitize_text(item)
        ]
        notes = self._sanitize_text(getattr(slide, "notes", ""))

        cjk_mode = self._contains_cjk(title + " " + " ".join(bullets))
        if cjk_mode:
            lines: List[str] = [f"这一页我们聚焦 {title}。"]
            for idx, bullet in enumerate(bullets[:4], start=1):
                lines.append(f"先看第{idx}个要点：{bullet}。")
            if notes:
                lines.append(f"工程上需要注意：{notes}。")
            if any(re.search(r"\d", item) for item in bullets):
                lines.append("请特别关注这些数字背后的测试口径和适用边界。")
            else:
                lines.append("落地时建议把这些要点映射到监控指标和回滚阈值。")
        else:
            lines = [f"This slide focuses on {title}."]
            for idx, bullet in enumerate(bullets[:4], start=1):
                lines.append(f"Key point {idx}: {bullet}.")
            if notes:
                lines.append(f"Implementation note: {notes}.")
            if any(re.search(r"\d", item) for item in bullets):
                lines.append("Pay attention to metric definitions and benchmark conditions.")
            else:
                lines.append("Map these points to observable signals before rollout.")

        text = " ".join(lines).strip()
        return text or title

    def build_narration_specs(self, slides: List[Any]) -> List[NarrationSpec]:
        return [
            NarrationSpec(
                slide_index=idx,
                slide_title=self._sanitize_text(getattr(slide, "title", "")) or f"Slide {idx}",
                duration_sec=max(1, int(getattr(slide, "duration_sec", 1) or 1)),
                text=self.build_narration_text_for_slide(slide, index=idx),
            )
            for idx, slide in enumerate(slides, start=1)
        ]

    def _extract_json_object(self, text: str) -> Dict[str, Any]:
        raw = self._sanitize_text(text)
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group())
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _fit_script_to_duration(self, text: str, duration_sec: int) -> str:
        cleaned = self._sanitize_text(text)
        if not cleaned:
            return cleaned
        target = max(6, int(duration_sec))
        is_cjk = self._contains_cjk(cleaned)
        # Keep script density high enough for ~3min brief while avoiding overlong sentences.
        units_per_second = (5.9 if is_cjk else 10.0) * max(0.8, min(1.5, self.tts_speed))
        max_chars = int(max(75, target * units_per_second))
        min_chars = int(max(55, max_chars * (0.62 if target >= 24 else 0.5)))
        max_sentences = 6 if target >= 42 else 5 if target >= 28 else 4 if target >= 18 else 3

        sentences = [self._sanitize_text(s) for s in self._split_sentences(cleaned) if self._sanitize_text(s)]
        if not sentences:
            return _clean_whitespace(cleaned)[:max_chars]

        selected: List[str] = []
        total_chars = 0
        for sentence in sentences:
            if len(selected) >= max_sentences:
                break
            proposed = total_chars + len(sentence)
            if selected and proposed > max_chars:
                break
            selected.append(sentence)
            total_chars = proposed

        if not selected:
            selected = [sentences[0][:max_chars]]
        script = _clean_whitespace(" ".join(selected))
        if len(script) >= min_chars:
            return script

        anchors = [sentence for sentence in sentences if len(sentence) >= 8][:4]
        extra_lines: List[str] = []
        for anchor in anchors:
            if is_cjk:
                extra_lines.append(f"{anchor}，落地时要明确输入输出边界与监控口径。")
                extra_lines.append(f"建议围绕 {anchor} 设计压测与回滚阈值，并记录异常样本。")
            else:
                extra_lines.append(
                    f"For {anchor}, define concrete input-output boundaries and monitoring signals."
                )
                extra_lines.append(
                    f"Validate {anchor} with stress tests, rollback thresholds, and failure-case logs."
                )

        for line in extra_lines:
            candidate = _clean_whitespace(f"{script} {line}") if script else _clean_whitespace(line)
            if len(candidate) > max_chars:
                break
            script = candidate
            if len(script) >= min_chars:
                break
        return script[:max_chars]

    def rewrite_narration_with_small_model(self, specs: List[NarrationSpec]) -> List[NarrationSpec]:
        if not specs:
            return specs

        try:
            from intelligence.llm import Message, get_llm
        except Exception:
            return specs

        payload = [
            {
                "slide_index": item.slide_index,
                "title": item.slide_title,
                "target_duration_sec": item.duration_sec,
                "seed_script": item.text,
            }
            for item in specs
        ]

        prompt = (
            "请把每一页 seed_script 改写为“自然讲解口播稿”，不要逐字念 PPT。"
            "要求：\n"
            "1) 保留关键技术信息和因果逻辑，优先解释“为什么这样设计”；\n"
            "2) 语气自然、有节奏变化，避免机械朗读、口号化和比喻堆砌；\n"
            "3) 每页输出 3-6 句完整句子，不要只保留一句；\n"
            "4) 每页必须包含至少一个具体实现点（组件/流程/参数/约束）；\n"
            "5) 每页至少包含一条可执行工程建议或验证动作；\n"
            "6) 不要添加与页面无关的新事实；\n"
            "7) 返回 JSON：{\"slides\":[{\"slide_index\":1,\"script\":\"...\"}]}。\n\n"
            f"输入数据：\n{json.dumps(payload, ensure_ascii=False)}"
        )
        messages = [
            Message.system("You are an expert technical script writer for spoken narration."),
            Message.user(prompt),
        ]

        try:
            model = (
                self._sanitize_text(os.getenv("VIDEO_NARRATION_MODEL") or self.narration_model)
                or self.narration_model
            )
            llm = get_llm(model=model)
            response = llm.complete(
                messages,
                temperature=0.25,
                max_tokens=min(2600, 450 + len(specs) * 260),
            )
            content = response.content or ""
        except Exception as e:
            logger.warning("Narration rewrite skipped: %s", e)
            return specs
        finally:
            try:
                close_fn = getattr(locals().get("llm"), "aclose", None)
                if callable(close_fn):
                    try:
                        asyncio.run(close_fn())
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        try:
                            loop.run_until_complete(close_fn())
                        finally:
                            loop.close()
            except Exception:
                pass

        parsed = self._extract_json_object(content)
        candidates = parsed.get("slides", []) if isinstance(parsed, dict) else []
        by_idx: Dict[int, str] = {}
        for item in candidates:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("slide_index"))
            except Exception:
                continue
            script = self._sanitize_text(item.get("script", ""))
            if script:
                by_idx[idx] = script

        return [
            NarrationSpec(
                slide_index=item.slide_index,
                slide_title=item.slide_title,
                duration_sec=item.duration_sec,
                text=by_idx.get(item.slide_index, item.text),
            )
            for item in specs
        ]

    def write_narration_script(self, specs: List[NarrationSpec], out_dir: Path) -> Path:
        lines: List[str] = ["# Video Narration Script", ""]
        lines.append(f"- slide_count: {len(specs)}")
        lines.append(f"- estimated_duration_sec: {sum(item.duration_sec for item in specs)}")
        lines.append("")
        for item in specs:
            lines.append(f"## Slide {item.slide_index}: {item.slide_title} ({item.duration_sec}s)")
            lines.append(item.text)
            lines.append("")

        path = out_dir / "video_narration_script.md"
        path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return path

    def tts_provider_candidates(self, text: str) -> List[str]:
        chain = ["edge_tts", "say", "espeak"]
        normalized = _normalize_tts_provider_name(self._sanitize_text, self.tts_provider)

        if normalized and normalized != "auto":
            if normalized in chain:
                start = chain.index(normalized)
                ordered = chain[start:]
                available = [item for item in ordered if self._is_provider_available(item)]
                return available or ordered
            return [normalized]

        candidates = [item for item in chain if self._is_provider_available(item)]

        deduped: List[str] = []
        seen = set()
        for item in candidates:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    def _is_provider_available(self, provider: str) -> bool:
        if provider == "edge_tts":
            return self._is_edge_tts_available()
        if provider == "say":
            return bool(shutil.which("say"))
        if provider == "espeak":
            return bool(shutil.which("espeak"))
        return False

    def _is_edge_tts_available(self) -> bool:
        if shutil.which("edge-tts"):
            return True
        try:
            import edge_tts  # noqa: F401
        except Exception:
            return False
        return True

    def _synthesize_speech_with_edge_tts(self, *, text: str, output_path: Path) -> None:
        voice = self.tts_voice or ("zh-CN-YunxiNeural" if self._contains_cjk(text) else "en-US-GuyNeural")
        rate_percent = int(round((self.tts_speed - 1.0) * 100))
        rate = f"{rate_percent:+d}%"

        edge_bin = shutil.which("edge-tts")
        if edge_bin:
            cmd = [
                edge_bin,
                "--voice",
                voice,
                "--rate",
                rate,
                "--text",
                text,
                "--write-media",
                str(output_path),
            ]
            try:
                self._run_subprocess(cmd, context="tts edge_tts")
                return
            except Exception:
                fallback = [
                    edge_bin,
                    "--rate",
                    rate,
                    "--text",
                    text,
                    "--write-media",
                    str(output_path),
                ]
                self._run_subprocess(fallback, context="tts edge_tts fallback")
                return

        try:
            import edge_tts
        except Exception as e:
            raise RuntimeError("Edge-TTS is unavailable. Install `edge-tts` package or binary.") from e

        async def _save_audio() -> None:
            communicator = edge_tts.Communicate(text=text, voice=voice, rate=rate)
            await communicator.save(str(output_path))

        try:
            asyncio.run(_save_audio())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(_save_audio())
            finally:
                loop.close()

    def _synthesize_speech_with_say(self, *, text: str, output_path: Path) -> None:
        say_bin = shutil.which("say")
        if not say_bin:
            raise RuntimeError("`say` is unavailable on current system.")

        preferred_voice = self.tts_voice or ("Tingting" if self._contains_cjk(text) else "Samantha")
        speaking_rate = str(int(220 * self.tts_speed))
        cmd = [
            say_bin,
            "-r",
            speaking_rate,
            "-v",
            preferred_voice,
            "-o",
            str(output_path),
            "--data-format=LEI16@22050",
            text,
        ]
        try:
            self._run_subprocess(cmd, context="tts say")
        except Exception:
            fallback = [
                say_bin,
                "-r",
                speaking_rate,
                "-o",
                str(output_path),
                "--data-format=LEI16@22050",
                text,
            ]
            self._run_subprocess(fallback, context="tts say fallback")

    def _synthesize_speech_with_espeak(self, *, text: str, output_path: Path) -> None:
        espeak_bin = shutil.which("espeak")
        if not espeak_bin:
            raise RuntimeError("`espeak` is unavailable on current system.")

        voice = self.tts_voice or ("zh" if self._contains_cjk(text) else "en-us")
        speed = str(int(190 * self.tts_speed))
        cmd = [espeak_bin, "-w", str(output_path), "-s", speed, "-v", voice, text]
        try:
            self._run_subprocess(cmd, context="tts espeak")
        except Exception:
            fallback = [espeak_bin, "-w", str(output_path), "-s", speed, text]
            self._run_subprocess(fallback, context="tts espeak fallback")

    def synthesize_speech(self, *, text: str, output_path: Path) -> str:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        providers = [p for p in self.tts_provider_candidates(text) if p not in self._tts_disabled_providers]
        if not providers:
            raise RuntimeError("No TTS provider available. Install edge-tts or system voice engines.")

        errors: List[str] = []
        for provider in providers:
            try:
                if provider == "edge_tts":
                    self._synthesize_speech_with_edge_tts(text=text, output_path=output_path)
                elif provider == "say":
                    self._synthesize_speech_with_say(text=text, output_path=output_path)
                elif provider == "espeak":
                    self._synthesize_speech_with_espeak(text=text, output_path=output_path)
                else:
                    raise RuntimeError(f"Unsupported TTS provider: {provider}")

                if output_path.exists() and output_path.stat().st_size > 0:
                    return provider
                raise RuntimeError("TTS finished but output file is empty.")
            except Exception as e:
                errors.append(f"{provider}: {e}")
                self._tts_disabled_providers.add(provider)
                output_path.unlink(missing_ok=True)

        raise RuntimeError("TTS synthesis failed: " + " | ".join(errors))

    def _probe_media_duration(self, path: Path) -> Optional[float]:
        ffprobe_bin = shutil.which("ffprobe")
        if not ffprobe_bin:
            return None
        process = subprocess.run(
            [
                ffprobe_bin,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if process.returncode != 0:
            return None
        try:
            value = float((process.stdout or "").strip())
        except Exception:
            return None
        return value if value > 0 else None

    def _retime_audio_to_target(
        self,
        *,
        source_path: Path,
        output_path: Path,
        target_duration_sec: int,
        ffmpeg_bin: str,
    ) -> Optional[float]:
        # Keep natural speaking pace. Timeline should follow actual TTS duration.
        _ = max(1, int(target_duration_sec))
        _ = ffmpeg_bin
        current = self._probe_media_duration(source_path)
        if not current or current <= 0:
            shutil.copy2(source_path, output_path)
            return self._probe_media_duration(output_path)
        shutil.copy2(source_path, output_path)
        return self._probe_media_duration(output_path)

    def _normalize_audio_track(self, *, source_path: Path, output_path: Path, ffmpeg_bin: str) -> None:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(source_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "48000",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            str(output_path),
        ]
        self._run_subprocess(cmd, context="ffmpeg narration normalize")

    def render_narration_segments(self, *, slides: List[Any], out_dir: Path, ffmpeg_bin: str) -> Dict[str, Any]:
        raw_specs = self.rewrite_narration_with_small_model(self.build_narration_specs(slides))
        specs = [
            NarrationSpec(
                slide_index=item.slide_index,
                slide_title=item.slide_title,
                duration_sec=item.duration_sec,
                text=self._fit_script_to_duration(item.text, item.duration_sec),
            )
            for item in raw_specs
        ]

        narration_dir = out_dir / "video_audio_segments"
        narration_dir.mkdir(parents=True, exist_ok=True)
        for old in narration_dir.glob("*"):
            if old.is_file():
                old.unlink(missing_ok=True)

        segment_paths: List[Path] = []
        segment_durations: List[float] = []
        providers_used: List[str] = []
        for spec in specs:
            raw_path = narration_dir / f"raw_{spec.slide_index:03d}.wav"
            segment_path = narration_dir / f"segment_{spec.slide_index:03d}.m4a"
            provider = self.synthesize_speech(text=spec.text, output_path=raw_path)
            if provider not in providers_used:
                providers_used.append(provider)
            self._normalize_audio_track(source_path=raw_path, output_path=segment_path, ffmpeg_bin=ffmpeg_bin)
            raw_path.unlink(missing_ok=True)
            segment_paths.append(segment_path)
            measured = self._probe_media_duration(segment_path)
            final_duration = measured if measured and measured > 0 else float(spec.duration_sec)
            segment_durations.append(final_duration)

        final_specs: List[NarrationSpec] = []
        for idx, spec in enumerate(specs):
            duration = segment_durations[idx] if idx < len(segment_durations) else float(spec.duration_sec)
            final_specs.append(
                NarrationSpec(
                    slide_index=spec.slide_index,
                    slide_title=spec.slide_title,
                    duration_sec=max(1, int(math.ceil(duration))),
                    text=spec.text,
                )
            )
        script_path = self.write_narration_script(final_specs, out_dir)

        return {
            "script_path": script_path,
            "segment_paths": segment_paths,
            "segment_durations": segment_durations,
            "providers_used": providers_used,
            "narration_dir": narration_dir,
            "narration_specs": final_specs,
        }

    def concat_audio_segments(self, audio_paths: List[Path], output_path: Path, ffmpeg_bin: str) -> None:
        if not audio_paths:
            raise RuntimeError("No narration audio segments to concatenate.")

        concat_path = output_path.parent / "video_audio_concat.txt"

        def _escape(path: str) -> str:
            return path.replace("'", "'\\''")

        concat_lines = [f"file '{_escape(str(path.resolve()))}'" for path in audio_paths]
        concat_path.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")

        cmd = [
            ffmpeg_bin,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-vn",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            str(output_path),
        ]
        self._run_subprocess(cmd, context="ffmpeg audio concat")


def _clean_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()
