#!/usr/bin/env python
"""
Phase 4 Demo: Real End-to-End Pipeline
关键词 -> 多源聚合 -> 分析 -> 深度输出 -> 导出 -> (可选)视频生成
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel

from intelligence.pipeline import run_research_end_to_end
from outputs.video_generator import SlidevVideoGenerator


console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 4 end-to-end demo: topic to deep technical outputs."
    )
    parser.add_argument("--topic", "-t", type=str, required=True, help="Research topic")
    parser.add_argument(
        "--max-results",
        "-n",
        type=int,
        default=8,
        help="Max results per source (default: 8)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory (default: auto timestamp dir under data/outputs)",
    )
    parser.add_argument(
        "--generate-video",
        action="store_true",
        help="Generate video artifact from retrieved content",
    )
    parser.add_argument(
        "--slides-target-duration-sec",
        type=int,
        default=420,
        help="Target duration for slides video in seconds (default: 420)",
    )
    parser.add_argument(
        "--slides-fps",
        type=int,
        default=24,
        help="FPS for slides video rendering (default: 24)",
    )
    parser.add_argument(
        "--disable-knowledge-indexing",
        action="store_true",
        help="Skip vector knowledge indexing step in analyst stage",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar while aggregating sources",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="arxiv,huggingface,github,semantic_scholar,stackoverflow,hackernews",
        help=(
            "Comma-separated sources. Available: "
            "arxiv,huggingface,twitter,reddit,github,semantic_scholar,stackoverflow,hackernews"
        ),
    )
    return parser


async def async_main(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir) if args.output_dir else None
    selected_sources = {
        s.strip().lower().replace("-", "_")
        for s in args.sources.split(",")
        if s.strip()
    }

    aggregator_kwargs = {
        "enable_arxiv": "arxiv" in selected_sources,
        "enable_huggingface": "huggingface" in selected_sources,
        "enable_twitter": "twitter" in selected_sources,
        "enable_reddit": "reddit" in selected_sources,
        "enable_github": "github" in selected_sources,
        "enable_semantic_scholar": "semantic_scholar" in selected_sources,
        "enable_stackoverflow": "stackoverflow" in selected_sources,
        "enable_hackernews": "hackernews" in selected_sources,
    }

    video_generator = None
    video_provider = "slidev"
    if args.generate_video:
        def _video_progress(message: str) -> None:
            console.print(f"[dim][video][/dim] {message}")

        video_generator = SlidevVideoGenerator(
            target_duration_sec=args.slides_target_duration_sec,
            fps=args.slides_fps,
            progress_callback=_video_progress,
        )

    console.print(
        Panel.fit(
            f"Topic: {args.topic}\n"
            f"Max results/source: {args.max_results}\n"
            f"Sources: {','.join(sorted(selected_sources))}\n"
            f"Generate video: {args.generate_video} ({video_provider})",
            title="Phase 4 E2E Run",
            border_style="blue",
        )
    )

    result = await run_research_end_to_end(
        topic=args.topic,
        max_results_per_source=args.max_results,
        out_dir=out_dir,
        show_progress=not args.no_progress,
        generate_video=args.generate_video,
        video_generator=video_generator,
        enable_knowledge_indexing=not args.disable_knowledge_indexing,
        aggregator_kwargs=aggregator_kwargs,
    )

    depth = result.get("depth_assessment", {})
    console.print("\n[bold]Run Summary[/bold]")
    console.print(f"- search_results: {result.get('search_results_count', 0)}")
    console.print(f"- facts: {len(result.get('facts', []))}")
    console.print(
        f"- depth_score: {depth.get('score', 0)}/{depth.get('max_score', 0)} "
        f"(pass={depth.get('pass', False)})"
    )
    console.print(f"- output_dir: {result.get('output_dir')}")

    written_files = result.get("written_files", {})
    if written_files:
        console.print("\n[bold]Artifacts[/bold]")
        for name, path in written_files.items():
            console.print(f"- {name}: {path}")

    video_artifact = result.get("video_artifact")
    video_error = result.get("video_error")
    if video_artifact:
        console.print("\n[bold]Video Artifact[/bold]")
        for k, v in video_artifact.items():
            console.print(f"- {k}: {v}")
    elif args.generate_video:
        console.print("\n[bold yellow]Video Status[/bold yellow]")
        if video_error:
            console.print(f"- failed: {video_error}")
            console.print("- documents were still exported successfully")
        else:
            console.print("- skipped or unavailable")

    if result.get("knowledge_gaps"):
        console.print("\n[bold]Knowledge Gaps[/bold]")
        for gap in result["knowledge_gaps"][:6]:
            console.print(f"- {gap}")

    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
