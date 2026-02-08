"""
Research Artifact Store
用于保存与检索可复用的研究产物索引（相似查询命中）。
"""

from __future__ import annotations

from datetime import datetime
import hashlib
from typing import Any, Dict, List, Optional

from .vector_store import QdrantVectorStore, get_vector_store


class ResearchArtifactStore:
    """Persistent semantic index for generated research artifacts."""

    def __init__(self, collection_name: str = "research_artifacts"):
        self.collection_name = collection_name
        self._store: QdrantVectorStore = get_vector_store(collection_name=collection_name)

    @staticmethod
    def _artifact_id(topic: str, output_dir: str) -> str:
        seed = f"{topic}|{output_dir}|{datetime.now().isoformat(timespec='seconds')}"
        digest = hashlib.md5(seed.encode("utf-8")).hexdigest()[:12]
        return f"artifact_{digest}"

    def index_artifact(
        self,
        *,
        topic: str,
        summary_text: str,
        output_dir: str,
        snapshot_path: str,
        manifest_path: Optional[str] = None,
        video_output_path: Optional[str] = None,
        quality_score: Optional[float] = None,
        quality_gate_pass: Optional[bool] = None,
        search_results_count: Optional[int] = None,
        artifact_schema_version: Optional[str] = None,
        artifact_id: Optional[str] = None,
    ) -> str:
        artifact_id = artifact_id or self._artifact_id(topic=topic, output_dir=output_dir)
        metadata: Dict[str, Any] = {
            "topic": topic,
            "output_dir": output_dir,
            "snapshot_path": snapshot_path,
            "manifest_path": manifest_path or "",
            "video_output_path": video_output_path or "",
            "has_video": bool(video_output_path),
            "quality_score": float(quality_score or 0.0),
            "quality_gate_pass": bool(quality_gate_pass) if quality_gate_pass is not None else False,
            "search_results_count": int(search_results_count or 0),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "artifact_schema_version": str(artifact_schema_version or "").strip(),
        }
        self._store.add(
            documents=[summary_text or topic],
            metadatas=[metadata],
            ids=[artifact_id],
        )
        return artifact_id

    def find_similar(
        self,
        *,
        query: str,
        score_threshold: float = 0.82,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        threshold = float(max(0.0, min(score_threshold, 1.0)))
        k = max(1, int(top_k))
        results = self._store.search(query=query, top_k=k)

        matches: List[Dict[str, Any]] = []
        for item in results:
            if float(item.score) < threshold:
                continue
            metadata = dict(item.metadata or {})
            metadata["artifact_id"] = item.id
            metadata["score"] = float(item.score)
            matches.append(metadata)

        matches.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return matches

    def close(self) -> None:
        self._store.close()
