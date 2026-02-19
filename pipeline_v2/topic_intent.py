"""Topic intent policy: hot new agents top picks vs infra/watchlist buckets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

from core import NormalizedItem


def parse_window_days(window: Optional[str]) -> int:
    token = str(window or "").strip().lower()
    if not token:
        return 1
    if token == "today":
        return 1
    if token.endswith("d"):
        try:
            return max(1, int(token[:-1]))
        except Exception:
            return 3
    if token.endswith("h"):
        try:
            return max(1, int((int(token[:-1]) + 23) // 24))
        except Exception:
            return 1
    if token in {"past_week", "last_week", "weekly"}:
        return 7
    if token in {"past_month", "monthly"}:
        return 30
    return 3


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _days_since(dt: datetime | None) -> float | None:
    if dt is None:
        return None
    return max(0.0, (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() / 86400.0)


def _contains_term(text: str, term: str) -> bool:
    payload = re.sub(r"[-_/]+", " ", str(text or "").lower())
    token = str(term or "").strip().lower()
    if not payload or not token:
        return False
    pattern = r"\b" + re.escape(token).replace(r"\ ", r"[\s\-_]+") + r"\b"
    return re.search(pattern, payload) is not None


def _any_term(text: str, terms: Sequence[str]) -> bool:
    return any(_contains_term(text, term) for term in list(terms or []))


def _repo_full_name(item: NormalizedItem) -> str:
    metadata = dict(item.metadata or {})
    repo_id = str(metadata.get("repo_id") or metadata.get("repo") or "").strip().lower()
    if repo_id and "/" in repo_id:
        return repo_id
    title = str(item.title or "").strip().lower()
    if "/" in title and re.match(r"^[a-z0-9_.-]+/[a-z0-9_.-]+$", title):
        return title
    canonical_url = str(item.url or "").strip()
    if canonical_url:
        parsed = urlparse(canonical_url)
        if str(parsed.netloc or "").strip().lower().endswith("github.com"):
            parts = [part for part in str(parsed.path or "").split("/") if part]
            if len(parts) >= 2:
                return f"{parts[0].lower()}/{parts[1].lower()}"
    return ""


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class TopicIntentConfig:
    key: str
    top_pick_bucket: str
    infra_watchlist_bucket: str
    background_bucket: str
    top_pick_max_infra: int
    top_pick_max_infra_exceptions: int
    infra_watchlist_max: int
    background_max: int
    min_hot_trend_score: float
    infra_repo_allowlist: Tuple[str, ...]
    infra_org_allowlist: Tuple[str, ...]
    infra_keywords: Tuple[str, ...]
    handbook_keywords: Tuple[str, ...]
    hot_agent_keywords: Tuple[str, ...]
    infra_exception_keywords: Tuple[str, ...]


AI_AGENT_INTENT = TopicIntentConfig(
    key="hot_new_agents",
    top_pick_bucket="hot_new_agents",
    infra_watchlist_bucket="infra_watchlist",
    background_bucket="background_reading",
    top_pick_max_infra=0,
    top_pick_max_infra_exceptions=1,
    infra_watchlist_max=3,
    background_max=3,
    min_hot_trend_score=0.36,
    infra_repo_allowlist=(
        "langchain-ai/langchain",
        "langchain-ai/langgraph",
        "microsoft/autogen",
        "crewaiinc/crewai",
        "run-llama/llama_index",
        "llamastack/llama-stack",
    ),
    infra_org_allowlist=(
        "langchain-ai",
        "microsoft",
        "crewaiinc",
        "run-llama",
        "llamastack",
    ),
    infra_keywords=(
        "framework",
        "sdk",
        "library",
        "agent framework",
        "agent runtime",
        "orchestration engine",
        "tooling sdk",
    ),
    handbook_keywords=(
        "handbook",
        "curated",
        "resources",
        "awesome list",
        "awesome-agent",
        "roadmap",
        "learning path",
    ),
    hot_agent_keywords=(
        "mcp",
        "agentic",
        "agent workflow",
        "workflow",
        "demo",
        "show hn",
        "vertical agent",
        "browser agent",
        "tool calling",
        "function calling",
        "agent app",
        "copilot app",
    ),
    infra_exception_keywords=(
        "cve",
        "vulnerability",
        "security advisory",
        "breaking change",
        "breaking",
        "migration guide",
        "incident",
        "critical bug",
    ),
)


class TopicIntent:
    """Intent policy and trend scoring for one topic/window pair."""

    def __init__(self, *, topic: str, time_window: str, config: TopicIntentConfig) -> None:
        self.topic = str(topic or "").strip()
        self.time_window = str(time_window or "").strip()
        self.window_days = parse_window_days(self.time_window)
        self.config = config

    @classmethod
    def for_request(cls, *, topic: str | None, time_window: str | None) -> TopicIntent | None:
        token = str(topic or "").strip().lower()
        if not token:
            return None
        if "agent" in token:
            return cls(topic=str(topic or ""), time_window=str(time_window or "today"), config=AI_AGENT_INTENT)
        return None

    @property
    def hot_new_agents_mode(self) -> bool:
        return self.config.key == "hot_new_agents" and int(self.window_days) >= 7

    def _text(self, item: NormalizedItem) -> str:
        metadata = dict(item.metadata or {})
        return " ".join(
            [
                str(item.title or ""),
                str(item.body_md or ""),
                str(metadata.get("clean_text") or ""),
                " ".join([str(value) for value in list(metadata.get("topics") or []) if str(value).strip()]),
            ]
        ).lower()

    def _infra_exception_event(self, item: NormalizedItem, *, text: str) -> bool:
        metadata = dict(item.metadata or {})
        if _any_term(text, self.config.infra_exception_keywords):
            return True

        cross_source_count = int(float(metadata.get("cross_source_corroboration_count", 0) or 0))
        points = int(float(metadata.get("points", 0) or 0))
        comments = int(float(metadata.get("comment_count", metadata.get("comments", 0)) or 0))
        discussion_spike = bool(points >= 220 or comments >= 80 or cross_source_count >= 3)

        release_dt = _parse_datetime(metadata.get("release_published_at"))
        if release_dt is None and str(metadata.get("item_type") or "").strip().lower() == "release":
            release_dt = _parse_datetime(metadata.get("published_at") or item.published_at)
        release_days = _days_since(release_dt)
        major_release = bool(
            release_days is not None
            and release_days <= 7.0
            and re.search(r"\bv\d+\.\d+(\.\d+)?\b", text)
        )
        return bool(discussion_spike and major_release)

    def trend_signal(self, item: NormalizedItem) -> Dict[str, Any]:
        metadata = dict(item.metadata or {})
        source = str(item.source or "").strip().lower()
        text = self._text(item)
        reasons: List[str] = []
        proxy_used = False

        quality = dict(metadata.get("quality_signals") or {})
        update_days_raw = quality.get("update_recency_days")
        update_days = None if update_days_raw in (None, "", "unknown") else float(update_days_raw)
        if update_days is None:
            update_days = _days_since(
                _parse_datetime(metadata.get("updated_at") or metadata.get("last_push") or metadata.get("publish_or_update_time"))
            )

        created_days = _days_since(_parse_datetime(metadata.get("created_at")))
        release_days = _days_since(_parse_datetime(metadata.get("release_published_at")))
        cross_source_count = int(float(metadata.get("cross_source_corroboration_count", 0) or 0))
        points = int(float(metadata.get("points", 0) or 0))
        comments = int(float(metadata.get("comment_count", metadata.get("comments", 0)) or 0))

        discussion_score = 0.0
        if cross_source_count >= 3:
            discussion_score = max(discussion_score, 1.0)
        elif cross_source_count >= 2:
            discussion_score = max(discussion_score, 0.75)
        if points >= 180 or comments >= 70:
            discussion_score = max(discussion_score, 1.0)
        elif points >= 80 or comments >= 30:
            discussion_score = max(discussion_score, 0.7)
        elif points >= 30 or comments >= 12:
            discussion_score = max(discussion_score, 0.45)

        rank_proxy_score = 0.0
        search_rank = int(float(metadata.get("search_rank", 0) or 0))
        search_pool_size = int(float(metadata.get("search_pool_size", 0) or 0))
        if search_rank > 0 and search_pool_size > 1:
            rank_proxy_score = _clamp(1.0 - float(search_rank - 1) / float(max(1, search_pool_size - 1)))
            proxy_used = True

        if source == "github":
            created_score = 0.0
            if created_days is not None:
                if created_days <= 7:
                    created_score = 1.0
                elif created_days <= 30:
                    created_score = 0.85
                elif created_days <= 90:
                    created_score = 0.45
                else:
                    created_score = 0.15

            release_score = 0.0
            if release_days is not None:
                if release_days <= 3:
                    release_score = 1.0
                elif release_days <= 7:
                    release_score = 0.85
                elif release_days <= 14:
                    release_score = 0.5
                else:
                    release_score = 0.2
            elif "/releases/" in str(item.url or "").lower() or "release" in text:
                release_score = 0.45 if (update_days is not None and update_days <= 7) else 0.2

            commit_score = 0.0
            if update_days is not None:
                if update_days <= 2:
                    commit_score = 1.0
                elif update_days <= 7:
                    commit_score = 0.8
                elif update_days <= 14:
                    commit_score = 0.45
                elif update_days <= 30:
                    commit_score = 0.2

            trend = 0.34 * created_score + 0.24 * release_score + 0.24 * commit_score + 0.18 * discussion_score
            if rank_proxy_score > 0.0:
                trend += 0.08 * rank_proxy_score
            trend = _clamp(trend)

            if created_score > 0:
                reasons.append(f"trend.created_at_recency={created_score:.2f}")
            if release_score > 0:
                reasons.append(f"trend.release_recency={release_score:.2f}")
            if commit_score > 0:
                reasons.append(f"trend.commit_recency={commit_score:.2f}")
            if discussion_score > 0:
                reasons.append(f"trend.discussion_signal={discussion_score:.2f}")
            if rank_proxy_score > 0:
                reasons.append(f"trend.search_rank_proxy={rank_proxy_score:.2f}")
        elif source == "hackernews":
            recency_score = 0.0
            published_days = _days_since(_parse_datetime(metadata.get("publish_or_update_time") or item.published_at))
            if published_days is not None:
                if published_days <= 2:
                    recency_score = 1.0
                elif published_days <= 7:
                    recency_score = 0.75
                elif published_days <= 14:
                    recency_score = 0.35
            discussion_score = max(discussion_score, _clamp((points / 300.0) * 0.6 + (comments / 120.0) * 0.4))
            trend = _clamp(0.55 * discussion_score + 0.45 * recency_score)
            if discussion_score > 0:
                reasons.append(f"trend.discussion_signal={discussion_score:.2f}")
            if recency_score > 0:
                reasons.append(f"trend.created_at_recency={recency_score:.2f}")
        elif source == "huggingface":
            agent_demo_score = 0.0
            if _any_term(text, ("agent", "mcp", "workflow", "demo", "tool calling", "space", "runtime")):
                agent_demo_score = 0.75
            if _any_term(text, ("model card", "embedding", "image classification")) and not _any_term(
                text, ("agent", "workflow", "mcp")
            ):
                agent_demo_score = min(agent_demo_score, 0.2)
            recency_score = 0.0
            if update_days is not None:
                recency_score = 1.0 if update_days <= 7 else (0.5 if update_days <= 14 else 0.2)
            trend = _clamp(0.65 * agent_demo_score + 0.35 * recency_score)
            if agent_demo_score > 0:
                reasons.append(f"trend.demo_signal={agent_demo_score:.2f}")
            if recency_score > 0:
                reasons.append(f"trend.commit_recency={recency_score:.2f}")
        else:
            trend = _clamp(0.4 * discussion_score + 0.3 * (1.0 if (update_days is not None and update_days <= 7) else 0.0))
            if discussion_score > 0:
                reasons.append(f"trend.discussion_signal={discussion_score:.2f}")

        if proxy_used:
            reasons.append("trend.proxy_used=search_rank_position")
        return {
            "score": float(trend),
            "proxy_used": bool(proxy_used),
            "reasons": reasons[:6],
        }

    def classify_and_annotate(self, item: NormalizedItem) -> Dict[str, Any]:
        metadata = dict(item.metadata or {})
        text = self._text(item)
        source = str(item.source or "").strip().lower()
        repo_full_name = _repo_full_name(item)
        repo_org = repo_full_name.split("/", 1)[0] if "/" in repo_full_name else ""

        infra_repo_match = repo_full_name in set(self.config.infra_repo_allowlist)
        infra_org_match = repo_org in set(self.config.infra_org_allowlist)
        infra_keyword_match = _any_term(text, self.config.infra_keywords)
        is_infra = bool(
            source == "github"
            and (infra_repo_match or (infra_org_match and infra_keyword_match))
        )
        is_handbook = bool(_any_term(text, self.config.handbook_keywords))
        hot_keyword_hits = [
            token
            for token in list(self.config.hot_agent_keywords)
            if _contains_term(text, token)
        ]
        infra_exception_event = bool(is_infra and self._infra_exception_event(item, text=text))

        trend_payload = self.trend_signal(item)
        trend_score = float(trend_payload.get("score", 0.0) or 0.0)
        hot_candidate = bool(
            (trend_score >= float(self.config.min_hot_trend_score) and hot_keyword_hits)
            or trend_score >= 0.75
        )

        top_pick_allowed = True
        bucket = self.config.top_pick_bucket
        if self.hot_new_agents_mode:
            if is_handbook:
                top_pick_allowed = False
                bucket = self.config.background_bucket
            elif is_infra and not infra_exception_event:
                top_pick_allowed = False
                bucket = self.config.infra_watchlist_bucket

        metadata["intent_key"] = self.config.key
        metadata["intent_mode"] = self.config.top_pick_bucket if self.hot_new_agents_mode else "default"
        metadata["intent_bucket"] = bucket
        metadata["intent_repo_full_name"] = repo_full_name
        metadata["intent_is_infra"] = bool(is_infra)
        metadata["intent_is_handbook"] = bool(is_handbook)
        metadata["infra_exception_event"] = bool(infra_exception_event)
        metadata["intent_hot_keyword_hits"] = list(hot_keyword_hits)
        metadata["intent_hot_candidate"] = bool(hot_candidate)
        metadata["intent_top_pick_allowed"] = bool(top_pick_allowed)
        metadata["trend_signal_score"] = float(trend_score)
        metadata["trend_signal_proxy_used"] = bool(trend_payload.get("proxy_used"))
        metadata["trend_signal_reasons"] = list(trend_payload.get("reasons") or [])
        item.metadata = metadata

        return {
            "repo_full_name": repo_full_name,
            "is_infra": bool(is_infra),
            "is_handbook": bool(is_handbook),
            "infra_exception_event": bool(infra_exception_event),
            "hot_candidate": bool(hot_candidate),
            "top_pick_allowed": bool(top_pick_allowed),
            "trend_signal_score": float(trend_score),
        }

    def filter_rows_for_top_picks(self, rows: Sequence[Any]) -> Tuple[List[Any], Dict[str, int]]:
        if not self.hot_new_agents_mode:
            return [row for row in list(rows or [])], {
                "infra_filtered_count": 0,
                "handbook_filtered_count": 0,
                "infra_exception_top_pick_count": 0,
            }

        filtered: List[Any] = []
        infra_filtered = 0
        handbook_filtered = 0
        infra_count = 0
        infra_exception_count = 0

        for row in list(rows or []):
            item = getattr(row, "item", None)
            if item is None:
                continue
            metadata = dict(getattr(item, "metadata", None) or {})
            is_infra = bool(metadata.get("intent_is_infra"))
            is_handbook = bool(metadata.get("intent_is_handbook"))
            infra_exception_event = bool(metadata.get("infra_exception_event"))
            top_pick_allowed = bool(metadata.get("intent_top_pick_allowed", True))

            if is_handbook:
                handbook_filtered += 1
                continue
            if not top_pick_allowed:
                if is_infra:
                    infra_filtered += 1
                continue
            if is_infra:
                if infra_exception_event:
                    if infra_exception_count >= int(self.config.top_pick_max_infra_exceptions):
                        infra_filtered += 1
                        continue
                    infra_exception_count += 1
                    infra_count += 1
                elif infra_count >= int(self.config.top_pick_max_infra):
                    infra_filtered += 1
                    continue
                else:
                    infra_count += 1
            filtered.append(row)

        return filtered, {
            "infra_filtered_count": int(infra_filtered),
            "handbook_filtered_count": int(handbook_filtered),
            "infra_exception_top_pick_count": int(infra_exception_count),
        }

    def build_watchlists(
        self,
        *,
        ranked_rows: Sequence[Any],
        selected_ids: Sequence[str],
    ) -> Dict[str, List[Dict[str, Any]]]:
        selected = {str(value).strip() for value in list(selected_ids or []) if str(value).strip()}
        infra: List[Dict[str, Any]] = []
        background: List[Dict[str, Any]] = []

        for row in list(ranked_rows or []):
            item = getattr(row, "item", None)
            if item is None:
                continue
            item_id = str(getattr(item, "id", "") or "").strip()
            if not item_id or item_id in selected:
                continue
            metadata = dict(getattr(item, "metadata", None) or {})
            bucket = str(metadata.get("intent_bucket") or "").strip().lower()
            payload = {
                "item_id": item_id,
                "title": str(getattr(item, "title", "") or ""),
                "url": str(getattr(item, "url", "") or ""),
                "source": str(getattr(item, "source", "") or ""),
                "trend_signal_score": float(metadata.get("trend_signal_score", 0.0) or 0.0),
                "trend_signal_reasons": list(metadata.get("trend_signal_reasons") or []),
                "infra_exception_event": bool(metadata.get("infra_exception_event", False)),
            }
            if bucket == self.config.infra_watchlist_bucket and len(infra) < int(self.config.infra_watchlist_max):
                infra.append(payload)
            elif bucket == self.config.background_bucket and len(background) < int(self.config.background_max):
                background.append(payload)
            if len(infra) >= int(self.config.infra_watchlist_max) and len(background) >= int(self.config.background_max):
                break

        return {
            "infra_watchlist": infra,
            "background_reading": background,
        }

