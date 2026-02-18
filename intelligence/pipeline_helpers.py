"""
Helper functions for research pipeline data transformation and normalization.
"""

from __future__ import annotations

from datetime import datetime
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from models import AggregatedResult

_REQUIRED_FACT_CATEGORIES = {
    "architecture",
    "performance",
    "training",
    "comparison",
    "limitation",
}

_PREFERRED_RESOURCE_SOURCES = (
    "arxiv",
    "semantic_scholar",
    "huggingface",
    "github",
    "stackoverflow",
)

_CONFLICT_HINTS = ("vs", "versus", "trade-off", "however", "but", "对比", "取舍", "风险", "缓解")

_ACTION_HINTS = (
    "deploy",
    "monitor",
    "rollback",
    "config",
    "pipeline",
    "instrument",
    "步骤",
    "监控",
    "回滚",
    "配置",
    "上线",
)

_PLACEHOLDER_MARKERS = (
    "n/a",
    "na",
    "none",
    "null",
    "todo",
    "tbd",
    "待补充",
    "暂无",
    "请补充",
    "placeholder",
    "unknown",
)

_TECHNICAL_SIGNAL_HINTS = (
    "architecture",
    "pipeline",
    "benchmark",
    "latency",
    "throughput",
    "token",
    "inference",
    "retrieval",
    "rag",
    "sft",
    "rl",
    "deployment",
    "monitor",
    "rollback",
    "kv",
    "moe",
    "agent",
    "架构",
    "链路",
    "延迟",
    "吞吐",
    "推理",
    "检索",
    "部署",
    "监控",
    "回滚",
    "指标",
    "实验",
    "参数",
    "训练",
)

_LOW_SIGNAL_MARKERS = (
    "what is",
    "what are",
    "how to",
    "i am trying",
    "permission problems",
    "question",
    "help me",
    "stackoverflow question",
    "如何",
    "是什么",
    "请问",
    "求助",
)

_STRUCTURED_TECH_FIELDS = (
    "SOTA_Metric",
    "Hardware_Requirement",
    "Core_Formula",
    "Key_Optimization",
)

_HARDWARE_HINTS = (
    "a100",
    "h100",
    "tpu",
    "gpu",
    "cpu",
    "vram",
    "显存",
    "内存",
    "memory",
    "hardware",
    "cluster",
    "nvidia",
)

_FORMULA_HINTS = (
    "formula",
    "equation",
    "objective",
    "loss",
    "l=",
    "o(",
    "θ",
    "λ",
    "∑",
    "∂",
    "公式",
    "损失",
    "目标函数",
)

_OPTIMIZATION_HINTS = (
    "optimization",
    "optimizer",
    "quantization",
    "sparsity",
    "pruning",
    "distillation",
    "routing",
    "cache",
    "fusion",
    "kernel",
    "shard",
    "checkpoint",
    "优化",
    "量化",
    "稀疏",
    "蒸馏",
    "路由",
    "缓存",
    "分片",
)

_TECH_DENSE_HINT_PATTERN = re.compile(
    r"\b("
    r"latency|throughput|benchmark|accuracy|f1|pass@1|tokens?/s|qps|rps|"
    r"memory|vram|params?|parameter|flops|cost|ablation|method|methodology|"
    r"experiment|evaluation|objective|loss|optimizer|complexity|"
    r"deploy|deployment|inference|serving|retrieval|rag|"
    r"延迟|吞吐|基准|准确率|实验|方法|公式|复杂度|成本|部署|推理"
    r")\b",
    re.IGNORECASE,
)

_FORMULA_SIGNAL_PATTERN = re.compile(
    r"(=|≈|∑|∂|O\(|Θ\(|%|ms\b|s\b|gb\b|tb\b|x\b|×|÷|±|->|→)",
    re.IGNORECASE,
)


def _safe_iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    text = str(value).strip()
    return text or None


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _strip_html_tags(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&nbsp;", " ").replace("&lt;", "<").replace("&gt;", ">")
    return _clean_text(text)


def _normalize_statement(value: Any, *, max_len: int = 260) -> str:
    text = _strip_html_tags(value)
    text = re.sub(r"^[`\"'“”‘’]+|[`\"'“”‘’]+$", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip(" ,.;:，。；：") + "..."


def _estimate_narration_duration_seconds(value: Any, *, buffer_sec: float = 2.0) -> int:
    text = _strip_html_tags(value)
    if not text:
        return 0
    zh_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    en_chars = len(re.findall(r"[A-Za-z]", text))
    estimate = (zh_chars / 4.5) + (en_chars / 15.0) + float(buffer_sec)
    if zh_chars <= 0 and en_chars <= 0:
        token_count = len([token for token in re.split(r"\s+", text) if token.strip()])
        estimate = (token_count / 2.5) + float(buffer_sec)
    return max(1, int(round(estimate)))


def _technical_signal_score(value: Any) -> int:
    text = _normalize_statement(value)
    if not text:
        return 0
    lowered = text.lower()
    score = 0
    if re.search(r"\d", text):
        score += 2
    if any(hint in lowered for hint in _TECHNICAL_SIGNAL_HINTS):
        score += 2
    if any(hint in lowered for hint in _ACTION_HINTS):
        score += 1
    if any(unit in lowered for unit in ["%", "ms", "rps", "qps", "token", "参数", "准确率", "召回"]):
        score += 1
    if 24 <= len(text) <= 220:
        score += 1
    return score


def _is_low_signal_statement(value: Any) -> bool:
    text = _normalize_statement(value)
    if not text or _is_placeholder_text(text):
        return True
    lowered = text.lower()
    if any(marker in lowered for marker in _LOW_SIGNAL_MARKERS):
        return True
    if text.endswith("?") or "？" in text:
        return True
    if _technical_signal_score(text) <= 0 and len(text) < 32:
        return True
    return False


def _infer_fact_category(text: Any) -> str:
    lowered = _normalize_statement(text, max_len=360).lower()
    category_hints = {
        "architecture": ("architecture", "mechanism", "pipeline", "protocol", "架构", "机制", "链路"),
        "performance": ("benchmark", "latency", "throughput", "性能", "延迟", "吞吐"),
        "training": ("train", "dataset", "optimizer", "loss", "训练", "数据集", "损失"),
        "comparison": ("compare", "versus", "vs", "trade-off", "对比", "取舍"),
        "deployment": ("deploy", "production", "monitor", "rollback", "部署", "监控", "回滚"),
        "limitation": ("limitation", "risk", "failure", "issue", "局限", "风险", "失效"),
    }
    for category, tokens in category_hints.items():
        if any(token in lowered for token in tokens):
            return category
    return "community"


def _ranked_unique_items(items: List[str], *, max_items: int) -> List[str]:
    staged: List[str] = []
    for item in items:
        text = _normalize_statement(item)
        if not text or _is_placeholder_text(text) or _is_low_signal_statement(text):
            continue
        staged.append(text)
    deduped = _dedupe_text_list(staged, max_items=max(max_items * 5, 40))
    ranked = sorted(
        deduped,
        key=lambda text: (_technical_signal_score(text), len(text)),
        reverse=True,
    )
    return ranked[:max_items]


def _is_placeholder_text(value: Any) -> bool:
    text = _clean_text(value).lower()
    if not text:
        return True
    if text in _PLACEHOLDER_MARKERS:
        return True
    return any(marker in text for marker in _PLACEHOLDER_MARKERS)


def _dedupe_text_list(items: List[str], *, max_items: int) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for item in items:
        cleaned = _clean_text(item)
        if not cleaned or _is_placeholder_text(cleaned):
            continue
        key = re.sub(r"\s+", "", cleaned.lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)
        if len(deduped) >= max_items:
            break
    return deduped


def _extract_metric_pairs_from_text(text: str) -> Dict[str, str]:
    cleaned = _clean_text(text)
    if not cleaned:
        return {}

    pairs: Dict[str, str] = {}
    colon_pattern = re.compile(
        r"([A-Za-z][A-Za-z0-9_\- /]{1,28}|[\u4e00-\u9fff]{2,18})\s*[:：=]\s*([<>~]?\s*[-+]?\d+(?:\.\d+)?\s*(?:%|ms|s|x|k|K|M|B|rps|qps|次|个|倍)?)",
        re.IGNORECASE,
    )
    for match in colon_pattern.finditer(cleaned):
        key = _clean_text(match.group(1))
        value = _clean_text(match.group(2))
        if key and value:
            pairs[key] = value
            if len(pairs) >= 5:
                return pairs

    percent_match = re.search(r"([-+]?\d+(?:\.\d+)?)\s*%", cleaned)
    if percent_match and len(pairs) < 5:
        pairs["change"] = f"{percent_match.group(1)}%"
    return pairs


def extract_technical_claim_candidates(
    text: Any,
    *,
    max_items: int = 4,
    max_len: int = 500,
) -> List[str]:
    raw = _strip_html_tags(text)
    if not raw:
        return []

    sentences = re.split(r"(?<=[。.!?；;])\s+|\n+", raw)
    scored: List[tuple[int, str]] = []
    for sentence in sentences:
        statement = _normalize_statement(sentence, max_len=max_len)
        if not statement or len(statement) < 36:
            continue

        lowered = statement.lower()
        has_numeric = bool(re.search(r"\d", statement))
        has_tech_hint = bool(_TECH_DENSE_HINT_PATTERN.search(lowered))
        has_formula_signal = bool(_FORMULA_SIGNAL_PATTERN.search(statement))
        if not ((has_numeric and has_tech_hint) or has_formula_signal):
            continue

        score = _technical_signal_score(statement)
        if has_numeric:
            score += 2
        if has_tech_hint:
            score += 2
        if has_formula_signal:
            score += 2
        if len(statement) >= 120:
            score += 1
        scored.append((score, statement))

    scored.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    selected: List[str] = []
    seen = set()
    for _, statement in scored:
        key = re.sub(r"\W+", "", statement.lower())
        if not key or key in seen:
            continue
        seen.add(key)
        selected.append(statement)
        if len(selected) >= max(1, int(max_items)):
            break
    return selected


def _pick_best_statement(
    candidates: List[str],
    *,
    keywords: tuple[str, ...],
    require_numeric: bool = False,
    max_len: int = 180,
) -> str:
    best = ""
    best_score = -1
    for raw in candidates:
        text = _normalize_statement(raw, max_len=max_len)
        if not text or _is_placeholder_text(text):
            continue
        lowered = text.lower()
        keyword_hits = sum(1 for token in keywords if token and token in lowered)
        if keyword_hits <= 0:
            continue
        numeric_hits = 1 if re.search(r"\d", text) else 0
        if require_numeric and numeric_hits <= 0:
            continue
        score = keyword_hits * 3 + numeric_hits * 2 + _technical_signal_score(text)
        if score > best_score:
            best = text
            best_score = score
    return best


def _derive_structured_tech_fields(
    *,
    candidates: List[str],
    metrics: Dict[str, str],
) -> Dict[str, str]:
    values: Dict[str, str] = {}

    values["SOTA_Metric"] = _pick_best_statement(
        candidates,
        keywords=("sota", "state-of-the-art", "benchmark", "score", "accuracy", "f1", "pass@1", "latency", "throughput", "基准", "准确率", "吞吐", "延迟"),
        require_numeric=True,
    )

    if not values["SOTA_Metric"]:
        for key in ("score", "accuracy", "latency_ms", "throughput", "citation_count", "downloads", "change"):
            value = _normalize_statement(metrics.get(key, ""), max_len=80)
            if value:
                values["SOTA_Metric"] = f"{key}: {value}"
                break

    values["Hardware_Requirement"] = _pick_best_statement(
        candidates,
        keywords=_HARDWARE_HINTS,
        require_numeric=False,
    )
    values["Core_Formula"] = _pick_best_statement(
        candidates,
        keywords=_FORMULA_HINTS,
        require_numeric=False,
    )
    values["Key_Optimization"] = _pick_best_statement(
        candidates,
        keywords=_OPTIMIZATION_HINTS,
        require_numeric=False,
    )

    for field in _STRUCTURED_TECH_FIELDS:
        value = _normalize_statement(values.get(field, ""), max_len=180)
        if value:
            values[field] = value
        else:
            values[field] = "当前证据未明确披露，需补充 Methods/Implementation 原文。"
    return values


def _normalize_date_label(value: Any) -> str:
    text = _clean_text(value)
    if not text:
        return ""

    match = re.search(r"(19|20)\d{2}(?:[-/.](\d{1,2}))?(?:[-/.](\d{1,2}))?", text)
    if not match:
        return ""

    year = int(match.group(0)[:4])
    current_year = datetime.now().year
    if year < 1900 or year > current_year + 1:
        return ""

    month_raw = match.group(2)
    day_raw = match.group(3)
    if not month_raw:
        return f"{year:04d}"

    month = max(1, min(12, int(month_raw)))
    if not day_raw:
        return f"{year:04d}-{month:02d}"

    day = max(1, min(31, int(day_raw)))
    return f"{year:04d}-{month:02d}-{day:02d}"


def _extract_verified_date_from_result(item: Dict[str, Any]) -> str:
    metadata = dict(item.get("metadata", {}) or {})
    for key in ("published_date", "created_at", "updated_at", "year"):
        normalized = _normalize_date_label(metadata.get(key))
        if normalized:
            return normalized
    for key in ("published_date", "created_at", "updated_at", "year"):
        normalized = _normalize_date_label(item.get(key))
        if normalized:
            return normalized
    return ""


def _matching_tokens(value: Any) -> set[str]:
    text = _clean_text(value).lower()
    if not text:
        return set()
    text = text.replace("_", " ").replace("/", " ").replace("-", " ")
    parts = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?|[\u4e00-\u9fff]{2,}", text)
    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "model",
        "models",
        "technical",
        "report",
        "chat",
        "agent",
        "open",
        "intelligence",
        "研究",
        "模型",
        "技术",
        "分析",
    }
    tokens: set[str] = set()
    for item in parts:
        token = item.strip()
        if not token or token in stopwords:
            continue
        if len(token) <= 1:
            continue
        tokens.add(token)
    return tokens


def _is_valid_http_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    url = value.strip()
    if not url:
        return False
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _looks_placeholder_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    url = value.strip().lower()
    if not url:
        return True
    return any(
        marker in url
        for marker in (
            "example.com",
            "your-repo",
            "your_org",
            "your-project",
            "placeholder",
            "todo",
        )
    )


def _resource_quality_penalty(url: str) -> int:
    text = str(url or "").lower()
    penalty = 0
    if any(token in text for token in ["/issues/", "/discussions/", "/pull/", "/questions/"]):
        penalty += 2
    if "stackoverflow.com/questions/" in text or "news.ycombinator.com/item?" in text:
        penalty += 1
    return penalty


def normalize_one_pager_resources(
    *,
    one_pager: Optional[Dict[str, Any]],
    facts: List[Dict[str, Any]],
    search_results: List[Dict[str, Any]],
    max_resources: int = 8,
) -> Optional[Dict[str, Any]]:
    if not one_pager:
        return one_pager

    normalized = dict(one_pager)
    id_to_result = {
        str(item.get("id", "")).strip(): item
        for item in search_results
        if str(item.get("id", "")).strip()
    }

    resources: List[Dict[str, str]] = []
    seen_urls = set()

    for resource in normalized.get("resources", []) or []:
        if not isinstance(resource, dict):
            continue
        url = str(resource.get("url", "")).strip()
        title = str(resource.get("title", "")).strip() or "Resource"
        if not _is_valid_http_url(url) or _looks_placeholder_url(url) or url in seen_urls:
            continue
        seen_urls.add(url)
        resources.append({"title": title, "url": url})
        if len(resources) >= max_resources:
            break

    evidence_ids: List[str] = []
    for fact in facts:
        for evidence in fact.get("evidence", []) or []:
            evidence_id = str(evidence).strip()
            if evidence_id and evidence_id not in evidence_ids:
                evidence_ids.append(evidence_id)

    for evidence_id in evidence_ids:
        item = id_to_result.get(evidence_id)
        if not item:
            continue
        url = str(item.get("url", "")).strip()
        if not _is_valid_http_url(url) or url in seen_urls:
            continue
        title = str(item.get("title", "")).strip() or evidence_id
        resources.append({"title": title, "url": url})
        seen_urls.add(url)
        if len(resources) >= max_resources:
            break

    if len(resources) < 3:
        source_rank = {source: idx for idx, source in enumerate(_PREFERRED_RESOURCE_SOURCES)}
        sorted_results = sorted(
            search_results,
            key=lambda item: (
                source_rank.get(str(item.get("source", "")).strip(), len(_PREFERRED_RESOURCE_SOURCES)),
                _resource_quality_penalty(str(item.get("url", "")).strip()),
                str(item.get("title", "")).lower(),
            ),
        )
        for item in sorted_results:
            url = str(item.get("url", "")).strip()
            if not _is_valid_http_url(url) or url in seen_urls:
                continue
            title = str(item.get("title", "")).strip() or str(item.get("id", "Resource")).strip()
            resources.append({"title": title, "url": url})
            seen_urls.add(url)
            if len(resources) >= max_resources:
                break

    normalized["resources"] = resources
    return normalized


def normalize_one_pager_content(
    *,
    topic: str,
    one_pager: Optional[Dict[str, Any]],
    facts: List[Dict[str, Any]],
    search_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    normalized = dict(one_pager or {})
    facts_sorted = sorted(list(facts or []), key=lambda item: float(item.get("confidence", 0.0) or 0.0), reverse=True)

    claims_by_category: Dict[str, List[str]] = {}
    all_claims: List[str] = []
    for fact in facts_sorted:
        claim = _normalize_statement(fact.get("claim", ""))
        if not claim or _is_placeholder_text(claim) or _is_low_signal_statement(claim):
            continue
        category = _clean_text(fact.get("category", "other")).lower() or "other"
        claims_by_category.setdefault(category, []).append(claim)
        all_claims.append(claim)

    search_snippets: List[str] = []
    for item in search_results:
        title = _normalize_statement(item.get("title", ""), max_len=120)
        content = _normalize_statement(item.get("content", ""), max_len=240)
        source = _clean_text(item.get("source", "")).lower()
        date = _extract_verified_date_from_result(item)
        prefix = f"[{source} {date}] " if source and date else (f"[{source}] " if source else "")
        snippet = title
        if content:
            sentence = re.split(r"[。.!?；;\n]+", content)[0]
            sentence = _normalize_statement(sentence, max_len=180)
            if sentence:
                snippet = f"{title}: {sentence}" if title else sentence
        snippet = _normalize_statement(f"{prefix}{snippet}", max_len=220)
        if not snippet or _is_placeholder_text(snippet) or _is_low_signal_statement(snippet):
            continue
        if _technical_signal_score(snippet) < 2 and source in {"reddit", "twitter"}:
            continue
        search_snippets.append(snippet)
        if len(search_snippets) >= 20:
            break

    def _section_list(name: str, *, max_items: int) -> List[str]:
        return _ranked_unique_items([_normalize_statement(item) for item in list(normalized.get(name) or [])], max_items=max_items)

    exec_summary = _normalize_statement(normalized.get("executive_summary", ""), max_len=220)
    if not exec_summary or _is_placeholder_text(exec_summary) or _is_low_signal_statement(exec_summary):
        summary_candidates = _ranked_unique_items(all_claims + search_snippets, max_items=6)
        exec_summary = (
            (summary_candidates[0] if summary_candidates else "")
            or f"{_clean_text(topic) or '该主题'} 当前结构化证据不足，需补充可验证数据源后再下结论。"
        )
    normalized["executive_summary"] = exec_summary

    key_findings = _section_list("key_findings", max_items=10)
    if len(key_findings) < 6:
        key_findings = _ranked_unique_items(key_findings + all_claims + search_snippets, max_items=10)
    normalized["key_findings"] = key_findings[:10]

    metrics = {
        _normalize_statement(k, max_len=40): _normalize_statement(v, max_len=50)
        for k, v in dict(normalized.get("metrics", {}) or {}).items()
        if _normalize_statement(k, max_len=40)
        and _normalize_statement(v, max_len=50)
        and not _is_placeholder_text(v)
    }
    if len(metrics) < 3:
        metric_candidates = _ranked_unique_items(all_claims + search_snippets, max_items=24)
        for text in metric_candidates:
            extracted = _extract_metric_pairs_from_text(text)
            for key, value in extracted.items():
                if key not in metrics:
                    metrics[key] = value
                if len(metrics) >= 6:
                    break
            if len(metrics) >= 6:
                break

    if len(metrics) < 3:
        metadata_metric_keys = (
            ("citation_count", "citation_count"),
            ("stars", "github_stars"),
            ("downloads", "downloads"),
            ("latency_ms", "latency_ms"),
            ("throughput", "throughput"),
            ("score", "score"),
            ("answer_count", "answer_count"),
            ("points", "hn_points"),
        )
        for item in search_results:
            metadata = dict(item.get("metadata", {}) or {})
            for key, metric_name in metadata_metric_keys:
                value = metadata.get(key)
                value_text = _clean_text(value)
                if not value_text or _is_placeholder_text(value_text):
                    continue
                if metric_name not in metrics:
                    metrics[metric_name] = value_text
                if len(metrics) >= 6:
                    break
            if len(metrics) >= 6:
                break
    structured_candidates = _ranked_unique_items(
        all_claims
        + search_snippets
        + list(normalized.get("technical_deep_dive") or [])
        + list(normalized.get("implementation_notes") or []),
        max_items=36,
    )
    structured_fields = _derive_structured_tech_fields(
        candidates=structured_candidates,
        metrics=metrics,
    )
    merged_metrics: Dict[str, str] = {}
    for key in _STRUCTURED_TECH_FIELDS:
        merged_metrics[key] = structured_fields.get(key, "")
    for key, value in metrics.items():
        if key in merged_metrics:
            continue
        merged_metrics[key] = value
    normalized["metrics"] = dict(list(merged_metrics.items())[:10])

    strengths = _section_list("strengths", max_items=6)
    if len(strengths) < 3:
        strengths = _ranked_unique_items(
            strengths
            + claims_by_category.get("architecture", [])[:3]
            + claims_by_category.get("performance", [])[:3]
            + key_findings[:4],
            max_items=6,
        )
    normalized["strengths"] = strengths[:6]

    weaknesses = _section_list("weaknesses", max_items=6)
    if len(weaknesses) < 2:
        weaknesses = _ranked_unique_items(
            weaknesses
            + claims_by_category.get("limitation", [])[:4]
            + claims_by_category.get("risk", [])[:2],
            max_items=6,
        )
    normalized["weaknesses"] = weaknesses[:6]

    deep_dive = _section_list("technical_deep_dive", max_items=7)
    if len(deep_dive) < 4:
        deep_dive = _ranked_unique_items(
            deep_dive
            + claims_by_category.get("architecture", [])[:4]
            + claims_by_category.get("training", [])[:3]
            + claims_by_category.get("performance", [])[:2]
            + claims_by_category.get("deployment", [])[:2],
            max_items=7,
        )
    if len(deep_dive) < 4:
        deep_dive = _ranked_unique_items(deep_dive + key_findings, max_items=7)
    normalized["technical_deep_dive"] = deep_dive[:7]

    implementation = _section_list("implementation_notes", max_items=7)
    if len(implementation) < 4:
        implementation = _ranked_unique_items(
            implementation
            + claims_by_category.get("deployment", [])[:5]
            + claims_by_category.get("performance", [])[:3]
            + claims_by_category.get("comparison", [])[:2],
            max_items=7,
        )
    if len(implementation) < 3 or not any(any(token in note.lower() for token in _ACTION_HINTS) for note in implementation):
        implementation = _ranked_unique_items(
            implementation
            + [
                "先定义上线监控口径（成功率、延迟、错误率）并设置回滚阈值。",
                "建立可回放评测集并固定评测脚本，避免版本迭代导致指标口径漂移。",
                "将关键路径拆为可观测步骤并对每一步记录可审计日志。",
                "对高风险改动采用灰度发布，触发阈值后自动回滚并保留事故复盘证据。",
            ],
            max_items=7,
        )
    normalized["implementation_notes"] = implementation[:7]

    risks = _section_list("risks_and_mitigations", max_items=6)
    if len(risks) < 3:
        risks = _ranked_unique_items(
            risks
            + claims_by_category.get("limitation", [])[:4]
            + claims_by_category.get("comparison", [])[:3]
            + weaknesses[:2],
            max_items=6,
        )
    if len(risks) < 2:
        risks = _ranked_unique_items(
            risks
            + ["关键证据覆盖不足 -> 增加跨来源验证并补充可复现实验后再发布结论。"],
            max_items=6,
        )
    normalized["risks_and_mitigations"] = risks[:6]

    if not _clean_text(normalized.get("title", "")) or _is_placeholder_text(normalized.get("title", "")):
        normalized["title"] = f"{_clean_text(topic) or 'Research'} One-Pager"

    return normalized


def normalize_timeline_dates(
    *,
    topic: str,
    timeline: Optional[Any],
    facts: List[Dict[str, Any]],
    search_results: List[Dict[str, Any]],
) -> Optional[Any]:
    if timeline is None:
        return timeline

    if isinstance(timeline, dict):
        events = [dict(item) for item in list(timeline.get("events") or []) if isinstance(item, dict)]
        output_type = "dict"
    elif isinstance(timeline, list):
        events = [dict(item) for item in timeline if isinstance(item, dict)]
        output_type = "list"
    else:
        return timeline

    result_date_by_id: Dict[str, str] = {}
    known_years = set()
    dated_result_rows: List[Dict[str, Any]] = []
    for item in search_results:
        result_id = _clean_text(item.get("id", ""))
        date = _extract_verified_date_from_result(item)
        if not date:
            continue
        if result_id:
            result_date_by_id[result_id] = date
        known_years.add(int(date[:4]))
        dated_result_rows.append(
            {
                "id": result_id,
                "date": date,
                "tokens": _matching_tokens(f"{item.get('title', '')} {item.get('content', '')}"),
            }
        )

    fact_by_id = {
        _clean_text(item.get("id", "")): item
        for item in facts
        if _clean_text(item.get("id", ""))
    }
    fact_token_rows: List[Dict[str, Any]] = []
    for fact in facts:
        fact_id = _clean_text(fact.get("id", ""))
        evidence_ids = [_clean_text(ev) for ev in list(fact.get("evidence") or []) if _clean_text(ev)]
        evidence_dates = sorted({result_date_by_id.get(ev, "") for ev in evidence_ids if result_date_by_id.get(ev, "")})
        fact_token_rows.append(
            {
                "id": fact_id,
                "tokens": _matching_tokens(fact.get("claim", "")),
                "evidence_dates": evidence_dates,
            }
        )

    normalized_events: List[Dict[str, Any]] = []
    for idx, event in enumerate(events, start=1):
        item = dict(event)
        refs = [
            _clean_text(ref)
            for ref in list(item.get("source_refs") or [])
            if _clean_text(ref)
        ]
        verified_candidates: List[str] = []
        for ref in refs:
            direct_date = result_date_by_id.get(ref)
            if direct_date:
                verified_candidates.append(direct_date)
                continue
            fact = fact_by_id.get(ref)
            if not fact:
                continue
            for evidence_id in list(fact.get("evidence") or []):
                date = result_date_by_id.get(_clean_text(evidence_id))
                if date:
                    verified_candidates.append(date)

        title = _clean_text(item.get("title", ""))
        desc = _clean_text(item.get("description", ""))
        event_tokens = _matching_tokens(f"{title} {desc}")
        if not verified_candidates and event_tokens:
            best_fact_dates: List[str] = []
            best_overlap = 0
            for row in fact_token_rows:
                overlap = len(event_tokens & set(row.get("tokens") or set()))
                if overlap < 2:
                    continue
                evidence_dates = list(row.get("evidence_dates") or [])
                if not evidence_dates:
                    continue
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_fact_dates = evidence_dates
            if best_fact_dates:
                verified_candidates.extend(best_fact_dates)

        if not verified_candidates and event_tokens:
            best_date = ""
            best_overlap = 0
            for row in dated_result_rows:
                overlap = len(event_tokens & set(row.get("tokens") or set()))
                if overlap < 2:
                    continue
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_date = str(row.get("date", "")).strip()
            if best_date:
                verified_candidates.append(best_date)

        verified_date = sorted(set(verified_candidates))[0] if verified_candidates else ""
        raw_date = _normalize_date_label(item.get("date"))
        has_resolvable_ref = any(ref in result_date_by_id or ref in fact_by_id for ref in refs)
        if not verified_date and raw_date and has_resolvable_ref:
            year = int(raw_date[:4])
            if year in known_years:
                verified_date = raw_date
        item["date"] = verified_date or "Unknown"
        if not title or _is_placeholder_text(title):
            title = _clean_text(f"{topic} milestone {idx}")
        if not desc or _is_placeholder_text(desc):
            desc = _clean_text(title)
        item["title"] = title
        item["description"] = desc
        item["source_refs"] = refs
        item["importance"] = int(item.get("importance", 3) or 3)
        normalized_events.append(item)

    if normalized_events and all(str(item.get("date", "")).strip().lower() == "unknown" for item in normalized_events):
        normalized_events = []

    if not normalized_events and result_date_by_id:
        dated_results = sorted(
            [
                (date, item)
                for item in search_results
                for date in [_extract_verified_date_from_result(item)]
                if date
            ],
            key=lambda pair: pair[0],
        )
        for idx, (date, source_item) in enumerate(dated_results[:8], start=1):
            title = _clean_text(source_item.get("title", "")) or f"{topic} event {idx}"
            desc = _clean_text(source_item.get("content", ""))
            if desc:
                desc = _clean_text(re.split(r"[。.!?；;\n]+", desc)[0])
            normalized_events.append(
                {
                    "date": date,
                    "title": title,
                    "description": desc or title,
                    "importance": 3,
                    "source_refs": [_clean_text(source_item.get("id", ""))] if _clean_text(source_item.get("id", "")) else [],
                }
            )

    if output_type == "dict":
        output = dict(timeline)
        output["events"] = normalized_events
        return output
    return normalized_events


def normalize_video_brief(
    *,
    topic: str,
    video_brief: Optional[Dict[str, Any]],
    facts: Optional[List[Dict[str, Any]]] = None,
    search_results: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    if not video_brief:
        return video_brief

    facts = list(facts or [])
    search_results = list(search_results or [])
    normalized = dict(video_brief)
    raw_segments = [dict(item) for item in list(normalized.get("segments") or []) if isinstance(item, dict)]

    category_titles = {
        "architecture": "机制实现与架构取舍",
        "performance": "指标与实验结论",
        "comparison": "替代方案对比与取舍",
        "deployment": "工程落地路径",
        "limitation": "风险边界与缓解",
        "training": "训练策略与数据依赖",
        "community": "社区反馈与实战问题",
    }

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

    evidence_segments: List[Dict[str, Any]] = []
    for fact in sorted(
        facts,
        key=lambda item: float(item.get("confidence", 0.0) or 0.0),
        reverse=True,
    ):
        claim = _normalize_statement(fact.get("claim", ""), max_len=220)
        if not claim or _is_placeholder_text(claim):
            continue
        category = _clean_text(fact.get("category", "")).lower() or "community"
        evidence_segments.append(
            {
                "title": category_titles.get(category, "关键技术结论"),
                "content": claim,
                "talking_points": [claim],
                "source_refs": list(fact.get("evidence") or []),
                "category": category,
            }
        )
        if len(evidence_segments) >= 10:
            break

    ranked_results = sorted(
        [item for item in search_results if isinstance(item, dict)],
        key=lambda item: (
            source_priority.get(_clean_text(item.get("source", "")).lower(), 99),
            -_technical_signal_score(f"{item.get('title', '')} {item.get('content', '')}"),
        ),
    )
    for item in ranked_results[:24]:
        title = _normalize_statement(item.get("title", ""), max_len=120)
        content = _normalize_statement(item.get("content", ""), max_len=240)
        if not title and not content:
            continue
        sentence = _normalize_statement(re.split(r"[。.!?；;\n]+", content)[0], max_len=180) if content else ""
        source = _clean_text(item.get("source", "")).lower()
        snippet = _normalize_statement(f"[{source}] {title}: {sentence or content}", max_len=220)
        if not snippet or _is_placeholder_text(snippet):
            continue
        if _technical_signal_score(snippet) < 2 and source in {"reddit", "twitter"}:
            continue
        evidence_segments.append(
            {
                "title": title or category_titles.get(_infer_fact_category(snippet), "关键技术结论"),
                "content": snippet,
                "talking_points": [snippet],
                "source_refs": [_clean_text(item.get("id", ""))] if _clean_text(item.get("id", "")) else [],
                "category": _infer_fact_category(snippet),
            }
        )
        if len(evidence_segments) >= 18:
            break

    if not raw_segments:
        raw_segments = evidence_segments[:6]

    evidence_strength = float(len(facts)) * 1.1 + float(len(search_results)) * 0.25
    if evidence_strength >= 12:
        target_min_segments = 3
    elif evidence_strength >= 5:
        target_min_segments = 2
    else:
        target_min_segments = 1

    if len(raw_segments) < target_min_segments:
        existing_titles = {_normalize_statement(item.get("title", ""), max_len=120).lower() for item in raw_segments}
        for candidate in evidence_segments:
            candidate_title = _normalize_statement(candidate.get("title", ""), max_len=120).lower()
            if candidate_title and candidate_title in existing_titles:
                continue
            raw_segments.append(candidate)
            if candidate_title:
                existing_titles.add(candidate_title)
            if len(raw_segments) >= target_min_segments:
                break

    if not raw_segments:
        raw_segments = [
            {
                "title": "证据缺口与补证路径",
                "content": f"{_clean_text(topic) or '当前主题'} 当前可验证证据不足，建议补齐论文/代码/线上指标后再生成完整脚本。",
                "talking_points": [
                    "明确实验口径与对照基线",
                    "补充生产指标与失效案例",
                    "完成跨来源交叉验证后再下结论",
                ],
            }
        ]

    segment_count = max(1, min(6, len(raw_segments)))
    fallback_duration = max(20, min(90, int(200 / segment_count)))
    segments: List[Dict[str, Any]] = []

    for idx, segment in enumerate(raw_segments[:6], 1):
        title = _normalize_statement(segment.get("title", ""), max_len=90) or f"Segment {idx}"
        content = _normalize_statement(segment.get("content", ""), max_len=260)
        if not content:
            content = f"{_clean_text(topic) or '该主题'} 的该部分证据不足，需补充可验证来源。"
        talking_points = [
            _normalize_statement(point, max_len=120)
            for point in list(segment.get("talking_points") or [])
            if _normalize_statement(point, max_len=120)
        ]
        if not talking_points:
            split_points = [
                _normalize_statement(item, max_len=120)
                for item in re.split(r"[。.!?；;\n]+", content)
                if _normalize_statement(item, max_len=120)
            ]
            talking_points = split_points[:3] or [content]
        talking_points = _dedupe_text_list(talking_points, max_items=4)

        duration_raw = segment.get("duration_sec")
        explicit_duration = 0
        try:
            explicit_duration = max(0, int(duration_raw))
        except Exception:
            explicit_duration = 0
        narration_basis = " ".join([title, content] + talking_points)
        estimated_duration = _estimate_narration_duration_seconds(narration_basis, buffer_sec=2.0)
        duration_sec = estimated_duration or explicit_duration or fallback_duration
        duration_sec = max(12, min(180, duration_sec))

        visual_prompt = _normalize_statement(segment.get("visual_prompt", ""), max_len=220)
        if not visual_prompt:
            visual_prompt = (
                f"technical explainer for {topic}, segment '{title}', "
                "cinematic macro hardware shots, architecture diagrams, "
                "benchmark overlays, precise engineering visual language"
            )

        segments.append(
            {
                "title": title,
                "content": content,
                "talking_points": talking_points,
                "duration_sec": duration_sec,
                "visual_prompt": visual_prompt,
            }
        )

    normalized["segments"] = segments
    if not _clean_text(normalized.get("duration_estimate", "")):
        total_duration = sum(max(1, int(item.get("duration_sec", 0) or 1)) for item in segments)
        minutes, seconds = divmod(total_duration, 60)
        normalized["duration_estimate"] = f"{minutes}m {seconds:02d}s"
    return normalized


def aggregated_result_to_search_results(result: AggregatedResult) -> List[Dict[str, Any]]:
    """
    将聚合层数据统一转换为 Agent 搜索结果格式。
    """
    items: List[Dict[str, Any]] = []

    for paper in result.papers:
        items.append(
            {
                "id": f"{paper.source.value}_{paper.id}",
                "source": paper.source.value,
                "title": paper.title,
                "content": paper.abstract,
                "url": paper.url,
                "metadata": {
                    "authors": [a.name for a in paper.authors],
                    "published_date": _safe_iso(paper.published_date),
                    "updated_date": _safe_iso(paper.updated_date),
                    "categories": paper.categories,
                    "citation_count": paper.citation_count,
                    "pdf_url": paper.pdf_url,
                    **paper.extra,
                },
            }
        )

    for model in result.models:
        items.append(
            {
                "id": f"hf_model_{model.id}",
                "source": "huggingface",
                "title": model.name,
                "content": model.description or "",
                "url": model.url,
                "metadata": {
                    "author": model.author,
                    "repo_id": model.id,
                    "downloads": model.downloads,
                    "likes": model.likes,
                    "tags": model.tags,
                    "created_at": _safe_iso(model.created_at),
                    "updated_at": _safe_iso(model.updated_at),
                    **model.extra,
                },
            }
        )

    for dataset in result.datasets:
        items.append(
            {
                "id": f"hf_dataset_{dataset.id}",
                "source": "huggingface",
                "title": dataset.name,
                "content": dataset.description or "",
                "url": dataset.url,
                "metadata": {
                    "author": dataset.author,
                    "repo_id": dataset.id,
                    "downloads": dataset.downloads,
                    "tags": dataset.tags,
                    **dataset.extra,
                },
            }
        )

    for post in result.social_posts:
        items.append(
            {
                "id": f"{post.source.value}_{post.id}",
                "source": post.source.value,
                "title": f"{post.author} @ {post.source.value}",
                "content": post.content,
                "url": post.url,
                "metadata": {
                    "author": post.author,
                    "likes": post.likes,
                    "comments": post.comments,
                    "reposts": post.reposts,
                    "created_at": _safe_iso(post.created_at),
                    **post.extra,
                },
            }
        )

    for repo in result.github_repos:
        items.append(
            {
                "id": f"github_repo_{repo.id}",
                "source": "github",
                "title": repo.full_name,
                "content": repo.description or "",
                "url": repo.url,
                "metadata": {
                    "owner": repo.owner,
                    "repo_full_name": repo.full_name,
                    "language": repo.language,
                    "stars": repo.stars,
                    "forks": repo.forks,
                    "watchers": repo.watchers,
                    "topics": repo.topics,
                    "updated_at": _safe_iso(repo.updated_at),
                    **repo.extra,
                },
            }
        )

    for question in result.stackoverflow_questions:
        items.append(
            {
                "id": f"stackoverflow_{question.id}",
                "source": "stackoverflow",
                "title": question.title,
                "content": question.body or "",
                "url": question.url,
                "metadata": {
                    "author": question.author,
                    "tags": question.tags,
                    "score": question.score,
                    "view_count": question.view_count,
                    "answer_count": question.answer_count,
                    "is_answered": question.is_answered,
                    **question.extra,
                },
            }
        )

    for hn in result.hackernews_items:
        items.append(
            {
                "id": f"hackernews_{hn.id}",
                "source": "hackernews",
                "title": hn.title,
                "content": hn.text or "",
                "url": hn.url or hn.hn_url,
                "metadata": {
                    "author": hn.author,
                    "points": hn.points,
                    "comment_count": hn.comment_count,
                    "created_at": _safe_iso(hn.created_at),
                    **hn.extra,
                },
            }
        )

    return items


def evaluate_output_depth(
    *,
    facts: List[Dict[str, Any]],
    one_pager: Optional[Dict[str, Any]],
    video_brief: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    对输出深度做简单可解释评分，便于判断是否满足“技术细节充足”。
    """
    score = 0
    max_score = 14

    fact_categories = {str(f.get("category", "")).strip().lower() for f in facts}
    covered_required = sorted(list(_REQUIRED_FACT_CATEGORIES.intersection(fact_categories)))
    facts_with_evidence = [f for f in facts if f.get("evidence")]
    high_confidence = [f for f in facts if float(f.get("confidence", 0.0)) >= 0.7]

    if len(facts) >= 8:
        score += 2
    if len(covered_required) >= 4:
        score += 2
    if len(facts_with_evidence) >= max(4, len(facts) // 3):
        score += 1
    if len(high_confidence) >= max(4, len(facts) // 3):
        score += 1

    one_pager = one_pager or {}
    metrics = dict(one_pager.get("metrics", {}) or {})
    if len(one_pager.get("key_findings", [])) >= 5:
        score += 1
    if len(metrics) >= 3:
        score += 1
    if len(one_pager.get("technical_deep_dive", [])) >= 2:
        score += 1
    if len(one_pager.get("implementation_notes", [])) >= 2:
        score += 1
    if len(one_pager.get("risks_and_mitigations", [])) >= 2:
        score += 1
    structured_fields_present = sum(
        1
        for key in _STRUCTURED_TECH_FIELDS
        if _normalize_statement(metrics.get(key, ""), max_len=200)
        and "未明确披露" not in str(metrics.get(key, ""))
    )
    if structured_fields_present >= 3:
        score += 1

    video_brief = video_brief or {}
    segments = list(video_brief.get("segments", []))
    if len(segments) >= 3:
        score += 1
    if segments and all(
        bool(segment.get("visual_prompt")) and bool(segment.get("duration_sec"))
        for segment in segments
    ):
        score += 1

    one_pager_statements: List[str] = []
    for field in (
        "key_findings",
        "strengths",
        "weaknesses",
        "technical_deep_dive",
        "implementation_notes",
        "risks_and_mitigations",
    ):
        one_pager_statements.extend([_normalize_statement(item) for item in list(one_pager.get(field) or [])])
    one_pager_statements = [item for item in one_pager_statements if item]
    technical_signal_items = [item for item in one_pager_statements if _technical_signal_score(item) >= 2]
    low_signal_items = [item for item in one_pager_statements if _is_low_signal_statement(item)]
    technical_density = (
        len(technical_signal_items) / len(one_pager_statements)
        if one_pager_statements
        else 0.0
    )

    return {
        "score": score,
        "max_score": max_score,
        "pass": score >= 8,
        "fact_count": len(facts),
        "fact_categories": sorted(list(fact_categories)),
        "covered_required_categories": covered_required,
        "facts_with_evidence": len(facts_with_evidence),
        "high_confidence_facts": len(high_confidence),
        "technical_density": round(float(technical_density), 3),
        "low_signal_items": len(low_signal_items),
        "structured_fields_present": structured_fields_present,
    }


def evaluate_research_quality(
    *,
    facts: List[Dict[str, Any]],
    search_results: List[Dict[str, Any]],
    one_pager: Optional[Dict[str, Any]],
    video_brief: Optional[Dict[str, Any]],
    knowledge_gaps: Optional[List[str]] = None,
    threshold: float = 0.65,
) -> Dict[str, Any]:
    """
    评估研究输出质量（用于 Critic Gate）。

    指标:
    - coverage_score
    - citation_density
    - cross_source_ratio
    - conflict_resolution_rate
    - actionability_score
    """
    facts = list(facts or [])
    search_results = list(search_results or [])
    one_pager = one_pager or {}
    video_brief = video_brief or {}
    knowledge_gaps = list(knowledge_gaps or [])

    source_by_id = {
        str(item.get("id", "")).strip(): str(item.get("source", "")).strip()
        for item in search_results
        if str(item.get("id", "")).strip()
    }

    fact_categories = {str(f.get("category", "")).strip().lower() for f in facts}
    covered_required = sorted(list(_REQUIRED_FACT_CATEGORIES.intersection(fact_categories)))
    missing_required = sorted(list(_REQUIRED_FACT_CATEGORIES - set(covered_required)))
    coverage_score = len(covered_required) / max(1, len(_REQUIRED_FACT_CATEGORIES))

    evidence_sizes: List[int] = []
    cross_source_count = 0
    for fact in facts:
        evidence_ids = [str(item).strip() for item in (fact.get("evidence", []) or []) if str(item).strip()]
        unique_evidence_ids = list(dict.fromkeys(evidence_ids))
        evidence_sizes.append(len(unique_evidence_ids))
        sources = {source_by_id.get(eid, "") for eid in unique_evidence_ids if source_by_id.get(eid, "")}
        if len(sources) >= 2:
            cross_source_count += 1

    avg_citations = (sum(evidence_sizes) / len(evidence_sizes)) if evidence_sizes else 0.0
    citation_density = min(1.0, avg_citations / 2.0)
    cross_source_ratio = (cross_source_count / len(facts)) if facts else 0.0

    conflict_candidates = [
        item
        for item in facts
        if str(item.get("category", "")).strip().lower() in {"comparison", "limitation"}
    ]
    resolved = 0
    for item in conflict_candidates:
        claim = str(item.get("claim", "")).lower()
        if any(token in claim for token in _CONFLICT_HINTS):
            resolved += 1
    conflict_resolution_rate = (resolved / len(conflict_candidates)) if conflict_candidates else 1.0

    implementation_notes = [str(item).strip() for item in (one_pager.get("implementation_notes", []) or []) if str(item).strip()]
    risks = [str(item).strip() for item in (one_pager.get("risks_and_mitigations", []) or []) if str(item).strip()]
    key_findings = [str(item).strip() for item in (one_pager.get("key_findings", []) or []) if str(item).strip()]
    metrics = dict(one_pager.get("metrics", {}) or {})
    structured_fields_present = sum(
        1
        for key in _STRUCTURED_TECH_FIELDS
        if _normalize_statement(metrics.get(key, ""), max_len=200)
        and "未明确披露" not in str(metrics.get(key, ""))
    )
    segments = list(video_brief.get("segments", []) or [])
    one_pager_statements = implementation_notes + risks + key_findings + [
        str(item).strip()
        for item in (one_pager.get("technical_deep_dive", []) or [])
        if str(item).strip()
    ]
    technical_density = (
        sum(1 for item in one_pager_statements if _technical_signal_score(item) >= 2) / len(one_pager_statements)
        if one_pager_statements
        else 0.0
    )

    action_signals = 0
    if len(implementation_notes) >= 2:
        action_signals += 1
    if len(risks) >= 2:
        action_signals += 1
    if len(metrics) >= 3:
        action_signals += 1
    if segments and all(bool(seg.get("talking_points")) for seg in segments[:3]):
        action_signals += 1
    if any(any(token in note.lower() for token in _ACTION_HINTS) for note in implementation_notes):
        action_signals += 1
    if structured_fields_present >= 3:
        action_signals += 1
    actionability_score = action_signals / 6.0

    gap_penalty = min(0.25, 0.03 * len(knowledge_gaps))
    overall_score = (
        0.25 * coverage_score
        + 0.20 * citation_density
        + 0.20 * cross_source_ratio
        + 0.15 * conflict_resolution_rate
        + 0.20 * actionability_score
    )
    overall_score = max(0.0, min(1.0, overall_score - gap_penalty))

    recommendations: List[str] = []
    if coverage_score < 0.7 and missing_required:
        recommendations.append(
            "补齐核心维度证据，优先补 architecture/performance/training/comparison/limitation 缺口。"
        )
    if citation_density < 0.6:
        recommendations.append("提高每条关键结论的证据密度，至少 2 个独立 evidence。")
    if cross_source_ratio < 0.5:
        recommendations.append("增加跨来源交叉验证（论文+代码+社区），降低单源偏差。")
    if technical_density < 0.6:
        recommendations.append("提升结论的技术密度，优先写入架构机制、实验口径、关键参数和工程约束。")
    if structured_fields_present < 3:
        recommendations.append("补齐结构化技术字段（SOTA_Metric/Hardware_Requirement/Core_Formula/Key_Optimization）。")
    if conflict_resolution_rate < 0.5:
        recommendations.append("补充争议点与取舍分析，明确风险及缓解策略。")
    if actionability_score < 0.6:
        recommendations.append("强化可执行落地信息（配置步骤、监控指标、回滚策略）。")
    if not recommendations:
        recommendations.append("质量门控通过，下一步可针对细分场景扩展样本与对比基线。")

    gate_pass = (
        overall_score >= threshold
        and coverage_score >= 0.6
        and citation_density >= 0.5
        and actionability_score >= 0.5
    )

    return {
        "coverage_score": round(coverage_score, 4),
        "citation_density": round(citation_density, 4),
        "cross_source_ratio": round(cross_source_ratio, 4),
        "conflict_resolution_rate": round(conflict_resolution_rate, 4),
        "actionability_score": round(actionability_score, 4),
        "overall_score": round(overall_score, 4),
        "threshold": float(threshold),
        "pass": gate_pass,
        "covered_required_categories": covered_required,
        "missing_required_categories": missing_required,
        "facts_with_cross_source_evidence": cross_source_count,
        "avg_citations_per_fact": round(avg_citations, 4),
        "knowledge_gap_count": len(knowledge_gaps),
        "recommendations": recommendations,
        "fact_count": len(facts),
        "search_result_count": len(search_results),
        "key_findings_count": len(key_findings),
        "implementation_note_count": len(implementation_notes),
        "structured_fields_present": structured_fields_present,
    }
