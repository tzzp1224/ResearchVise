"""Unit tests for DataAggregator helper behaviors."""

from aggregator.data_aggregator import DataAggregator


def test_topic_query_candidates_for_mcp():
    aggregator = DataAggregator(
        enable_arxiv=False,
        enable_huggingface=False,
        enable_twitter=False,
        enable_reddit=False,
        enable_github=False,
        enable_semantic_scholar=False,
        enable_stackoverflow=False,
        enable_hackernews=False,
    )

    candidates = aggregator._topic_query_candidates("MCP production deployment")

    assert candidates[0] == "MCP production deployment"
    assert "MCP" in candidates
    assert "Model Context Protocol" in candidates
