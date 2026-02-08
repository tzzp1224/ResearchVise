"""
Unit tests for GitHub scraper pagination robustness.
"""

from __future__ import annotations

from scrapers.social.github_scraper import GitHubScraper


class _IterOnlyPaginated:
    """
    Mimics a paginated list that supports iteration but can fail on slice access.
    """

    def __init__(self, items):
        self._items = list(items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, item):
        if isinstance(item, slice):
            raise IndexError("slice access is not supported")
        return self._items[item]


class _DummyGithub:
    def search_repositories(self, query, sort, order):
        return _IterOnlyPaginated(["repo_a", "repo_b", "repo_c"])

    def search_issues(self, query):
        return _IterOnlyPaginated(["issue_a", "issue_b"])


def test_sync_search_repos_works_with_iter_only_paginated():
    scraper = GitHubScraper()
    scraper._github = _DummyGithub()

    results = scraper._sync_search_repos(
        query="test",
        max_results=2,
        sort="stars",
        order="desc",
    )

    assert results == ["repo_a", "repo_b"]


def test_sync_search_issues_works_with_iter_only_paginated():
    scraper = GitHubScraper()
    scraper._github = _DummyGithub()

    results = scraper._sync_search_issues(
        query="test",
        max_results=1,
    )

    assert results == ["issue_a"]
