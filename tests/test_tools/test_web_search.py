"""Tests for deeplens.tools.web_search — Tavily-based web search and extraction."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from deeplens.config import get_settings
from deeplens.tools.web_search import (
    _deduplicate_results,
    _get_tavily_client,
    extract_urls,
    multi_query_search,
    web_search,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tavily_result(title: str, url: str, score: float = 0.8) -> dict:
    """Create a single Tavily-style search result dict."""
    return {
        "title": title,
        "url": url,
        "content": f"Snippet for {title}",
        "score": score,
    }


def _make_tavily_extract(url: str, raw_content: str = "Full article text") -> dict:
    """Create a single Tavily-style extract result dict."""
    return {
        "url": url,
        "title": f"Page at {url}",
        "raw_content": raw_content,
    }


def _setup_mock_client() -> MagicMock:
    """Clear caches and set up a mock TavilyClient via _get_tavily_client."""
    get_settings.cache_clear()
    _get_tavily_client.cache_clear()
    return MagicMock()


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------

@patch("deeplens.tools.web_search._get_tavily_client")
def test_web_search_basic(mock_get_client: MagicMock) -> None:
    """web_search returns a list of WebResult dicts with correct keys."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.search.return_value = {
        "results": [
            _make_tavily_result("Result 1", "https://example.com/1", 0.9),
            _make_tavily_result("Result 2", "https://example.com/2", 0.7),
        ],
    }

    results = web_search("test query")

    assert len(results) == 2
    for r in results:
        assert set(r.keys()) == {"title", "url", "snippet", "score"}
    assert results[0]["title"] == "Result 1"
    assert results[0]["url"] == "https://example.com/1"
    assert results[0]["score"] == 0.9


@patch("deeplens.tools.web_search._get_tavily_client")
def test_web_search_respects_max_results(mock_get_client: MagicMock) -> None:
    """max_results parameter is forwarded to the Tavily client."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.search.return_value = {"results": []}

    web_search("query", max_results=3)

    mock_client.search.assert_called_once_with(
        query="query", max_results=3, search_depth="basic",
    )


# ---------------------------------------------------------------------------
# multi_query_search
# ---------------------------------------------------------------------------

@patch("deeplens.tools.web_search._get_tavily_client")
def test_multi_query_search(mock_get_client: MagicMock) -> None:
    """multi_query_search calls web_search per query and deduplicates by URL."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    # Two queries, each returns different URLs
    mock_client.search.side_effect = [
        {"results": [_make_tavily_result("A", "https://a.com", 0.9)]},
        {"results": [_make_tavily_result("B", "https://b.com", 0.8)]},
    ]

    results = multi_query_search(["query1", "query2"], max_results_per_query=1)

    assert len(results) == 2
    assert mock_client.search.call_count == 2
    urls = {r["url"] for r in results}
    assert urls == {"https://a.com", "https://b.com"}


@patch("deeplens.tools.web_search._get_tavily_client")
def test_multi_query_search_dedup(mock_get_client: MagicMock) -> None:
    """Same URL from two queries is kept once with the highest score."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    shared_url = "https://shared.com/page"
    mock_client.search.side_effect = [
        {"results": [_make_tavily_result("Low", shared_url, 0.5)]},
        {"results": [_make_tavily_result("High", shared_url, 0.9)]},
    ]

    results = multi_query_search(["q1", "q2"], max_results_per_query=1)

    assert len(results) == 1
    assert results[0]["score"] == 0.9
    assert results[0]["title"] == "High"


# ---------------------------------------------------------------------------
# extract_urls
# ---------------------------------------------------------------------------

@patch("deeplens.tools.web_search._get_tavily_client")
def test_extract_urls(mock_get_client: MagicMock) -> None:
    """extract_urls calls client.extract and returns WebArticle list."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.extract.return_value = {
        "results": [
            _make_tavily_extract("https://example.com/article"),
        ],
    }

    articles = extract_urls(["https://example.com/article"])

    assert len(articles) == 1
    art = articles[0]
    assert set(art.keys()) == {"url", "title", "content", "source_domain"}
    assert art["url"] == "https://example.com/article"
    assert art["source_domain"] == "example.com"
    assert art["content"] == "Full article text"


def test_extract_urls_empty() -> None:
    """Empty URL list returns empty without calling the API."""
    result = extract_urls([])
    assert result == []


@patch("deeplens.tools.web_search._get_tavily_client")
def test_extract_urls_graceful_failure(mock_get_client: MagicMock) -> None:
    """If extract raises an exception, returns empty list without crashing."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.extract.side_effect = RuntimeError("Paid plan required")

    result = extract_urls(["https://example.com"])

    assert result == []


# ---------------------------------------------------------------------------
# _deduplicate_results
# ---------------------------------------------------------------------------

def test_deduplicate_results() -> None:
    """Keeps highest score per URL and sorts descending by score."""
    from deeplens.models import WebResult

    results: list[WebResult] = [
        WebResult(title="A", url="https://a.com", snippet="...", score=0.5),
        WebResult(title="B", url="https://b.com", snippet="...", score=0.9),
        WebResult(title="A2", url="https://a.com", snippet="...", score=0.8),
    ]

    deduped = _deduplicate_results(results)

    assert len(deduped) == 2
    # Sorted descending by score
    assert deduped[0]["score"] == 0.9
    assert deduped[0]["url"] == "https://b.com"
    assert deduped[1]["score"] == 0.8
    assert deduped[1]["url"] == "https://a.com"
    # Higher-scored "A2" kept, not "A"
    assert deduped[1]["title"] == "A2"
