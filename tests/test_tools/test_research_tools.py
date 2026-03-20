"""Tests for deeplens.tools.research_tools — LLM-callable research tool wrappers."""

from unittest.mock import MagicMock, patch

from deeplens.models import WebArticle, WebResult
from deeplens.tools.research_tools import ResearchCollector, create_research_tools


def _mock_settings(youtube_available: bool = False) -> MagicMock:
    s = MagicMock()
    s.youtube_available = youtube_available
    return s


_FAKE_WEB: list[WebResult] = [
    {"title": "Result 1", "url": "https://example.com/1", "snippet": "Snippet 1", "score": 0.9},
    {"title": "Result 2", "url": "https://example.com/2", "snippet": "Snippet 2", "score": 0.8},
]

_FAKE_ARTICLES: list[WebArticle] = [
    {
        "url": "https://example.com/1", "title": "Article 1",
        "content": "Full text...", "source_domain": "example.com",
    },
]


# ── Tool creation ────────────────────────────────────────────────────────


def test_create_research_tools_returns_six_tools():
    """Factory returns exactly 6 tools."""
    collector = ResearchCollector()
    tools = create_research_tools(collector, _mock_settings())
    assert len(tools) == 6
    names = {t.name for t in tools}
    assert names == {
        "search_web", "search_web_multi", "extract_article_content",
        "search_youtube", "get_youtube_comments", "get_youtube_channel",
    }


# ── search_web ───────────────────────────────────────────────────────────


@patch("deeplens.tools.research_tools.web_search", return_value=_FAKE_WEB)
def test_search_web_accumulates(mock_ws):
    """search_web appends results to collector."""
    collector = ResearchCollector()
    tools = create_research_tools(collector, _mock_settings())
    tool_map = {t.name: t for t in tools}

    result = tool_map["search_web"].invoke({"query": "test query"})

    assert "Found 2 results" in result
    assert len(collector.web_results) == 2
    assert "test query" in collector.executed_queries
    assert len(collector.sources) == 2


@patch("deeplens.tools.research_tools.web_search", side_effect=RuntimeError("API down"))
def test_search_web_error_handling(mock_ws):
    """search_web catches errors and appends to collector.errors."""
    collector = ResearchCollector()
    tools = create_research_tools(collector, _mock_settings())
    tool_map = {t.name: t for t in tools}

    result = tool_map["search_web"].invoke({"query": "test"})

    assert "error" in result.lower()
    assert len(collector.errors) == 1


# ── search_web_multi ─────────────────────────────────────────────────────


@patch("deeplens.tools.research_tools.multi_query_search", return_value=_FAKE_WEB)
def test_search_web_multi(mock_mqs):
    """search_web_multi accepts a list of queries."""
    collector = ResearchCollector()
    tools = create_research_tools(collector, _mock_settings())
    tool_map = {t.name: t for t in tools}

    result = tool_map["search_web_multi"].invoke(
        {"queries": ["query1", "query2"]}
    )

    assert "Found 2" in result
    mock_mqs.assert_called_once_with(
        ["query1", "query2"], max_results_per_query=5,
    )
    assert len(collector.executed_queries) == 2


# ── extract_article_content ──────────────────────────────────────────────


@patch("deeplens.tools.research_tools.extract_urls", return_value=_FAKE_ARTICLES)
def test_extract_articles(mock_ext):
    """extract_article_content parses comma-separated URLs."""
    collector = ResearchCollector()
    tools = create_research_tools(collector, _mock_settings())
    tool_map = {t.name: t for t in tools}

    result = tool_map["extract_article_content"].invoke(
        {"urls": "https://example.com/1, https://example.com/2"}
    )

    assert "Extracted 1" in result
    assert len(collector.web_articles) == 1


# ── YouTube tools unavailability ─────────────────────────────────────────


def test_youtube_tools_unavailable():
    """YouTube tools return 'unavailable' message when no API key."""
    collector = ResearchCollector()
    tools = create_research_tools(collector, _mock_settings(youtube_available=False))
    tool_map = {t.name: t for t in tools}

    for name in ["search_youtube", "get_youtube_comments", "get_youtube_channel"]:
        result = tool_map[name].invoke({"query": "test"} if "search" in name
                                       else {"video_id": "abc"} if "comments" in name
                                       else {"channel_name": "test"})
        assert "not available" in result.lower()

    assert len(collector.videos) == 0
    assert len(collector.comments) == 0
    assert collector.channel_data is None


# ── YouTube tools with API key ───────────────────────────────────────────


@patch("deeplens.tools.research_tools.youtube_search")
def test_search_youtube_with_key(mock_yt):
    """search_youtube calls youtube_search when API key available."""
    mock_yt.return_value = [
        {"video_id": "v1", "title": "Video 1", "view_count": 1000,
         "like_count": 100, "comment_count": 10, "published_at": "2025-01-01"},
    ]
    collector = ResearchCollector()
    tools = create_research_tools(collector, _mock_settings(youtube_available=True))
    tool_map = {t.name: t for t in tools}

    result = tool_map["search_youtube"].invoke({"query": "test"})

    assert "Found 1 videos" in result
    assert len(collector.videos) == 1
    assert len(collector.sources) == 1
