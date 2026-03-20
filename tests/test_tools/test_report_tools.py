"""Tests for deeplens.tools.report_tools — Report Agent tool wrappers."""

from deeplens.models import Source, WebArticle, WebResult
from deeplens.state import DeepLensState
from deeplens.tools.report_tools import create_report_tools


def _state_with_data(**overrides) -> DeepLensState:
    """Build a state dict with some default research data."""
    base: dict = {
        "user_query": "Research Baby Monster",
        "research_plan": [],
        "web_results": [
            WebResult(
                title="BM Wiki", url="https://wiki.example.com/bm",
                snippet="K-pop group", score=0.9,
            ),
            WebResult(
                title="BM News", url="https://news.example.com/bm",
                snippet="Latest comeback", score=0.8,
            ),
        ],
        "web_articles": [
            WebArticle(
                url="https://wiki.example.com/bm",
                title="BM Wiki",
                content=(
                    "Baby Monster is a K-pop girl group formed by "
                    "YG Entertainment in 2023. They debuted with "
                    "seven members and quickly gained global "
                    "popularity."
                ),
                source_domain="wiki.example.com",
            ),
        ],
        "sources": [
            Source(url="https://wiki.example.com/bm", title="BM Wiki", source_type="web"),
        ],
        "channel_data": None,
        "videos": [],
        "comments": [],
        "statistics": None,
        "sentiment": None,
        "web_analysis": None,
        "report_markdown": "",
        "charts": [],
        "next_agent": "",
        "iteration_count": 0,
        "max_iterations": 5,
        "errors": [],
        "executed_queries": [],
    }
    base.update(overrides)
    return DeepLensState(**base)


def _empty_state() -> DeepLensState:
    return _state_with_data(
        web_results=[], web_articles=[], sources=[],
    )


# ── Tool creation ────────────────────────────────────────────────────────


def test_create_report_tools_returns_eight():
    """Factory returns exactly 8 tools."""
    tools = create_report_tools(_state_with_data(), "output")
    assert len(tools) == 8
    names = {t.name for t in tools}
    assert names == {
        "search_articles", "get_web_results_summary", "get_video_data",
        "get_statistics", "get_sentiment", "get_web_analysis",
        "get_sources", "generate_report_charts",
    }


# ── search_articles ──────────────────────────────────────────────────────


def test_search_articles_finds_relevant():
    """search_articles returns chunks matching query."""
    tools = create_report_tools(_state_with_data(), "output")
    tool_map = {t.name: t for t in tools}

    result = tool_map["search_articles"].invoke({"query": "Baby Monster K-pop"})
    assert "relevant chunks" in result.lower() or "found" in result.lower()


def test_search_articles_empty_state():
    """search_articles returns 'not available' on empty state."""
    tools = create_report_tools(_empty_state(), "output")
    tool_map = {t.name: t for t in tools}

    result = tool_map["search_articles"].invoke({"query": "anything"})
    assert "no web articles" in result.lower()


# ── get_web_results_summary ──────────────────────────────────────────────


def test_get_web_results_summary():
    """Returns web result summaries with titles and URLs."""
    tools = create_report_tools(_state_with_data(), "output")
    tool_map = {t.name: t for t in tools}

    result = tool_map["get_web_results_summary"].invoke({})
    assert "BM Wiki" in result
    assert "2 of 2" in result


def test_get_web_results_summary_empty():
    tools = create_report_tools(_empty_state(), "output")
    tool_map = {t.name: t for t in tools}

    result = tool_map["get_web_results_summary"].invoke({})
    assert "no web search results" in result.lower()


# ── get_video_data ───────────────────────────────────────────────────────


def test_get_video_data_no_data():
    tools = create_report_tools(_state_with_data(), "output")
    tool_map = {t.name: t for t in tools}

    result = tool_map["get_video_data"].invoke({})
    assert "no youtube" in result.lower()


def test_get_video_data_with_channel():
    state = _state_with_data(
        channel_data={
            "channel_id": "c1", "title": "BM Channel",
            "subscriber_count": 1000000,
            "view_count": 50000000, "video_count": 100,
        },
        videos=[{
            "video_id": "v1", "title": "MV",
            "view_count": 5000000, "like_count": 100000,
            "comment_count": 5000, "published_at": "2025-01-01",
        }],
    )
    tools = create_report_tools(state, "output")
    tool_map = {t.name: t for t in tools}

    result = tool_map["get_video_data"].invoke({})
    assert "BM Channel" in result
    assert "MV" in result


# ── get_statistics / get_sentiment / get_web_analysis ────────────────────


def test_data_tools_empty():
    """All data tools return 'not available' when state is empty."""
    tools = create_report_tools(_state_with_data(), "output")
    tool_map = {t.name: t for t in tools}

    for name in ["get_statistics", "get_sentiment", "get_web_analysis"]:
        result = tool_map[name].invoke({})
        assert "not available" in result.lower() or "no " in result.lower()


def test_get_sentiment_with_data():
    state = _state_with_data(
        sentiment={
            "positive": 0.65, "neutral": 0.25, "negative": 0.10,
            "total_analyzed": 50,
            "sample_positive": ["Great!", "Love it!"],
            "sample_negative": ["Not good"],
        }
    )
    tools = create_report_tools(state, "output")
    tool_map = {t.name: t for t in tools}

    result = tool_map["get_sentiment"].invoke({})
    assert "0.65" in result
    assert "50" in result


# ── get_sources ──────────────────────────────────────────────────────────


def test_get_sources():
    tools = create_report_tools(_state_with_data(), "output")
    tool_map = {t.name: t for t in tools}

    result = tool_map["get_sources"].invoke({})
    assert "BM Wiki" in result
    assert "1 of 1" in result
