"""Tests for deeplens.agents.research — ReACT-based research agent."""

from unittest.mock import MagicMock, patch

from deeplens.agents.research import _deduplicate_videos, research_agent
from deeplens.models import WebArticle, WebResult, YouTubeVideoData
from deeplens.state import DeepLensState


def _empty_state(**overrides) -> DeepLensState:
    base: dict = {
        "user_query": "Research Baby Monster",
        "research_plan": [],
        "web_results": [],
        "web_articles": [],
        "sources": [],
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


_FAKE_WEB: list[WebResult] = [
    {
        "title": "Baby Monster Wikipedia",
        "url": "https://en.wikipedia.org/wiki/Baby_Monster",
        "snippet": "K-pop group",
        "score": 0.95,
    },
    {
        "title": "Baby Monster latest news",
        "url": "https://news.example.com/bm",
        "snippet": "Recent comeback",
        "score": 0.88,
    },
]

_FAKE_ARTICLES: list[WebArticle] = [
    {
        "url": "https://en.wikipedia.org/wiki/Baby_Monster",
        "title": "Baby Monster Wikipedia",
        "content": "Full article text...",
        "source_domain": "en.wikipedia.org",
    },
]


def _make_ai_message(tool_calls=None, content=""):
    """Create a mock AIMessage with optional tool_calls."""
    msg = MagicMock()
    msg.tool_calls = tool_calls or []
    msg.content = content
    return msg


def _mock_llm_with_tools(responses):
    """Create a mock LLM whose bind_tools().invoke() returns responses in sequence."""
    llm_with_tools = MagicMock()
    llm_with_tools.invoke = MagicMock(side_effect=responses)

    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=llm_with_tools)
    return llm


# ── ReACT flow tests ─────────────────────────────────────────────────────


@patch("deeplens.agents.research.web_search")
@patch("deeplens.agents.research.create_research_tools")
@patch("deeplens.agents.research.get_settings")
@patch("deeplens.agents.research.get_llm")
def test_research_agent_react_basic(
    mock_get_llm, mock_settings, mock_create_tools, mock_ws,
):
    """ReACT loop: LLM calls search_web_multi, then extract_article_content, then stops."""
    mock_settings.return_value = MagicMock(
        youtube_available=False, max_research_tool_calls=15,
    )

    # Mock the tools themselves
    mock_search = MagicMock(name="search_web_multi")
    mock_search.name = "search_web_multi"
    mock_search.invoke.return_value = "Found 2 results"

    mock_extract = MagicMock(name="extract_article_content")
    mock_extract.name = "extract_article_content"
    mock_extract.invoke.return_value = "Extracted 1 article"

    mock_create_tools.return_value = [mock_search, mock_extract]

    # LLM response sequence: call search → call extract → done (no tool_calls)
    responses = [
        _make_ai_message(tool_calls=[{
            "name": "search_web_multi",
            "args": {"queries": "Baby Monster overview, Baby Monster news"},
            "id": "tc1",
        }]),
        _make_ai_message(tool_calls=[{
            "name": "extract_article_content",
            "args": {"urls": "https://example.com/1"},
            "id": "tc2",
        }]),
        _make_ai_message(content="Research complete."),  # No tool_calls → loop ends
    ]
    mock_get_llm.return_value = _mock_llm_with_tools(responses)

    state = _empty_state()
    result = research_agent(state)

    # Verify tool invocations
    mock_search.invoke.assert_called_once()
    mock_extract.invoke.assert_called_once()

    # State contract: returns expected keys
    assert "web_results" in result
    assert "web_articles" in result
    assert "sources" in result
    assert "errors" in result
    assert "executed_queries" in result


@patch("deeplens.agents.research.create_research_tools")
@patch("deeplens.agents.research.get_settings")
@patch("deeplens.agents.research.get_llm")
def test_research_agent_max_steps(mock_get_llm, mock_settings, mock_create_tools):
    """ReACT loop stops at max_research_tool_calls even if LLM keeps calling tools."""
    mock_settings.return_value = MagicMock(
        youtube_available=False, max_research_tool_calls=2,
    )

    mock_tool = MagicMock(name="search_web")
    mock_tool.name = "search_web"
    mock_tool.invoke.return_value = "Results"
    mock_create_tools.return_value = [mock_tool]

    # LLM always returns tool calls (never stops)
    never_stops = _make_ai_message(tool_calls=[
        {"name": "search_web", "args": {"query": "test"}, "id": "tc1"},
    ])
    mock_get_llm.return_value = _mock_llm_with_tools([never_stops, never_stops, never_stops])

    state = _empty_state()
    result = research_agent(state)

    # Should have called tool exactly 2 times (max_research_tool_calls=2)
    assert mock_tool.invoke.call_count == 2
    assert "web_results" in result


@patch("deeplens.agents.research.web_search", return_value=_FAKE_WEB)
@patch("deeplens.agents.research.create_research_tools")
@patch("deeplens.agents.research.get_settings")
@patch("deeplens.agents.research.get_llm")
def test_research_agent_fallback_on_failure(
    mock_get_llm, mock_settings, mock_create_tools, mock_ws,
):
    """When the ReACT loop fails completely, falls back to basic web_search."""
    mock_settings.return_value = MagicMock(
        youtube_available=False, max_research_tool_calls=15,
    )
    mock_create_tools.return_value = []

    # LLM raises an exception
    llm = MagicMock()
    llm.bind_tools.side_effect = RuntimeError("LLM initialization failed")
    mock_get_llm.return_value = llm

    state = _empty_state()
    result = research_agent(state)

    # Fallback web_search should have been called
    mock_ws.assert_called_once_with("Research Baby Monster", max_results=10)
    assert len(result["web_results"]) == 2
    assert any("ReACT error" in e for e in result["errors"])


@patch("deeplens.agents.research.create_research_tools")
@patch("deeplens.agents.research.get_settings")
@patch("deeplens.agents.research.get_llm")
def test_research_agent_tool_error_handling(mock_get_llm, mock_settings, mock_create_tools):
    """Tool execution errors are caught and recorded, loop continues."""
    mock_settings.return_value = MagicMock(
        youtube_available=False, max_research_tool_calls=15,
    )

    mock_tool = MagicMock(name="search_web")
    mock_tool.name = "search_web"
    mock_tool.invoke.side_effect = RuntimeError("Tool crashed")
    mock_create_tools.return_value = [mock_tool]

    responses = [
        _make_ai_message(tool_calls=[
            {"name": "search_web", "args": {"query": "test"}, "id": "tc1"},
        ]),
        _make_ai_message(content="Done."),
    ]
    mock_get_llm.return_value = _mock_llm_with_tools(responses)

    state = _empty_state()
    result = research_agent(state)

    # Error should be recorded
    assert any("error" in e.lower() for e in result["errors"])
    # But result should still be valid
    assert "web_results" in result


@patch("deeplens.agents.research.create_research_tools")
@patch("deeplens.agents.research.get_settings")
@patch("deeplens.agents.research.get_llm")
def test_research_agent_state_contract(mock_get_llm, mock_settings, mock_create_tools):
    """Result dict has all expected keys matching the state contract."""
    mock_settings.return_value = MagicMock(
        youtube_available=False, max_research_tool_calls=15,
    )
    mock_create_tools.return_value = []

    # LLM immediately stops (no tool calls)
    mock_get_llm.return_value = _mock_llm_with_tools([
        _make_ai_message(content="No research needed."),
    ])

    state = _empty_state()
    result = research_agent(state)

    expected_keys = {
        "web_results", "web_articles", "videos", "comments",
        "sources", "errors", "executed_queries",
    }
    assert expected_keys.issubset(set(result.keys()))


@patch("deeplens.agents.research.create_research_tools")
@patch("deeplens.agents.research.get_settings")
@patch("deeplens.agents.research.get_llm")
def test_research_agent_passes_supervisor_instructions(
    mock_get_llm, mock_settings, mock_create_tools,
):
    """Supervisor instructions are included in the system prompt."""
    mock_settings.return_value = MagicMock(
        youtube_available=False, max_research_tool_calls=15,
    )
    mock_create_tools.return_value = []

    llm_with_tools = MagicMock()
    llm_with_tools.invoke.return_value = _make_ai_message(content="Done.")
    llm = MagicMock()
    llm.bind_tools.return_value = llm_with_tools
    mock_get_llm.return_value = llm

    state = _empty_state(
        research_plan=["Search for overview", "Focus on controversies"],
    )
    research_agent(state)

    # Check the system message contains supervisor instructions
    call_args = llm_with_tools.invoke.call_args[0][0]
    system_content = call_args[0].content
    assert "Search for overview" in system_content
    assert "Focus on controversies" in system_content


# ── Deduplication across iterations ──────────────────────────────────────


@patch("deeplens.agents.research.create_research_tools")
@patch("deeplens.agents.research.get_settings")
@patch("deeplens.agents.research.get_llm")
def test_web_results_deduplicated_across_iterations(mock_get_llm, mock_settings, mock_create_tools):
    """web_results from multiple iterations are deduplicated by URL."""
    mock_settings.return_value = MagicMock(
        youtube_available=False, max_research_tool_calls=15,
    )
    mock_create_tools.return_value = []

    mock_get_llm.return_value = _mock_llm_with_tools([
        _make_ai_message(content="Done."),
    ])

    existing: list[WebResult] = [
        {"title": "Article 1", "url": "https://example.com/a1", "snippet": "...", "score": 0.8},
    ]
    state = _empty_state(web_results=existing)
    result = research_agent(state)

    urls = [r["url"] for r in result["web_results"]]
    assert len(urls) == len(set(urls))


# ── _deduplicate_videos ──────────────────────────────────────────────────


def test_deduplicate_videos():
    """Removes duplicate video_ids, keeping first occurrence."""
    videos: list[YouTubeVideoData] = [
        {
            "video_id": "a", "title": "A", "view_count": 100,
            "like_count": 10, "comment_count": 1,
            "published_at": "2025-01-01T00:00:00Z",
        },
        {
            "video_id": "b", "title": "B", "view_count": 200,
            "like_count": 20, "comment_count": 2,
            "published_at": "2025-02-01T00:00:00Z",
        },
        {
            "video_id": "a", "title": "A duplicate", "view_count": 999,
            "like_count": 99, "comment_count": 9,
            "published_at": "2025-03-01T00:00:00Z",
        },
    ]
    deduped = _deduplicate_videos(videos)

    assert len(deduped) == 2
    ids = [v["video_id"] for v in deduped]
    assert ids == ["a", "b"]
    assert deduped[0]["title"] == "A"
