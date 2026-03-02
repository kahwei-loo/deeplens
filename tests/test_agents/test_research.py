"""Tests for deeplens.agents.research — web-first data collection."""

from unittest.mock import MagicMock, patch

from deeplens.agents.research import (
    ResearchPlan,
    SearchQuery,
    _deduplicate_videos,
    research_agent,
)
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


def _mock_plan_llm(plan: ResearchPlan) -> MagicMock:
    """Create a mock LLM whose .with_structured_output().invoke() returns *plan*."""
    structured = MagicMock()
    structured.invoke.return_value = plan
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


# ── research_agent tests ─────────────────────────────────────────────────


@patch("deeplens.agents.research.extract_urls", return_value=_FAKE_ARTICLES)
@patch("deeplens.agents.research.multi_query_search", return_value=_FAKE_WEB)
@patch("deeplens.agents.research.get_settings")
@patch("deeplens.agents.research.get_llm")
def test_research_agent_basic(mock_get_llm, mock_settings, mock_search, mock_extract):
    """LLM plan + web search + extract → returns web_results, web_articles, sources."""
    mock_settings.return_value = MagicMock(youtube_available=False)
    plan = ResearchPlan(
        entity_type="artist/group",
        search_queries=[
            SearchQuery(query="Baby Monster overview", angle="overview"),
            SearchQuery(query="Baby Monster news", angle="news"),
        ],
        youtube_enrichment=False,
    )
    mock_get_llm.return_value = _mock_plan_llm(plan)

    state = _empty_state()
    result = research_agent(state)

    assert len(result["web_results"]) == 2
    assert len(result["web_articles"]) == 1
    assert len(result["sources"]) == 2
    assert all(s["source_type"] == "web" for s in result["sources"])
    mock_search.assert_called_once_with(
        ["Baby Monster overview", "Baby Monster news"], max_results_per_query=5
    )


@patch("deeplens.agents.research.web_search", return_value=_FAKE_WEB)
@patch("deeplens.agents.research.get_settings")
@patch("deeplens.agents.research.get_llm")
def test_research_agent_plan_failure_fallback(mock_get_llm, mock_settings, mock_web_search):
    """When LLM planning fails, falls back to basic web_search."""
    mock_settings.return_value = MagicMock(youtube_available=False)
    llm = MagicMock()
    structured = MagicMock()
    structured.invoke.side_effect = RuntimeError("LLM failed")
    llm.with_structured_output.return_value = structured
    mock_get_llm.return_value = llm

    state = _empty_state()
    result = research_agent(state)

    # Falls back to web_search with original query
    mock_web_search.assert_called_once_with("Research Baby Monster", max_results=10)
    assert len(result["web_results"]) == 2
    assert any("Research planning error" in e for e in result["errors"])


@patch("deeplens.agents.research.youtube_search")
@patch("deeplens.agents.research.extract_urls", return_value=[])
@patch("deeplens.agents.research.multi_query_search", return_value=_FAKE_WEB)
@patch("deeplens.agents.research.get_settings")
@patch("deeplens.agents.research.get_llm")
def test_research_agent_no_youtube(
    mock_get_llm, mock_settings, mock_search, mock_extract, mock_yt_search
):
    """youtube_available=False → no YouTube API calls made."""
    mock_settings.return_value = MagicMock(youtube_available=False)
    plan = ResearchPlan(
        entity_type="artist/group",
        search_queries=[SearchQuery(query="Baby Monster", angle="overview")],
        youtube_enrichment=True,  # Requested but not available
    )
    mock_get_llm.return_value = _mock_plan_llm(plan)

    state = _empty_state()
    result = research_agent(state)

    # YouTube search should NOT have been called
    mock_yt_search.assert_not_called()
    assert len(result["videos"]) == 0
    assert len(result["comments"]) == 0


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
    # First occurrence kept — title should be "A", not "A duplicate"
    assert deduped[0]["title"] == "A"
