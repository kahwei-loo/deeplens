"""Tests for deeplens.agents.analysis — statistics, sentiment, and web analysis."""

from unittest.mock import MagicMock, patch

from deeplens.agents.analysis import WebAnalysisResult, analysis_agent
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


_FAKE_VIDEOS = [
    {
        "video_id": "v1",
        "title": "V1",
        "view_count": 1000,
        "like_count": 100,
        "comment_count": 10,
        "published_at": "2025-01-01T00:00:00Z",
    },
]

_FAKE_COMMENTS = [
    {"text": "Great video!", "like_count": 5, "author": "user1", "video_id": "v1"},
    {"text": "Not bad", "like_count": 1, "author": "user2", "video_id": "v1"},
]

_FAKE_ARTICLES = [
    {
        "url": "https://example.com/article1",
        "title": "Baby Monster Overview",
        "content": "Baby Monster is a K-pop group formed by YG Entertainment.",
        "source_domain": "example.com",
    },
    {
        "url": "https://news.example.com/bm",
        "title": "Baby Monster Comeback",
        "content": "The group recently made a comeback with a new album.",
        "source_domain": "news.example.com",
    },
]

_FAKE_WEB_ANALYSIS = WebAnalysisResult(
    key_themes=["K-pop debut", "YG Entertainment", "Music industry"],
    entity_mentions=["Baby Monster", "YG Entertainment", "BLACKPINK"],
    summary="Baby Monster is a rising K-pop group under YG Entertainment.",
)


# ── No data at all ──────────────────────────────────────────────────────


def test_analysis_no_data():
    """No videos, comments, or articles — appends error, returns empty."""
    state = _empty_state()
    result = analysis_agent(state)

    assert "Analysis skipped: no data to process" in result["errors"]
    assert "statistics" not in result
    assert "sentiment" not in result
    assert "web_analysis" not in result


# ── YouTube path (videos + comments) ────────────────────────────────────


@patch("deeplens.agents.analysis.sentiment_analyzer")
@patch("deeplens.agents.analysis.compute_statistics")
def test_analysis_youtube_path(mock_stats, mock_sentiment):
    """Videos + comments → runs statistics and sentiment, no web analysis."""
    mock_stats.return_value = {
        "avg_views": 1000.0,
        "avg_likes": 100.0,
        "avg_engagement_rate": 0.1,
        "top_videos": _FAKE_VIDEOS,
        "upload_frequency_days": 7.0,
    }
    mock_sentiment.return_value = {
        "positive": 0.7,
        "neutral": 0.2,
        "negative": 0.1,
        "total_analyzed": 2,
        "sample_positive": ["Great video!"],
        "sample_negative": [],
    }

    state = _empty_state(videos=_FAKE_VIDEOS, comments=_FAKE_COMMENTS)
    result = analysis_agent(state)

    assert result["statistics"]["avg_views"] == 1000.0
    assert result["sentiment"]["total_analyzed"] == 2
    assert "web_analysis" not in result
    mock_stats.assert_called_once_with(_FAKE_VIDEOS)
    mock_sentiment.assert_called_once_with(_FAKE_COMMENTS)


# ── Web-only path (articles, no comments) ───────────────────────────────


@patch("deeplens.agents.analysis.get_llm")
def test_analysis_web_only_path(mock_get_llm):
    """Web articles + no comments → runs web analysis via LLM."""
    structured_mock = MagicMock()
    structured_mock.invoke.return_value = _FAKE_WEB_ANALYSIS
    llm_mock = MagicMock()
    llm_mock.with_structured_output.return_value = structured_mock
    mock_get_llm.return_value = llm_mock

    state = _empty_state(web_articles=_FAKE_ARTICLES)
    result = analysis_agent(state)

    assert result["web_analysis"] is not None
    assert "K-pop debut" in result["web_analysis"]["key_themes"]
    assert "Baby Monster" in result["web_analysis"]["entity_mentions"]
    assert len(result["web_analysis"]["summary"]) > 0
    assert "statistics" not in result
    assert "sentiment" not in result


@patch("deeplens.agents.analysis.get_llm")
def test_analysis_web_only_llm_failure(mock_get_llm):
    """Web analysis LLM failure → error recorded, no crash."""
    structured_mock = MagicMock()
    structured_mock.invoke.side_effect = RuntimeError("LLM API timeout")
    llm_mock = MagicMock()
    llm_mock.with_structured_output.return_value = structured_mock
    mock_get_llm.return_value = llm_mock

    state = _empty_state(web_articles=_FAKE_ARTICLES)
    result = analysis_agent(state)

    assert any("Web article analysis failed" in e for e in result["errors"])
    assert "web_analysis" not in result


# ── Mixed path (articles + comments) ────────────────────────────────────


@patch("deeplens.agents.analysis.get_llm")
@patch("deeplens.agents.analysis.sentiment_analyzer")
@patch("deeplens.agents.analysis.compute_statistics")
def test_analysis_mixed_runs_all_analyses(mock_stats, mock_sentiment, mock_get_llm):
    """Articles + comments → runs sentiment AND web analysis (mixed path).

    Regression test for bug where web_analysis was silently skipped when
    comments were present (condition was `if web_articles and not comments`).
    """
    mock_stats.return_value = None  # No videos
    mock_sentiment.return_value = {
        "positive": 0.6,
        "neutral": 0.3,
        "negative": 0.1,
        "total_analyzed": 2,
        "sample_positive": ["Nice!"],
        "sample_negative": [],
    }
    structured_mock = MagicMock()
    structured_mock.invoke.return_value = _FAKE_WEB_ANALYSIS
    llm_mock = MagicMock()
    llm_mock.with_structured_output.return_value = structured_mock
    mock_get_llm.return_value = llm_mock

    state = _empty_state(
        web_articles=_FAKE_ARTICLES,
        comments=_FAKE_COMMENTS,
    )
    result = analysis_agent(state)

    assert result["sentiment"]["total_analyzed"] == 2
    # Web analysis now runs even when comments exist (fixed behavior)
    assert "web_analysis" in result
    assert "K-pop debut" in result["web_analysis"]["key_themes"]


# ── Statistics failure ──────────────────────────────────────────────────


@patch("deeplens.agents.analysis.compute_statistics")
def test_analysis_statistics_failure(mock_stats):
    """Statistics exception → error recorded, no crash."""
    mock_stats.side_effect = ValueError("Bad video data")

    state = _empty_state(videos=_FAKE_VIDEOS)
    result = analysis_agent(state)

    assert any("Statistics computation failed" in e for e in result["errors"])
    assert "statistics" not in result


# ── Sentiment failure ───────────────────────────────────────────────────


@patch("deeplens.agents.analysis.sentiment_analyzer")
def test_analysis_sentiment_failure(mock_sentiment):
    """Sentiment exception → error recorded, no crash."""
    mock_sentiment.side_effect = RuntimeError("API error")

    state = _empty_state(comments=_FAKE_COMMENTS)
    result = analysis_agent(state)

    assert any("Sentiment analysis failed" in e for e in result["errors"])
    assert "sentiment" not in result
