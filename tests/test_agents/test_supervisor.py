"""Tests for deeplens.agents.supervisor — routing logic and fallbacks."""

from unittest.mock import MagicMock, patch

from deeplens.agents.supervisor import (
    SupervisorDecision,
    route_decision,
    supervisor_agent,
)
from deeplens.state import DeepLensState


def _empty_state(**overrides) -> DeepLensState:
    """Return a minimal DeepLensState with sensible defaults."""
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


def _mock_llm_returning(decision: SupervisorDecision) -> MagicMock:
    """Create a mock LLM whose .with_structured_output().invoke() returns *decision*."""
    structured = MagicMock()
    structured.invoke.return_value = decision
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


# ── Routing tests ────────────────────────────────────────────────────────


@patch("deeplens.agents.supervisor.get_settings")
@patch("deeplens.agents.supervisor.get_llm")
def test_supervisor_first_iteration(mock_get_llm, mock_settings):
    """iter=0 with no data — LLM routes to research."""
    mock_settings.return_value = MagicMock(youtube_available=False, max_iterations=5)
    decision = SupervisorDecision(
        next_agent="research",
        reason="No data yet",
        research_instructions="Search for Baby Monster overview",
    )
    mock_get_llm.return_value = _mock_llm_returning(decision).return_value  # wrong
    # We need get_llm() to return an object whose .with_structured_output() works
    llm_mock = MagicMock()
    structured_mock = MagicMock()
    structured_mock.invoke.return_value = decision
    llm_mock.with_structured_output.return_value = structured_mock
    mock_get_llm.return_value = llm_mock

    state = _empty_state(iteration_count=0)
    result = supervisor_agent(state)

    assert result["next_agent"] == "research"
    assert result["iteration_count"] == 1
    assert "Search for Baby Monster overview" in result["research_plan"]


@patch("deeplens.agents.supervisor.get_settings")
@patch("deeplens.agents.supervisor.get_llm")
def test_supervisor_routes_to_report(mock_get_llm, mock_settings):
    """Has web_results + web_articles, LLM decides report."""
    mock_settings.return_value = MagicMock(youtube_available=False, max_iterations=5)
    decision = SupervisorDecision(
        next_agent="report",
        reason="Sufficient web data collected",
    )
    llm_mock = MagicMock()
    structured_mock = MagicMock()
    structured_mock.invoke.return_value = decision
    llm_mock.with_structured_output.return_value = structured_mock
    mock_get_llm.return_value = llm_mock

    state = _empty_state(
        iteration_count=1,
        web_results=[{"title": "T", "url": "http://x", "snippet": "s", "score": 0.9}],
        web_articles=[
            {"url": "http://x", "title": "T", "content": "body", "source_domain": "x.com"},
        ],
    )
    result = supervisor_agent(state)

    assert result["next_agent"] == "report"


@patch("deeplens.agents.supervisor.get_settings")
@patch("deeplens.agents.supervisor.get_llm")
def test_supervisor_routes_to_analysis(mock_get_llm, mock_settings):
    """Has videos + comments but no statistics — routes to analysis."""
    mock_settings.return_value = MagicMock(youtube_available=True, max_iterations=5)
    decision = SupervisorDecision(
        next_agent="analysis",
        reason="Need to process video data",
    )
    llm_mock = MagicMock()
    structured_mock = MagicMock()
    structured_mock.invoke.return_value = decision
    llm_mock.with_structured_output.return_value = structured_mock
    mock_get_llm.return_value = llm_mock

    state = _empty_state(
        iteration_count=1,
        videos=[{
            "video_id": "v1", "title": "V", "view_count": 100,
            "like_count": 10, "comment_count": 5, "published_at": "2025-01-01T00:00:00Z",
        }],
        comments=[{"text": "Great", "like_count": 1, "author": "u", "video_id": "v1"}],
        statistics=None,
    )
    result = supervisor_agent(state)

    assert result["next_agent"] == "analysis"


@patch("deeplens.agents.supervisor.get_settings")
def test_supervisor_max_iterations(mock_settings):
    """At max_iterations the supervisor forces report — no LLM call needed."""
    mock_settings.return_value = MagicMock(max_iterations=5)

    state = _empty_state(iteration_count=5, max_iterations=5)
    result = supervisor_agent(state)

    assert result["next_agent"] == "report"
    assert result["iteration_count"] == 6


@patch("deeplens.agents.supervisor.get_settings")
@patch("deeplens.agents.supervisor.get_llm")
def test_supervisor_llm_failure_fallback(mock_get_llm, mock_settings):
    """LLM raises exception; has web data → falls back to report."""
    mock_settings.return_value = MagicMock(youtube_available=False, max_iterations=5)
    llm_mock = MagicMock()
    structured_mock = MagicMock()
    structured_mock.invoke.side_effect = RuntimeError("API timeout")
    llm_mock.with_structured_output.return_value = structured_mock
    mock_get_llm.return_value = llm_mock

    state = _empty_state(
        iteration_count=1,
        web_results=[{"title": "T", "url": "http://x", "snippet": "s", "score": 0.9}],
    )
    result = supervisor_agent(state)

    assert result["next_agent"] == "report"
    assert any("Supervisor LLM error" in e for e in result["errors"])


# ── route_decision ───────────────────────────────────────────────────────


def test_route_decision():
    """route_decision() extracts next_agent from state."""
    assert route_decision(_empty_state(next_agent="research")) == "research"
    assert route_decision(_empty_state(next_agent="analysis")) == "analysis"
    assert route_decision(_empty_state(next_agent="report")) == "report"
    assert route_decision(_empty_state(next_agent="done")) == "done"
    # Missing / empty defaults to "done"
    assert route_decision(_empty_state(next_agent="")) == ""
