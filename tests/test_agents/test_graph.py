"""Tests for deeplens.graph — graph creation and initial state."""

from unittest.mock import MagicMock, patch

from deeplens.graph import build_initial_state, create_graph

# ── build_initial_state ──────────────────────────────────────────────────


@patch("deeplens.graph.get_settings")
def test_build_initial_state(mock_settings):
    """Verify all fields present with correct defaults."""
    mock_settings.return_value = MagicMock(max_iterations=5)

    state = build_initial_state("Research Baby Monster")

    assert state["user_query"] == "Research Baby Monster"
    assert state["research_plan"] == []
    assert state["web_results"] == []
    assert state["sources"] == []
    assert state["channel_data"] is None
    assert state["videos"] == []
    assert state["comments"] == []
    assert state["statistics"] is None
    assert state["sentiment"] is None
    assert state["web_analysis"] is None
    assert state["report_markdown"] == ""
    assert state["charts"] == []
    assert state["next_agent"] == ""
    assert state["iteration_count"] == 0
    assert state["max_iterations"] == 5
    assert state["errors"] == []


@patch("deeplens.graph.get_settings")
def test_build_initial_state_web_articles_empty(mock_settings):
    """web_articles starts as an empty list."""
    mock_settings.return_value = MagicMock(max_iterations=5)

    state = build_initial_state("Research Elon Musk")

    assert state["web_articles"] == []
    assert isinstance(state["web_articles"], list)


# ── create_graph ─────────────────────────────────────────────────────────


def test_create_graph():
    """Graph compiles without error and has expected nodes."""
    compiled = create_graph()

    # LangGraph compiled graph exposes a `get_graph()` method
    graph_repr = compiled.get_graph()
    # .nodes may be a dict (node_id -> node) or iterable of node objects
    node_ids = set(graph_repr.nodes)

    assert "supervisor" in node_ids
    assert "research" in node_ids
    assert "analysis" in node_ids
    assert "report" in node_ids
