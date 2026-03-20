"""Tests for deeplens.agents.report — ReACT-based report generation."""

from unittest.mock import MagicMock, patch

from deeplens.agents.report import _build_fallback_report, report_agent
from deeplens.models import Source, WebArticle, WebResult
from deeplens.state import DeepLensState


def _state_with_data(**overrides) -> DeepLensState:
    base: dict = {
        "user_query": "Research Baby Monster",
        "research_plan": [],
        "web_results": [
            WebResult(
                title="BM Wiki", url="https://wiki.example.com/bm",
                snippet="K-pop group", score=0.9,
            ),
        ],
        "web_articles": [
            WebArticle(
                url="https://wiki.example.com/bm",
                title="BM Wiki",
                content="Baby Monster is a K-pop girl group formed by YG Entertainment.",
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


def _make_ai_message(tool_calls=None, content=""):
    msg = MagicMock()
    msg.tool_calls = tool_calls or []
    msg.content = content
    return msg


def _mock_llm_with_tools(responses):
    llm_with_tools = MagicMock()
    llm_with_tools.invoke = MagicMock(side_effect=responses)
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=llm_with_tools)
    return llm


# ── ReACT flow tests ─────────────────────────────────────────────────────


@patch("deeplens.agents.report.create_report_tools")
@patch("deeplens.agents.report.get_settings")
@patch("deeplens.agents.report.get_llm")
def test_report_agent_react_basic(
    mock_get_llm, mock_settings, mock_create_tools, tmp_path,
):
    """ReACT loop: LLM queries data tools, then writes report."""
    mock_settings.return_value = MagicMock(
        output_dir=str(tmp_path), max_report_tool_calls=10,
    )

    mock_summary = MagicMock(name="get_web_results_summary")
    mock_summary.name = "get_web_results_summary"
    mock_summary.invoke.return_value = "2 web results found"

    mock_sources = MagicMock(name="get_sources")
    mock_sources.name = "get_sources"
    mock_sources.invoke.return_value = "1 source: BM Wiki"

    mock_create_tools.return_value = [mock_summary, mock_sources]

    report_text = (
        "# Research Report: Baby Monster\n\n"
        "## Executive Summary\nBaby Monster is a K-pop group."
    )

    responses = [
        _make_ai_message(tool_calls=[
            {"name": "get_web_results_summary", "args": {}, "id": "tc1"},
        ]),
        _make_ai_message(tool_calls=[
            {"name": "get_sources", "args": {}, "id": "tc2"},
        ]),
        _make_ai_message(content=report_text),
    ]
    mock_get_llm.return_value = _mock_llm_with_tools(responses)

    state = _state_with_data()
    result = report_agent(state)

    assert result["report_markdown"] == report_text
    assert "errors" in result
    assert "charts" in result

    # Report file should be saved
    report_file = tmp_path / "report.md"
    assert report_file.exists()
    assert report_file.read_text(encoding="utf-8") == report_text


@patch("deeplens.agents.report.create_report_tools")
@patch("deeplens.agents.report.get_settings")
@patch("deeplens.agents.report.get_llm")
def test_report_agent_max_steps(
    mock_get_llm, mock_settings, mock_create_tools, tmp_path,
):
    """ReACT loop stops at max_report_tool_calls."""
    mock_settings.return_value = MagicMock(
        output_dir=str(tmp_path), max_report_tool_calls=2,
    )

    mock_tool = MagicMock(name="get_web_results_summary")
    mock_tool.name = "get_web_results_summary"
    mock_tool.invoke.return_value = "data"
    mock_create_tools.return_value = [mock_tool]

    never_stops = _make_ai_message(tool_calls=[
        {"name": "get_web_results_summary", "args": {}, "id": "tc1"},
    ])
    # After max steps, the last response content becomes the report
    mock_get_llm.return_value = _mock_llm_with_tools([never_stops, never_stops, never_stops])

    state = _state_with_data()
    result = report_agent(state)

    assert mock_tool.invoke.call_count == 2
    # Should still produce some result (fallback if empty)
    assert "report_markdown" in result


@patch("deeplens.agents.report.create_report_tools")
@patch("deeplens.agents.report.get_settings")
@patch("deeplens.agents.report.get_llm")
def test_report_agent_fallback_on_failure(
    mock_get_llm, mock_settings, mock_create_tools, tmp_path,
):
    """Falls back to _build_fallback_report when ReACT loop fails."""
    mock_settings.return_value = MagicMock(
        output_dir=str(tmp_path), max_report_tool_calls=10,
    )
    mock_create_tools.return_value = []

    llm = MagicMock()
    llm.bind_tools.side_effect = RuntimeError("LLM init failed")
    mock_get_llm.return_value = llm

    state = _state_with_data()
    result = report_agent(state)

    # Fallback report should contain the query
    assert "Baby Monster" in result["report_markdown"]
    has_error = (
        "error" in result["report_markdown"].lower()
        or any("error" in e.lower() for e in result["errors"])
    )
    assert has_error


@patch("deeplens.agents.report.create_report_tools")
@patch("deeplens.agents.report.get_settings")
@patch("deeplens.agents.report.get_llm")
def test_report_agent_state_contract(
    mock_get_llm, mock_settings, mock_create_tools, tmp_path,
):
    """Result dict has report_markdown, charts, and errors keys."""
    mock_settings.return_value = MagicMock(
        output_dir=str(tmp_path), max_report_tool_calls=10,
    )
    mock_create_tools.return_value = []

    mock_get_llm.return_value = _mock_llm_with_tools([
        _make_ai_message(content="# Report"),
    ])

    state = _state_with_data()
    result = report_agent(state)

    assert "report_markdown" in result
    assert "charts" in result
    assert "errors" in result
    assert isinstance(result["charts"], list)
    assert isinstance(result["errors"], list)


@patch("deeplens.agents.report.create_report_tools")
@patch("deeplens.agents.report.get_settings")
@patch("deeplens.agents.report.get_llm")
def test_report_agent_chart_tracking(
    mock_get_llm, mock_settings, mock_create_tools, tmp_path,
):
    """Chart paths are tracked via the accumulator, not string parsing."""
    mock_settings.return_value = MagicMock(
        output_dir=str(tmp_path), max_report_tool_calls=10,
    )

    mock_chart = MagicMock(name="generate_report_charts")
    mock_chart.name = "generate_report_charts"
    mock_chart.invoke.return_value = (
        "Generated 1 charts:\n"
        "- ![Chart](output/sentiment_pie.png)"
    )

    # Capture the chart_paths_out kwarg and write to it
    def fake_create_tools(state, output_dir, chart_paths_out=None):
        if chart_paths_out is not None:
            # Simulate what the real tool does: write paths
            orig_invoke = mock_chart.invoke

            def invoke_and_accumulate(args):
                result = orig_invoke(args)
                chart_paths_out.append("output/sentiment_pie.png")
                return result

            mock_chart.invoke = invoke_and_accumulate
        return [mock_chart]

    mock_create_tools.side_effect = fake_create_tools

    responses = [
        _make_ai_message(tool_calls=[{
            "name": "generate_report_charts",
            "args": {}, "id": "tc1",
        }]),
        _make_ai_message(content="# Report with charts"),
    ]
    mock_get_llm.return_value = _mock_llm_with_tools(responses)

    state = _state_with_data()
    result = report_agent(state)

    assert "output/sentiment_pie.png" in result["charts"]


# ── Fallback report ──────────────────────────────────────────────────────


def test_build_fallback_report_basic():
    """Fallback report includes query and available data."""
    state = _state_with_data()
    report = _build_fallback_report(state, [])

    assert "Baby Monster" in report
    assert "Executive Summary" in report
    assert "BM Wiki" in report


def test_build_fallback_report_with_youtube():
    """Fallback report includes YouTube data when available."""
    state = _state_with_data(
        channel_data={
            "channel_id": "c1", "title": "BM Official",
            "subscriber_count": 5000000, "view_count": 100000000, "video_count": 50,
        },
        videos=[{
            "video_id": "v1", "title": "MV", "view_count": 10000000,
            "like_count": 500000, "comment_count": 20000, "published_at": "2025-01-01",
        }],
    )
    report = _build_fallback_report(state, ["output/chart.png"])

    assert "BM Official" in report
    assert "5,000,000" in report
    assert "MV" in report
    assert "![Chart](output/chart.png)" in report
