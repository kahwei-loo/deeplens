"""Tests for deeplens.tools.validation — report quality checks."""

from deeplens.models import Source, WebArticle, WebResult
from deeplens.state import DeepLensState
from deeplens.tools.validation import (
    report_quality_summary,
    validate_report,
)

_GOOD_REPORT = """\
# Research Report: Baby Monster

## Executive Summary

Baby Monster is a K-pop girl group formed by YG Entertainment in 2023.

## Key Findings

- The group debuted with seven members
- Their music videos gained millions of views
- Strong presence on social media

## Sources

- [BM Wiki](https://wiki.example.com/bm)
- [BM News](https://news.example.com/bm)
"""


def _state(**overrides) -> DeepLensState:
    base: dict = {
        "user_query": "Research Baby Monster",
        "research_plan": [],
        "web_results": [
            WebResult(
                title="BM Wiki",
                url="https://wiki.example.com/bm",
                snippet="K-pop group", score=0.9,
            ),
        ],
        "web_articles": [
            WebArticle(
                url="https://wiki.example.com/bm",
                title="BM Wiki", content="...",
                source_domain="wiki.example.com",
            ),
        ],
        "sources": [
            Source(
                url="https://wiki.example.com/bm",
                title="BM Wiki", source_type="web",
            ),
            Source(
                url="https://news.example.com/bm",
                title="BM News", source_type="web",
            ),
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


def test_good_report_passes_all_checks():
    """A well-structured report passes all validation checks."""
    checks = validate_report(_GOOD_REPORT, _state())
    _, _, failed = report_quality_summary(checks)
    assert failed == [], f"Unexpected failures: {failed}"


def test_empty_report_fails():
    """Empty report fails basic checks."""
    checks = validate_report("", _state())
    assert checks["has_content"] is False
    assert checks["min_length"] is False


def test_no_title_detected():
    """Report without heading fails has_title."""
    report = "This report has no title header.\nJust plain text."
    checks = validate_report(report, _state())
    assert checks["has_title"] is False


def test_no_summary_detected():
    """Report without summary section fails has_summary."""
    report = "# Report\n\nJust raw content with no overview."
    checks = validate_report(report, _state())
    assert checks["has_summary"] is False


def test_urls_grounded():
    """URLs in report that match sources pass grounding check."""
    report = (
        "# Report\n\n## Summary\n\nBased on "
        "[source](https://wiki.example.com/bm).\n\n"
        "## Sources\n\nhttps://wiki.example.com/bm"
    )
    checks = validate_report(report, _state())
    assert checks["urls_grounded"] is True


def test_ungrounded_urls_detected():
    """URLs in report that don't match any source fail grounding."""
    report = (
        "# Report\n\n## Summary\n\nBased on "
        "https://fake-source.com/made-up and "
        "https://another-fake.com/hallucinated.\n\n"
        "## Sources\n"
    )
    checks = validate_report(report, _state())
    assert checks["urls_grounded"] is False


def test_fallback_report_detected():
    """Reports containing error markers are flagged."""
    report = (
        "# Report\n\n## Summary\n\n"
        "This report encountered an error and is simplified.\n\n"
        "## Sources\n"
    )
    checks = validate_report(report, _state())
    assert checks["not_fallback"] is False


def test_quality_summary():
    """report_quality_summary returns correct counts."""
    checks = {"a": True, "b": True, "c": False, "d": True}
    passed, total, failed = report_quality_summary(checks)
    assert passed == 3
    assert total == 4
    assert failed == ["c"]
