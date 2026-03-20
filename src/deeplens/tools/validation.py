"""Structural quality validation for generated research reports.

Provides automated quality gates that verify report structure, content
grounding, and citation integrity — without requiring LLM-as-judge.
"""

from __future__ import annotations

import logging
import re

from deeplens.state import DeepLensState

logger = logging.getLogger(__name__)


def validate_report(report: str, state: DeepLensState) -> dict[str, bool]:
    """Run structural quality checks on a generated report.

    Returns a dict of check_name → pass/fail. All checks are deterministic
    and fast (no LLM calls). Designed to catch common report generation
    failures without requiring human review.
    """
    checks: dict[str, bool] = {}

    # Basic structure
    checks["has_content"] = bool(report.strip())
    checks["has_title"] = report.lstrip().startswith("#")
    checks["min_length"] = len(report) >= 200
    checks["max_length"] = len(report) <= 20000

    # Expected sections (case-insensitive)
    report_lower = report.lower()
    checks["has_summary"] = (
        "executive summary" in report_lower
        or "summary" in report_lower
    )
    checks["has_sources_section"] = bool(
        re.search(r"^#{1,3}\s+.*source", report_lower, re.MULTILINE)
    )

    # Citation grounding — URLs in report should come from actual sources
    report_urls = set(re.findall(r"https?://[^\s\)\]\"<>,]+", report))
    source_urls = {
        s.get("url", "") for s in (state.get("sources") or [])
    }
    web_urls = {
        r.get("url", "") for r in (state.get("web_results") or [])
    }
    known_urls = source_urls | web_urls
    if report_urls:
        grounded = report_urls & known_urls
        checks["urls_grounded"] = len(grounded) >= len(report_urls) * 0.5
    else:
        # No URLs in report is acceptable (some reports use plain text)
        checks["urls_grounded"] = True

    # Data utilization — report should reference available data
    has_web = bool(state.get("web_results"))
    has_articles = bool(state.get("web_articles"))
    if has_web or has_articles:
        # Report should be substantially longer than the fallback
        checks["data_utilized"] = len(report) >= 300
    else:
        checks["data_utilized"] = True

    # No fallback markers (indicates LLM failure path was taken)
    checks["not_fallback"] = (
        "encountered an error" not in report_lower
    )

    return checks


def report_quality_summary(
    checks: dict[str, bool],
) -> tuple[int, int, list[str]]:
    """Summarize validation results.

    Returns (passed, total, list_of_failed_check_names).
    """
    passed = sum(1 for v in checks.values() if v)
    total = len(checks)
    failed = [k for k, v in checks.items() if not v]
    return passed, total, failed
