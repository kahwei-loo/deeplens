"""Report agent — LLM-driven ReACT loop for research report generation.

Upgraded from L1 (single LLM call with monolithic context) to L4 (autonomous
tool calling): the LLM queries data on demand via tools, generates charts,
and writes the final report.

Tools available to the LLM:
- search_articles — keyword-based RAG over extracted web articles
- get_web_results_summary — overview of all web search results
- get_video_data — YouTube channel info and top video metrics
- get_statistics — computed video statistics
- get_sentiment — sentiment analysis results
- get_web_analysis — LLM-extracted themes and entities
- get_sources — all research sources for citation
- generate_report_charts — create PNG charts from analysis data
"""

import logging
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from deeplens.config import get_llm, get_settings
from deeplens.state import DeepLensState
from deeplens.tools.report_tools import create_report_tools

logger = logging.getLogger(__name__)

REPORT_REACT_PROMPT = """\
You are a research report writer for DeepLens. Your job is to generate a
well-structured markdown report based on research data you query via tools.

## How to Work
1. **Query data first**: Use your tools to gather the information you need.
   Start with get_web_results_summary and search_articles for key topics.
2. **Check for optional data**: Use get_video_data, get_statistics,
   get_sentiment, get_web_analysis to see what's available.
3. **Generate charts**: Call generate_report_charts before writing the report
   so you can reference chart file paths.
4. **Get sources**: Call get_sources to get all citations.
5. **Write the report**: Your final message (with no tool calls) should be
   the complete markdown report.

## Report Structure Guidelines

### For any entity — adapt based on available data:
1. **Title**: "Research Report: [Entity/Topic Name]"
2. **Executive Summary**: 2-3 sentence overview of key findings
3. **Overview**: Key background information synthesized from web sources.
   Include channel overview if YouTube channel data exists.
4. **Key Findings**: Synthesize the most important facts and insights from
   web articles and search results. Organize by theme (news, opinion, etc.).
5. **Content & Engagement** (if YouTube video data exists): Top videos,
   engagement patterns, metrics analysis.
6. **Public Sentiment** (if sentiment analysis exists): Sentiment breakdown
   with representative comments.
7. **Key Insights**: 3-5 bullet points of the most important findings
8. **Sources**: All sources used with URLs

### Adaptation rules
- **Web-only research** (no YouTube data): Focus sections 1-4, 7-8.
- **With YouTube data**: Include sections 5-6 with video metrics and sentiment.
- **With channel data**: Add channel subscriber/view counts to Overview.

## Rules
- Use markdown formatting with headers, bullet points, and bold text
- Include chart references as ![Chart Title](chart_path) where paths are provided
- Synthesize across multiple sources — don't just list what each source says
- Include actual numbers and statistics when available
- Quote representative comments when sentiment data exists
- Be factual and evidence-based — do not speculate beyond the data
- Keep the report concise but comprehensive (aim for 500-1000 words)
- List all sources at the end with URLs
"""


def report_agent(state: DeepLensState) -> dict:
    """LLM-driven ReACT report generation with tool-based data access.

    1. Create report tools with closures over state
    2. Run ReACT loop: LLM queries data tools, generates charts
    3. LLM's final message (no tool_calls) = the report markdown
    4. Save report to output directory
    """
    settings = get_settings()
    errors = list(state.get("errors") or [])
    query = state.get("user_query", "")

    # Collect chart paths via accumulator (not string parsing)
    charts: list[str] = []

    # Create tools — chart_paths_out writes directly to charts list
    tools = create_report_tools(
        state, settings.output_dir, chart_paths_out=charts,
    )
    tool_map = {t.name: t for t in tools}
    report_markdown = ""

    # ── ReACT Loop ─────────────────────────────────────────────────
    try:
        llm_with_tools = get_llm(temperature=0.3).bind_tools(tools)
        messages: list = [
            SystemMessage(content=REPORT_REACT_PROMPT),
            HumanMessage(content=f"Write a research report for: {query}"),
        ]

        for step in range(settings.max_report_tool_calls):
            response: AIMessage = llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                # Final message is the report
                report_markdown = str(response.content)
                logger.info(
                    "[Report] ReACT loop ended at step %d, report length=%d chars",
                    step, len(report_markdown),
                )
                break

            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                logger.info("[Report] Step %d: calling %s(%s)", step, tool_name, tool_args)

                if tool_name in tool_map:
                    try:
                        result = tool_map[tool_name].invoke(tool_args)
                    except Exception as e:
                        result = f"Tool error: {e}"
                        errors.append(f"Report tool {tool_name} error: {e}")
                else:
                    result = f"Unknown tool: {tool_name}"

                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        else:
            # Loop ended without a final non-tool message
            logger.warning(
                "[Report] ReACT loop hit max steps (%d)",
                settings.max_report_tool_calls,
            )
            # Use the last AIMessage content if available
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content:
                    report_markdown = str(msg.content)
                    break

    except Exception as e:
        error_msg = f"Report ReACT loop failed: {e}"
        logger.error("[Report] %s", error_msg)
        errors.append(error_msg)

    # Fallback if report is empty
    if not report_markdown.strip():
        logger.warning("[Report] Empty report from ReACT, using fallback")
        report_markdown = _build_fallback_report(state, charts)

    # Save report to output directory
    out = Path(settings.output_dir)
    try:
        out.mkdir(parents=True, exist_ok=True)
        report_path = out / "report.md"
        report_path.write_text(report_markdown, encoding="utf-8")
        logger.info("[Report] Saved to %s", report_path)
    except OSError as e:
        error_msg = f"Report file write failed: {e}"
        logger.error("[Report] %s", error_msg)
        errors.append(error_msg)

    return {
        "report_markdown": report_markdown,
        "charts": charts,
        "errors": errors,
    }


def _build_fallback_report(state: DeepLensState, chart_paths: list[str]) -> str:
    """Generate a basic report without LLM if the ReACT loop fails."""
    query = state.get("user_query", "Unknown")
    lines = [
        f"# Research Report: {query}",
        "",
        "## Executive Summary",
        "",
        f"This report was generated for the query: \"{query}\". "
        "Note: The AI report writer encountered an error, "
        "so this is a simplified report.",
        "",
    ]

    # Web articles
    articles = state.get("web_articles") or []
    if articles:
        lines.extend(["## Key Sources", ""])
        for a in articles[:5]:
            title = a.get("title", "Untitled")
            domain = a.get("source_domain", "")
            lines.append(f"- **{title}** ({domain})")
            content = a.get("content", "")
            if content:
                lines.append(f"  {content[:200]}...")
        lines.append("")

    channel = state.get("channel_data")
    if channel:
        lines.extend([
            "## Channel Overview",
            "",
            f"- **Channel**: {channel.get('title', 'Unknown')}",
            f"- **Subscribers**: {channel.get('subscriber_count', 0):,}",
            f"- **Total Views**: {channel.get('view_count', 0):,}",
            f"- **Videos**: {channel.get('video_count', 0)}",
            "",
        ])

    videos = state.get("videos") or []
    if videos:
        lines.extend(["## Videos Found", ""])
        sorted_vids = sorted(videos, key=lambda v: v.get("view_count", 0), reverse=True)
        for v in sorted_vids[:5]:
            lines.append(
                f"- **{v.get('title', '?')}** — "
                f"{v.get('view_count', 0):,} views"
            )
        lines.append("")

    statistics = state.get("statistics")
    if statistics:
        avg_views = statistics.get("avg_views") or 0
        avg_eng = statistics.get("avg_engagement_rate") or 0
        lines.extend([
            "## Statistics",
            "",
            f"- Average views: {avg_views:,.0f}",
            f"- Average engagement rate: {avg_eng:.2%}",
            "",
        ])

    sentiment = state.get("sentiment")
    if sentiment:
        pos = sentiment.get("positive") or 0
        neu = sentiment.get("neutral") or 0
        neg = sentiment.get("negative") or 0
        lines.extend([
            "## Sentiment",
            "",
            f"- Positive: {pos:.0%}",
            f"- Neutral: {neu:.0%}",
            f"- Negative: {neg:.0%}",
            f"- Comments analyzed: {sentiment.get('total_analyzed', 0)}",
            "",
        ])

    for path in chart_paths:
        lines.append(f"![Chart]({path})")
        lines.append("")

    sources = state.get("sources") or []
    if sources:
        lines.extend(["## Sources", ""])
        for s in sources:
            lines.append(f"- [{s.get('title', 'Source')}]({s.get('url', '')})")

    return "\n".join(lines)
