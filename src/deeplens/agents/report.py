"""Report agent — generates charts and a structured markdown research report.

Produces a comprehensive report that adapts its structure based on what data
is available. Entity type determines the report layout: artist profiles get
channel overviews, public figures get perception analysis, topics get
landscape comparisons.
"""

import logging
from pathlib import Path

from deeplens.config import get_llm, get_settings
from deeplens.state import DeepLensState
from deeplens.tools.chart import generate_charts

logger = logging.getLogger(__name__)

REPORT_SYSTEM_PROMPT = """\
You are a research report writer for DeepLens. Generate a well-structured
markdown report based on the provided research data. Adapt the report
structure based on the entity type and available data.

## Report structure guidelines

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
  The web articles and search results ARE your primary evidence.
- **With YouTube data**: Include sections 5-6 with video metrics and sentiment.
- **With channel data**: Add channel subscriber/view counts to Overview.

## Rules
- Use markdown formatting with headers, bullet points, and bold text
- Include chart references as ![Chart Title](chart_path) where chart paths are provided
- Synthesize across multiple sources — don't just list what each source says
- Include actual numbers and statistics when available
- Quote representative comments when sentiment data exists
- Be factual and evidence-based — do not speculate beyond the data
- Keep the report concise but comprehensive (aim for 500-1000 words)
- List all sources at the end with URLs
"""


def report_agent(state: DeepLensState) -> dict:
    """Generate charts and a structured markdown research report.

    1. Generate charts from statistics and sentiment data
    2. Build comprehensive report context from all state data
    3. Use LLM to generate a structured markdown report
    4. Save report to output directory
    """
    settings = get_settings()
    errors = list(state.get("errors") or [])

    # Generate charts
    charts: list[str] = []
    statistics = state.get("statistics")
    sentiment = state.get("sentiment")

    try:
        charts = generate_charts(statistics=statistics, sentiment=sentiment)
        logger.info("[Report] Generated %d charts", len(charts))
    except Exception as e:
        error_msg = f"Chart generation failed: {e}"
        logger.error("[Report] %s", error_msg)
        errors.append(error_msg)

    # Build report context for the LLM
    report_context = _build_report_context(state, charts)

    # Generate report with LLM
    llm = get_llm(temperature=0.3)

    try:
        response = llm.invoke(
            [
                {"role": "system", "content": REPORT_SYSTEM_PROMPT},
                {"role": "user", "content": report_context},
            ]
        )
        report_markdown = response.content
        logger.info("[Report] Generated report (%d chars)", len(report_markdown))
    except Exception as e:
        error_msg = f"Report generation LLM failed: {e}"
        logger.error("[Report] %s", error_msg)
        errors.append(error_msg)
        report_markdown = _build_fallback_report(state, charts)

    # Save report to output directory
    out = Path(settings.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "report.md"
    report_path.write_text(report_markdown, encoding="utf-8")
    logger.info("[Report] Saved to %s", report_path)

    return {
        "report_markdown": report_markdown,
        "charts": charts,
        "errors": errors,
    }


def _build_report_context(state: DeepLensState, chart_paths: list[str]) -> str:
    """Build comprehensive context from all state data for report generation."""
    lines: list[str] = []

    query = state.get("user_query", "")
    lines.append(f"Research query: {query}")
    lines.append("")

    # Web articles (primary data source)
    articles = state.get("web_articles") or []
    if articles:
        lines.append(f"## Web Articles ({len(articles)} extracted)")
        for a in articles:
            lines.append(f"### [{a.get('source_domain', 'Unknown')}] {a.get('title', 'Untitled')}")
            lines.append(f"URL: {a.get('url', '')}")
            content = a.get("content", "")
            # Include up to 2000 chars per article for the LLM context
            if len(content) > 2000:
                content = content[:2000] + "..."
            lines.append(content)
            lines.append("")

    # Channel data (optional YouTube enrichment)
    channel = state.get("channel_data")
    if channel:
        lines.append("## YouTube Channel Data")
        lines.append(f"- Name: {channel.get('title', 'Unknown')}")
        lines.append(f"- Channel ID: {channel.get('channel_id', 'N/A')}")
        lines.append(f"- Subscribers: {channel.get('subscriber_count', 0):,}")
        lines.append(f"- Total views: {channel.get('view_count', 0):,}")
        lines.append(f"- Video count: {channel.get('video_count', 0)}")
        lines.append("")

    # Videos
    videos = state.get("videos") or []
    if videos:
        lines.append(f"## Videos ({len(videos)} found)")
        sorted_vids = sorted(videos, key=lambda v: v.get("view_count", 0), reverse=True)
        for v in sorted_vids[:10]:
            lines.append(
                f"- \"{v.get('title', '?')}\" — "
                f"{v.get('view_count', 0):,} views, "
                f"{v.get('like_count', 0):,} likes, "
                f"{v.get('comment_count', 0):,} comments "
                f"(published: {v.get('published_at', 'unknown')})"
            )
        lines.append("")

    # Statistics
    statistics = state.get("statistics")
    if statistics:
        lines.append("## Video Statistics")
        lines.append(f"- Average views: {statistics.get('avg_views', 0):,.0f}")
        lines.append(f"- Average likes: {statistics.get('avg_likes', 0):,.0f}")
        lines.append(f"- Average engagement rate: {statistics.get('avg_engagement_rate', 0):.2%}")
        freq = statistics.get("upload_frequency_days")
        if freq:
            lines.append(f"- Upload frequency: ~{freq:.1f} days between uploads")
        top = statistics.get("top_videos") or []
        if top:
            lines.append("- Top performing videos:")
            for tv in top[:5]:
                lines.append(f"  - \"{tv.get('title', '?')}\" ({tv.get('view_count', 0):,} views)")
        lines.append("")

    # Sentiment
    sentiment = state.get("sentiment")
    if sentiment:
        lines.append("## Sentiment Analysis")
        lines.append(f"- Total comments analyzed: {sentiment.get('total_analyzed', 0)}")
        lines.append(f"- Positive: {sentiment.get('positive', 0):.0%}")
        lines.append(f"- Neutral: {sentiment.get('neutral', 0):.0%}")
        lines.append(f"- Negative: {sentiment.get('negative', 0):.0%}")
        pos_samples = sentiment.get("sample_positive") or []
        if pos_samples:
            lines.append("- Sample positive comments:")
            for c in pos_samples[:3]:
                lines.append(f'  - "{c}"')
        neg_samples = sentiment.get("sample_negative") or []
        if neg_samples:
            lines.append("- Sample negative comments:")
            for c in neg_samples[:3]:
                lines.append(f'  - "{c}"')
        lines.append("")

    # Web analysis (LLM-extracted themes and entities from web articles)
    web_analysis = state.get("web_analysis")
    if web_analysis:
        lines.append("## Web Analysis")
        summary = web_analysis.get("summary", "")
        if summary:
            lines.append(f"Summary: {summary}")
        themes = web_analysis.get("key_themes") or []
        if themes:
            lines.append("Key themes:")
            for t in themes:
                lines.append(f"- {t}")
        entities = web_analysis.get("entity_mentions") or []
        if entities:
            lines.append("Notable entities mentioned:")
            for e in entities:
                lines.append(f"- {e}")
        lines.append("")

    # Web results (capped to avoid token blowout)
    web = state.get("web_results") or []
    if web:
        capped = web[:20]
        lines.append(f"## Web Results ({len(capped)} of {len(web)} shown)")
        for w in capped:
            lines.append(f"- [{w.get('title', 'Untitled')}]({w.get('url', '')})")
            snippet = w.get("snippet", "")
            if snippet:
                lines.append(f"  {snippet}")
        lines.append("")

    # Charts
    if chart_paths:
        lines.append("## Generated Charts")
        lines.append("Include these charts in the report using markdown image syntax:")
        for path in chart_paths:
            lines.append(f"- ![Chart]({path})")
        lines.append("")

    # Sources (capped to avoid token blowout)
    sources = state.get("sources") or []
    if sources:
        capped = sources[:30]
        lines.append(f"## Sources ({len(capped)} of {len(sources)})")
        for s in capped:
            title = s.get("title", "Source")
            url = s.get("url", "")
            stype = s.get("source_type", "")
            lines.append(f"- [{title}]({url}) [{stype}]")
        lines.append("")

    # Errors
    errs = state.get("errors") or []
    if errs:
        lines.append(f"## Errors encountered ({len(errs)})")
        lines.append("Note any data limitations in the report:")
        for e in errs:
            lines.append(f"- {e}")

    return "\n".join(lines)


def _build_fallback_report(state: DeepLensState, chart_paths: list[str]) -> str:
    """Generate a basic report without LLM if the LLM call fails."""
    query = state.get("user_query", "Unknown")
    lines = [
        f"# Research Report: {query}",
        "",
        "## Executive Summary",
        "",
        f"This report was generated for the query: \"{query}\". "
        "Note: The AI report writer encountered an error, so this is a simplified report.",
        "",
    ]

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
        lines.extend([
            "## Statistics",
            "",
            f"- Average views: {statistics.get('avg_views', 0):,.0f}",
            f"- Average engagement rate: {statistics.get('avg_engagement_rate', 0):.2%}",
            "",
        ])

    sentiment = state.get("sentiment")
    if sentiment:
        lines.extend([
            "## Sentiment",
            "",
            f"- Positive: {sentiment.get('positive', 0):.0%}",
            f"- Neutral: {sentiment.get('neutral', 0):.0%}",
            f"- Negative: {sentiment.get('negative', 0):.0%}",
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
