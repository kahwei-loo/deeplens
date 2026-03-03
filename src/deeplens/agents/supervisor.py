"""Supervisor agent — the adaptive routing brain of DeepLens.

The Supervisor evaluates the current research state and decides which agent
runs next. This is NOT a fixed pipeline — the LLM makes context-dependent
routing decisions based on information completeness, entity type, and findings.

Architecture: **Web-first** — web search is the primary data source;
YouTube API is optional enrichment when configured and relevant.
"""

import logging
from typing import Literal, cast

from pydantic import BaseModel, Field

from deeplens.config import get_llm, get_settings
from deeplens.state import DeepLensState

logger = logging.getLogger(__name__)

SUPERVISOR_SYSTEM_PROMPT = """\
You are the Supervisor agent for DeepLens, a multi-agent entity research system.
Your job is to evaluate the current research state and decide which agent should
run next. You make context-dependent routing decisions based on information
completeness, entity type, and research quality.

## Architecture: Web-First Research

DeepLens follows a **web-first** research approach (similar to Perplexity, Manus):
- **Web search is the primary data source** — covers any entity, any platform.
  The Research agent searches from multiple angles (overview, news, opinion,
  controversy) and extracts full content from top URLs.
- **YouTube API is optional enrichment** — provides precise video metrics and
  comments when the entity has a significant YouTube presence AND the API key
  is configured. If not configured, the system works fully on web data alone.

## Agents you can route to

- **research**: Data collection agent. Primary: multi-angle web search +
  URL content extraction. Optional: YouTube API enrichment.
  Route here when more information is needed.
- **analysis**: Data processing agent. Computes video statistics (if video
  data exists), runs LLM-based sentiment analysis on comments, and extracts
  key themes and entities from web articles (web-only or mixed path).
  Route here when raw data has been collected and needs processing.
- **report**: Report generation agent. Creates charts and a structured markdown
  report. Route here when you are satisfied with the data or analysis is done.
- **done**: Terminal state. Only for unrecoverable errors or empty queries.

## Routing rules

### First iteration (no data collected yet)
Always route to **research**. Provide research_instructions describing:
- The entity type and what angles to search from
- Whether YouTube enrichment would add value (only for entities with strong
  YouTube presence: artists, creators, brands)

### Subsequent iterations — evaluate information completeness
Check what data exists:
- web_results: How many? Do they cover the topic from multiple angles?
- web_articles: Have we extracted deeper content from key sources?
- videos: Any YouTube video data? (empty is fine if no YouTube API)
- comments: Any comments for sentiment analysis?
- errors: Any failures that need workaround?

**Key principle**: A good report needs BREADTH of information, not just depth.
Multiple angles (news, opinions, facts, controversies) matter more than
exhaustive data from a single source.

### When to route to analysis
Route to **analysis** when:
- Research data has been collected (web_results exist, optionally videos)
- There are videos for statistics, OR comments for sentiment, OR
  web_articles for theme/entity extraction (web analysis)
- The relevant analysis has NOT been done yet (statistics, sentiment,
  or web_analysis is still missing)

Only skip analysis if there are NO web_articles AND NO videos AND NO
comments — i.e., nothing to analyze at all.

### When to route to report
Route to **report** when:
- Analysis is complete (statistics/sentiment/web_analysis computed as applicable)
- You have sufficient web results and/or articles for a comprehensive report
- OR max iterations are approaching — produce output with what you have

### When to loop back to research (non-linear routing)
This is a KEY capability. Route back to **research** when:
- **Insufficient coverage**: Web results only cover one angle of the topic
- **Contradictions found**: Different sources say different things
- **New leads discovered**: Results reveal an unexpected aspect worth exploring
- **Missing YouTube data**: Entity clearly has YouTube presence but we haven't
  collected video data yet (and YouTube API is available)
- **Too few sources**: Less than 5 total web results

### When to use "done"
Only for empty/nonsensical queries or unrecoverable errors. Very rare.

## Important notes
- Always provide a clear reason for your routing decision
- When routing to research, always provide research_instructions
- Consider the iteration count — if approaching max, prioritize producing output
- The goal is a comprehensive, multi-angle research report
"""


class SupervisorDecision(BaseModel):
    """Structured output for supervisor routing decisions."""

    next_agent: Literal["research", "analysis", "report", "done"] = Field(
        description="Which agent should run next"
    )
    reason: str = Field(description="Why this routing decision was made")
    research_instructions: str | None = Field(
        default=None,
        description="Specific instructions for the research agent if next_agent is research",
    )


def supervisor_agent(state: DeepLensState) -> dict:
    """Supervisor evaluates current state and decides which agent runs next.

    This is the core differentiator of DeepLens — the LLM-based routing
    adapts strategy based on entity type, data availability, and findings.
    """
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 5)

    # Hard stop at max iterations — force report generation
    if iteration >= max_iter:
        logger.warning(
            "[Supervisor] Max iterations (%d) reached, forcing report", max_iter
        )
        return {"next_agent": "report", "iteration_count": iteration + 1}

    # Build a concise summary of current state for the LLM
    state_summary = _build_state_summary(state)

    llm = get_llm(temperature=0)
    structured_llm = llm.with_structured_output(SupervisorDecision)

    try:
        decision = cast(SupervisorDecision, structured_llm.invoke(
            [
                {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT},
                {"role": "user", "content": state_summary},
            ]
        ))
    except Exception as e:
        logger.error("[Supervisor] LLM call failed: %s", e)
        errors = state.get("errors", [])
        has_analysis = state.get("statistics") or state.get("sentiment")
        has_web_data = len(state.get("web_results") or []) > 0
        # Fallback: report if we have data, otherwise try research
        fallback = "report" if (has_analysis or has_web_data) else "research"
        return {
            "next_agent": fallback,
            "iteration_count": iteration + 1,
            "errors": errors + [f"Supervisor LLM error: {e}"],
        }

    logger.info(
        '[Supervisor] iter=%d, decision=%s, reason="%s"',
        iteration, decision.next_agent, decision.reason,
    )

    result: dict = {
        "next_agent": decision.next_agent,
        "iteration_count": iteration + 1,
    }

    # Pass research instructions via research_plan if routing to research
    if decision.next_agent == "research" and decision.research_instructions:
        existing_plan = state.get("research_plan", []) or []
        result["research_plan"] = existing_plan + [decision.research_instructions]

    return result


def route_decision(state: DeepLensState) -> str:
    """Conditional edge function for LangGraph.

    Returns the next node name based on the Supervisor's decision.
    Maps "done" to the END sentinel in the graph definition.
    """
    next_agent = state.get("next_agent", "done")
    if next_agent == "done":
        return "done"
    return next_agent


def _build_state_summary(state: DeepLensState) -> str:
    """Build a concise summary of current state for the Supervisor LLM.

    Includes: user query, iteration info, data counts, analysis status,
    YouTube API availability, errors, and key findings.
    """
    settings = get_settings()
    lines: list[str] = []

    # Query and iteration context
    lines.append(f"User query: {state.get('user_query', '(empty)')}")
    lines.append(
        f"Iteration: {state.get('iteration_count', 0)} "
        f"of {state.get('max_iterations', 5)}"
    )
    lines.append(f"YouTube API available: {settings.youtube_available}")

    # Research plan history
    plan = state.get("research_plan") or []
    if plan:
        lines.append(f"Research plan so far: {'; '.join(plan)}")

    lines.append("")
    lines.append("--- Web data (primary) ---")

    # Web results
    web = state.get("web_results") or []
    lines.append(f"Web results: {len(web)}")
    if web:
        for w in web[:5]:
            lines.append(f"  - \"{w.get('title', '?')}\" ({w.get('url', '')})")

    # Web articles (extracted full content)
    articles = state.get("web_articles") or []
    lines.append(f"Web articles extracted: {len(articles)}")
    if articles:
        for a in articles[:3]:
            lines.append(
                f"  - [{a.get('source_domain', '?')}] {a.get('title', '?')} "
                f"({len(a.get('content', ''))} chars)"
            )

    lines.append("")
    lines.append("--- YouTube data (optional enrichment) ---")

    # Channel data
    channel = state.get("channel_data")
    if channel:
        lines.append(
            f"Channel: {channel.get('title', 'unknown')} — "
            f"{channel.get('subscriber_count', 0):,} subscribers, "
            f"{channel.get('view_count', 0):,} views"
        )
    else:
        lines.append("Channel data: None")

    # Videos
    videos = state.get("videos") or []
    lines.append(f"Videos collected: {len(videos)}")
    if videos:
        top = sorted(videos, key=lambda v: v.get("view_count", 0), reverse=True)[:3]
        for v in top:
            lines.append(
                f"  - \"{v.get('title', '?')}\" "
                f"({v.get('view_count', 0):,} views, "
                f"{v.get('like_count', 0):,} likes)"
            )

    # Comments
    comments = state.get("comments") or []
    lines.append(f"Comments collected: {len(comments)}")

    lines.append("")
    lines.append("--- Analysis status ---")

    # Statistics
    stats = state.get("statistics")
    if stats:
        lines.append(
            f"Statistics: computed — avg views {stats.get('avg_views', 0):,.0f}, "
            f"avg engagement {stats.get('avg_engagement_rate', 0):.2%}"
        )
    else:
        lines.append("Statistics: not computed yet")

    # Sentiment
    sentiment = state.get("sentiment")
    if sentiment:
        lines.append(
            f"Sentiment: analyzed {sentiment.get('total_analyzed', 0)} comments — "
            f"positive {sentiment.get('positive', 0):.0%}, "
            f"neutral {sentiment.get('neutral', 0):.0%}, "
            f"negative {sentiment.get('negative', 0):.0%}"
        )
    else:
        lines.append("Sentiment: not analyzed yet")

    # Web analysis
    web_analysis = state.get("web_analysis")
    if web_analysis:
        themes = web_analysis.get("key_themes") or []
        lines.append(
            f"Web analysis: complete — {len(themes)} themes, "
            f"summary: {web_analysis.get('summary', '')[:100]}"
        )
    else:
        lines.append("Web analysis: not done yet")

    # Report
    report = state.get("report_markdown")
    if report:
        lines.append(f"Report: generated ({len(report)} chars)")
    else:
        lines.append("Report: not generated yet")

    # Errors
    errors = state.get("errors") or []
    if errors:
        lines.append("")
        lines.append(f"--- Errors ({len(errors)}) ---")
        for err in errors[-5:]:
            lines.append(f"  - {err}")

    return "\n".join(lines)
