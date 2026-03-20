"""Research agent — LLM-driven ReACT loop for web-first data collection.

Upgraded from L2 (hardcoded pipeline) to L4 (autonomous tool calling):
the LLM decides which tools to call, in what order, and when to stop.

Tools available to the LLM:
- search_web / search_web_multi — Tavily web search
- extract_article_content — full article extraction from URLs
- search_youtube / get_youtube_comments / get_youtube_channel — YouTube API

The ResearchCollector pattern accumulates tool results as side-effects,
avoiding fragile message-history parsing.
"""

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from deeplens.config import get_llm, get_settings
from deeplens.models import Source, WebResult, YouTubeVideoData
from deeplens.state import DeepLensState
from deeplens.tools.research_tools import ResearchCollector, create_research_tools
from deeplens.tools.web_search import web_search

logger = logging.getLogger(__name__)

RESEARCH_REACT_PROMPT = """\
You are a research agent for DeepLens. Your job is to collect comprehensive
data about the entity or topic the user is researching.

## Available Tools
You have access to web search, article extraction, and YouTube tools.
Use them strategically to gather thorough, multi-angle coverage.

## Research Strategy
1. **Start with multi-angle web search**: Generate 2-5 diverse search queries
   approaching the entity from different angles (overview, news, opinion,
   controversy, technical, comparison). Use search_web_multi for efficiency.
2. **Extract top articles**: After searching, extract full content from the
   2-3 most promising URLs using extract_article_content.
3. **YouTube enrichment** (optional): Only if the entity has strong YouTube
   presence (artists, creators, brands) AND YouTube API is available,
   search for videos and get comments from the top video.
4. **Fill gaps**: If initial results miss important angles, do targeted
   follow-up searches.

## Rules
- Prioritize web search — it's your primary data source
- Generate DIVERSE queries, not variations of the same search
- Don't repeat queries that have already been executed
- Extract articles from the highest-scoring URLs
- For YouTube: only use if the entity genuinely has YouTube presence
- Stop when you have comprehensive coverage (don't over-search)
- If Supervisor gave specific instructions, prioritize those angles

## Current Context
{context}
"""


def research_agent(state: DeepLensState) -> dict:
    """LLM-driven ReACT research loop with autonomous tool calling.

    1. Pre-populate a ResearchCollector with existing state data
    2. Run a ReACT loop: LLM decides tool calls, tools write to collector
    3. Deduplicate and merge collector data with existing state
    4. Return the same dict shape as before (state contract preserved)
    """
    settings = get_settings()
    query = state.get("user_query", "")
    plan_instructions = state.get("research_plan") or []
    errors = list(state.get("errors") or [])

    # Existing data — we append, not replace
    existing_web = list(state.get("web_results") or [])
    existing_articles = list(state.get("web_articles") or [])
    existing_videos = list(state.get("videos") or [])
    existing_comments = list(state.get("comments") or [])
    existing_sources = list(state.get("sources") or [])
    existing_channel = state.get("channel_data")
    executed_queries = list(state.get("executed_queries") or [])

    # Build collector and tools
    collector = ResearchCollector(
        executed_queries=list(executed_queries),
    )
    tools = create_research_tools(collector, settings)
    tool_map = {t.name: t for t in tools}

    # Build context for the system prompt
    context_parts = [f"User query: {query}"]
    if plan_instructions:
        context_parts.append(f"Supervisor instructions: {'; '.join(plan_instructions)}")
    if existing_web:
        context_parts.append(f"Already collected {len(existing_web)} web results")
    if existing_articles:
        context_parts.append(f"Already extracted {len(existing_articles)} full articles")
    if existing_videos:
        context_parts.append(f"Already collected {len(existing_videos)} YouTube videos")
    if existing_comments:
        context_parts.append(f"Already collected {len(existing_comments)} comments")
    if existing_channel:
        context_parts.append(
            f"Already have channel data for: {existing_channel.get('title', 'unknown')}"
        )
    if executed_queries:
        # Truncate each query to prevent prompt injection from prior LLM output
        safe_queries = [q[:200] for q in executed_queries]
        context_parts.append(f"Already executed queries: {safe_queries}")
    context_parts.append(f"YouTube API available: {settings.youtube_available}")

    system_prompt = RESEARCH_REACT_PROMPT.format(context="\n".join(context_parts))

    # ── ReACT Loop ─────────────────────────────────────────────────
    try:
        llm_with_tools = get_llm(temperature=0).bind_tools(tools)
        messages: list = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Research this entity/topic: {query}"),
        ]

        for step in range(settings.max_research_tool_calls):
            response: AIMessage = llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                logger.info("[Research] ReACT loop ended at step %d (no more tool calls)", step)
                break

            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                logger.info("[Research] Step %d: calling %s(%s)", step, tool_name, tool_args)

                if tool_name in tool_map:
                    try:
                        result = tool_map[tool_name].invoke(tool_args)
                    except Exception as e:
                        result = f"Tool execution error: {e}"
                        collector.errors.append(f"{tool_name} error: {e}")
                else:
                    result = f"Unknown tool: {tool_name}"

                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        else:
            logger.warning(
                "[Research] ReACT loop hit max steps (%d)",
                settings.max_research_tool_calls,
            )

    except Exception as e:
        logger.error("[Research] ReACT loop failed: %s", e)
        errors.append(f"Research ReACT error: {e}")
        # Fallback: basic web search with the original query
        fallback_results = web_search(query, max_results=10)
        return {
            "web_results": existing_web + fallback_results,
            "web_articles": existing_articles,
            "sources": existing_sources + [
                Source(url=r["url"], title=r["title"], source_type="web")
                for r in fallback_results
            ],
            "errors": errors,
        }

    # ── Merge collector with existing data ─────────────────────────
    # Deduplicate web results by URL, keeping highest score
    _seen_urls: dict[str, WebResult] = {}
    for r in existing_web + collector.web_results:
        url = r["url"]
        if url not in _seen_urls or r["score"] > _seen_urls[url]["score"]:
            _seen_urls[url] = r
    all_web = sorted(_seen_urls.values(), key=lambda x: x["score"], reverse=True)

    # Deduplicate articles by URL, keeping first occurrence
    _seen_article_urls: set[str] = set()
    all_articles: list = []
    for a in existing_articles + collector.web_articles:
        if a["url"] not in _seen_article_urls:
            _seen_article_urls.add(a["url"])
            all_articles.append(a)

    all_videos = _deduplicate_videos(existing_videos + collector.videos)
    all_comments = existing_comments + collector.comments

    # Deduplicate sources by URL, keeping first occurrence
    _seen_source_urls: dict[str, Source] = {}
    for s in existing_sources + collector.sources:
        _seen_source_urls.setdefault(s["url"], s)
    all_sources = list(_seen_source_urls.values())
    all_queries = list(dict.fromkeys(executed_queries + collector.executed_queries))

    # Channel data: prefer new if found
    channel_data = collector.channel_data or existing_channel

    logger.info(
        "[Research] Done. web=%d, articles=%d, videos=%d, comments=%d",
        len(all_web), len(all_articles), len(all_videos), len(all_comments),
    )

    result: dict = {
        "web_results": all_web,
        "web_articles": all_articles,
        "videos": all_videos,
        "comments": all_comments,
        "sources": all_sources,
        "errors": errors + collector.errors,
        "executed_queries": all_queries,
    }

    if channel_data is not None:
        result["channel_data"] = channel_data

    return result


def _deduplicate_videos(videos: list[YouTubeVideoData]) -> list[YouTubeVideoData]:
    """Remove duplicate videos by video_id, keeping the first occurrence."""
    seen: set[str] = set()
    deduped: list[YouTubeVideoData] = []
    for v in videos:
        vid = v["video_id"]
        if vid not in seen:
            seen.add(vid)
            deduped.append(v)
    return deduped
