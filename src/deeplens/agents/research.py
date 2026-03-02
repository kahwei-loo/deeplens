"""Research agent — web-first data collection with optional YouTube enrichment.

Follows a multi-angle search strategy inspired by production research agents:

- **Perplexity**: Multi-angle search queries for comprehensive coverage.
  Instead of one search, generate 2-5 queries approaching the entity from
  different angles (overview, news, opinion, controversy, etc.).
- **Manus**: Deep extraction from top URLs. Go beyond snippets — extract
  the full article content for richer analysis and synthesis.
- **Open Deep Research**: Parallel sub-queries with deduplication. Multiple
  researchers with isolated context, merged at the end.

YouTube API is treated as optional enrichment — the system produces
meaningful research reports using web data alone.
"""

import logging

from pydantic import BaseModel, Field

from deeplens.config import get_llm, get_settings
from deeplens.models import (
    CommentData,
    Source,
    WebArticle,
    YouTubeVideoData,
)
from deeplens.state import DeepLensState
from deeplens.tools.web_search import extract_urls, multi_query_search, web_search
from deeplens.tools.youtube import youtube_channel, youtube_comments, youtube_search

logger = logging.getLogger(__name__)

RESEARCH_PLAN_PROMPT = """\
You are a research planning assistant for DeepLens. Given a user query and
optional Supervisor instructions, produce a research plan with multi-angle
search queries.

## Research Strategy

Your PRIMARY tool is **web search**. Generate 2-5 search queries that approach
the entity from different angles to ensure comprehensive coverage.

### Search Angles (choose 2-5 most relevant):
- **overview**: General information, background, key facts
- **news**: Recent news, events, developments
- **opinion**: Public opinion, reviews, community discussion
- **technical**: Technical details, specifications, methodology
- **controversy**: Controversies, criticisms, challenges
- **comparison**: How it compares to alternatives or competitors
- **history**: Timeline, origin story, key milestones

### YouTube Enrichment (optional)
Set youtube_enrichment to true ONLY when ALL of these are true:
1. The entity has a significant YouTube presence (artists, creators, brands)
2. Video metrics and comments would add UNIQUE value beyond web data
3. YouTube API is available (indicated in the context)

For public figures, topics, or concepts → youtube_enrichment should be false.
Web search already captures YouTube-related info from search results.

### Rules
- Generate DIVERSE queries, not variations of the same search
- Each query should target a DIFFERENT research angle
- Keep queries concise and search-engine-friendly
- If Supervisor instructions focus on something specific, prioritize that angle
- If existing data already covers an angle, focus on uncovered angles
"""


class SearchQuery(BaseModel):
    """A single search query targeting a specific research angle."""

    query: str = Field(description="The search query to execute")
    angle: str = Field(
        description="Research angle: overview, news, opinion, technical, "
        "controversy, comparison, history"
    )


class ResearchPlan(BaseModel):
    """Structured research plan from the planning LLM."""

    entity_type: str = Field(
        description="Detected entity type: artist/group, public_figure, "
        "topic, brand, other"
    )
    search_queries: list[SearchQuery] = Field(
        description="2-5 search queries from different angles"
    )
    youtube_enrichment: bool = Field(
        default=False,
        description="Whether to use YouTube API for video data and comments",
    )
    youtube_search_query: str | None = Field(
        default=None,
        description="YouTube-specific search query if enrichment is enabled",
    )


def research_agent(state: DeepLensState) -> dict:
    """Web-first data collection with optional YouTube enrichment.

    Phases:
    1. LLM generates multi-angle search queries (Perplexity pattern)
    2. Execute web searches across all angles, deduplicate
    3. Extract full content from top URLs (Manus pattern)
    4. Optionally enrich with YouTube API data
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

    # Build context for planning LLM
    context_parts = [f"User query: {query}"]
    if plan_instructions:
        context_parts.append(f"Supervisor instructions: {plan_instructions[-1]}")
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
    context_parts.append(f"YouTube API available: {settings.youtube_available}")

    context = "\n".join(context_parts)

    # ── Phase 1: Get research plan from LLM ──────────────────────
    llm = get_llm(temperature=0)
    structured_llm = llm.with_structured_output(ResearchPlan)

    try:
        plan: ResearchPlan = structured_llm.invoke(
            [
                {"role": "system", "content": RESEARCH_PLAN_PROMPT},
                {"role": "user", "content": context},
            ]
        )
        logger.info(
            "[Research] entity_type=%s, queries=%d, youtube=%s",
            plan.entity_type,
            len(plan.search_queries),
            plan.youtube_enrichment,
        )
    except Exception as e:
        logger.error("[Research] Planning LLM failed: %s", e)
        errors.append(f"Research planning error: {e}")
        # Fallback: basic web search with the original query
        fallback_results = web_search(query, max_results=10)
        return {
            "web_results": existing_web + fallback_results,
            "web_articles": existing_articles,
            "sources": existing_sources
            + [
                Source(url=r["url"], title=r["title"], source_type="web")
                for r in fallback_results
            ],
            "errors": errors,
        }

    # ── Phase 2: Multi-angle web search (Perplexity pattern) ─────
    search_queries = [sq.query for sq in plan.search_queries]
    new_web_results = multi_query_search(search_queries, max_results_per_query=5)
    new_sources: list[Source] = [
        Source(url=r["url"], title=r["title"], source_type="web")
        for r in new_web_results
    ]
    logger.info(
        "[Research] Web search: %d results from %d queries",
        len(new_web_results),
        len(search_queries),
    )

    # ── Phase 3: Deep extraction from top URLs (Manus pattern) ───
    new_articles: list[WebArticle] = []
    extracted_urls = {a["url"] for a in existing_articles}
    urls_to_extract = [
        r["url"] for r in new_web_results if r["url"] not in extracted_urls
    ][:3]  # Top 3 by score (already sorted)

    if urls_to_extract:
        new_articles = extract_urls(urls_to_extract)
        logger.info(
            "[Research] Extracted %d articles from top URLs", len(new_articles)
        )

    # ── Phase 4: Optional YouTube enrichment ─────────────────────
    new_videos: list[YouTubeVideoData] = []
    new_comments: list[CommentData] = []
    channel_data = existing_channel

    if plan.youtube_enrichment and settings.youtube_available:
        logger.info("[Research] YouTube enrichment enabled")
        yt_query = plan.youtube_search_query or query

        try:
            # Search for videos
            videos = youtube_search(query=yt_query, max_results=10)
            new_videos.extend(videos)
            if videos:
                new_sources.append(
                    Source(
                        url=f"https://www.youtube.com/results?search_query={yt_query}",
                        title=f"YouTube search: {yt_query}",
                        source_type="youtube",
                    )
                )
                logger.info("[Research] YouTube search: %d videos", len(videos))

            # Get comments from top video
            all_vids = existing_videos + new_videos
            if all_vids:
                top_video = max(all_vids, key=lambda v: v.get("view_count", 0))
                video_id = top_video.get("video_id", "")
                if video_id:
                    comments = youtube_comments(
                        video_id=video_id, max_results=100
                    )
                    new_comments.extend(comments)
                    if comments:
                        new_sources.append(
                            Source(
                                url=f"https://www.youtube.com/watch?v={video_id}",
                                title=f"Comments: {top_video.get('title', video_id)}",
                                source_type="youtube",
                            )
                        )
                        logger.info(
                            "[Research] YouTube comments: %d", len(comments)
                        )

            # Channel lookup for artist/group/brand entities
            if (
                plan.entity_type in ("artist/group", "brand")
                and not existing_channel
            ):
                channel = youtube_channel(channel_name=query)
                if channel:
                    channel_data = channel
                    new_sources.append(
                        Source(
                            url=f"https://www.youtube.com/channel/{channel.get('channel_id', '')}",
                            title=f"Channel: {channel.get('title', query)}",
                            source_type="youtube",
                        )
                    )
                    logger.info(
                        "[Research] YouTube channel: %s", channel.get("title")
                    )
        except Exception as e:
            error_msg = f"YouTube enrichment failed: {e}"
            logger.error("[Research] %s", error_msg)
            errors.append(error_msg)

    elif plan.youtube_enrichment and not settings.youtube_available:
        logger.info(
            "[Research] YouTube enrichment requested but no API key, skipping"
        )

    # ── Merge with existing data ─────────────────────────────────
    all_web = existing_web + new_web_results
    all_articles = existing_articles + new_articles
    all_videos = _deduplicate_videos(existing_videos + new_videos)
    all_comments = existing_comments + new_comments
    all_sources = existing_sources + new_sources

    logger.info(
        "[Research] Done. web=%d, articles=%d, videos=%d, comments=%d",
        len(all_web),
        len(all_articles),
        len(all_videos),
        len(all_comments),
    )

    result: dict = {
        "web_results": all_web,
        "web_articles": all_articles,
        "videos": all_videos,
        "comments": all_comments,
        "sources": all_sources,
        "errors": errors,
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
