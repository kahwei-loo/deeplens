"""LLM-callable tool wrappers for the Research Agent's ReACT loop.

Each tool wraps an existing function from web_search.py or youtube.py,
accumulates results in a ResearchCollector dataclass, and returns a
truncated JSON summary for the LLM (to save tokens).

The factory pattern (create_research_tools) creates closures over the
collector so tools can write side-effects without globals or thread-locals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langchain_core.tools import BaseTool, tool

from deeplens.config import Settings
from deeplens.models import (
    CommentData,
    Source,
    WebArticle,
    WebResult,
    YouTubeChannelData,
    YouTubeVideoData,
)
from deeplens.tools.rag import truncate_for_llm
from deeplens.tools.web_search import (
    extract_urls,
    multi_query_search,
    web_search,
)
from deeplens.tools.youtube import (
    youtube_channel,
    youtube_comments,
    youtube_search,
)

logger = logging.getLogger(__name__)


@dataclass
class ResearchCollector:
    """Mutable accumulator for research tool results.

    Tools write to this during the ReACT loop; after the loop ends,
    the agent reads it to build the state update dict.
    """

    web_results: list[WebResult] = field(default_factory=list)
    web_articles: list[WebArticle] = field(default_factory=list)
    videos: list[YouTubeVideoData] = field(default_factory=list)
    comments: list[CommentData] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    executed_queries: list[str] = field(default_factory=list)
    channel_data: YouTubeChannelData | None = None


def create_research_tools(
    collector: ResearchCollector,
    settings: Settings,
) -> list[BaseTool]:
    """Factory returning @tool-decorated wrappers bound to *collector*.

    YouTube tools are always present so the LLM can reason about
    their availability (they return "unavailable" when no API key).
    """

    @tool
    def search_web(query: str) -> str:
        """Search the web for a single query.

        Returns top results with titles, URLs, and snippets.
        """
        try:
            results = web_search(query, max_results=5)
            collector.web_results.extend(results)
            collector.executed_queries.append(query)
            collector.sources.extend(
                Source(
                    url=r["url"], title=r["title"],
                    source_type="web",
                )
                for r in results
            )
            preview = [
                {
                    "title": r["title"],
                    "url": r["url"],
                    "snippet": r["snippet"][:100],
                }
                for r in results
            ]
            return (
                f"Found {len(results)} results:\n"
                f"{truncate_for_llm(preview)}"
            )
        except Exception as e:
            msg = f"search_web error: {e}"
            collector.errors.append(msg)
            logger.error("[ResearchTools] %s", msg)
            return msg

    @tool
    def search_web_multi(queries: list[str]) -> str:
        """Execute multiple web searches at once.

        Pass a list of search queries. Returns deduplicated results
        across all queries.
        """
        try:
            query_list = [q.strip() for q in queries if q.strip()]
            if not query_list:
                return "No valid queries provided."
            results = multi_query_search(
                query_list, max_results_per_query=5,
            )
            collector.web_results.extend(results)
            collector.executed_queries.extend(query_list)
            collector.sources.extend(
                Source(
                    url=r["url"], title=r["title"],
                    source_type="web",
                )
                for r in results
            )
            preview = [
                {"title": r["title"], "url": r["url"]}
                for r in results
            ]
            return (
                f"Found {len(results)} unique results from "
                f"{len(query_list)} queries:\n{truncate_for_llm(preview)}"
            )
        except Exception as e:
            msg = f"search_web_multi error: {e}"
            collector.errors.append(msg)
            logger.error("[ResearchTools] %s", msg)
            return msg

    @tool
    def extract_article_content(urls: str) -> str:
        """Extract full article content from URLs.

        Pass comma-separated URLs (max 3). Returns article text
        for deeper analysis.
        """
        try:
            url_list = [
                u.strip() for u in urls.split(",") if u.strip()
            ][:3]
            if not url_list:
                return "No valid URLs provided."
            articles = extract_urls(url_list)
            collector.web_articles.extend(articles)
            preview = [
                {
                    "title": a["title"],
                    "url": a["url"],
                    "content_preview": a["content"][:200],
                }
                for a in articles
            ]
            return (
                f"Extracted {len(articles)} articles:\n"
                f"{truncate_for_llm(preview)}"
            )
        except Exception as e:
            msg = f"extract_article_content error: {e}"
            collector.errors.append(msg)
            logger.error("[ResearchTools] %s", msg)
            return msg

    @tool
    def search_youtube(query: str) -> str:
        """Search YouTube for videos matching the query.

        Returns video titles, view counts, and IDs.
        Only works if YouTube API is configured.
        """
        if not settings.youtube_available:
            return (
                "YouTube API is not available (no API key "
                "configured). Use web search instead."
            )
        try:
            videos = youtube_search(query=query, max_results=10)
            collector.videos.extend(videos)
            if videos:
                collector.sources.append(
                    Source(
                        url=(
                            "https://www.youtube.com/"
                            f"results?search_query={query}"
                        ),
                        title=f"YouTube search: {query}",
                        source_type="youtube",
                    )
                )
            preview = [
                {
                    "title": v["title"],
                    "video_id": v["video_id"],
                    "views": v["view_count"],
                }
                for v in videos
            ]
            return (
                f"Found {len(videos)} videos:\n"
                f"{truncate_for_llm(preview)}"
            )
        except Exception as e:
            msg = f"search_youtube error: {e}"
            collector.errors.append(msg)
            logger.error("[ResearchTools] %s", msg)
            return msg

    @tool
    def get_youtube_comments(video_id: str) -> str:
        """Get top comments from a YouTube video by video_id.

        Only works if YouTube API is configured.
        """
        if not settings.youtube_available:
            return (
                "YouTube API is not available "
                "(no API key configured)."
            )
        try:
            comments = youtube_comments(
                video_id=video_id, max_results=100,
            )
            collector.comments.extend(comments)
            if comments:
                collector.sources.append(
                    Source(
                        url=(
                            "https://www.youtube.com/"
                            f"watch?v={video_id}"
                        ),
                        title=f"Comments: {video_id}",
                        source_type="youtube",
                    )
                )
            preview = [
                {"text": c["text"][:80], "likes": c["like_count"]}
                for c in comments[:10]
            ]
            return (
                f"Retrieved {len(comments)} comments:\n"
                f"{truncate_for_llm(preview)}"
            )
        except Exception as e:
            msg = f"get_youtube_comments error: {e}"
            collector.errors.append(msg)
            logger.error("[ResearchTools] %s", msg)
            return msg

    @tool
    def get_youtube_channel(channel_name: str) -> str:
        """Look up a YouTube channel by name.

        Returns subscriber count, view count, and video count.
        Only works if YouTube API is configured.
        """
        if not settings.youtube_available:
            return (
                "YouTube API is not available "
                "(no API key configured)."
            )
        try:
            channel = youtube_channel(channel_name=channel_name)
            if channel:
                collector.channel_data = channel
                cid = channel.get("channel_id", "")
                collector.sources.append(
                    Source(
                        url=(
                            "https://www.youtube.com/"
                            f"channel/{cid}"
                        ),
                        title=(
                            f"Channel: "
                            f"{channel.get('title', channel_name)}"
                        ),
                        source_type="youtube",
                    )
                )
                return f"Channel found: {truncate_for_llm(channel)}"
            return f"No channel found for '{channel_name}'."
        except Exception as e:
            msg = f"get_youtube_channel error: {e}"
            collector.errors.append(msg)
            logger.error("[ResearchTools] %s", msg)
            return msg

    return [
        search_web,
        search_web_multi,
        extract_article_content,
        search_youtube,
        get_youtube_comments,
        get_youtube_channel,
    ]
