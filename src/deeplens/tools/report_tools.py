"""LLM-callable tool wrappers for the Report Agent's ReACT loop.

Each tool provides read-only access to a slice of DeepLensState via closures.
The Report Agent LLM decides which data to query, when, and in what order —
replacing the old monolithic _build_report_context() string builder.
"""

from __future__ import annotations

import logging

from langchain_core.tools import BaseTool, tool

from deeplens.config import get_settings
from deeplens.models import Source
from deeplens.state import DeepLensState
from deeplens.tools.chart import generate_charts
from deeplens.tools.rag import (
    chunk_articles,
    embed_chunks,
    embedding_search,
    keyword_search,
    truncate_for_llm,
)

logger = logging.getLogger(__name__)


def create_report_tools(
    state: DeepLensState,
    output_dir: str,
    chart_paths_out: list[str] | None = None,
) -> list[BaseTool]:
    """Factory returning @tool-decorated wrappers with closures over state.

    Tools are read-only views into the state — the Report Agent queries
    them during its ReACT loop to gather data for the report.
    """
    # Pre-compute article chunks + embeddings for RAG search
    articles = state.get("web_articles") or []
    _chunks = chunk_articles(articles) if articles else []
    settings = get_settings()
    _embeddings = embed_chunks(
        _chunks, api_key=settings.openai_api_key,
    )

    @tool
    def search_articles(query: str) -> str:
        """Search extracted web articles using semantic similarity.

        Uses embedding-based search with keyword fallback.
        Find specific information about topics, events, or
        entities mentioned in the research data.
        """
        if not _chunks:
            return "No web articles available to search."

        # Try embedding search first, fall back to keyword
        results = []
        if _embeddings is not None:
            results = embedding_search(
                _chunks, _embeddings, query,
                api_key=settings.openai_api_key, top_k=5,
            )
        if not results:
            results = keyword_search(_chunks, query, top_k=5)

        if not results:
            return f"No article chunks matched '{query}'."
        formatted = []
        for r in results:
            formatted.append({
                "source": r.get("title", "Unknown"),
                "url": r.get("url", ""),
                "text": r["text"][:300],
            })
        return (
            f"Found {len(results)} relevant chunks:\n"
            f"{truncate_for_llm(formatted)}"
        )

    @tool
    def get_web_results_summary() -> str:
        """Get a summary of all web search results.

        Returns titles, URLs, and snippets for an overview
        of all sources found during research.
        """
        web = state.get("web_results") or []
        if not web:
            return "No web search results available."
        capped = web[:20]
        summary = []
        for w in capped:
            summary.append({
                "title": w.get("title", "Untitled"),
                "url": w.get("url", ""),
                "snippet": w.get("snippet", "")[:150],
            })
        total = f"{len(capped)} of {len(web)} web results:\n"
        return total + truncate_for_llm(summary)

    @tool
    def get_video_data() -> str:
        """Get YouTube channel info and top video metrics.

        Returns channel stats and up to 10 top videos by view count.
        """
        channel = state.get("channel_data")
        videos = state.get("videos") or []

        if not channel and not videos:
            return "No YouTube video data available."

        data: dict = {}
        if channel:
            data["channel"] = {
                "title": channel.get("title", "Unknown"),
                "subscribers": channel.get("subscriber_count", 0),
                "total_views": channel.get("view_count", 0),
                "video_count": channel.get("video_count", 0),
            }
        if videos:
            sorted_vids = sorted(
                videos, key=lambda v: v.get("view_count", 0), reverse=True
            )
            data["top_videos"] = [
                {
                    "title": v.get("title", "?"),
                    "views": v.get("view_count", 0),
                    "likes": v.get("like_count", 0),
                    "comments": v.get("comment_count", 0),
                    "published": v.get("published_at", "unknown"),
                }
                for v in sorted_vids[:10]
            ]
        return truncate_for_llm(data)

    @tool
    def get_statistics() -> str:
        """Get computed video statistics.

        Returns average views, engagement rates, upload frequency,
        and top performers.
        """
        statistics = state.get("statistics")
        if not statistics:
            return "No video statistics available."
        return truncate_for_llm({
            "avg_views": statistics.get("avg_views", 0),
            "avg_likes": statistics.get("avg_likes", 0),
            "avg_engagement_rate": statistics.get(
                "avg_engagement_rate", 0
            ),
            "upload_frequency_days": statistics.get(
                "upload_frequency_days"
            ),
            "top_videos": [
                {
                    "title": v.get("title", "?"),
                    "views": v.get("view_count", 0),
                }
                for v in (statistics.get("top_videos") or [])[:5]
            ],
        })

    @tool
    def get_sentiment() -> str:
        """Get sentiment analysis results.

        Returns positive/neutral/negative percentages and sample
        comments.
        """
        sentiment = state.get("sentiment")
        if not sentiment:
            return "No sentiment analysis available."
        return truncate_for_llm({
            "positive": sentiment.get("positive", 0),
            "neutral": sentiment.get("neutral", 0),
            "negative": sentiment.get("negative", 0),
            "total_analyzed": sentiment.get("total_analyzed", 0),
            "sample_positive": (
                sentiment.get("sample_positive") or []
            )[:3],
            "sample_negative": (
                sentiment.get("sample_negative") or []
            )[:3],
        })

    @tool
    def get_web_analysis() -> str:
        """Get LLM-extracted themes, entities, and summary."""
        web_analysis = state.get("web_analysis")
        if not web_analysis:
            return "No web analysis available."
        return truncate_for_llm({
            "summary": web_analysis.get("summary", ""),
            "key_themes": web_analysis.get("key_themes") or [],
            "entity_mentions": (
                web_analysis.get("entity_mentions") or []
            ),
        })

    @tool
    def get_sources() -> str:
        """Get all research sources for citation.

        Returns URLs, titles, and types (web/youtube).
        """
        sources: list[Source] = state.get("sources") or []
        if not sources:
            return "No sources available."
        capped = sources[:30]
        formatted = [
            {
                "title": s.get("title", "Source"),
                "url": s.get("url", ""),
                "type": s.get("source_type", ""),
            }
            for s in capped
        ]
        total = f"{len(capped)} of {len(sources)} sources:\n"
        return total + truncate_for_llm(formatted)

    @tool
    def generate_report_charts() -> str:
        """Generate charts from analysis data.

        Creates sentiment pie and top videos bar charts. Returns
        file paths of generated PNGs. Call before writing report.
        """
        try:
            statistics = state.get("statistics")
            sentiment = state.get("sentiment")
            paths = generate_charts(
                statistics=statistics,
                sentiment=sentiment,
                output_dir=output_dir,
            )
            # Write paths to accumulator for the caller
            if paths and chart_paths_out is not None:
                chart_paths_out.extend(paths)
            if paths:
                return (
                    f"Generated {len(paths)} charts:\n"
                    + "\n".join(f"- ![Chart]({p})" for p in paths)
                )
            return (
                "No charts generated (no statistics or "
                "sentiment data available)."
            )
        except Exception as e:
            return f"Chart generation failed: {e}"

    return [
        search_articles,
        get_web_results_summary,
        get_video_data,
        get_statistics,
        get_sentiment,
        get_web_analysis,
        get_sources,
        generate_report_charts,
    ]
