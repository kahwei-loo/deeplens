"""DeepLensState — shared state schema that all agents read and write."""

from __future__ import annotations

from typing import TypedDict

from deeplens.models import (
    CommentData,
    SentimentResult,
    Source,
    VideoStatistics,
    WebAnalysis,
    WebArticle,
    WebResult,
    YouTubeChannelData,
    YouTubeVideoData,
)


class DeepLensState(TypedDict):
    """Central state contract for the LangGraph agent graph.

    Every agent node receives this state and returns a partial update.
    """

    # Input
    user_query: str
    research_plan: list[str]

    # Research data (web-first)
    web_results: list[WebResult]
    web_articles: list[WebArticle]
    sources: list[Source]

    # YouTube enrichment (optional — empty when no API key)
    channel_data: YouTubeChannelData | None
    videos: list[YouTubeVideoData]
    comments: list[CommentData]

    # Analysis
    statistics: VideoStatistics | None
    sentiment: SentimentResult | None
    web_analysis: WebAnalysis | None

    # Report
    report_markdown: str
    charts: list[str]

    # Control flow
    next_agent: str
    iteration_count: int
    max_iterations: int
    errors: list[str]
