"""Typed data models for DeepLens entity research system."""

from __future__ import annotations

from typing import TypedDict


class YouTubeChannelData(TypedDict):
    """YouTube channel metadata."""

    channel_id: str
    title: str
    subscriber_count: int
    view_count: int
    video_count: int


class YouTubeVideoData(TypedDict):
    """YouTube video metadata and engagement metrics."""

    video_id: str
    title: str
    view_count: int
    like_count: int
    comment_count: int
    published_at: str


class CommentData(TypedDict):
    """A single YouTube comment."""

    text: str
    like_count: int
    author: str
    video_id: str


class WebResult(TypedDict):
    """A web search result from Tavily."""

    title: str
    url: str
    snippet: str
    score: float


class Source(TypedDict):
    """A tracked research source with its origin type."""

    url: str
    title: str
    source_type: str  # "youtube" | "web"


class SentimentResult(TypedDict):
    """Aggregated sentiment analysis of comments."""

    positive: float
    neutral: float
    negative: float
    total_analyzed: int
    sample_positive: list[str]
    sample_negative: list[str]


class VideoStatistics(TypedDict):
    """Computed statistics across collected videos."""

    avg_views: float
    avg_likes: float
    avg_engagement_rate: float
    top_videos: list[YouTubeVideoData]
    upload_frequency_days: float | None


class WebArticle(TypedDict):
    """Full content extracted from a web page via Tavily extract API."""

    url: str
    title: str
    content: str
    source_domain: str


class WebAnalysis(TypedDict):
    """LLM-extracted analysis of web articles."""

    key_themes: list[str]
    entity_mentions: list[str]
    summary: str
