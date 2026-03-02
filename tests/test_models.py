"""Tests for deeplens.models — TypedDict data models."""

import pytest

from deeplens.models import (
    CommentData,
    SentimentResult,
    Source,
    WebAnalysis,
    WebArticle,
    WebResult,
    YouTubeVideoData,
)


def test_web_result_creation() -> None:
    result: WebResult = {
        "title": "Example Article",
        "url": "https://example.com/article",
        "snippet": "A short summary of the article.",
        "score": 0.95,
    }
    assert result["title"] == "Example Article"
    assert result["score"] == 0.95


def test_web_article_creation() -> None:
    article: WebArticle = {
        "url": "https://example.com/full",
        "title": "Full Article",
        "content": "The complete extracted text of the article.",
        "source_domain": "example.com",
    }
    assert article["content"].startswith("The complete")
    assert article["source_domain"] == "example.com"


def test_youtube_video_data_creation() -> None:
    video: YouTubeVideoData = {
        "video_id": "abc123",
        "title": "Demo Video",
        "view_count": 100_000,
        "like_count": 5_000,
        "comment_count": 200,
        "published_at": "2025-01-15T00:00:00Z",
    }
    assert video["video_id"] == "abc123"
    assert video["view_count"] == 100_000


def test_comment_data_creation() -> None:
    comment: CommentData = {
        "text": "Great video!",
        "like_count": 42,
        "author": "user123",
        "video_id": "abc123",
    }
    assert comment["author"] == "user123"
    assert comment["like_count"] == 42


def test_sentiment_result_creation() -> None:
    sentiment: SentimentResult = {
        "positive": 0.6,
        "neutral": 0.3,
        "negative": 0.1,
        "total_analyzed": 50,
        "sample_positive": ["Love it!"],
        "sample_negative": ["Not great."],
    }
    total = sentiment["positive"] + sentiment["neutral"] + sentiment["negative"]
    assert total == pytest.approx(1.0)
    assert sentiment["total_analyzed"] == 50


def test_source_creation() -> None:
    source: Source = {
        "url": "https://example.com",
        "title": "Example Source",
        "source_type": "web",
    }
    assert source["source_type"] in ("web", "youtube")


def test_web_analysis_creation() -> None:
    analysis: WebAnalysis = {
        "key_themes": ["AI", "Machine Learning"],
        "entity_mentions": ["OpenAI", "Google"],
        "summary": "An overview of AI developments.",
    }
    assert len(analysis["key_themes"]) == 2
    assert "OpenAI" in analysis["entity_mentions"]
    assert analysis["summary"].startswith("An overview")
