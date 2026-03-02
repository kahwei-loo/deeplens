"""Tests for deeplens.tools.youtube — YouTube Data API v3 tools."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from deeplens.config import get_settings
from deeplens.tools.youtube import (
    _get_youtube_client,
    youtube_channel,
    youtube_comments,
    youtube_search,
)

# ---------------------------------------------------------------------------
# No API key tests — youtube_available == False
# ---------------------------------------------------------------------------

def test_youtube_search_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """youtube_search returns empty list when YOUTUBE_API_KEY is empty."""
    monkeypatch.setenv("YOUTUBE_API_KEY", "")
    get_settings.cache_clear()
    _get_youtube_client.cache_clear()

    result = youtube_search("test query")
    assert result == []


def test_youtube_channel_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """youtube_channel returns None when YOUTUBE_API_KEY is empty."""
    monkeypatch.setenv("YOUTUBE_API_KEY", "")
    get_settings.cache_clear()
    _get_youtube_client.cache_clear()

    result = youtube_channel("SomeChannel")
    assert result is None


def test_youtube_comments_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """youtube_comments returns empty list when YOUTUBE_API_KEY is empty."""
    monkeypatch.setenv("YOUTUBE_API_KEY", "")
    get_settings.cache_clear()
    _get_youtube_client.cache_clear()

    result = youtube_comments("vid123")
    assert result == []


# ---------------------------------------------------------------------------
# With API key tests — mock _get_youtube_client
# ---------------------------------------------------------------------------

def _mock_youtube_client() -> MagicMock:
    """Build a fully mocked YouTube API resource with chainable methods."""
    client = MagicMock()

    # search().list().execute() chain
    client.search.return_value.list.return_value.execute.return_value = {
        "items": []
    }
    # videos().list().execute() chain
    client.videos.return_value.list.return_value.execute.return_value = {
        "items": []
    }
    # channels().list().execute() chain
    client.channels.return_value.list.return_value.execute.return_value = {
        "items": []
    }
    # commentThreads().list().execute() chain
    client.commentThreads.return_value.list.return_value.execute.return_value = {
        "items": []
    }

    return client


@patch("deeplens.tools.youtube._get_youtube_client")
def test_youtube_search_with_key(mock_get_client: MagicMock) -> None:
    """youtube_search returns a list of YouTubeVideoData when API key is set."""
    get_settings.cache_clear()

    mock_client = _mock_youtube_client()
    mock_get_client.return_value = mock_client

    # search returns one video ID
    mock_client.search.return_value.list.return_value.execute.return_value = {
        "items": [
            {"id": {"videoId": "abc123"}},
        ],
    }
    # videos.list returns stats for that video
    mock_client.videos.return_value.list.return_value.execute.return_value = {
        "items": [
            {
                "id": "abc123",
                "snippet": {
                    "title": "Test Video",
                    "publishedAt": "2024-01-15T00:00:00Z",
                },
                "statistics": {
                    "viewCount": "1000",
                    "likeCount": "50",
                    "commentCount": "10",
                },
            },
        ],
    }

    results = youtube_search("test query")

    assert len(results) == 1
    video = results[0]
    assert video["video_id"] == "abc123"
    assert video["title"] == "Test Video"
    assert video["view_count"] == 1000
    assert video["like_count"] == 50
    assert video["comment_count"] == 10
    assert video["published_at"] == "2024-01-15T00:00:00Z"


@patch("deeplens.tools.youtube._get_youtube_client")
def test_youtube_channel_with_key(mock_get_client: MagicMock) -> None:
    """youtube_channel returns YouTubeChannelData when API key is set."""
    get_settings.cache_clear()

    mock_client = _mock_youtube_client()
    mock_get_client.return_value = mock_client

    # search returns a channel
    mock_client.search.return_value.list.return_value.execute.return_value = {
        "items": [
            {"snippet": {"channelId": "UC12345"}},
        ],
    }
    # channels.list returns channel details
    mock_client.channels.return_value.list.return_value.execute.return_value = {
        "items": [
            {
                "snippet": {"title": "Test Channel"},
                "statistics": {
                    "subscriberCount": "50000",
                    "viewCount": "2000000",
                    "videoCount": "120",
                },
            },
        ],
    }

    result = youtube_channel("Test Channel")

    assert result is not None
    assert result["channel_id"] == "UC12345"
    assert result["title"] == "Test Channel"
    assert result["subscriber_count"] == 50000
    assert result["view_count"] == 2000000
    assert result["video_count"] == 120


@patch("deeplens.tools.youtube._get_youtube_client")
def test_youtube_comments_with_key(mock_get_client: MagicMock) -> None:
    """youtube_comments returns a list of CommentData when API key is set."""
    get_settings.cache_clear()

    mock_client = _mock_youtube_client()
    mock_get_client.return_value = mock_client

    mock_client.commentThreads.return_value.list.return_value.execute.return_value = {
        "items": [
            {
                "snippet": {
                    "topLevelComment": {
                        "snippet": {
                            "textOriginal": "Great video!",
                            "likeCount": 5,
                            "authorDisplayName": "User1",
                        },
                    },
                },
            },
            {
                "snippet": {
                    "topLevelComment": {
                        "snippet": {
                            "textOriginal": "Very informative",
                            "likeCount": 2,
                            "authorDisplayName": "User2",
                        },
                    },
                },
            },
        ],
    }

    results = youtube_comments("vid456")

    assert len(results) == 2
    assert results[0]["text"] == "Great video!"
    assert results[0]["like_count"] == 5
    assert results[0]["author"] == "User1"
    assert results[0]["video_id"] == "vid456"
    assert results[1]["text"] == "Very informative"
    assert results[1]["author"] == "User2"
