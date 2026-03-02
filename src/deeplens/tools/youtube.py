"""YouTube Data API v3 tools for video search, channel info, and comment retrieval."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from deeplens.config import get_settings
from deeplens.models import CommentData, YouTubeChannelData, YouTubeVideoData

if TYPE_CHECKING:
    from googleapiclient._apis.youtube.v3 import YouTubeResource

logger = logging.getLogger(__name__)


@lru_cache
def _get_youtube_client() -> YouTubeResource:
    """Build and return a cached YouTube Data API v3 client."""
    settings = get_settings()
    return build("youtube", "v3", developerKey=settings.youtube_api_key)


# ---------------------------------------------------------------------------
# Retry-wrapped internal API helpers
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    reraise=True,
)
def _yt_search_list(client: YouTubeResource, **kwargs) -> dict:
    """Execute a YouTube search.list call with retry on transient errors."""
    return client.search().list(**kwargs).execute()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    reraise=True,
)
def _yt_videos_list(client: YouTubeResource, **kwargs) -> dict:
    """Execute a YouTube videos.list call with retry on transient errors."""
    return client.videos().list(**kwargs).execute()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    reraise=True,
)
def _yt_channels_list(client: YouTubeResource, **kwargs) -> dict:
    """Execute a YouTube channels.list call with retry on transient errors."""
    return client.channels().list(**kwargs).execute()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    reraise=True,
)
def _yt_comment_threads_list(client: YouTubeResource, **kwargs) -> dict:
    """Execute a YouTube commentThreads.list call with retry on transient errors."""
    return client.commentThreads().list(**kwargs).execute()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def youtube_search(query: str, max_results: int = 15) -> list[YouTubeVideoData]:
    """Search YouTube for videos matching *query* and return enriched metadata.

    Uses two API calls:
    1. search.list to find video IDs matching the query.
    2. videos.list to fetch full statistics for those videos.

    Returns an empty list when the YouTube API key is not configured or on errors.
    """
    settings = get_settings()
    if not settings.youtube_available:
        logger.info("youtube_search: skipped (no YouTube API key configured)")
        return []

    logger.info("youtube_search: query=%r max_results=%d", query, max_results)
    try:
        client = _get_youtube_client()

        # Step 1: search for video IDs
        search_resp = _yt_search_list(
            client, part="snippet", q=query, type="video", maxResults=max_results,
        )

        video_ids = [item["id"]["videoId"] for item in search_resp.get("items", [])]
        if not video_ids:
            logger.warning("youtube_search: no results for query=%r", query)
            return []

        # Step 2: fetch full statistics for each video
        videos_resp = _yt_videos_list(
            client, part="statistics,snippet", id=",".join(video_ids),
        )

        results: list[YouTubeVideoData] = []
        for item in videos_resp.get("items", []):
            stats = item.get("statistics", {})
            results.append(
                YouTubeVideoData(
                    video_id=item["id"],
                    title=item["snippet"]["title"],
                    view_count=int(stats.get("viewCount", 0)),
                    like_count=int(stats.get("likeCount", 0)),
                    comment_count=int(stats.get("commentCount", 0)),
                    published_at=item["snippet"]["publishedAt"],
                )
            )

        logger.info("youtube_search: returned %d videos", len(results))
        return results

    except HttpError as exc:
        logger.error("youtube_search API error: %s", exc)
        return []
    except Exception as exc:
        logger.error("youtube_search unexpected error: %s", exc)
        return []


def youtube_channel(channel_name: str) -> YouTubeChannelData | None:
    """Look up a YouTube channel by name and return its metadata.

    Returns ``None`` when the API key is not configured or the channel cannot be found.
    """
    settings = get_settings()
    if not settings.youtube_available:
        logger.info("youtube_channel: skipped (no YouTube API key configured)")
        return None

    logger.info("youtube_channel: channel_name=%r", channel_name)
    try:
        client = _get_youtube_client()

        # Step 1: search for the channel
        search_resp = _yt_search_list(
            client, part="snippet", q=channel_name, type="channel", maxResults=1,
        )

        items = search_resp.get("items", [])
        if not items:
            logger.warning("youtube_channel: no channel found for %r", channel_name)
            return None

        channel_id = items[0]["snippet"]["channelId"]

        # Step 2: fetch full channel statistics
        channel_resp = _yt_channels_list(
            client, part="statistics,snippet", id=channel_id,
        )

        channel_items = channel_resp.get("items", [])
        if not channel_items:
            logger.warning("youtube_channel: channel details unavailable for id=%s", channel_id)
            return None

        ch = channel_items[0]
        stats = ch.get("statistics", {})

        result = YouTubeChannelData(
            channel_id=channel_id,
            title=ch["snippet"]["title"],
            subscriber_count=int(stats.get("subscriberCount", 0)),
            view_count=int(stats.get("viewCount", 0)),
            video_count=int(stats.get("videoCount", 0)),
        )

        logger.info("youtube_channel: found %r (id=%s)", result["title"], channel_id)
        return result

    except HttpError as exc:
        logger.error("youtube_channel API error: %s", exc)
        return None
    except Exception as exc:
        logger.error("youtube_channel unexpected error: %s", exc)
        return None


def youtube_comments(video_id: str, max_results: int = 100) -> list[CommentData]:
    """Fetch top-level comment threads for a video, ordered by relevance.

    Returns an empty list when the API key is not configured or comments are disabled.
    """
    settings = get_settings()
    if not settings.youtube_available:
        logger.info("youtube_comments: skipped (no YouTube API key configured)")
        return []

    logger.info("youtube_comments: video_id=%s max_results=%d", video_id, max_results)
    try:
        client = _get_youtube_client()

        # The API caps maxResults at 100 per page
        capped = min(max_results, 100)

        resp = _yt_comment_threads_list(
            client,
            part="snippet",
            videoId=video_id,
            maxResults=capped,
            order="relevance",
        )

        results: list[CommentData] = []
        for item in resp.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            results.append(
                CommentData(
                    text=top["textOriginal"],
                    like_count=int(top.get("likeCount", 0)),
                    author=top.get("authorDisplayName", ""),
                    video_id=video_id,
                )
            )

        logger.info("youtube_comments: retrieved %d comments for video=%s", len(results), video_id)
        return results

    except HttpError as exc:
        # 403 with "commentsDisabled" is expected for some videos
        if exc.resp.status == 403:
            logger.warning("youtube_comments: comments disabled for video=%s", video_id)
            return []
        logger.error("youtube_comments API error: %s", exc)
        return []
    except Exception as exc:
        logger.error("youtube_comments unexpected error: %s", exc)
        return []
