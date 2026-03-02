"""Pandas-based statistics computation for collected YouTube video data."""

from __future__ import annotations

import logging

import pandas as pd

from deeplens.models import VideoStatistics, YouTubeVideoData

logger = logging.getLogger(__name__)


def compute_statistics(videos: list[YouTubeVideoData]) -> VideoStatistics | None:
    """Compute aggregate statistics from a list of YouTube video data.

    Returns ``None`` when the input list is empty.
    """
    if not videos:
        logger.info("compute_statistics: no videos provided")
        return None

    logger.info("compute_statistics: processing %d videos", len(videos))

    df = pd.DataFrame(videos)

    avg_views = float(df["view_count"].mean())
    avg_likes = float(df["like_count"].mean())

    # Engagement rate per video: (likes + comments) / views.
    # Guard against division by zero for videos with 0 views.
    df["engagement_rate"] = df.apply(
        lambda row: (row["like_count"] + row["comment_count"]) / row["view_count"]
        if row["view_count"] > 0
        else 0.0,
        axis=1,
    )
    avg_engagement_rate = float(df["engagement_rate"].mean())

    # Top 5 videos by view count
    top_df = df.nlargest(5, "view_count")
    top_videos: list[YouTubeVideoData] = [
        YouTubeVideoData(
            video_id=row["video_id"],
            title=row["title"],
            view_count=int(row["view_count"]),
            like_count=int(row["like_count"]),
            comment_count=int(row["comment_count"]),
            published_at=row["published_at"],
        )
        for _, row in top_df.iterrows()
    ]

    # Upload frequency: average days between consecutive uploads.
    upload_frequency_days: float | None = None
    if len(df) >= 2:
        dates = pd.to_datetime(df["published_at"]).sort_values()
        deltas = dates.diff().dropna()
        if not deltas.empty:
            upload_frequency_days = float(deltas.mean().total_seconds() / 86400)

    stats = VideoStatistics(
        avg_views=avg_views,
        avg_likes=avg_likes,
        avg_engagement_rate=avg_engagement_rate,
        top_videos=top_videos,
        upload_frequency_days=upload_frequency_days,
    )

    logger.info(
        "compute_statistics: avg_views=%.0f avg_likes=%.0f engagement=%.4f freq=%s days",
        avg_views,
        avg_likes,
        avg_engagement_rate,
        f"{upload_frequency_days:.1f}" if upload_frequency_days is not None else "N/A",
    )
    return stats
