"""Matplotlib chart generation for DeepLens reports."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for server/CLI usage
import matplotlib.pyplot as plt

from deeplens.models import SentimentResult, VideoStatistics

logger = logging.getLogger(__name__)


def generate_charts(
    statistics: VideoStatistics | None,
    sentiment: SentimentResult | None,
    output_dir: str = "output",
) -> list[str]:
    """Generate PNG charts from analysis results and return their file paths.

    Creates up to two charts:
    1. Sentiment pie chart (if sentiment data exists with analyzed comments).
    2. Top videos bar chart (if statistics with top_videos exist).

    Returns an empty list when there is no data to visualize.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    chart_paths: list[str] = []

    # --- Sentiment pie chart ---
    if sentiment and sentiment["total_analyzed"] > 0:
        path = _sentiment_pie_chart(sentiment, output_dir)
        if path:
            chart_paths.append(path)

    # --- Top videos bar chart ---
    if statistics and statistics["top_videos"]:
        path = _top_videos_bar_chart(statistics, output_dir)
        if path:
            chart_paths.append(path)

    logger.info("generate_charts: created %d charts in %s", len(chart_paths), output_dir)
    return chart_paths


def _sentiment_pie_chart(sentiment: SentimentResult, output_dir: str) -> str | None:
    """Create a sentiment distribution pie chart."""
    try:
        labels = ["Positive", "Neutral", "Negative"]
        sizes = [sentiment["positive"], sentiment["neutral"], sentiment["negative"]]
        colors = ["#4CAF50", "#9E9E9E", "#F44336"]

        fig, ax = plt.subplots(figsize=(7, 5))
        wedges, texts, autotexts = ax.pie(  # type: ignore[misc]
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=140,
            textprops={"fontsize": 11},
        )
        for t in autotexts:
            t.set_fontweight("bold")

        ax.set_title("Comment Sentiment Distribution", fontsize=14, fontweight="bold", pad=15)
        fig.tight_layout()

        filepath = os.path.join(output_dir, "sentiment_pie.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("_sentiment_pie_chart: saved to %s", filepath)
        return filepath

    except Exception as exc:
        logger.error("_sentiment_pie_chart error: %s", exc)
        return None


def _top_videos_bar_chart(statistics: VideoStatistics, output_dir: str) -> str | None:
    """Create a horizontal bar chart of top videos by view count."""
    try:
        top = statistics["top_videos"]
        # Truncate long titles for readability
        titles = [v["title"][:45] + "..." if len(v["title"]) > 45 else v["title"] for v in top]
        views = [v["view_count"] for v in top]

        # Reverse so the highest-viewed video is at the top
        titles = titles[::-1]
        views = views[::-1]

        fig, ax = plt.subplots(figsize=(9, max(4, len(titles) * 0.8)))
        bars = ax.barh(titles, views, color="#1976D2")
        ax.set_xlabel("View Count", fontsize=11)
        ax.set_title("Top Videos by View Count", fontsize=14, fontweight="bold", pad=15)
        ax.ticklabel_format(style="plain", axis="x")

        # Add value labels on bars
        for bar in bars:
            width = bar.get_width()
            label = f"{width:,.0f}"
            ax.text(
                width,
                bar.get_y() + bar.get_height() / 2,
                f"  {label}",
                va="center",
                fontsize=9,
            )

        fig.tight_layout()

        filepath = os.path.join(output_dir, "top_videos_bar.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("_top_videos_bar_chart: saved to %s", filepath)
        return filepath

    except Exception as exc:
        logger.error("_top_videos_bar_chart error: %s", exc)
        return None
