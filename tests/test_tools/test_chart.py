"""Tests for deeplens.tools.chart — uses tmp_path to verify file I/O."""

import os

from deeplens.models import SentimentResult, VideoStatistics, YouTubeVideoData
from deeplens.tools.chart import generate_charts


def _make_sentiment(
    positive: float = 0.6,
    neutral: float = 0.3,
    negative: float = 0.1,
    total: int = 10,
) -> SentimentResult:
    return SentimentResult(
        positive=positive,
        neutral=neutral,
        negative=negative,
        total_analyzed=total,
        sample_positive=["Great!"],
        sample_negative=["Bad!"],
    )


def _make_statistics(num_videos: int = 3) -> VideoStatistics:
    top_videos = [
        YouTubeVideoData(
            video_id=f"v{i}",
            title=f"Video {i}",
            view_count=(num_videos - i) * 1000,
            like_count=(num_videos - i) * 100,
            comment_count=(num_videos - i) * 10,
            published_at=f"2025-01-{i + 1:02d}T00:00:00Z",
        )
        for i in range(num_videos)
    ]
    return VideoStatistics(
        avg_views=2000.0,
        avg_likes=200.0,
        avg_engagement_rate=0.11,
        top_videos=top_videos,
        upload_frequency_days=5.0,
    )


class TestGenerateChartsNoData:
    def test_both_none_returns_empty(self, tmp_path: str) -> None:
        result = generate_charts(None, None, output_dir=str(tmp_path))
        assert result == []

    def test_sentiment_zero_analyzed(self, tmp_path: str) -> None:
        sentiment = _make_sentiment(total=0)
        result = generate_charts(None, sentiment, output_dir=str(tmp_path))
        assert result == []

    def test_statistics_empty_top_videos(self, tmp_path: str) -> None:
        stats = VideoStatistics(
            avg_views=0.0,
            avg_likes=0.0,
            avg_engagement_rate=0.0,
            top_videos=[],
            upload_frequency_days=None,
        )
        result = generate_charts(stats, None, output_dir=str(tmp_path))
        assert result == []


class TestGenerateChartsWithStatistics:
    def test_creates_bar_chart(self, tmp_path: str) -> None:
        stats = _make_statistics(3)
        result = generate_charts(stats, None, output_dir=str(tmp_path))
        assert len(result) == 1
        assert "top_videos_bar.png" in result[0]
        assert os.path.isfile(result[0])


class TestGenerateChartsWithSentiment:
    def test_creates_pie_chart(self, tmp_path: str) -> None:
        sentiment = _make_sentiment()
        result = generate_charts(None, sentiment, output_dir=str(tmp_path))
        assert len(result) == 1
        assert "sentiment_pie.png" in result[0]
        assert os.path.isfile(result[0])


class TestGenerateChartsBothPresent:
    def test_creates_both_charts(self, tmp_path: str) -> None:
        stats = _make_statistics(3)
        sentiment = _make_sentiment()
        result = generate_charts(stats, sentiment, output_dir=str(tmp_path))
        assert len(result) == 2
        filenames = [os.path.basename(p) for p in result]
        assert "sentiment_pie.png" in filenames
        assert "top_videos_bar.png" in filenames


class TestGenerateChartsOutputDir:
    def test_charts_saved_to_correct_directory(self, tmp_path: str) -> None:
        out = str(tmp_path / "charts")
        stats = _make_statistics(2)
        sentiment = _make_sentiment()
        result = generate_charts(stats, sentiment, output_dir=out)
        for path in result:
            assert os.path.isfile(path)
            assert str(tmp_path) in path
