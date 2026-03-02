"""Tests for deeplens.tools.statistics — pure pandas computation, no mocks."""

from deeplens.models import YouTubeVideoData
from deeplens.tools.statistics import compute_statistics


def _make_video(
    video_id: str = "v1",
    title: str = "Video",
    view_count: int = 1000,
    like_count: int = 100,
    comment_count: int = 10,
    published_at: str = "2025-01-01T00:00:00Z",
) -> YouTubeVideoData:
    return YouTubeVideoData(
        video_id=video_id,
        title=title,
        view_count=view_count,
        like_count=like_count,
        comment_count=comment_count,
        published_at=published_at,
    )


class TestComputeStatisticsEmpty:
    def test_returns_none(self) -> None:
        assert compute_statistics([]) is None


class TestComputeStatisticsBasic:
    def test_averages(self) -> None:
        videos = [
            _make_video(video_id="a", view_count=1000, like_count=100, comment_count=10),
            _make_video(video_id="b", view_count=2000, like_count=200, comment_count=20),
            _make_video(video_id="c", view_count=3000, like_count=300, comment_count=30),
        ]
        result = compute_statistics(videos)
        assert result is not None
        assert result["avg_views"] == 2000.0
        assert result["avg_likes"] == 200.0

        # engagement_rate per video = (likes + comments) / views
        # a: 110/1000=0.11, b: 220/2000=0.11, c: 330/3000=0.11 => avg=0.11
        assert abs(result["avg_engagement_rate"] - 0.11) < 1e-9


class TestComputeStatisticsTopVideos:
    def test_sorted_by_views_descending(self) -> None:
        videos = [
            _make_video(video_id="low", title="Low Views", view_count=100),
            _make_video(video_id="high", title="High Views", view_count=5000),
            _make_video(video_id="mid", title="Mid Views", view_count=1000),
        ]
        result = compute_statistics(videos)
        assert result is not None
        top = result["top_videos"]
        assert len(top) == 3
        assert top[0]["video_id"] == "high"
        assert top[1]["video_id"] == "mid"
        assert top[2]["video_id"] == "low"

    def test_max_five_returned(self) -> None:
        videos = [
            _make_video(video_id=f"v{i}", view_count=i * 100) for i in range(8)
        ]
        result = compute_statistics(videos)
        assert result is not None
        assert len(result["top_videos"]) == 5


class TestComputeStatisticsUploadFrequency:
    def test_frequency_calculation(self) -> None:
        videos = [
            _make_video(video_id="a", published_at="2025-01-01T00:00:00Z"),
            _make_video(video_id="b", published_at="2025-01-11T00:00:00Z"),
            _make_video(video_id="c", published_at="2025-01-21T00:00:00Z"),
        ]
        result = compute_statistics(videos)
        assert result is not None
        # Deltas: 10 days, 10 days => avg 10.0
        assert result["upload_frequency_days"] is not None
        assert abs(result["upload_frequency_days"] - 10.0) < 1e-9

    def test_single_video_no_frequency(self) -> None:
        videos = [_make_video(video_id="only")]
        result = compute_statistics(videos)
        assert result is not None
        assert result["upload_frequency_days"] is None


class TestComputeStatisticsSingleVideo:
    def test_single_video_averages(self) -> None:
        videos = [_make_video(view_count=500, like_count=50, comment_count=5)]
        result = compute_statistics(videos)
        assert result is not None
        assert result["avg_views"] == 500.0
        assert result["avg_likes"] == 50.0
        # engagement: (50+5)/500 = 0.11
        assert abs(result["avg_engagement_rate"] - 0.11) < 1e-9
        assert len(result["top_videos"]) == 1
