"""Tests for deeplens.tools.sentiment — mock LLM calls."""

from unittest.mock import MagicMock, patch

import pytest

from deeplens.models import CommentData
from deeplens.tools.sentiment import (
    ClassificationEntry,
    CommentSentiment,
    sentiment_analyzer,
)


def _make_comment(text: str, video_id: str = "v1") -> CommentData:
    return CommentData(text=text, like_count=0, author="user", video_id=video_id)


def _mock_structured_llm(classifications: list[ClassificationEntry]) -> MagicMock:
    """Create a mock that behaves like llm.with_structured_output(...)."""
    result = CommentSentiment(classifications=classifications)
    structured = MagicMock()
    structured.invoke.return_value = result
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


class TestAnalyzeSentimentEmpty:
    def test_empty_comments_returns_default(self) -> None:
        result = sentiment_analyzer([])
        assert result["total_analyzed"] == 0
        assert result["positive"] == 0.0
        assert result["neutral"] == 0.0
        assert result["negative"] == 0.0
        assert result["sample_positive"] == []
        assert result["sample_negative"] == []


class TestAnalyzeSentimentBasic:
    @patch("deeplens.tools.sentiment.get_llm")
    @patch("deeplens.tools.sentiment.get_settings")
    def test_basic_classification(
        self, mock_settings: MagicMock, mock_get_llm: MagicMock
    ) -> None:
        mock_settings.return_value = MagicMock(comment_batch_size=20)

        classifications = [
            ClassificationEntry(text="Great video!", sentiment="positive"),
            ClassificationEntry(text="Meh", sentiment="neutral"),
            ClassificationEntry(text="Terrible", sentiment="negative"),
            ClassificationEntry(text="Love it", sentiment="positive"),
        ]
        llm_mock = _mock_structured_llm(classifications)
        mock_get_llm.return_value = llm_mock

        comments = [
            _make_comment("Great video!"),
            _make_comment("Meh"),
            _make_comment("Terrible"),
            _make_comment("Love it"),
        ]
        result = sentiment_analyzer(comments)

        assert result["total_analyzed"] == 4
        assert result["positive"] == pytest.approx(0.5)  # 2/4
        assert result["neutral"] == pytest.approx(0.25)  # 1/4
        assert result["negative"] == pytest.approx(0.25)  # 1/4
        assert "Great video!" in result["sample_positive"]
        assert "Terrible" in result["sample_negative"]


class TestAnalyzeSentimentBatching:
    @patch("deeplens.tools.sentiment.get_llm")
    @patch("deeplens.tools.sentiment.get_settings")
    def test_two_batches_for_30_comments(
        self, mock_settings: MagicMock, mock_get_llm: MagicMock
    ) -> None:
        mock_settings.return_value = MagicMock(comment_batch_size=20)

        # Create a structured mock that tracks invoke calls
        structured_mock = MagicMock()
        structured_mock.invoke.return_value = CommentSentiment(
            classifications=[ClassificationEntry(text="ok", sentiment="neutral")]
        )
        llm_mock = MagicMock()
        llm_mock.with_structured_output.return_value = structured_mock
        mock_get_llm.return_value = llm_mock

        comments = [_make_comment(f"Comment {i}") for i in range(30)]
        sentiment_analyzer(comments)

        # batch_size=20 and 30 comments => 2 batches (0-19, 20-29)
        assert structured_mock.invoke.call_count == 2


class TestAnalyzeSentimentLLMFailure:
    @patch("deeplens.tools.sentiment.get_llm")
    @patch("deeplens.tools.sentiment.get_settings")
    def test_llm_error_returns_neutral_fallback(
        self, mock_settings: MagicMock, mock_get_llm: MagicMock
    ) -> None:
        mock_settings.return_value = MagicMock(comment_batch_size=20)

        structured_mock = MagicMock()
        structured_mock.invoke.side_effect = RuntimeError("LLM API down")
        llm_mock = MagicMock()
        llm_mock.with_structured_output.return_value = structured_mock
        mock_get_llm.return_value = llm_mock

        comments = [_make_comment("Hello"), _make_comment("World")]
        result = sentiment_analyzer(comments)

        # On failure, comments fall back to neutral
        assert result["total_analyzed"] == 2
        assert result["neutral"] == pytest.approx(1.0)
        assert result["positive"] == pytest.approx(0.0)
        assert result["negative"] == pytest.approx(0.0)
