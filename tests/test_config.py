"""Tests for deeplens.config — Settings, get_settings, get_llm."""

from unittest.mock import patch

import pytest

from deeplens.config import Settings, get_llm, get_settings, validate_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear lru_cache and reset env vars that .env may override."""
    get_settings.cache_clear()
    # Ensure .env file values don't leak into tests
    monkeypatch.setenv("MODEL_NAME", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_BASE", "")


# ── Settings defaults ────────────────────────────────────────────────────


def test_default_settings() -> None:
    """Verify field values from test environment."""
    s = get_settings()
    assert s.model_name == "gpt-4o-mini"
    assert s.max_iterations == 5
    assert s.output_dir == "output"


# ── youtube_available property ───────────────────────────────────────────


def test_youtube_available_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YOUTUBE_API_KEY", "real-key")
    s = get_settings()
    assert s.youtube_available is True


def test_youtube_available_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YOUTUBE_API_KEY", "")
    s = get_settings()
    assert s.youtube_available is False


# ── get_llm ──────────────────────────────────────────────────────────────


@patch("deeplens.config.ChatOpenAI")
def test_get_llm_default(mock_chat: object) -> None:
    """get_llm() uses model_name and api_key from settings, temperature=0."""
    get_llm()
    mock_chat.assert_called_once_with(  # type: ignore[attr-defined]
        model="gpt-4o-mini",
        api_key="test-key",
        temperature=0,
    )


@patch("deeplens.config.ChatOpenAI")
def test_get_llm_with_base_url(
    mock_chat: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """get_llm() forwards openai_api_base as base_url when set."""
    monkeypatch.setenv("OPENAI_API_BASE", "https://custom.api/v1")
    get_llm()
    mock_chat.assert_called_once_with(  # type: ignore[attr-defined]
        model="gpt-4o-mini",
        api_key="test-key",
        temperature=0,
        base_url="https://custom.api/v1",
    )


@patch("deeplens.config.ChatOpenAI")
def test_get_llm_temperature(mock_chat: object) -> None:
    """get_llm(temperature=0.5) passes the temperature through."""
    get_llm(temperature=0.5)
    mock_chat.assert_called_once_with(  # type: ignore[attr-defined]
        model="gpt-4o-mini",
        api_key="test-key",
        temperature=0.5,
    )


# ── get_settings caching ────────────────────────────────────────────────


def test_settings_cache() -> None:
    """get_settings returns the same instance on repeated calls (lru_cache)."""
    a = get_settings()
    b = get_settings()
    assert a is b


# ── Truncation constants ─────────────────────────────────────────────────


def test_article_content_chars_default() -> None:
    """article_content_chars defaults to 5000."""
    s = get_settings()
    assert s.article_content_chars == 5000


def test_article_llm_chars_default() -> None:
    """article_llm_chars defaults to 2000."""
    s = get_settings()
    assert s.article_llm_chars == 2000


def test_article_llm_chars_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    """article_llm_chars can be overridden via environment variable."""
    monkeypatch.setenv("ARTICLE_LLM_CHARS", "1000")
    s = get_settings()
    assert s.article_llm_chars == 1000


# ── validate_settings ────────────────────────────────────────────────────


def test_validate_settings_passes_with_keys() -> None:
    """No exception when both required keys are present."""
    s = Settings(openai_api_key="sk-test", tavily_api_key="tvly-test")
    validate_settings(s)  # Should not raise


def test_validate_settings_missing_openai_key() -> None:
    """ValueError raised when OPENAI_API_KEY is missing."""
    s = Settings(openai_api_key="", tavily_api_key="tvly-test")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        validate_settings(s)


def test_validate_settings_missing_tavily_key() -> None:
    """ValueError raised when TAVILY_API_KEY is missing."""
    s = Settings(openai_api_key="sk-test", tavily_api_key="")
    with pytest.raises(ValueError, match="TAVILY_API_KEY"):
        validate_settings(s)
