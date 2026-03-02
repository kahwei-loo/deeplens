"""Shared test fixtures for DeepLens."""

import pytest


@pytest.fixture(autouse=True)
def _env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent tests from using real API keys."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("YOUTUBE_API_KEY", "test-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
