"""Application configuration via pydantic-settings and .env."""

from functools import lru_cache

from langchain_openai import ChatOpenAI
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """DeepLens configuration loaded from environment variables / .env file."""

    openai_api_key: str = ""
    openai_api_base: str = ""  # Custom base URL (e.g. OpenRouter, Azure)
    youtube_api_key: str = ""  # Optional — YouTube enrichment disabled if empty
    tavily_api_key: str = ""
    model_name: str = "gpt-4o-mini"
    max_iterations: int = 5
    youtube_max_results: int = 15
    comment_batch_size: int = 20
    output_dir: str = "output"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def youtube_available(self) -> bool:
        """Whether YouTube API enrichment is available."""
        return bool(self.youtube_api_key)


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()


def get_llm(temperature: float = 0) -> ChatOpenAI:
    """Create a ChatOpenAI instance with the correct settings."""
    settings = get_settings()
    kwargs: dict = {
        "model": settings.model_name,
        "api_key": settings.openai_api_key,
        "temperature": temperature,
    }
    if settings.openai_api_base:
        kwargs["base_url"] = settings.openai_api_base
    return ChatOpenAI(**kwargs)
