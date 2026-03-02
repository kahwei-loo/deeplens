"""Analysis agent — processes collected data into statistics, sentiment, and web analysis.

This agent processes data collected by the Research agent. It handles three paths:
- YouTube path: compute video statistics and comment sentiment
- Web-only path: extract key themes and entities from web articles via LLM
- Mixed path: both YouTube and web analysis
"""

import logging

from pydantic import BaseModel, Field

from deeplens.config import get_llm
from deeplens.models import WebAnalysis
from deeplens.state import DeepLensState
from deeplens.tools.sentiment import sentiment_analyzer
from deeplens.tools.statistics import compute_statistics

logger = logging.getLogger(__name__)


class WebAnalysisResult(BaseModel):
    """Structured output for LLM-based web article analysis."""

    key_themes: list[str] = Field(
        description="3-7 key themes or topics identified across the web articles"
    )
    entity_mentions: list[str] = Field(
        description="Notable people, organizations, or entities mentioned"
    )
    summary: str = Field(
        description="A 2-4 sentence synthesis of the main findings from web articles"
    )


def _analyze_web_articles(articles: list[dict]) -> WebAnalysis | None:
    """Extract key themes, entities, and summary from web articles using LLM.

    Returns a dict with key_themes, entity_mentions, and summary, or None on failure.
    """
    if not articles:
        return None

    # Build article context for the LLM
    article_texts: list[str] = []
    for a in articles:
        title = a.get("title", "Untitled")
        domain = a.get("source_domain", "unknown")
        content = a.get("content", "")
        if len(content) > 1500:
            content = content[:1500] + "..."
        article_texts.append(f"[{domain}] {title}\n{content}")

    combined = "\n\n---\n\n".join(article_texts)

    llm = get_llm(temperature=0)
    structured_llm = llm.with_structured_output(WebAnalysisResult)

    response = structured_llm.invoke(
        [
            {
                "role": "system",
                "content": (
                    "You are an analytical research assistant. Analyze the following "
                    "web articles and extract key themes, notable entity mentions, "
                    "and a concise synthesis. Be factual and evidence-based."
                ),
            },
            {
                "role": "user",
                "content": f"Analyze these web articles:\n\n{combined}",
            },
        ]
    )

    return response.model_dump()


def analysis_agent(state: DeepLensState) -> dict:
    """Process collected data: compute video statistics, sentiment, and web analysis.

    Handles three scenarios:
    - YouTube data available: compute statistics and sentiment
    - Web articles available (no comments): run LLM-based web analysis
    - Both: run all analyses
    """
    errors = list(state.get("errors") or [])
    videos = state.get("videos") or []
    comments = state.get("comments") or []
    web_articles = state.get("web_articles") or []

    statistics = None
    sentiment = None
    web_analysis = None

    # Compute video statistics if we have video data
    if videos:
        try:
            statistics = compute_statistics(videos)
            logger.info(
                "[Analysis] Statistics computed for %d videos — "
                "avg views: %s, avg engagement: %s",
                len(videos),
                f"{statistics.get('avg_views', 0):,.0f}",
                f"{statistics.get('avg_engagement_rate', 0):.2%}",
            )
        except Exception as e:
            error_msg = f"Statistics computation failed: {e}"
            logger.error("[Analysis] %s", error_msg)
            errors.append(error_msg)
    else:
        logger.info("[Analysis] No videos available, skipping statistics")

    # Run sentiment analysis if we have comments
    if comments:
        try:
            sentiment = sentiment_analyzer(comments)
            logger.info(
                "[Analysis] Sentiment analyzed for %d comments — "
                "positive: %s, neutral: %s, negative: %s",
                sentiment.get("total_analyzed", 0),
                f"{sentiment.get('positive', 0):.0%}",
                f"{sentiment.get('neutral', 0):.0%}",
                f"{sentiment.get('negative', 0):.0%}",
            )
        except Exception as e:
            error_msg = f"Sentiment analysis failed: {e}"
            logger.error("[Analysis] %s", error_msg)
            errors.append(error_msg)
    else:
        logger.info("[Analysis] No comments available, skipping sentiment analysis")

    # Run web article analysis if we have articles but no comments
    # (web-only path, or as supplementary analysis)
    if web_articles and not comments:
        try:
            web_analysis = _analyze_web_articles(web_articles)
            if web_analysis:
                logger.info(
                    "[Analysis] Web analysis complete — %d themes, %d entities",
                    len(web_analysis.get("key_themes", [])),
                    len(web_analysis.get("entity_mentions", [])),
                )
            else:
                logger.warning("[Analysis] Web analysis returned no results")
        except Exception as e:
            error_msg = f"Web article analysis failed: {e}"
            logger.error("[Analysis] %s", error_msg)
            errors.append(error_msg)

    if not videos and not comments and not web_articles:
        logger.warning(
            "[Analysis] No data to analyze — videos, comments, and web articles all empty"
        )
        errors.append("Analysis skipped: no data to process")

    result: dict = {"errors": errors}

    if statistics is not None:
        result["statistics"] = statistics
    if sentiment is not None:
        result["sentiment"] = sentiment
    if web_analysis is not None:
        result["web_analysis"] = web_analysis

    return result
