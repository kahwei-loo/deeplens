"""LLM-based sentiment analysis tool with batch processing."""

from __future__ import annotations

import logging
import re
from typing import Literal, cast

from pydantic import BaseModel

from deeplens.config import get_llm, get_settings
from deeplens.models import CommentData, SentimentResult

logger = logging.getLogger(__name__)

_MAX_COMMENT_LENGTH = 500

SENTIMENT_PROMPT = """You are a sentiment analysis expert.\
 Classify each YouTube comment as exactly one of: "positive", "neutral", or "negative".

Handle multilingual text, sarcasm, emoji, internet slang, and cultural context.
- Sarcasm like "Great video, I only fell asleep twice" is NEGATIVE.
- Simple emoji reactions like "🔥🔥🔥" are POSITIVE.
- Factual observations without opinion are NEUTRAL.

Analyze EVERY comment in the list below and return your classifications.

Comments:
{comments}"""


def _sanitize_comment(text: str) -> str:
    """Strip control characters and truncate to prevent prompt injection."""
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return cleaned[:_MAX_COMMENT_LENGTH]


class ClassificationEntry(BaseModel):
    """A single comment sentiment classification."""

    text: str
    sentiment: Literal["positive", "neutral", "negative"]


class CommentSentiment(BaseModel):
    """Structured LLM output for a batch of comment sentiment classifications."""

    classifications: list[ClassificationEntry]


def sentiment_analyzer(comments: list[CommentData]) -> SentimentResult:
    """Classify comments into positive/neutral/negative using an LLM.

    Comments are processed in batches of ``config.comment_batch_size`` (default 20)
    to balance cost and latency. Results are aggregated into a single
    :class:`SentimentResult`.
    """
    if not comments:
        logger.info("sentiment_analyzer: no comments to analyze")
        return SentimentResult(
            positive=0.0,
            neutral=0.0,
            negative=0.0,
            total_analyzed=0,
            sample_positive=[],
            sample_negative=[],
        )

    settings = get_settings()
    batch_size = settings.comment_batch_size

    llm = get_llm(temperature=0)
    structured_llm = llm.with_structured_output(CommentSentiment)

    # Collect all classifications across batches
    all_positive: list[str] = []
    all_neutral: list[str] = []
    all_negative: list[str] = []

    for i in range(0, len(comments), batch_size):
        batch = comments[i : i + batch_size]
        batch_texts = "\n".join(
            f"{idx + 1}. {_sanitize_comment(c['text'])}"
            for idx, c in enumerate(batch)
        )

        logger.info(
            "sentiment_analyzer: processing batch %d–%d of %d",
            i + 1,
            min(i + batch_size, len(comments)),
            len(comments),
        )

        try:
            result = cast(CommentSentiment, structured_llm.invoke(
                SENTIMENT_PROMPT.format(comments=batch_texts)
            ))

            for entry in result.classifications:
                if entry.sentiment == "positive":
                    all_positive.append(entry.text)
                elif entry.sentiment == "negative":
                    all_negative.append(entry.text)
                else:
                    all_neutral.append(entry.text)

        except Exception as exc:
            logger.error("sentiment_analyzer batch error: %s", exc)
            # Count the failed batch as neutral rather than dropping data
            for c in batch:
                all_neutral.append(c["text"])

    total = len(all_positive) + len(all_neutral) + len(all_negative)

    sentiment = SentimentResult(
        positive=len(all_positive) / total if total else 0.0,
        neutral=len(all_neutral) / total if total else 0.0,
        negative=len(all_negative) / total if total else 0.0,
        total_analyzed=total,
        sample_positive=all_positive[:5],
        sample_negative=all_negative[:5],
    )

    logger.info(
        "sentiment_analyzer: total=%d pos=%.1f%% neu=%.1f%% neg=%.1f%%",
        total,
        sentiment["positive"] * 100,
        sentiment["neutral"] * 100,
        sentiment["negative"] * 100,
    )
    return sentiment
