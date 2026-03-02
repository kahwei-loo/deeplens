"""Web search tools using the Tavily API — primary data source for DeepLens.

Provides multi-query search, result deduplication, and URL content extraction.

Design patterns borrowed from production research agents:
- **Perplexity**: Multi-angle search queries for comprehensive coverage.
- **Manus**: Deep extraction from top URLs for richer analysis.
- **Open Deep Research**: Parallel sub-queries with deduplication.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse

from tavily import TavilyClient
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from deeplens.config import get_settings
from deeplens.models import WebArticle, WebResult

logger = logging.getLogger(__name__)


@lru_cache
def _get_tavily_client() -> TavilyClient:
    """Return a cached TavilyClient singleton."""
    settings = get_settings()
    return TavilyClient(api_key=settings.tavily_api_key)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    reraise=True,
)
def _tavily_search(query: str, max_results: int, search_depth: str) -> dict[str, Any]:
    """Execute a single Tavily search call with retry on transient errors."""
    client = _get_tavily_client()
    return client.search(query=query, max_results=max_results, search_depth=search_depth)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    reraise=True,
)
def _tavily_extract(urls: list[str]) -> dict[str, Any]:
    """Execute a Tavily extract call with retry on transient errors."""
    client = _get_tavily_client()
    return client.extract(urls=urls)


def web_search(
    query: str, max_results: int = 5, search_depth: str = "basic"
) -> list[WebResult]:
    """Search the web via Tavily and return structured results.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.
        search_depth: ``"basic"`` for fast search, ``"advanced"`` for richer snippets.

    Returns an empty list on API errors.
    """
    logger.info("web_search: query=%r max_results=%d depth=%s", query, max_results, search_depth)
    try:
        response = _tavily_search(query, max_results, search_depth)

        results: list[WebResult] = []
        for item in response.get("results", []):
            results.append(
                WebResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    score=float(item.get("score", 0.0)),
                )
            )

        logger.info("web_search: returned %d results", len(results))
        return results

    except Exception as exc:
        logger.error("web_search error: %s", exc)
        return []


def multi_query_search(
    queries: list[str], max_results_per_query: int = 5
) -> list[WebResult]:
    """Execute multiple search queries and return deduplicated results.

    Inspired by Perplexity's multi-angle search: approaching an entity from
    different angles (overview, news, opinion, controversy) yields broader
    coverage than a single query.
    """
    all_results: list[WebResult] = []

    with ThreadPoolExecutor(max_workers=min(len(queries), 5)) as executor:
        futures = {
            executor.submit(web_search, q, max_results_per_query): q
            for q in queries
        }
        for future in as_completed(futures):
            query = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as exc:
                logger.error("multi_query_search: query %r failed: %s", query, exc)

    deduped = _deduplicate_results(all_results)
    logger.info(
        "multi_query_search: %d queries → %d unique results (from %d total)",
        len(queries), len(deduped), len(all_results),
    )
    return deduped


def extract_urls(urls: list[str]) -> list[WebArticle]:
    """Extract full content from URLs using Tavily's extract API.

    Inspired by Manus's deep extraction: going beyond search snippets to get
    the full article content enables richer analysis and better reports.

    Returns an empty list if extraction fails or is unavailable in the
    current Tavily plan.
    """
    if not urls:
        return []

    logger.info("extract_urls: extracting %d URLs", len(urls))
    try:
        response = _tavily_extract(urls)

        articles: list[WebArticle] = []
        for item in response.get("results", []):
            url = item.get("url", "")
            raw = item.get("raw_content", "")
            # Limit content length to avoid token explosion downstream
            content = raw[:5000] if raw else ""
            articles.append(
                WebArticle(
                    url=url,
                    title=item.get("title", urlparse(url).netloc),
                    content=content,
                    source_domain=urlparse(url).netloc,
                )
            )

        logger.info("extract_urls: extracted %d articles", len(articles))
        return articles

    except Exception as exc:
        # extract() may not be available on all Tavily plans — degrade gracefully
        logger.warning("extract_urls failed (may require paid plan): %s", exc)
        return []


def _deduplicate_results(results: list[WebResult]) -> list[WebResult]:
    """Remove duplicate results by URL, keeping the highest-scored version."""
    seen: dict[str, WebResult] = {}
    for r in results:
        url = r["url"]
        if url not in seen or r["score"] > seen[url]["score"]:
            seen[url] = r
    return sorted(seen.values(), key=lambda x: x["score"], reverse=True)
