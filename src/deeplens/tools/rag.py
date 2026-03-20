"""RAG retrieval module — embedding search with keyword fallback.

Chunks web articles into overlapping segments and retrieves relevant chunks
using OpenAI embeddings + cosine similarity. Falls back to keyword matching
when embeddings are unavailable (e.g. in tests or when API key is missing).

Also provides ``truncate_for_llm``, the shared JSON-truncation helper used
by both research_tools and report_tools to cap tool return sizes.
"""

from __future__ import annotations

import json
import logging
import re

import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr

from deeplens.models import WebArticle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_articles(
    articles: list[WebArticle],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    """Split web articles into overlapping text chunks with source metadata.

    Each chunk is a dict with keys: text, url, title, source_domain, chunk_index.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []
    for article in articles:
        content = article.get("content", "")
        if not content.strip():
            continue

        texts = splitter.split_text(content)
        for i, text in enumerate(texts):
            chunks.append({
                "text": text,
                "url": article.get("url", ""),
                "title": article.get("title", ""),
                "source_domain": article.get("source_domain", ""),
                "chunk_index": i,
            })

    logger.info(
        "chunk_articles: %d articles → %d chunks (size=%d, overlap=%d)",
        len(articles), len(chunks), chunk_size, overlap,
    )
    return chunks


# ---------------------------------------------------------------------------
# Embedding search (primary)
# ---------------------------------------------------------------------------

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vector *a* and matrix *b*."""
    norm_a = np.linalg.norm(a)
    norms_b = np.linalg.norm(b, axis=1)
    # Guard against zero-norm vectors
    denom = norm_a * norms_b
    denom = np.where(denom == 0, 1.0, denom)
    result: np.ndarray = (b @ a) / denom
    return result


def embed_chunks(
    chunks: list[dict],
    api_key: str,
    model: str = "text-embedding-3-small",
) -> np.ndarray | None:
    """Compute embeddings for chunk texts using OpenAI API.

    Returns an (N, dim) numpy array of embeddings, or None on failure.
    """
    if not chunks or not api_key:
        return None

    try:
        embedder = OpenAIEmbeddings(
            model=model, api_key=SecretStr(api_key),
        )
        texts = [c["text"] for c in chunks]
        vectors = embedder.embed_documents(texts)
        arr = np.array(vectors, dtype=np.float32)
        logger.info(
            "embed_chunks: embedded %d chunks → shape %s",
            len(chunks), arr.shape,
        )
        return arr
    except Exception as e:
        logger.warning(
            "embed_chunks failed: %s (falling back to keyword)", e,
        )
        return None


def embedding_search(
    chunks: list[dict],
    chunk_embeddings: np.ndarray,
    query: str,
    api_key: str,
    top_k: int = 5,
    model: str = "text-embedding-3-small",
) -> list[dict]:
    """Retrieve top_k chunks by cosine similarity to query embedding.

    Returns empty list on any failure (caller should fall back to keyword).
    """
    if not chunks or chunk_embeddings is None or not query.strip():
        return []

    if len(chunks) != len(chunk_embeddings):
        logger.warning(
            "embedding_search: chunk count (%d) != embedding count (%d)",
            len(chunks), len(chunk_embeddings),
        )
        return []

    try:
        embedder = OpenAIEmbeddings(
            model=model, api_key=SecretStr(api_key),
        )
        query_vec = np.array(
            embedder.embed_query(query), dtype=np.float32,
        )
        scores = _cosine_similarity(query_vec, chunk_embeddings)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = [chunks[i] for i in top_indices if scores[i] > 0]
        logger.info(
            "embedding_search: query=%r, top_score=%.3f, returning=%d",
            query,
            float(scores[top_indices[0]]) if len(top_indices) > 0 else 0,
            len(results),
        )
        return results
    except Exception as e:
        logger.warning("embedding_search failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Keyword search (fallback)
# ---------------------------------------------------------------------------

def keyword_search(
    chunks: list[dict],
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """Rank chunks by keyword match count (fallback when embeddings unavailable).

    Matching is case-insensitive. Query is split into individual terms;
    each chunk is scored by how many distinct query terms appear in its text.
    """
    if not chunks or not query.strip():
        return []

    terms = [
        t.lower() for t in re.split(r"\s+", query.strip())
        if len(t) >= 2
    ]
    if not terms:
        return []

    scored: list[tuple[int, int, dict]] = []
    for idx, chunk in enumerate(chunks):
        text_lower = chunk["text"].lower()
        score = sum(1 for term in terms if term in text_lower)
        if score > 0:
            scored.append((score, -idx, chunk))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    results = [item[2] for item in scored[:top_k]]

    logger.info(
        "keyword_search: query=%r, terms=%d, matched=%d, returning=%d",
        query, len(terms), len(scored), len(results),
    )
    return results


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def truncate_for_llm(obj: object, max_len: int = 1500) -> str:
    """JSON-serialize *obj* and truncate to *max_len* characters.

    Used by tool wrappers to cap the size of data returned to the LLM
    so that tool results don't blow up the context window.
    """
    text = json.dumps(obj, ensure_ascii=False, default=str)
    if len(text) > max_len:
        return text[:max_len] + "... [truncated]"
    return text
