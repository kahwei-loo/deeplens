"""Tests for deeplens.tools.rag — embedding RAG with keyword fallback."""

from unittest.mock import MagicMock, patch

import numpy as np

from deeplens.models import WebArticle
from deeplens.tools.rag import (
    _cosine_similarity,
    chunk_articles,
    embed_chunks,
    embedding_search,
    keyword_search,
)

_ARTICLES: list[WebArticle] = [
    {
        "url": "https://example.com/article1",
        "title": "Baby Monster debut album",
        "content": (
            "Baby Monster is a K-pop girl group formed by YG Entertainment. "
            "The group debuted in November 2023 with seven members. "
            "Their debut single was a massive hit in South Korea and globally. "
            "The group quickly gained millions of fans worldwide."
        ),
        "source_domain": "example.com",
    },
    {
        "url": "https://example.com/article2",
        "title": "K-pop industry overview",
        "content": (
            "The K-pop industry has seen rapid growth in recent years. "
            "Major entertainment companies include YG Entertainment, SM, and JYP. "
            "Groups like BTS, BLACKPINK, and Baby Monster represent different "
            "generations. Global streaming platforms have accelerated "
            "K-pop's international reach."
        ),
        "source_domain": "example.com",
    },
]


# ── Chunking ─────────────────────────────────────────────────────────


def test_chunk_articles_basic():
    """Chunks articles and tags each chunk with source metadata."""
    chunks = chunk_articles(_ARTICLES, chunk_size=100, overlap=20)

    assert len(chunks) > 0
    for chunk in chunks:
        assert "text" in chunk
        assert "url" in chunk
        assert "title" in chunk
        assert "source_domain" in chunk
        assert "chunk_index" in chunk
        assert len(chunk["text"]) > 0


def test_chunk_articles_preserves_source():
    """Each chunk carries the source article's URL and title."""
    chunks = chunk_articles(_ARTICLES, chunk_size=200, overlap=20)

    urls = {c["url"] for c in chunks}
    assert "https://example.com/article1" in urls
    assert "https://example.com/article2" in urls


def test_chunk_articles_empty():
    """Empty input → empty output."""
    assert chunk_articles([]) == []
    empty_article: WebArticle = {
        "url": "", "title": "", "content": "",
        "source_domain": "",
    }
    assert chunk_articles([empty_article]) == []


def test_chunk_articles_respects_size():
    """Chunks should not exceed chunk_size (approximately)."""
    chunks = chunk_articles(_ARTICLES, chunk_size=100, overlap=10)
    for chunk in chunks:
        assert len(chunk["text"]) <= 150


# ── Cosine similarity ────────────────────────────────────────────────


def test_cosine_similarity_identical():
    """Identical vectors → similarity 1.0."""
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    scores = _cosine_similarity(a, b)
    assert abs(scores[0] - 1.0) < 1e-6


def test_cosine_similarity_orthogonal():
    """Orthogonal vectors → similarity 0.0."""
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([[0.0, 1.0]], dtype=np.float32)
    scores = _cosine_similarity(a, b)
    assert abs(scores[0]) < 1e-6


def test_cosine_similarity_batch():
    """Handles multiple vectors in matrix b."""
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([
        [1.0, 0.0],  # identical → 1.0
        [0.0, 1.0],  # orthogonal → 0.0
        [-1.0, 0.0],  # opposite → -1.0
    ], dtype=np.float32)
    scores = _cosine_similarity(a, b)
    assert abs(scores[0] - 1.0) < 1e-6
    assert abs(scores[1]) < 1e-6
    assert abs(scores[2] + 1.0) < 1e-6


# ── Embedding search ─────────────────────────────────────────────────


@patch("deeplens.tools.rag.OpenAIEmbeddings")
def test_embed_chunks_returns_array(mock_cls):
    """embed_chunks returns numpy array of correct shape."""
    mock_embedder = MagicMock()
    mock_embedder.embed_documents.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]
    mock_cls.return_value = mock_embedder

    chunks = [{"text": "hello"}, {"text": "world"}]
    result = embed_chunks(chunks, api_key="test-key")

    assert result is not None
    assert result.shape == (2, 3)
    mock_embedder.embed_documents.assert_called_once_with(
        ["hello", "world"],
    )


def test_embed_chunks_no_api_key():
    """embed_chunks returns None when no API key."""
    result = embed_chunks([{"text": "hello"}], api_key="")
    assert result is None


def test_embed_chunks_empty():
    """embed_chunks returns None for empty input."""
    result = embed_chunks([], api_key="test-key")
    assert result is None


@patch("deeplens.tools.rag.OpenAIEmbeddings")
def test_embedding_search_returns_ranked(mock_cls):
    """embedding_search returns chunks ranked by similarity."""
    chunks = [
        {"text": "K-pop music", "url": "a"},
        {"text": "Weather forecast", "url": "b"},
        {"text": "Baby Monster debut", "url": "c"},
    ]
    # Pre-computed chunk embeddings
    chunk_embs = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.9, 0.1, 0.0],  # most similar to query
    ], dtype=np.float32)

    mock_embedder = MagicMock()
    # Query embedding: similar to chunks 0 and 2
    mock_embedder.embed_query.return_value = [1.0, 0.0, 0.0]
    mock_cls.return_value = mock_embedder

    results = embedding_search(
        chunks, chunk_embs, "K-pop groups",
        api_key="test-key", top_k=2,
    )

    assert len(results) == 2
    # First result should be the one most similar to [1,0,0]
    assert results[0]["url"] == "a"


def test_embedding_search_empty():
    """embedding_search returns empty on empty input."""
    assert embedding_search([], None, "test", api_key="k") == []
    assert embedding_search(
        [{"text": "x"}], np.array([[1.0]]), "",
        api_key="k",
    ) == []


# ── Keyword search ───────────────────────────────────────────────────


def test_keyword_search_basic():
    """Finds chunks matching query terms."""
    chunks = chunk_articles(_ARTICLES, chunk_size=200, overlap=20)
    results = keyword_search(chunks, "Baby Monster debut", top_k=3)

    assert len(results) > 0
    assert "baby monster" in results[0]["text"].lower()


def test_keyword_search_ranking():
    """Chunks with more distinct matching terms rank higher."""
    chunks = [
        {
            "text": "Baby Monster is a K-pop group from YG",
            "url": "a", "title": "a",
            "source_domain": "a", "chunk_index": 0,
        },
        {
            "text": "The weather is nice today",
            "url": "b", "title": "b",
            "source_domain": "b", "chunk_index": 0,
        },
        {
            "text": "Baby Monster debut was huge in K-pop scene",
            "url": "c", "title": "c",
            "source_domain": "c", "chunk_index": 0,
        },
    ]
    results = keyword_search(
        chunks, "Baby Monster K-pop debut", top_k=3,
    )

    assert len(results) >= 2
    assert "debut" in results[0]["text"].lower()


def test_keyword_search_empty():
    """Empty query or empty chunks → empty results."""
    assert keyword_search([], "test") == []
    assert keyword_search([{"text": "hello"}], "") == []


def test_keyword_search_top_k():
    """Respects top_k limit."""
    chunks = chunk_articles(_ARTICLES, chunk_size=80, overlap=10)
    results = keyword_search(chunks, "K-pop", top_k=2)
    assert len(results) <= 2
