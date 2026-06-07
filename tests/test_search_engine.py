import numpy as np
import pytest
import src.search_engine as search_engine_module
from unittest.mock import MagicMock
from src.search_engine import HybridSearchEngine


SAMPLE_DOCS = [
    {"content": "transformer models use attention mechanisms", "id": "1"},
    {"content": "retrieval augmented generation improves accuracy", "id": "2"},
    {"content": "bm25 is a sparse ranking algorithm for search", "id": "3"},
]


def make_mock_engine(docs=None):
    if docs is None:
        docs = SAMPLE_DOCS

    mock_model = MagicMock()
    mock_model.encode.side_effect = lambda texts, normalize_embeddings=False: (
        np.random.rand(len(texts), 384)
        if isinstance(texts, list)
        else np.random.rand(1, 384)
    )

    mock_bm25 = MagicMock()
    mock_bm25.get_scores.return_value = np.array([0.5, 0.8, 0.3])

    return HybridSearchEngine(docs, model=mock_model, bm25=mock_bm25)


def test_search_returns_correct_number_of_results():
    engine = make_mock_engine()
    results = engine.search("transformer attention", top_k=2)
    assert len(results) == 2


def test_search_results_have_rrf_score():
    engine = make_mock_engine()
    results = engine.search("bm25 ranking", top_k=3)
    for r in results:
        assert "rrf_score" in r


def test_search_top_k_one():
    engine = make_mock_engine()
    results = engine.search("any query", top_k=1)
    assert len(results) == 1


def test_search_results_contain_content_and_id():
    engine = make_mock_engine()
    results = engine.search("retrieval generation")
    for r in results:
        assert "content" in r
        assert "id" in r


def test_rrf_scores_are_positive():
    engine = make_mock_engine()
    results = engine.search("test query", top_k=3)
    for r in results:
        assert r["rrf_score"] > 0


def test_search_with_single_document():
    docs = [{"content": "only one document here", "id": "solo"}]
    engine = make_mock_engine(docs=docs)
    engine.bm25.get_scores.return_value = np.array([1.0])
    results = engine.search("document", top_k=1)
    assert len(results) == 1


def test_init_with_real_model_classes():
    """Covers the else-branch of __init__ by patching module-level classes."""
    docs = [{"content": "hello world", "id": "1"}]

    mock_model = MagicMock()
    mock_model.encode.return_value = np.random.rand(1, 384)

    mock_bm25_instance = MagicMock()
    mock_bm25_class = MagicMock(return_value=mock_bm25_instance)

    original_st = search_engine_module._SentenceTransformer
    original_bm25 = search_engine_module._BM25Okapi
    search_engine_module._SentenceTransformer = MagicMock(return_value=mock_model)
    search_engine_module._BM25Okapi = mock_bm25_class

    try:
        engine = HybridSearchEngine(docs)
        assert engine.model is not None
        assert engine.bm25 is not None
    finally:
        search_engine_module._SentenceTransformer = original_st
        search_engine_module._BM25Okapi = original_bm25
