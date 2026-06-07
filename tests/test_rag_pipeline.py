import pytest
from unittest.mock import MagicMock
from src.rag_pipeline import RAGPipeline, RAGResponse


def make_mock_pipeline(content_override=None):
    content = content_override or "the transformer model uses attention mechanisms bert is pretrained"
    mock_engine = MagicMock()
    mock_engine.search.return_value = [
        {"content": content, "id": "1", "rrf_score": 0.9},
        {"content": "bert is a pretrained language model", "id": "2", "rrf_score": 0.7},
    ]
    return RAGPipeline(engine=mock_engine)


def test_query_returns_rag_response():
    pipeline = make_mock_pipeline()
    response = pipeline.query("what is a transformer?", "attention mechanisms")
    assert isinstance(response, RAGResponse)


def test_recall_is_between_0_and_1():
    pipeline = make_mock_pipeline()
    response = pipeline.query("transformer model", "transformer model uses attention")
    assert 0.0 <= response.recall_score <= 1.0


def test_latency_is_non_negative():
    pipeline = make_mock_pipeline()
    response = pipeline.query("some query", "some ground truth")
    assert response.latency_ms >= 0


def test_query_stored_in_response():
    pipeline = make_mock_pipeline()
    q = "what is RAG?"
    response = pipeline.query(q, "retrieval augmented generation")
    assert response.query == q


def test_empty_ground_truth_gives_zero_recall():
    pipeline = make_mock_pipeline()
    response = pipeline.query("some query", "")
    assert response.recall_score == 0.0


def test_perfect_recall_when_all_tokens_in_context():
    pipeline = make_mock_pipeline(content_override="transformer bert")
    response = pipeline.query("test", "transformer bert")
    assert response.recall_score == 1.0


def test_partial_recall():
    pipeline = make_mock_pipeline(content_override="only transformer is here")
    response = pipeline.query("test", "transformer missing_word")
    assert 0.0 < response.recall_score < 1.0
