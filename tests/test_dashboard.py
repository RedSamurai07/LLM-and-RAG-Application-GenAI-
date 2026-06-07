import pytest
from src.rag_pipeline import RAGResponse
from src.dashboard import build_monitoring_dashboard


@pytest.fixture
def sample_responses():
    return [
        RAGResponse(query="What is RAG?", latency_ms=120.5, recall_score=0.85),
        RAGResponse(query="Explain transformers", latency_ms=95.2, recall_score=0.72),
        RAGResponse(query="What is BM25?", latency_ms=110.0, recall_score=0.91),
    ]


def test_dashboard_returns_dict(sample_responses):
    result = build_monitoring_dashboard(sample_responses)
    assert isinstance(result, dict)


def test_dashboard_has_avg_latency_key(sample_responses):
    result = build_monitoring_dashboard(sample_responses)
    assert "avg_latency_ms" in result


def test_dashboard_has_avg_recall_key(sample_responses):
    result = build_monitoring_dashboard(sample_responses)
    assert "avg_recall" in result


def test_dashboard_avg_recall_in_range(sample_responses):
    result = build_monitoring_dashboard(sample_responses)
    assert 0 <= result["avg_recall"] <= 1


def test_dashboard_dataframe_row_count(sample_responses):
    result = build_monitoring_dashboard(sample_responses)
    assert result["dataframe"].shape[0] == 3


def test_dashboard_dataframe_has_query_column(sample_responses):
    result = build_monitoring_dashboard(sample_responses)
    assert "Query" in result["dataframe"].columns


def test_dashboard_single_response():
    responses = [RAGResponse(query="single test", latency_ms=50.0, recall_score=0.6)]
    result = build_monitoring_dashboard(responses)
    assert result["avg_latency_ms"] == pytest.approx(50.0)


def test_dashboard_avg_latency_value(sample_responses):
    result = build_monitoring_dashboard(sample_responses)
    assert result["avg_latency_ms"] == pytest.approx(108.567, abs=0.1)
