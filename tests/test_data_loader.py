import src.data_loader as data_loader_module
from unittest.mock import patch, MagicMock
from src.data_loader import load_markrai_dataset


def _run_loader(mock_corpus, mock_qa):
    mock_fn = MagicMock(side_effect=[mock_corpus, mock_qa])
    original = data_loader_module.load_dataset
    data_loader_module.load_dataset = mock_fn
    try:
        docs, test_set = load_markrai_dataset()
    finally:
        data_loader_module.load_dataset = original
    return docs, test_set


def test_load_returns_docs_and_test_set():
    mock_corpus = [{"contents": "text about transformers", "doc_id": "d1"}]
    mock_qa = [{"query": "What is a transformer?", "generation_gt": ["A deep learning model"]}]
    docs, test_set = _run_loader(mock_corpus, mock_qa)
    assert len(docs) == 1
    assert len(test_set) == 1


def test_docs_have_content_and_id():
    mock_corpus = [{"contents": "some content", "doc_id": "doc_001"}]
    mock_qa = [{"query": "a question", "generation_gt": ["an answer"]}]
    docs, _ = _run_loader(mock_corpus, mock_qa)
    assert docs[0]["content"] == "some content"
    assert docs[0]["id"] == "doc_001"


def test_test_set_has_question_and_ground_truth():
    mock_corpus = [{"contents": "content", "doc_id": "d1"}]
    mock_qa = [{"query": "What is RAG?", "generation_gt": ["Retrieval Augmented Generation"]}]
    _, test_set = _run_loader(mock_corpus, mock_qa)
    assert test_set[0]["question"] == "What is RAG?"
    assert test_set[0]["ground_truth"] == "Retrieval Augmented Generation"


def test_multiple_docs_loaded():
    mock_corpus = [
        {"contents": "doc one content", "doc_id": "d1"},
        {"contents": "doc two content", "doc_id": "d2"},
    ]
    mock_qa = [{"query": "q1", "generation_gt": ["a1"]}]
    docs, _ = _run_loader(mock_corpus, mock_qa)
    assert len(docs) == 2
