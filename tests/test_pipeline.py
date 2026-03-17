import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.rag_pipeline import HybridSearchEngine, RAGPipeline
def test_query():

    engine = HybridSearchEngine(["RAG is retrieval augmented generation"])

    pipeline = RAGPipeline(engine)

    result, latency = pipeline.query("What is RAG?")

    assert latency >= 0