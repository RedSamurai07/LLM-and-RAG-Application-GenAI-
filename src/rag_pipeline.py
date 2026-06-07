import time
from dataclasses import dataclass
from src.search_engine import HybridSearchEngine
from src.config import CONFIG


@dataclass
class RAGResponse:
    query: str
    latency_ms: float
    recall_score: float


class RAGPipeline:
    def __init__(self, engine: HybridSearchEngine):
        self.engine = engine

    def query(self, user_query: str, ground_truth: str) -> RAGResponse:
        t0 = time.time()
        results = self.engine.search(user_query, top_k=CONFIG["top_k"])
        latency = (time.time() - t0) * 1000

        context = " ".join([r["content"] for r in results]).lower()
        gt_tokens = set(ground_truth.lower().split())
        recall = (
            sum(1 for t in gt_tokens if t in context) / len(gt_tokens)
            if gt_tokens
            else 0
        )

        return RAGResponse(query=user_query, latency_ms=latency, recall_score=recall)
