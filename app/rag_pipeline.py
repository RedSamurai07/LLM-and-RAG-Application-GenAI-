import numpy as np
import time
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

class HybridSearchEngine:
    """Combines Dense (Embeddings) and Sparse (BM25) for precision."""
    def __init__(self, documents: List[Dict], embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.docs = documents
        self.model = SentenceTransformer(embedding_model)
        
        # Pre-compute dense embeddings
        print(f"Encoding {len(documents)} documents...")
        self.embeddings = self.model.encode([d["content"] for d in documents], normalize_embeddings=True)
        
        # Initialize BM25
        print("Initializing BM25...")
        tokenized_corpus = [d["content"].lower().split() for d in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 5, dense_weight: float = 0.7) -> List[Dict]:
        # Dense Rank
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        dense_scores = np.dot(self.embeddings, q_emb)
        dense_ranks = np.argsort(-dense_scores)
        
        # Sparse Rank
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_ranks = np.argsort(-bm25_scores)
        
        # Reciprocal Rank Fusion (RRF)
        k = 60
        scores = np.zeros(len(self.docs))
        
        # We use ranks for RRF
        for rank, idx in enumerate(dense_ranks):
            scores[idx] += dense_weight * (1 / (rank + k))
            
        for rank, idx in enumerate(bm25_ranks):
            scores[idx] += (1 - dense_weight) * (1 / (rank + k))
            
        # Get top-k indices
        top_indices = np.argsort(-scores)[:top_k]
        
        results = []
        for i in top_indices:
            doc = self.docs[i].copy()
            doc["rrf_score"] = scores[i]
            results.append(doc)
            
        return results

class RAGPipeline:
    def __init__(self, engine: HybridSearchEngine):
        self.engine = engine

    def query(self, question: str) -> tuple:
        start_time = time.time()
        results = self.engine.search(question)
        latency = (time.time() - start_time) * 1000
        return results, latency

if __name__ == "__main__":
    # Simple smoke test for CI/CD
    docs = [{"content": "Retrieval Augmented Generation (RAG) is a powerful technique.", "id": "smoke_1"}]
    engine = HybridSearchEngine(docs)
    pipeline = RAGPipeline(engine)
    res, lat = pipeline.query("What is RAG?")
    print(f"Smoke test successful. Latency: {lat:.2f}ms")