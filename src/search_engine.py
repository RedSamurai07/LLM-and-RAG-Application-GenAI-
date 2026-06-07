import numpy as np
from typing import List, Dict
from src.config import CONFIG

try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
except ImportError:  # pragma: no cover
    _SentenceTransformer = None  # type: ignore

try:
    from rank_bm25 import BM25Okapi as _BM25Okapi
except ImportError:  # pragma: no cover
    _BM25Okapi = None  # type: ignore


class HybridSearchEngine:
    """Combines Dense (Embeddings) and Sparse (BM25) for precision."""

    def __init__(self, documents: List[Dict], model=None, bm25=None):
        self.docs = documents

        if model is not None and bm25 is not None:
            self.model = model
            self.bm25 = bm25
            self.embeddings = self.model.encode(
                [d["content"] for d in documents], normalize_embeddings=True
            )
        else:
            self.model = _SentenceTransformer(CONFIG["embedding_model"])
            self.embeddings = self.model.encode(
                [d["content"] for d in documents], normalize_embeddings=True
            )
            tokenized_corpus = [d["content"].lower().split() for d in documents]
            self.bm25 = _BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        dense_ranks = np.argsort(-np.dot(self.embeddings, q_emb))
        bm25_ranks = np.argsort(-self.bm25.get_scores(query.lower().split()))

        k = 60
        scores = np.zeros(len(self.docs))
        for rank, idx in enumerate(dense_ranks):
            scores[idx] += CONFIG["dense_weight"] * (1 / (rank + k))
        for rank, idx in enumerate(bm25_ranks):
            scores[idx] += (1 - CONFIG["dense_weight"]) * (1 / (rank + k))

        return [
            dict(self.docs[i], rrf_score=scores[i])
            for i in np.argsort(-scores)[:top_k]
        ]
