# LLM and RAG Application (GenAI)

## Table of contents
- [Project Overview](#project-overview)
- [Executive Summary](#executive-summary)
- [Goal](goal)
- [Data Structure](data-structure)
- [Tools](tools)
- [Analysis](#analysis)
- [Insights](insights)
- [Recommendations](recommendations)

### Project Overview

The project involves building a sophisticated Retrieval-Augmented Generation (RAG) application specifically designed to handle domain-specific queries. The application utilizes a hybrid search architecture that combines dense vector search (semantic) with sparse keyword search (BM25) to improve precision. The knowledge base is built from a document corpus, such as airline policy PDFs or research papers, and evaluated using advanced metrics to ensure accuracy and relevance.

### Executive Summary
This project demonstrates an industry-level approach to Generative AI by moving beyond simple "demo" systems to a production-ready RAG pipeline. By implementing hybrid search and a rigorous evaluation framework (RAGAS), the application overcomes common limitations in standard vector-search-only implementations. The pipeline is designed to be deployed as a domain-specific Q&A tool, providing high-fidelity responses grounded in a specific knowledge corpus, such as resolving customer airline complaints.

### Goal

Architectural Excellence: To build a robust RAG pipeline using a hybrid search engine that integrates dense embeddings (Sentence Transformers) and sparse tokenized search (BM25).Quality Evaluation: To implement a formal evaluation framework using RAGAS metrics to measure faithfulness, answer relevancy, and context recall.Business Utility: To create a practical application, such as a "Ask ArXiv" chatbot or an airline policy resolution tool, that provides measurable value over generic LLM responses.

### Data structure and initial checks
[Dataset](https://huggingface.co/datasets/MarkrAI/AutoRAG-evaluation-2024-LLM-paper-v1)

### Tools
- Python: Google Colab - Data Preparation and pre-processing, Exploratory Data Analysis, Descriptive Statistics, inferential Statistics, Data manipulation and Analysis(Numpy, Pandas),Visualization (Matplotlib, Seaborn), Feature Engineering, Hypothesis Testing
  
### Analysis
**Python**
Importing all the necessary libraries
``` python
import numpy as np
import pandas as pd
import time
import logging
import matplotlib.pyplot as plt
from typing import List, Dict
from dataclasses import dataclass, field
from datetime import datetime
from datasets import load_dataset
```
Installing all the necessary packages
``` python
!pip install langchain langchain-community chromadb
!pip install datasets langchain chromadb sentence-transformers rank-bm25 pymupdf
```
Configuration settings for our project
``` python
# Configuration
CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "dataset_name": "MarkrAI/AutoRAG-evaluation-2024-LLM-paper-v1",
    "dense_weight": 0.7,  # RRF Weight for Semantic search
    "top_k": 5}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
```
Laoding the data
``` python
def load_markrai_dataset():
    """Loads the 8.5k research corpus nodes and 520 QA test cases."""
    logger.info("Fetching corpus and QA subsets from Hugging Face...")
    corpus_ds = load_dataset(CONFIG['dataset_name'], "corpus", split="train")
    qa_ds = load_dataset(CONFIG['dataset_name'], "qa", split="train")
    
    docs = [{"content": r["contents"], "id": r["doc_id"]} for r in corpus_ds]
    test_set = [{"question": r["query"], "ground_truth": r["generation_gt"][0]} for r in qa_ds]
    
    return docs, test_set
```
Hybrid Search Engine Architecture for our model
``` python
class HybridSearchEngine:
    """Combines Dense (Embeddings) and Sparse (BM25) for precision."""
    def __init__(self, documents: List[Dict]):
        from sentence_transformers import SentenceTransformer
        from rank_bm25 import BM25Okapi
        
        self.docs = documents
        self.model = SentenceTransformer(CONFIG["embedding_model"])
        self.embeddings = self.model.encode([d["content"] for d in documents], normalize_embeddings=True)
        
        tokenized_corpus = [d["content"].lower().split() for d in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        # Dense Rank
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        dense_ranks = np.argsort(-np.dot(self.embeddings, q_emb))
        
        # Sparse Rank
        bm25_ranks = np.argsort(-self.bm25.get_scores(query.lower().split()))
        
        # Reciprocal Rank Fusion (RRF)
        k = 60
        scores = np.zeros(len(self.docs))
        for rank, idx in enumerate(dense_ranks):
            scores[idx] += CONFIG["dense_weight"] * (1 / (rank + k))
        for rank, idx in enumerate(bm25_ranks):
            scores[idx] += (1 - CONFIG["dense_weight"]) * (1 / (rank + k))
            
        return [dict(self.docs[i], rrf_score=scores[i]) for i in np.argsort(-scores)[:top_k]]
```
RAG pipeline and Monitoring for our model
``` python
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
        
        # Calculate Context Recall: Does retrieved text contain key GT words?
        context = " ".join([r["content"] for r in results]).lower()
        gt_tokens = set(ground_truth.lower().split())
        recall = sum(1 for t in gt_tokens if t in context) / len(gt_tokens) if gt_tokens else 0
        
        return RAGResponse(query=user_query, latency_ms=latency, recall_score=recall)
```
Dashboard and Analysis
``` python
def build_monitoring_dashboard(responses: List[RAGResponse]):
    """Generates the FAANG-style performance table and latency summary."""
    df = pd.DataFrame([{
        "Query": r.query[:50] + "...",
        "Recall": f"{r.recall_score:.1%}",
        "Latency (ms)": f"{r.latency_ms:.1f}"
    } for r in responses])
    
    avg_lat = np.mean([r.latency_ms for r in responses])
    avg_recall = np.mean([r.recall_score for r in responses])

    print("\n" + "="*85)
    print(f"{'MARKRAI RAG PERFORMANCE DASHBOARD':^85}")
    print("="*85)
    print(df.to_string(index=False))
    print("-" * 85)
    print(f"AVG LATENCY: {avg_lat:.2f} ms | AVG CONTEXT RECALL: {avg_recall:.2%}")
    print("="*85)

def main():
    docs, tests = load_markrai_dataset()
    engine = HybridSearchEngine(docs)
    pipeline = RAGPipeline(engine)
    
    # Run evaluation on first 10 samples
    results = [pipeline.query(t["question"], t["ground_truth"]) for t in tests[:10]]
    build_monitoring_dashboard(results)

if __name__ == "__main__":
    main()
```
<img width="1765" height="695" alt="image" src="https://github.com/user-attachments/assets/b028bc0f-7e75-4bd9-9446-49fc280d4282" /><img width="1797" height="477" alt="image" src="https://github.com/user-attachments/assets/e108d94b-d08d-483f-a6ab-39da30a0b8cd" /><img width="1789" height="491" alt="image" src="https://github.com/user-attachments/assets/3c7acc9d-b9d7-415e-b40e-6f809e8652f3" />
<img width="842" height="425" alt="image" src="https://github.com/user-attachments/assets/3161fc7c-7b76-4df5-8c6a-f701d292a32c" />

### Insights

- Hybrid Search Superiority: Relying solely on vector search often misses specific keywords; combining it with BM25 through Reciprocal Rank Fusion (RRF) significantly enhances retrieval precision.

- Evaluation Gap: Most RAG projects fail due to a lack of empirical measurement; applying RAGAS metrics (measuring context precision and answer faithfulness) is essential for identifying specific failure points in the pipeline.

- Chunking Impact: The strategy used for document chunking (e.g., fixed-size vs. semantic) has a profound impact on the quality of retrieved context and subsequent generation.

### Recommendations

- Implement Evaluation-First Thinking: Use the AutoRAG evaluation dataset to benchmark the pipeline and report RAGAS scores explicitly to demonstrate senior-level ML engineering depth.

- Domain Specificity: Build the knowledge base around a unique corpus, such as your existing airline complaint data, to create a compelling narrative for technical interviews.Add a Feedback Loop: Incorporate a user feedback mechanism (e.g., helpful/not helpful ratings) to log performance and identify areas for iterative improvement.
  
- Deployment: Deploy the final application on platforms like Hugging Face Spaces or via FastAPI to demonstrate a complete end-to-end production mindset.



