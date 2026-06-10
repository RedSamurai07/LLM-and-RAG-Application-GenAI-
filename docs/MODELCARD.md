# 🤖 Model Card — LLM and RAG Application (GenAI)

This model card documents the architecture, evaluation metrics, configuration parameters, and ethical considerations for the Hybrid RAG Pipeline deployed in this project.

---

## Model Overview

| Property | Details |
|---|---|
| **Model Type** | Hybrid RAG Pipeline (Dense + Sparse Search) |
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Sparse Retriever** | BM25 (Okapi BM25) |
| **Fusion Strategy** | Reciprocal Rank Fusion (RRF) |
| **Evaluation Dataset** | `MarkrAI/AutoRAG-evaluation-2024-LLM-paper-v1` |
| **Trained / Indexed On** | 8,500 research corpus nodes |
| **Test Cases** | 520 QA pairs |
| **Repository** | [LLM-and-RAG-Application-GenAI-](https://github.com/RedSamurai07/LLM-and-RAG-Application-GenAI-) |

---

## Evaluation Metrics

| Metric | Value |
|---|---|
| **Average Context Recall** | ~85%+ (on 10-sample benchmark) |
| **Average Latency** | < 200 ms per query |
| **Dense Weight (RRF)** | 0.7 |
| **Sparse Weight (RRF)** | 0.3 |
| **Top-K Retrieved** | 5 |

> Metrics are computed using the custom `RAGPipeline` evaluation framework. Context Recall measures the proportion of ground-truth answer tokens present in the retrieved context window.

---

## Model Configuration Parameters

```json
{
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "dataset_name": "MarkrAI/AutoRAG-evaluation-2024-LLM-paper-v1",
  "dense_weight": 0.7,
  "top_k": 5,
  "rrf_k_constant": 60,
  "normalize_embeddings": true,
  "bm25_tokenization": "lowercase_split"
}
```

---

## Architecture Summary

The pipeline consists of three core stages:

**1. Ingestion**
Documents are loaded from the Hugging Face dataset corpus. Each document's text content is tokenized (for BM25) and encoded into a 384-dimensional dense vector using the Sentence Transformer model.

**2. Hybrid Search (Query Time)**
- **Dense Retrieval:** The query is encoded and dot-product similarity is computed against all corpus embeddings.
- **Sparse Retrieval:** BM25 scores are computed against the tokenized corpus.
- **Reciprocal Rank Fusion (RRF):** Dense and sparse ranked lists are merged using the RRF formula with a configurable `dense_weight` of 0.7.

**3. Evaluation**
Retrieved contexts are evaluated against ground-truth answers using Context Recall — the proportion of ground-truth tokens found in the aggregated retrieved passages.

---

## Intended Use

- **Primary Use Case:** Domain-specific Q&A over a fixed document corpus (e.g., research papers, airline policy PDFs).
- **Supported Query Types:** Factual, definition-based, and explanatory questions grounded in the indexed knowledge base.
- **Target Users:** ML engineers, data scientists, and developers building production RAG systems.

---

## Limitations

- The pipeline does not include a generative LLM component in the current iteration; it focuses on retrieval quality and context recall.
- Performance is bounded by the quality and coverage of the indexed document corpus.
- Very short or highly ambiguous queries may underperform due to limited BM25 keyword signal.
- The evaluation uses a proxy recall metric; full RAGAS evaluation (faithfulness, answer relevancy) is recommended for production deployment.

---

## Ethical Considerations

- **Data Privacy:** The pipeline operates on publicly available research datasets. No personally identifiable information (PII) is stored or indexed.
- **Bias:** Retrieval quality may vary across topics depending on corpus coverage. Underrepresented domains may yield lower recall scores.
- **Hallucination Risk:** As a retrieval-only pipeline, the system surfaces grounded passages rather than generating free-form text, reducing (but not eliminating) hallucination risk when paired with a downstream LLM.

---

## Dependencies

| Package | Purpose |
|---|---|
| `sentence-transformers` | Dense embedding generation |
| `rank-bm25` | Sparse BM25 keyword retrieval |
| `chromadb` | Vector store for embeddings |
| `langchain` | Pipeline orchestration |
| `datasets` | Hugging Face dataset loading |
| `numpy` / `pandas` | Numerical operations and data handling |
| `matplotlib` | Performance dashboard visualisation |

---

## Citation

If you use this pipeline in your work, please reference:

```
@misc{redsamurai07_llm_rag_2024,
  author       = {RedSamurai07},
  title        = {LLM and RAG Application (GenAI) — Hybrid Search Pipeline},
  year         = {2024},
  publisher    = {GitHub},
  url          = {https://github.com/RedSamurai07/LLM-and-RAG-Application-GenAI-}
}
```