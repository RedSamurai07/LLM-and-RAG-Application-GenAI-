import numpy as np
import pandas as pd
from typing import List
from src.rag_pipeline import RAGResponse


def build_monitoring_dashboard(responses: List[RAGResponse]) -> dict:
    """Generates the performance table and returns summary stats."""
    df = pd.DataFrame([
        {
            "Query": r.query[:50] + "...",
            "Recall": f"{r.recall_score:.1%}",
            "Latency (ms)": f"{r.latency_ms:.1f}",
        }
        for r in responses
    ])

    avg_lat = np.mean([r.latency_ms for r in responses])
    avg_recall = np.mean([r.recall_score for r in responses])

    print("\n" + "=" * 85)
    print(f"{'MARKRAI RAG PERFORMANCE DASHBOARD':^85}")
    print("=" * 85)
    print(df.to_string(index=False))
    print("-" * 85)
    print(f"AVG LATENCY: {avg_lat:.2f} ms | AVG CONTEXT RECALL: {avg_recall:.2%}")
    print("=" * 85)

    return {"avg_latency_ms": avg_lat, "avg_recall": avg_recall, "dataframe": df}
