import mlflow
import time
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

class HybridSearchEngine:

    def __init__(self, docs):
        self.docs = docs
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def search(self, query):
        embeddings = self.model.encode([query])
        return embeddings


class RAGPipeline:

    def __init__(self, engine):
        self.engine = engine

    def query(self, question):
        start = time.time()
        result = self.engine.search(question)
        latency = (time.time() - start) * 1000
        return result, latency


def train_pipeline():

    with mlflow.start_run():

        dataset = load_dataset(
            "MarkrAI/AutoRAG-evaluation-2024-LLM-paper-v1",
            "corpus",
            split="train"
        )

        docs = [{"content": d["contents"], "id": d["doc_id"]} for d in dataset]

        engine = HybridSearchEngine(docs)
        rag = RAGPipeline(engine)

        result, latency = rag.query("What is retrieval augmented generation?")

        mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
        mlflow.log_metric("latency_ms", latency)

        mlflow.sklearn.log_model(engine, "rag_model")

        print("Experiment logged")


if __name__ == "__main__":
    train_pipeline()