from fastapi import FastAPI
from app.rag_pipeline import HybridSearchEngine, RAGPipeline

app = FastAPI()

engine = HybridSearchEngine(["Example document about RAG"])
pipeline = RAGPipeline(engine)


@app.get("/")
def health():
    return {"status": "running"}


@app.post("/query")
def query_model(question: str):

    result, latency = pipeline.query(question)

    return {
        "question": question,
        "latency_ms": latency,
        "result": str(result)
    }