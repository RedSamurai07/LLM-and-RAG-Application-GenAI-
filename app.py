import gradio as gr
import numpy as np
from app.rag_pipeline import HybridSearchEngine, RAGPipeline
from datasets import load_dataset

# Initialize the engine and pipeline with a subset of the data for the demo
print("Loading dataset...")
dataset = load_dataset(
    "MarkrAI/AutoRAG-evaluation-2024-LLM-paper-v1",
    "corpus",
    split="train"
)
# Using a subset for the live demo to ensure quick response times
subset = dataset.select(range(min(500, len(dataset))))
docs = [{"content": d["contents"], "id": d["doc_id"]} for d in subset]
print(f"Initializing Hybrid Search Engine with {len(docs)} documents...")
engine = HybridSearchEngine(docs)
pipeline = RAGPipeline(engine)

def predict(question):
    if not question:
        return "Please enter a question.", "0.00 ms"
    
    result, latency = pipeline.query(question)
    
    # Format the result nicely
    if isinstance(result, (list, np.ndarray)):
        # If it's a list of results (from hybrid search)
        formatted_result = ""
        for i, res in enumerate(result[:3]): # Show top 3
            if isinstance(res, dict) and "content" in res:
                formatted_result += f"Result {i+1}:\n{res['content'][:500]}...\n\n"
            else:
                formatted_result += f"Result {i+1}:\n{str(res)[:500]}...\n\n"
        return formatted_result or "No results found.", f"{latency:.2f} ms"
    
    return str(result), f"{latency:.2f} ms"

# Create the Gradio Interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        label="Query", 
        placeholder="e.g., What is retrieval augmented generation?",
        lines=2
    ),
    outputs=[
        gr.Textbox(label="Top Retrieved Context", lines=10),
        gr.Textbox(label="Retrieval Latency")
    ],
    title="🚀 Hybrid RAG Search Engine",
    description="""
    This application demonstrates a **Hybrid Retrieval-Augmented Generation (RAG)** pipeline. 
    It combines **Dense Semantic Search** (using Sentence Transformers) with **Sparse Keyword Search** (BM25) 
    to provide highly accurate document retrieval.
    """,
    examples=[
        ["What is retrieval augmented generation?"],
        ["How does hybrid search improve precision?"],
        ["Explain Reciprocal Rank Fusion (RRF)."]
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()
