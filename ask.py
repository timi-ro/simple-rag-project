"""
Ask a question about your ingested documents.

Usage:
    python ask.py "What is this document about?"
    python ask.py "Summarize the key points"

Set OLLAMA_MODEL env var to change the model (default: llama3.2).
Requires Ollama running: https://ollama.com
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys

import ollama

from rag_utils import (
    COLLECTION_NAME,
    get_embedder,
    get_qdrant_client,
)

TOP_K = 5
MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")


def ask(question: str) -> None:
    embedder = get_embedder()
    query_vector = embedder.encode(question).tolist()

    client = get_qdrant_client()

    try:
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=TOP_K,
        ).points
    except Exception:
        print("No documents found. Run 'python ingest.py <file>' first.")
        sys.exit(1)

    if not results:
        print("No relevant chunks found. Try ingesting more documents.")
        return

    context_blocks = []
    for i, hit in enumerate(results, 1):
        source = hit.payload.get("source", "unknown")
        text = hit.payload.get("text", "")
        score = hit.score
        context_blocks.append(f"[{i}] (source: {source}, relevance: {score:.2f})\n{text}")

    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""You are a helpful assistant. Answer the question using only the provided context.
If the answer is not in the context, say "I couldn't find that in the documents."

Context:
{context}

Question: {question}"""

    print(f"\nQ: {question}\n")
    print(f"A: ", end="", flush=True)

    for chunk in ollama.generate(model=MODEL, prompt=prompt, stream=True):
        print(chunk["response"], end="", flush=True)

    print("\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python ask.py "Your question here"')
        sys.exit(1)
    ask(" ".join(sys.argv[1:]))