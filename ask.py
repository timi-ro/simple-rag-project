"""
Ask a question about your ingested documents.

Usage:
    python ask.py "What is this document about?"
    python ask.py "Summarize the key points"

Optional filters (can be combined):
    --source report.pdf          scope to a specific file
    --type pdf                   scope to a file type (pdf, docx, md, ...)
    --pages 2-5                  scope to a page range (PDF only)

Examples:
    python ask.py --source report.pdf "What are the risks?"
    python ask.py --type docx "What is the summary?"
    python ask.py --source report.pdf --pages 1-3 "What is on the first pages?"

Set OLLAMA_MODEL env var to change the model (default: llama3.2).
Requires Ollama running: https://ollama.com
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import sys

import ollama
from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

from rag_utils import (
    COLLECTION_NAME,
    get_embedder,
    get_qdrant_client,
)

TOP_K = 5
MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")


def build_filter(source: str | None, file_type: str | None, pages: str | None) -> Filter | None:
    conditions = []

    if source:
        conditions.append(FieldCondition(key="source", match=MatchValue(value=source)))

    if file_type:
        conditions.append(FieldCondition(key="file_type", match=MatchValue(value=file_type)))

    if pages:
        parts = pages.split("-")
        if len(parts) == 2:
            start, end = int(parts[0]), int(parts[1])
        else:
            start = end = int(parts[0])
        conditions.append(FieldCondition(key="page", range=Range(gte=start, lte=end)))

    return Filter(must=conditions) if conditions else None


def ask(question: str, source: str | None, file_type: str | None, pages: str | None) -> None:
    embedder = get_embedder()
    query_vector = embedder.encode(question).tolist()

    client = get_qdrant_client()
    query_filter = build_filter(source, file_type, pages)

    try:
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
            limit=TOP_K,
        ).points
    except Exception:
        print("No documents found. Run 'python ingest.py <file>' first.")
        sys.exit(1)

    if not results:
        print("No relevant chunks found. Try ingesting more documents or relaxing your filters.")
        return

    context_blocks = []
    for i, hit in enumerate(results, 1):
        source_val = hit.payload.get("source", "unknown")
        text = hit.payload.get("text", "")
        page = hit.payload.get("page")
        section = hit.payload.get("section_title")
        file_type_val = hit.payload.get("file_type", "unknown")
        score = hit.score
        location = f"p.{page}" if page is not None else "n/a"
        section_str = f", section: {section}" if section else ""
        context_blocks.append(f"[{i}] (source: {source_val}, type: {file_type_val}, page: {location}{section_str}, relevance: {score:.2f})\n{text}")

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
    parser = argparse.ArgumentParser(description="Ask questions about your ingested documents.")
    parser.add_argument("question", nargs="+", help="The question to ask")
    parser.add_argument("--source", help="Filter by filename (e.g. report.pdf)")
    parser.add_argument("--type", dest="file_type", help="Filter by file type (e.g. pdf, docx, md)")
    parser.add_argument("--pages", help="Filter by page range (e.g. 2-5 or 3)")
    args = parser.parse_args()

    ask(" ".join(args.question), args.source, args.file_type, args.pages)