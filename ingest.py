"""
Ingest a PDF or Word document into Qdrant.

Usage:
    python ingest.py path/to/document.pdf
    python ingest.py path/to/document.docx
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import uuid

from qdrant_client.models import PointStruct

from rag_utils import (
    COLLECTION_NAME,
    chunk_text,
    ensure_collection,
    get_embedder,
    get_qdrant_client,
    load_document,
)

def ingest(file_path: str) -> None:
    print(f"Loading '{file_path}'...")
    text = load_document(file_path)
    print(f"  Extracted {len(text):,} characters")

    chunks = chunk_text(text)
    print(f"  Split into {len(chunks)} chunks")

    print("  Embedding chunks (first run downloads the model ~90 MB)...")
    embedder = get_embedder()
    embeddings = embedder.encode(chunks, show_progress_bar=True, batch_size=64)

    client = get_qdrant_client()
    ensure_collection(client)

    source_name = file_path.rsplit("/", 1)[-1]
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb.tolist(),
            payload={"text": chunk, "source": source_name, "chunk_id": i},
        )
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"  Stored {len(points)} chunks from '{source_name}' in Qdrant ✓")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <document.pdf|document.docx>")
        sys.exit(1)
    ingest(sys.argv[1])