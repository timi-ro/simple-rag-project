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
    load_docx_sections,
    load_pdf_pages,
)


def build_chunks(file_path: str) -> tuple[list[str], list[dict]]:
    """Return (chunks, metadata) where metadata[i] holds payload fields for chunk i."""
    suffix = file_path.rsplit(".", 1)[-1].lower()

    if suffix == "pdf":
        pages = load_pdf_pages(file_path)
        chunks, metadata = [], []
        for page_num, page_text in pages:
            for chunk in chunk_text(page_text):
                chunks.append(chunk)
                metadata.append({"page": page_num, "section_title": None})
        return chunks, metadata

    if suffix in ("docx", "doc"):
        sections = load_docx_sections(file_path)
        chunks, metadata = [], []
        for section_title, para_text in sections:
            for chunk in chunk_text(para_text):
                chunks.append(chunk)
                metadata.append({"page": None, "section_title": section_title})
        return chunks, metadata

    text = load_document(file_path)
    chunks = chunk_text(text)
    return chunks, [{"page": None, "section_title": None}] * len(chunks)


def ingest(file_path: str) -> None:
    print(f"Loading '{file_path}'...")
    chunks, metadata = build_chunks(file_path)
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
            payload={"text": chunk, "source": source_name, "chunk_id": i, **meta},
        )
        for i, (chunk, emb, meta) in enumerate(zip(chunks, embeddings, metadata))
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"  Stored {len(points)} chunks from '{source_name}' in Qdrant ✓")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <document.pdf|document.docx>")
        sys.exit(1)
    ingest(sys.argv[1])