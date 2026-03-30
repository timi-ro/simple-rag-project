"""
Ingest a PDF or Word document into Qdrant.

Usage:
    python ingest.py path/to/document.pdf
    python ingest.py path/to/document.docx
"""

import sys
import uuid
import warnings

from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
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


def _page_for_offset(page_offsets: list[tuple[int, int]], offset: int) -> int:
    """Return the page number that contains the given character offset."""
    page = page_offsets[0][1]
    for start, pnum in page_offsets:
        if start <= offset:
            page = pnum
        else:
            break
    return page


def build_chunks(file_path: str) -> tuple[list[str], list[dict]]:
    """Return (chunks, metadata) where metadata[i] holds payload fields for chunk i."""
    suffix = file_path.rsplit(".", 1)[-1].lower()

    if suffix == "pdf":
        pages = load_pdf_pages(file_path)
        # Concatenate all pages before chunking so concepts spanning page
        # boundaries are not split; record where each page starts.
        page_offsets: list[tuple[int, int]] = []
        parts: list[str] = []
        pos = 0
        for page_num, page_text in pages:
            page_offsets.append((pos, page_num))
            parts.append(page_text)
            pos += len(page_text) + 2  # +2 for the "\n\n" join separator
        full_text = "\n\n".join(parts)

        chunks, metadata = [], []
        search_from = 0
        for chunk in chunk_text(full_text):
            idx = full_text.find(chunk, search_from)
            chunk_start = idx if idx != -1 else search_from
            search_from = chunk_start + len(chunk)
            chunks.append(chunk)
            metadata.append({
                "page": _page_for_offset(page_offsets, chunk_start),
                "section_title": None,
                "file_type": "pdf",
            })
        return chunks, metadata

    if suffix in ("docx", "doc"):
        sections = load_docx_sections(file_path)
        # Group all paragraphs under the same section heading into one block
        # before chunking so short paragraphs are merged across boundaries.
        from itertools import groupby
        chunks, metadata = [], []
        for section_title, paras in groupby(sections, key=lambda x: x[0]):
            section_text = "\n\n".join(p for _, p in paras)
            for chunk in chunk_text(section_text):
                chunks.append(chunk)
                metadata.append({"page": None, "section_title": section_title, "file_type": suffix})
        return chunks, metadata

    text = load_document(file_path)
    chunks = chunk_text(text)
    return chunks, [{"page": None, "section_title": None, "file_type": suffix} for _ in chunks]


def _already_ingested(client, source_name: str) -> bool:
    """Return True if any points with this source filename exist in the collection."""
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME not in existing:
        return False
    results, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source_name))]),
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    return len(results) > 0


def ingest(file_path: str) -> None:
    print(f"Loading '{file_path}'...")
    source_name = file_path.rsplit("/", 1)[-1]

    client = get_qdrant_client()
    ensure_collection(client)

    if _already_ingested(client, source_name):
        print(f"  '{source_name}' is already ingested — skipping. Delete its points first to re-ingest.")
        return

    chunks, metadata = build_chunks(file_path)
    print(f"  Split into {len(chunks)} chunks")

    print("  Embedding chunks (first run downloads the model ~90 MB)...")
    embedder = get_embedder()
    embeddings = embedder.encode(chunks, show_progress_bar=True, batch_size=64)

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