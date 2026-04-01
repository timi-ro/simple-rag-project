from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
QDRANT_PATH = "./qdrant_db"


def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(path=QDRANT_PATH)


def ensure_collection(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


def load_pdf_pages(path: str) -> list[tuple[int, str]]:
    # TODO: extract section/heading titles to improve chunk context
    import fitz  # PyMuPDF

    doc = fitz.open(path)
    return [(i + 1, page.get_text()) for i, page in enumerate(doc) if page.get_text().strip()]


def load_docx_sections(path: str) -> list[tuple[str | None, str]]:
    """Return [(section_title, paragraph_text), ...] tracking Heading styles."""
    # TODO: preserve page number metadata if needed
    from docx import Document

    doc = Document(path)
    current_section = None
    results = []
    for p in doc.paragraphs:
        if not p.text.strip():
            continue
        if p.style.name.startswith("Heading"):
            current_section = p.text.strip()
        else:
            results.append((current_section, p.text.strip()))
    return results


def load_docx(path: str) -> str:
    from docx import Document

    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def load_document(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    suffix = p.suffix.lower()
    if suffix in (".docx", ".doc"):
        return load_docx(str(p))
    if suffix in (".md", ".txt", ".rst", ""):
        return p.read_text(encoding="utf-8", errors="replace")
    raise ValueError(f"Unsupported file type '{suffix}'. Supported: .pdf, .docx, .md, .txt")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on punctuation boundaries."""
    import re
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 80) -> list[str]:
    """
    Paragraph- and sentence-aware chunking:
    1. Split on paragraph boundaries first.
    2. Group short paragraphs together up to chunk_size.
    3. Split oversized paragraphs on sentence boundaries.
    4. Fall back to character slicing only for sentences that exceed chunk_size.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Flatten into sentence-level units so we never split mid-sentence
    units: list[str] = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            units.append(para)
        else:
            units.extend(_split_sentences(para))

    chunks: list[str] = []
    current = ""

    for unit in units:
        # Unit alone exceeds chunk_size — slice it with overlap as a last resort
        if len(unit) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(unit):
                chunks.append(unit[start : start + chunk_size])
                start += chunk_size - overlap
            continue

        candidate = f"{current}\n\n{unit}".strip() if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = unit

    if current:
        chunks.append(current)

    return chunks