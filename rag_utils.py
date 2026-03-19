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


def load_pdf(path: str) -> str:
    import fitz  # PyMuPDF

    doc = fitz.open(path)
    return "".join(page.get_text() for page in doc)


def load_docx(path: str) -> str:
    from docx import Document

    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def load_document(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(str(p))
    if suffix in (".docx", ".doc"):
        return load_docx(str(p))
    if suffix in (".md", ".txt", ".rst", ""):
        return p.read_text(encoding="utf-8", errors="replace")
    raise ValueError(f"Unsupported file type '{suffix}'. Supported: .pdf, .docx, .md, .txt")


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 80) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks