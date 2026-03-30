# Simple RAG Project

A local Retrieval-Augmented Generation (RAG) system that lets you embed PDF, Word, and Markdown documents into a vector database and ask questions about them, entirely offline.

**Stack:**
- **Embeddings:** `all-MiniLM-L6-v2` via sentence-transformers (local, no API key)
- **Vector DB:** Qdrant (local file storage, no Docker needed)
- **LLM:** Ollama (local)

---

## Setup

```bash
pip install -r requirements.txt
```

Make sure Ollama is running with a model pulled:

```bash
ollama pull llama3.2
ollama serve
```

---

## Usage

**Ingest a document:**

```bash
python ingest.py path/to/document.pdf
python ingest.py path/to/document.docx
python ingest.py path/to/notes.md
```

**Ask a question:**

```bash
python ask.py "What is this document about?"
python ask.py "Summarize the key points"
```

**Filter results by file, type, or page range:**

```bash
python ask.py --source report.pdf "What are the risks?"
python ask.py --type docx "What is the summary?"
python ask.py --source report.pdf --pages 1-3 "What is on the first pages?"
```

**Use a different Ollama model:**

```bash
OLLAMA_MODEL=mistral python ask.py "What are the main topics?"
```

---

## Project Structure

```
simple-rag-project/
├── ingest.py       # Load, chunk, embed, and store documents in Qdrant
├── ask.py          # Embed a question, retrieve chunks, generate an answer
├── rag_utils.py    # Shared utilities: loaders, chunker, embedder, Qdrant client
├── requirements.txt
└── qdrant_db/      # Local vector database (auto-created, git-ignored)
```

---

## Future Improvement in the Curriculum

- [x] **Page tracking** — Extract text per page in `load_pdf` and store the page number in each chunk's Qdrant payload.
- [x] **Section awareness** — Parse `docx` heading styles (`Heading 1`, `Heading 2`) to attach a `section_title` to each paragraph chunk.
- [x] **Smarter chunking** — Replace character-based chunking with heading-aware and paragraph-boundary strategies to avoid splitting mid-sentence or mid-concept.
- [x] **Richer payload** — Extend the Qdrant point payload from `{text, source, chunk_id}` to include `page`, `section_title`, and `file_type`.
- [x] **Payload filtering** — Use Qdrant's `Filter` with `must` conditions to scope searches to a specific file, file type, or page range.
- [ ] **Retrieval evaluation** — Build a test set of question/answer pairs and measure retrieval precision (are the right chunks in the top-k?) using metrics like MRR and Recall@k.

---

## Mentor Feedback — Planned Fixes

### Bugs (correctness & safety)

- [x] **Shared-dict reference** (`ingest.py:53`) — `[{...}] * n` creates `n` references to the same dict; replace with a list comprehension so each chunk gets its own metadata dict.
- [x] **Bare `except Exception`** (`ingest.py`, `ask.py`) — Replace with specific exceptions (`FileNotFoundError`, `QdrantException`, etc.) so real errors aren't silently swallowed.
- [x] **Scoped warning suppression** (`ingest.py`, `ask.py`) — Replace the global `warnings.filterwarnings("ignore")` with targeted `warnings.catch_warnings()` blocks around the specific imports that emit noise.

### Retrieval quality

- [x] **Cross-page chunking for PDFs** — Currently each page is chunked independently, splitting concepts that span a page boundary. Concatenate full-document text first, then chunk, while still recording the starting page number per chunk.
- [x] **Group DOCX paragraphs before chunking** — Single paragraphs are passed to `chunk_text` one at a time; short paragraphs never get merged across section boundaries. Collect all paragraphs under a section into one block before chunking.

### Data integrity

- [x] **Duplicate detection on ingest** — Before upserting, query Qdrant for existing points with the same `source` filename and skip (or warn) if they are already present, preventing doubled results on re-ingest.

### Testing

- [ ] **Unit tests for core functions** — Add `tests/` with pytest cases covering all main functionalities: `chunk_text` (boundary conditions), `build_filter` (filter construction), `load_pdf_pages`, `load_docx_sections`, `build_chunks`, and `ingest`. Minimum coverage target: **40%** (enforced via `pytest --cov`).