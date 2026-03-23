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
- [ ] **Smarter chunking** — Replace character-based chunking with heading-aware and paragraph-boundary strategies to avoid splitting mid-sentence or mid-concept.
- [ ] **Richer payload** — Extend the Qdrant point payload from `{text, source, chunk_id}` to include `page`, `section_title`, and `file_type`.
- [ ] **Payload filtering** — Use Qdrant's `Filter` with `must` conditions to scope searches to a specific file, file type, or page range.
- [ ] **Retrieval evaluation** — Build a test set of question/answer pairs and measure retrieval precision (are the right chunks in the top-k?) using metrics like MRR and Recall@k.