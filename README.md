# Simple RAG Project

A fully local Retrieval-Augmented Generation (RAG) system. Embed PDF, Word, and Markdown documents into a vector database and ask questions about them — no cloud, no API keys.

> **Archived as learning artifact. Pivoted to frontend role at Scorealytics, May 2026.**

---

## What it does

1. **Ingest** — Load a document, split it into chunks, embed each chunk with a local sentence-transformer model, and store the vectors + metadata in a local Qdrant database.
2. **Ask** — Embed the question, retrieve the most relevant chunks by cosine similarity, and pass them as context to a local LLM (Ollama) to generate an answer.
3. **Filter** — Scope retrieval to a specific file, file type, or page range using Qdrant payload filters.
4. **Manage** — List all ingested sources with chunk counts, or delete a source to re-ingest.

---

## Architecture

```
simple-rag-project/
├── ingest.py       # CLI: load → chunk → embed → upsert into Qdrant
├── ask.py          # CLI: embed question → retrieve chunks → generate answer
├── rag_utils.py    # Shared: document loaders, chunker, embedder, Qdrant client
├── eval/           # Retrieval evaluation: Recall@k and MRR metrics
├── tests/          # pytest unit tests (~66% coverage)
└── qdrant_db/      # Local vector DB (auto-created, git-ignored)
```

**Data flow:**

```
Document file
    → loader (PDF / DOCX / MD)          rag_utils.py
    → paragraph- and section-aware chunker
    → sentence-transformer encoder      (all-MiniLM-L6-v2, 384-dim)
    → Qdrant upsert                     ingest.py

Question
    → sentence-transformer encoder
    → Qdrant cosine search              (score_threshold=0.3, top-5)
    → prompt assembly
    → Ollama LLM                        ask.py
    → streamed answer
```

**Stack:**

| Layer      | Tool                                                        |
|------------|-------------------------------------------------------------|
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers                |
| Vector DB  | Qdrant (local file, no Docker)                              |
| LLM        | Ollama (`llama3.2` default, swappable via `OLLAMA_MODEL`)   |

---

## Setup

```bash
pip install -r requirements.txt
ollama pull llama3.2
ollama serve
```

## Usage

```bash
# Ingest
python ingest.py path/to/document.pdf
python ingest.py path/to/notes.md

# Ask
python ask.py "What are the key risks?"
python ask.py --source report.pdf --pages 1-3 "What is covered here?"
python ask.py --type docx "Summarize the main points"

# Manage
python ingest.py --list
python ingest.py --delete report.pdf

# Use a different model
OLLAMA_MODEL=mistral python ask.py "What is this about?"
```

---

## What I learned

- **Chunking strategy matters more than model choice.** Naive character-based chunking split sentences mid-thought and hurt retrieval quality noticeably. Switching to paragraph-boundary chunking with sentence-level fallback improved answer coherence without changing anything else.
- **Metadata in the vector store unlocks filtering.** Storing `page`, `section_title`, and `file_type` alongside each vector made it possible to scope queries to specific parts of a document — a feature that's nearly impossible to bolt on after the fact.
- **Score thresholds prevent hallucination.** Without a minimum similarity cutoff, the LLM received low-relevance chunks and confidently fabricated answers. A `score_threshold=0.3` filter meant the model received nothing rather than something misleading.
- **Retrieval evaluation is a different discipline from generation evaluation.** Measuring Recall@k and MRR on a hand-labeled question set revealed retrieval failures that weren't visible when eyeballing LLM answers — the LLM was often papering over bad retrieval with plausible-sounding text.
- **Local-first tooling is viable for real workloads.** Running embeddings, vector search, and a 3B LLM entirely on a laptop was fast enough for document-scale tasks, with no latency from network calls or rate limits.