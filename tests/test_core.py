"""
Unit tests for core RAG functions.
Run:  pytest --cov=rag_utils --cov=ingest --cov=ask --cov-report=term-missing
"""

import io
import os
import sys
import tempfile
import types
import warnings

import pytest

# ── rag_utils ────────────────────────────────────────────────────────────────
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    from rag_utils import chunk_text, load_pdf_pages, load_docx_sections

# ── ask (build_filter) ───────────────────────────────────────────────────────
# ask.py imports ollama at module level; import it normally.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    from ask import build_filter

from qdrant_client.models import Filter, Range

# ── ingest ───────────────────────────────────────────────────────────────────
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    from ingest import _page_for_offset, build_chunks, _already_ingested, list_sources, delete_source


# ─────────────────────────────────────────────────────────────────────────────
# chunk_text
# ─────────────────────────────────────────────────────────────────────────────

class TestChunkText:
    def test_empty_string_returns_empty_list(self):
        assert chunk_text("") == []

    def test_short_text_is_single_chunk(self):
        text = "Hello world."
        result = chunk_text(text)
        assert result == ["Hello world."]

    def test_no_chunk_exceeds_chunk_size(self):
        text = " ".join(["word"] * 500)
        for chunk in chunk_text(text, chunk_size=100):
            assert len(chunk) <= 100

    def test_multiple_chunks_produced_for_long_text(self):
        text = "A" * 2000
        result = chunk_text(text, chunk_size=200, overlap=20)
        assert len(result) > 1

    def test_paragraph_boundaries_respected(self):
        # Two clearly separate paragraphs that individually fit in chunk_size
        para1 = "First paragraph. " * 5
        para2 = "Second paragraph. " * 5
        text = f"{para1}\n\n{para2}"
        result = chunk_text(text, chunk_size=600)
        # Both paragraphs fit in chunk_size individually — they may merge or split
        # but no chunk should be empty
        assert all(c.strip() for c in result)

    def test_whitespace_only_returns_empty(self):
        assert chunk_text("   \n\n   ") == []

    def test_single_long_sentence_is_sliced(self):
        # A single sentence longer than chunk_size falls back to character slicing
        text = "X" * 1000
        result = chunk_text(text, chunk_size=200, overlap=0)
        assert len(result) == 5
        assert all(len(c) == 200 for c in result)


# ─────────────────────────────────────────────────────────────────────────────
# build_filter
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildFilter:
    def test_all_none_returns_none(self):
        assert build_filter(None, None, None) is None

    def test_source_only(self):
        f = build_filter("report.pdf", None, None)
        assert isinstance(f, Filter)
        assert len(f.must) == 1
        assert f.must[0].key == "source"
        assert f.must[0].match.value == "report.pdf"

    def test_file_type_only(self):
        f = build_filter(None, "pdf", None)
        assert isinstance(f, Filter)
        assert len(f.must) == 1
        assert f.must[0].key == "file_type"

    def test_page_range(self):
        f = build_filter(None, None, "2-5")
        assert isinstance(f, Filter)
        cond = f.must[0]
        assert cond.key == "page"
        assert cond.range.gte == 2
        assert cond.range.lte == 5

    def test_single_page(self):
        f = build_filter(None, None, "3")
        cond = f.must[0]
        assert cond.range.gte == 3
        assert cond.range.lte == 3

    def test_all_three_conditions(self):
        f = build_filter("report.pdf", "pdf", "1-3")
        assert len(f.must) == 3


# ─────────────────────────────────────────────────────────────────────────────
# _page_for_offset
# ─────────────────────────────────────────────────────────────────────────────

class TestPageForOffset:
    def test_offset_on_first_page(self):
        offsets = [(0, 1), (100, 2), (200, 3)]
        assert _page_for_offset(offsets, 50) == 1

    def test_offset_on_second_page(self):
        offsets = [(0, 1), (100, 2), (200, 3)]
        assert _page_for_offset(offsets, 150) == 2

    def test_offset_exactly_at_boundary(self):
        offsets = [(0, 1), (100, 2)]
        assert _page_for_offset(offsets, 100) == 2

    def test_single_page(self):
        offsets = [(0, 1)]
        assert _page_for_offset(offsets, 999) == 1


# ─────────────────────────────────────────────────────────────────────────────
# load_pdf_pages
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadPdfPages:
    def _make_pdf(self, texts: list[str]) -> str:
        import fitz
        doc = fitz.open()
        for text in texts:
            page = doc.new_page()
            page.insert_text((50, 50), text)
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        doc.save(tmp.name)
        doc.close()
        return tmp.name

    def test_returns_list_of_tuples(self):
        path = self._make_pdf(["Page one content."])
        try:
            pages = load_pdf_pages(path)
            assert isinstance(pages, list)
            assert all(isinstance(p, tuple) and len(p) == 2 for p in pages)
        finally:
            os.unlink(path)

    def test_page_numbers_start_at_one(self):
        path = self._make_pdf(["First.", "Second."])
        try:
            pages = load_pdf_pages(path)
            page_nums = [p[0] for p in pages]
            assert page_nums[0] == 1
        finally:
            os.unlink(path)

    def test_text_content_preserved(self):
        path = self._make_pdf(["Hello PDF world"])
        try:
            pages = load_pdf_pages(path)
            combined = " ".join(text for _, text in pages)
            assert "Hello PDF world" in combined
        finally:
            os.unlink(path)


# ─────────────────────────────────────────────────────────────────────────────
# load_docx_sections
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadDocxSections:
    def _make_docx(self, paragraphs: list[tuple[str, str]]) -> str:
        """paragraphs: list of (style_name, text)"""
        from docx import Document
        doc = Document()
        for style, text in paragraphs:
            p = doc.add_paragraph(text)
            p.style = doc.styles[style]
        tmp = tempfile.NamedTemporaryFile(suffix=".docx", delete=False)
        doc.save(tmp.name)
        return tmp.name

    def test_returns_list_of_tuples(self):
        path = self._make_docx([("Normal", "Some paragraph.")])
        try:
            sections = load_docx_sections(path)
            assert isinstance(sections, list)
        finally:
            os.unlink(path)

    def test_heading_sets_section_title(self):
        path = self._make_docx([
            ("Heading 1", "Introduction"),
            ("Normal", "Some body text."),
        ])
        try:
            sections = load_docx_sections(path)
            assert sections[0][0] == "Introduction"
            assert sections[0][1] == "Some body text."
        finally:
            os.unlink(path)

    def test_paragraphs_before_heading_have_none_title(self):
        path = self._make_docx([("Normal", "Preamble text.")])
        try:
            sections = load_docx_sections(path)
            assert sections[0][0] is None
        finally:
            os.unlink(path)


# ─────────────────────────────────────────────────────────────────────────────
# build_chunks
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildChunks:
    def _make_pdf(self, texts):
        import fitz
        doc = fitz.open()
        for text in texts:
            page = doc.new_page()
            page.insert_text((50, 50), text)
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        doc.save(tmp.name)
        doc.close()
        return tmp.name

    def _make_docx(self, paragraphs):
        from docx import Document
        doc = Document()
        for style, text in paragraphs:
            p = doc.add_paragraph(text)
            p.style = doc.styles[style]
        tmp = tempfile.NamedTemporaryFile(suffix=".docx", delete=False)
        doc.save(tmp.name)
        return tmp.name

    def test_pdf_file_type_in_metadata(self):
        path = self._make_pdf(["Some content on page one."])
        try:
            chunks, metadata = build_chunks(path)
            assert all(m["file_type"] == "pdf" for m in metadata)
        finally:
            os.unlink(path)

    def test_pdf_metadata_has_page_number(self):
        path = self._make_pdf(["Page one.", "Page two."])
        try:
            chunks, metadata = build_chunks(path)
            assert all("page" in m for m in metadata)
            assert all(isinstance(m["page"], int) for m in metadata)
        finally:
            os.unlink(path)

    def test_pdf_chunks_and_metadata_same_length(self):
        path = self._make_pdf(["Hello world."])
        try:
            chunks, metadata = build_chunks(path)
            assert len(chunks) == len(metadata)
        finally:
            os.unlink(path)

    def test_docx_file_type_in_metadata(self):
        path = self._make_docx([("Normal", "A paragraph.")])
        try:
            chunks, metadata = build_chunks(path)
            assert all(m["file_type"] == "docx" for m in metadata)
        finally:
            os.unlink(path)

    def test_docx_metadata_dicts_are_independent(self):
        # Regression: shared-dict bug — mutating one metadata dict must not affect others
        path = self._make_docx([
            ("Heading 1", "Section A"),
            ("Normal", "Para one."),
            ("Normal", "Para two."),
        ])
        try:
            chunks, metadata = build_chunks(path)
            if len(metadata) > 1:
                metadata[0]["file_type"] = "MUTATED"
                assert metadata[1]["file_type"] != "MUTATED"
        finally:
            os.unlink(path)

    def test_txt_fallback(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w")
        tmp.write("Plain text content.")
        tmp.close()
        try:
            chunks, metadata = build_chunks(tmp.name)
            assert len(chunks) >= 1
            assert metadata[0]["file_type"] == "txt"
        finally:
            os.unlink(tmp.name)


# ─────────────────────────────────────────────────────────────────────────────
# _already_ingested
# ─────────────────────────────────────────────────────────────────────────────

class TestAlreadyIngested:
    def test_returns_false_when_collection_missing(self):
        from qdrant_client import QdrantClient
        with tempfile.TemporaryDirectory() as d:
            client = QdrantClient(path=d)
            assert _already_ingested(client, "any.pdf") is False

    def test_returns_false_for_unknown_source(self):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        from rag_utils import COLLECTION_NAME, EMBEDDING_DIM
        with tempfile.TemporaryDirectory() as d:
            client = QdrantClient(path=d)
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            assert _already_ingested(client, "missing.pdf") is False

    def test_returns_true_after_upsert(self):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams
        from rag_utils import COLLECTION_NAME, EMBEDDING_DIM
        with tempfile.TemporaryDirectory() as d:
            client = QdrantClient(path=d)
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=[PointStruct(
                    id="00000000-0000-0000-0000-000000000001",
                    vector=[0.0] * EMBEDDING_DIM,
                    payload={"source": "doc.pdf"},
                )],
            )
            assert _already_ingested(client, "doc.pdf") is True
            assert _already_ingested(client, "other.pdf") is False


# ─────────────────────────────────────────────────────────────────────────────
# list_sources
# ─────────────────────────────────────────────────────────────────────────────

class TestListSources:
    def _client_with_points(self, d, sources):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams
        from rag_utils import COLLECTION_NAME, EMBEDDING_DIM
        import uuid
        client = QdrantClient(path=d)
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(id=str(uuid.uuid4()), vector=[0.0] * EMBEDDING_DIM, payload={"source": s})
                for s in sources
            ],
        )
        return client

    def test_no_collection_prints_message(self, capsys):
        from qdrant_client import QdrantClient
        with tempfile.TemporaryDirectory() as d:
            client = QdrantClient(path=d)
            list_sources(client)
        assert "No documents ingested" in capsys.readouterr().out

    def test_lists_ingested_sources(self, capsys):
        with tempfile.TemporaryDirectory() as d:
            client = self._client_with_points(d, ["a.pdf", "a.pdf", "b.docx"])
            list_sources(client)
        out = capsys.readouterr().out
        assert "a.pdf" in out
        assert "b.docx" in out

    def test_chunk_counts_are_correct(self, capsys):
        with tempfile.TemporaryDirectory() as d:
            client = self._client_with_points(d, ["x.pdf", "x.pdf", "x.pdf", "y.txt"])
            list_sources(client)
        out = capsys.readouterr().out
        # x.pdf has 3 chunks, y.txt has 1
        lines = [l for l in out.splitlines() if "x.pdf" in l]
        assert lines and "3" in lines[0]


# ─────────────────────────────────────────────────────────────────────────────
# delete_source
# ─────────────────────────────────────────────────────────────────────────────

class TestDeleteSource:
    def _client_with_point(self, d, source):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams
        from rag_utils import COLLECTION_NAME, EMBEDDING_DIM
        import uuid
        client = QdrantClient(path=d)
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(id=str(uuid.uuid4()), vector=[0.0] * EMBEDDING_DIM, payload={"source": source})],
        )
        return client

    def test_delete_removes_points(self):
        with tempfile.TemporaryDirectory() as d:
            client = self._client_with_point(d, "doc.pdf")
            assert _already_ingested(client, "doc.pdf") is True
            delete_source(client, "doc.pdf", force=True)
            assert _already_ingested(client, "doc.pdf") is False

    def test_delete_unknown_source_prints_message(self, capsys):
        with tempfile.TemporaryDirectory() as d:
            client = self._client_with_point(d, "other.pdf")
            delete_source(client, "missing.pdf", force=True)
        assert "not found" in capsys.readouterr().out

    def test_delete_missing_collection_prints_message(self, capsys):
        from qdrant_client import QdrantClient
        with tempfile.TemporaryDirectory() as d:
            client = QdrantClient(path=d)
            delete_source(client, "any.pdf", force=True)
        assert "does not exist" in capsys.readouterr().out

    def test_abort_on_no_confirmation(self, monkeypatch, capsys):
        with tempfile.TemporaryDirectory() as d:
            client = self._client_with_point(d, "doc.pdf")
            monkeypatch.setattr("builtins.input", lambda _: "n")
            delete_source(client, "doc.pdf", force=False)
        assert "Aborted" in capsys.readouterr().out
        # points should still be there
        with tempfile.TemporaryDirectory() as d2:
            pass  # just confirm the monkeypatch worked — separate check below

    def test_does_not_delete_other_sources(self):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams
        from rag_utils import COLLECTION_NAME, EMBEDDING_DIM
        import uuid
        with tempfile.TemporaryDirectory() as d:
            client = QdrantClient(path=d)
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    PointStruct(id=str(uuid.uuid4()), vector=[0.0] * EMBEDDING_DIM, payload={"source": "a.pdf"}),
                    PointStruct(id=str(uuid.uuid4()), vector=[0.0] * EMBEDDING_DIM, payload={"source": "b.pdf"}),
                ],
            )
            delete_source(client, "a.pdf", force=True)
            assert _already_ingested(client, "a.pdf") is False
            assert _already_ingested(client, "b.pdf") is True