"""End-to-end tests for the ragkit pipeline. Run: python -m pytest -q"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragkit import chunk_documents, load_docs
from ragkit.ingest import Document
from ragkit.retriever import Retriever, tokenize


def test_load_sample_docs():
    docs = load_docs(Path(__file__).resolve().parents[1] / "sample_docs")
    assert docs, "expected sample docs to load"
    assert all(d.text.strip() for d in docs)


def test_chunking_preserves_source_and_splits():
    doc = Document(source="mem.md", text="# A\n\npara one.\n\n# B\n\npara two.")
    chunks = chunk_documents([doc], target_chars=50, overlap_chars=0)
    assert len(chunks) >= 2
    assert {c.heading for c in chunks} == {"A", "B"}
    assert all(c.source == "mem.md" for c in chunks)


def test_retriever_ranks_relevant_chunk_first():
    docs = [
        Document(source="a.md", text="Retrieval uses TF-IDF cosine similarity to rank chunks."),
        Document(source="b.md", text="Chunking splits documents on Markdown headings."),
    ]
    retriever = Retriever(chunk_documents(docs))
    hits = retriever.search("how does ranking similarity work", k=2)
    assert hits, "expected at least one hit"
    assert "cosine" in hits[0].chunk.text.lower()


def test_empty_query_returns_no_hits():
    docs = [Document(source="a.md", text="some indexed content here")]
    retriever = Retriever(chunk_documents(docs))
    assert retriever.search("the a an of to", k=3) == []


def test_tokenize_drops_stopwords():
    assert "the" not in tokenize("the cosine of the angle")
    assert "cosine" in tokenize("the cosine of the angle")
