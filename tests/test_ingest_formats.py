"""PDF / DOCX ingest: real parsing when the extra is installed, a clear skip when not."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragkit import ingest
from ragkit.ingest import load_docs

SHOP = Path(__file__).resolve().parents[1] / "sample_docs" / "shop"


def test_pdf_ingest_extracts_text():
    pytest.importorskip("pypdf")
    docs = load_docs(SHOP / "brewing-guide.pdf")
    assert docs and "cold brew" in docs[0].text.lower()


def test_docx_ingest_extracts_text(tmp_path):
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_heading("Wholesale", level=1)
    d.add_paragraph("Cafes get 20 percent off bulk beans.")
    path = tmp_path / "wholesale.docx"
    d.save(path)
    docs = load_docs(tmp_path)
    assert len(docs) == 1
    assert "# Wholesale" in docs[0].text
    assert "bulk beans" in docs[0].text


def test_missing_extra_skips_with_warning(tmp_path, monkeypatch, capsys):
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4 not really")
    (tmp_path / "b.md").write_text("plain markdown still loads")
    monkeypatch.setitem(ingest.READERS, ".pdf", ingest._missing_reader("pypdf"))
    docs = load_docs(tmp_path)
    assert [Path(d.source).name for d in docs] == ["b.md"]
    err = capsys.readouterr().err
    assert "pip install pypdf" in err
    assert err.count("\n") == 1  # one line, not one per file spam


def test_pdf_chunks_cite_page_numbers():
    pytest.importorskip("pypdf")
    from ragkit import chunk_documents
    chunks = chunk_documents(load_docs(SHOP / "brewing-guide.pdf"))
    assert chunks and all(c.heading.startswith("Page ") for c in chunks)
