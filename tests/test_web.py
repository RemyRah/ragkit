"""The chat web UI: exercise the real request handler over a loopback socket."""

import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragkit import chunk_documents
from ragkit.ingest import Document
from ragkit.retriever import Retriever
from ragkit.web import make_server

EVIL = '<script>alert("x")</script>'


@pytest.fixture()
def server(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    docs = [
        Document(source=f"policy{EVIL}.md", text=f"# Refunds\n\nFull refund within 30 days. {EVIL}"),
        Document(source="shipping.md", text="# Shipping\n\nOrders over $40 ship free."),
    ]
    srv = make_server(Retriever(chunk_documents(docs)), title=f"Shop {EVIL}", port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _get(url):
    with urllib.request.urlopen(url) as r:
        return r.status, r.headers.get("Content-Type"), r.read().decode()


def _ask(base, payload):
    req = urllib.request.Request(
        base + "/api/ask",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def test_page_serves_and_escapes_title(server):
    status, ctype, body = _get(server + "/")
    assert status == 200 and ctype.startswith("text/html")
    assert EVIL not in body
    assert "&lt;script&gt;" in body


def test_ask_returns_passages_and_sources_without_key(server):
    data = _ask(server, {"question": "can I get my money back?"})
    assert data["answer"] is None
    assert "ANTHROPIC_API_KEY" in data["note"]
    assert data["sources"][0]["citation"].startswith("policy")
    assert "refund" in data["sources"][0]["text"].lower()


def test_html_fallback_escapes_document_content(server):
    _, _, body = _get(server + "/?q=refund")
    assert "Full refund within 30 days" in body
    assert EVIL not in body
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;" in body


def test_ask_uses_claude_when_available(server, monkeypatch):
    import ragkit.web as web

    monkeypatch.setattr(web, "synthesize", lambda q, hits, model=None: ("Yes, within 30 days [1].", "ok"))
    data = _ask(server, {"question": "refund?"})
    assert data["answer"] == "Yes, within 30 days [1]."


def test_bad_requests_are_rejected(server):
    with pytest.raises(urllib.error.HTTPError) as e:
        _ask(server, {"question": "   "})
    assert e.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(server + "/nope")
    assert e.value.code == 404
