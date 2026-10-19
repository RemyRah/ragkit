"""Private mode: an OpenAI-compatible local server answers; if it's down we
fall back to passages instead of crashing."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from ragkit import chunk_documents, load_docs
from ragkit.answer import synthesize
from ragkit.retriever import Retriever


def _hits():
    chunks = chunk_documents(load_docs("sample_docs/shop"))
    return Retriever(chunks).search("can I get my money back?", k=3)


def test_local_server_answers(monkeypatch):
    seen = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            out = json.dumps({"choices": [{"message": {"content": "Yes, email us [1]."}}]}).encode()
            self.send_response(200); self.end_headers(); self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.handle_request, daemon=True).start()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("RAGKIT_LLM_URL", f"http://127.0.0.1:{srv.server_port}/v1")
    answer, note = synthesize("can I get my money back?", _hits())
    assert answer == "Yes, email us [1]." and "local model" in note
    assert "refund" in seen["body"]["messages"][1]["content"].lower()


def test_local_server_down_falls_back(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("RAGKIT_LLM_URL", "http://127.0.0.1:9/v1")
    answer, note = synthesize("refund?", _hits())
    assert answer is None and "unavailable" in note


def test_empty_local_answer_falls_back(monkeypatch):
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            out = json.dumps({"choices": [{"message": {"content": "<think>hmm</think>"}}]}).encode()
            self.send_response(200); self.end_headers(); self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.handle_request, daemon=True).start()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("RAGKIT_LLM_URL", f"http://127.0.0.1:{srv.server_port}/v1")
    answer, note = synthesize("refund?", _hits())
    assert answer is None and "no answer" in note
