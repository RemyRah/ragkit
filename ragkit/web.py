"""A single-page chat UI over your docs, using only the standard library.

    python -m ragkit.web --docs sample_docs/shop
    # then open http://127.0.0.1:8000

GET /            the chat page (also answers ?q=... server-side, so it works without JS)
POST /api/ask    {"question": "..."} -> {"answer", "note", "sources": [...]}

Uses Claude when ANTHROPIC_API_KEY is set (model: RAGKIT_MODEL, default
claude-haiku-4-5); otherwise it shows the most relevant passages. Everything
user- or document-derived is HTML-escaped on the server, and the page's script
only ever inserts text via textContent.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .answer import synthesize
from .ingest import chunk_documents, load_docs
from .retriever import Retriever

MAX_QUESTION = 500


def ask(retriever: Retriever, question: str, k: int = 4) -> dict:
    """Retrieve, then answer with Claude if available. JSON-ready result."""

    hits = retriever.search(question, k=k)
    sources = [
        {"citation": h.chunk.citation, "text": h.chunk.text, "score": round(h.score, 3)}
        for h in hits
    ]
    if not hits:
        return {"answer": None, "note": "Nothing in the documents matches that question.", "sources": []}
    answer, note = synthesize(question, hits)
    return {"answer": answer, "note": note, "sources": sources}


def _render_result(question: str, result: dict) -> str:
    e = html.escape
    parts = [f'<div class="msg user">{e(question)}</div>', '<div class="msg bot">']
    if result["answer"]:
        parts.append(f'<div class="answer">{e(result["answer"])}</div>')
    parts.append(f'<div class="note">{e(result["note"])}</div>')
    for i, s in enumerate(result["sources"], start=1):
        parts.append(
            f'<details class="src"{" open" if i == 1 and not result["answer"] else ""}>'
            f"<summary>[{i}] {e(s['citation'])}</summary><p>{e(s['text'])}</p></details>"
        )
    parts.append("</div>")
    return "".join(parts)


def render_page(title: str, question: str = "", result: dict | None = None) -> str:
    log = _render_result(question, result) if result else ""
    return PAGE.replace("{{TITLE}}", html.escape(title)).replace("{{LOG}}", log)


def make_handler(retriever: Retriever, title: str):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: str, ctype: str) -> None:
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", f"{ctype}; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            url = urlparse(self.path)
            if url.path != "/":
                return self._send(404, "Not found", "text/plain")
            q = parse_qs(url.query).get("q", [""])[0].strip()[:MAX_QUESTION]
            result = ask(retriever, q) if q else None
            self._send(200, render_page(title, q, result), "text/html")

        def do_POST(self) -> None:
            if self.path != "/api/ask":
                return self._send(404, json.dumps({"error": "not found"}), "application/json")
            try:
                length = min(int(self.headers.get("Content-Length", 0)), 10_000)
                question = str(json.loads(self.rfile.read(length) or b"{}").get("question", ""))
            except (ValueError, AttributeError):
                question = ""
            question = question.strip()[:MAX_QUESTION]
            if not question:
                return self._send(400, json.dumps({"error": "question is required"}), "application/json")
            try:
                result = ask(retriever, question)
            except Exception as exc:  # e.g. a bad API key: report it, don't crash
                return self._send(502, json.dumps({"error": f"answer failed: {exc}"}), "application/json")
            self._send(200, json.dumps(result), "application/json")

        def log_message(self, fmt: str, *args) -> None:
            sys.stderr.write("ragkit.web: " + (fmt % args) + "\n")

    return Handler


def make_server(retriever: Retriever, title: str = "Ask our docs", host: str = "127.0.0.1", port: int = 8000):
    return ThreadingHTTPServer((host, port), make_handler(retriever, title))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chat with your docs in the browser.")
    parser.add_argument("--docs", default="sample_docs/shop", help="File or folder of docs to index.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (0.0.0.0 to serve beyond this machine).")
    parser.add_argument("--title", default=None, help="Heading shown on the page.")
    args = parser.parse_args(argv)

    docs = load_docs(args.docs)
    if not docs:
        print(f"No readable documents found under {args.docs!r}.", file=sys.stderr)
        return 1
    title = args.title or f"Ask {Path(args.docs).name}"
    server = make_server(Retriever(chunk_documents(docs)), title=title, host=args.host, port=args.port)
    print(f"ragkit: {len(docs)} documents indexed. Open http://{args.host}:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{TITLE}}</title>
<style>
  :root { --bg:#f6f3ee; --card:#fff; --ink:#2b2420; --muted:#7a6e66; --accent:#8a4b2a; --line:#e6ded5; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.5 system-ui, sans-serif; }
  main { max-width:760px; margin:0 auto; padding:24px 16px 120px; }
  h1 { font-size:1.4rem; margin:0 0 4px; }
  .sub { color:var(--muted); margin:0 0 20px; font-size:.95rem; }
  .msg { padding:12px 16px; border-radius:14px; margin:10px 0; white-space:pre-wrap; }
  .user { background:var(--accent); color:#fff; margin-left:20%; }
  .bot { background:var(--card); border:1px solid var(--line); margin-right:8%; }
  .answer { margin-bottom:8px; }
  .note { color:var(--muted); font-size:.85rem; margin-bottom:6px; }
  .src { border-top:1px solid var(--line); padding:6px 0; font-size:.9rem; }
  .src summary { cursor:pointer; color:var(--accent); }
  .src p { margin:6px 0 0; white-space:pre-wrap; }
  form { position:fixed; left:0; right:0; bottom:0; background:var(--bg); border-top:1px solid var(--line); padding:12px 16px; }
  .row { max-width:760px; margin:0 auto; display:flex; gap:8px; }
  input { flex:1; padding:12px 14px; border:1px solid var(--line); border-radius:10px; font:inherit; }
  button { padding:12px 18px; border:0; border-radius:10px; background:var(--accent); color:#fff; font:inherit; cursor:pointer; }
</style></head>
<body><main>
<h1>{{TITLE}}</h1>
<p class="sub">Ask a question. Answers come only from this business's documents, with sources.</p>
<div id="log">{{LOG}}</div>
</main>
<form id="f" method="get" action="/"><div class="row">
  <input id="q" name="q" autocomplete="off" maxlength="500" placeholder="e.g. Can I get my money back?" autofocus>
  <button>Ask</button>
</div></form>
<script>
const log = document.getElementById("log"), q = document.getElementById("q");
function el(tag, cls, text) { const n = document.createElement(tag); if (cls) n.className = cls; if (text) n.textContent = text; return n; }
document.getElementById("f").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const question = q.value.trim(); if (!question) return;
  q.value = "";
  log.appendChild(el("div", "msg user", question));
  const bot = log.appendChild(el("div", "msg bot"));
  bot.appendChild(el("div", "note", "Searching..."));
  try {
    const r = await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question }) });
    const data = await r.json();
    bot.textContent = "";
    if (!r.ok) { bot.appendChild(el("div", "note", data.error || "Something went wrong.")); return; }
    if (data.answer) bot.appendChild(el("div", "answer", data.answer));
    bot.appendChild(el("div", "note", data.note));
    data.sources.forEach((s, i) => {
      const d = el("details", "src"); if (i === 0 && !data.answer) d.open = true;
      d.appendChild(el("summary", null, `[${i + 1}] ${s.citation}`));
      d.appendChild(el("p", null, s.text));
      bot.appendChild(d);
    });
  } catch (e) { bot.textContent = "Could not reach the server."; }
  bot.scrollIntoView({ behavior: "smooth", block: "end" });
});
</script>
</body></html>
"""

if __name__ == "__main__":
    raise SystemExit(main())
