# ragkit

A tiny, readable **retrieval-augmented generation (RAG)** kit for asking
questions against your own documents. It runs from a clean clone with **no API
key and no model download** — the retriever is pure Python standard library.
Add an Anthropic API key and it will synthesize a cited answer with Claude.

Built to be *understood in one sitting*: four small modules, one clear contract
per stage, no framework to learn.

## Why this exists

Most RAG demos drown you in a vector database, an embedding service, and a
framework before you can ask a single question. ragkit strips that back to the
essentials so you can see exactly how the pieces fit — and swap any one of them
out. The retriever, chunker, and answerer are independent; each has a surface
small enough to replace in an afternoon.

## Quick start

```bash
git clone <this repo>
cd ragkit
python cli.py --docs sample_docs "How do I add a new retriever?"
```

That works with zero setup — it prints the most relevant passages. To get a
synthesized, cited answer:

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-...          # your key
python cli.py --docs sample_docs "How do I add a new retriever?"
```

Point `--docs` at any file or folder of Markdown/text to query your own notes.

## How it works

```
load_docs → chunk_documents → Retriever.search → answer_question
  (disk)      (structure-aware)   (TF-IDF cosine)    (Claude, optional)
```

- **Load** — walk a path, read text/Markdown, skip binaries.
- **Chunk** — split on headings, pack paragraphs to ~900 chars with overlap so
  facts that straddle a boundary still land whole in a chunk.
- **Retrieve** — TF-IDF + cosine similarity, pure stdlib. The only method
  callers depend on is `search`, so an embedding retriever drops straight in.
- **Answer** — Claude answers *only* from the retrieved passages and cites
  them; with no key set, ragkit returns the ranked passages instead.

## Use it as a library

```python
from ragkit import load_docs, chunk_documents, answer_question
from ragkit.retriever import Retriever

chunks = chunk_documents(load_docs("sample_docs"))
hits = Retriever(chunks).search("how do I tune chunk size?", k=4)
print(answer_question("how do I tune chunk size?", hits))
```

## Extending

See [`sample_docs/extending.md`](sample_docs/extending.md) — swap the retriever
for embeddings, change the answer model via `RAGKIT_MODEL`, or tune chunking.

## Tests

```bash
python -m pytest -q        # or: python tests/test_pipeline.py
```

## License

MIT — see [LICENSE](LICENSE).
