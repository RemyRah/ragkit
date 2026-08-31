"""ragkit — a tiny, dependency-light RAG over your own docs.

Ingest a folder of Markdown/text, retrieve the most relevant chunks for a
question, and (optionally) synthesize a cited answer with Claude.

The retriever is pure standard library so the demo runs from a clean clone
with no API key and no model download. Swap in embeddings when you want
semantic recall — the interface is one method (`Retriever.search`).
"""

from .ingest import Document, Chunk, load_docs, chunk_documents
from .retriever import Retriever
from .answer import answer_question

__all__ = [
    "Document",
    "Chunk",
    "load_docs",
    "chunk_documents",
    "Retriever",
    "answer_question",
]
