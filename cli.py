#!/usr/bin/env python3
"""ragkit CLI — ask questions against a folder of docs.

Examples:
    python cli.py --docs sample_docs "How do I add a new retriever?"
    python cli.py --docs ./my_notes --k 6 "What did we decide about caching?"
"""

from __future__ import annotations

import argparse
import sys

from ragkit import answer_question, chunk_documents, load_docs
from ragkit.retriever import Retriever


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask questions against your docs (RAG).")
    parser.add_argument("question", help="The question to answer.")
    parser.add_argument("--docs", default="sample_docs", help="File or folder of docs to index.")
    parser.add_argument("--k", type=int, default=4, help="How many chunks to retrieve.")
    parser.add_argument("--show-context", action="store_true", help="Print retrieved chunks too.")
    args = parser.parse_args(argv)

    docs = load_docs(args.docs)
    if not docs:
        print(f"No readable documents found under {args.docs!r}.", file=sys.stderr)
        return 1

    chunks = chunk_documents(docs)
    retriever = Retriever(chunks)
    hits = retriever.search(args.question, k=args.k)

    if args.show_context:
        print("--- retrieved ---")
        for i, hit in enumerate(hits, start=1):
            print(f"[{i}] {hit.chunk.citation}  (score={hit.score:.3f})")
        print("--- answer ---")

    print(answer_question(args.question, hits))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
