# ragkit architecture

## Pipeline overview

ragkit runs a classic retrieval-augmented-generation pipeline in four stages:
load, chunk, retrieve, answer. Each stage is a separate module so you can
replace one without disturbing the others.

## Loading

The loader walks a file or directory and reads every Markdown, text, or reST
file it finds. Binary files are skipped. Empty files are dropped.

## Chunking

Chunking is structure-aware. It splits on Markdown headings first, then packs
paragraphs up to a target character budget with a small overlap. The overlap
means a fact that straddles a boundary still lands whole inside at least one
chunk, which improves recall.

## Retrieval

The default retriever is TF-IDF with cosine similarity, implemented in pure
standard library. It needs no model download and no API key, so the demo runs
anywhere Python does. To get semantic recall, swap in an embedding retriever —
the only method callers depend on is `search`.

## Answering

If an Anthropic API key is present, Claude synthesizes an answer constrained to
the retrieved passages and cites them. Without a key, ragkit returns the ranked
passages directly, so it stays useful and honest with zero setup.
