# Extending ragkit

## How to add a new retriever

The retriever contract is a single method: `search(query: str, k: int) -> list[Hit]`.
To add a semantic retriever, embed each chunk once at index time, embed the
query at search time, and rank chunks by cosine similarity of the embeddings.
Return the top `k` as `Hit` objects. Because callers only depend on `search`,
nothing else in the pipeline needs to change.

## How to change the answer model

Set the `RAGKIT_MODEL` environment variable to any current Claude model id.
The default is a small, cheap model because this is a demo, not a production
grader. For higher-stakes synthesis, point it at a larger model.

## Tuning chunk size

`chunk_documents` takes `target_chars` and `overlap_chars`. Smaller chunks give
sharper retrieval but more of them; larger chunks give more context per hit but
blurrier matches. Overlap trades a little redundancy for better boundary recall.
