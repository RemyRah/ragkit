"""A pure-standard-library TF-IDF retriever.

No numpy, no torch, no model download — so the demo runs anywhere Python does.
The public surface is intentionally one method, `search`, so you can drop in a
sentence-transformers or API-embedding retriever later without touching callers.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .ingest import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A small, ordinary English stoplist. Kept short on purpose — over-aggressive
# stopping hurts recall on technical docs where words like "state" matter.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "for",
    "on", "with", "as", "at", "by", "it", "this", "that", "be", "from",
    # Conversational filler from customer questions.
    "i", "me", "my", "you", "your", "we", "our", "do", "does", "can", "will",
    "what", "whats", "how", "have", "get", "if", "there", "any",
}


# How customers ask vs. how policies are written. Each key (a word or phrase
# found in the query) adds its terms to the query. Small and editable on purpose:
# add your own business's vocabulary here.
SYNONYMS: dict[str, list[str]] = {
    "money back": ["refund"],
    "reimburse": ["refund"],
    "send back": ["return"],
    "exchange": ["return", "replacement"],
    "cancel": ["cancel", "refund"],
    "delivered": ["ship", "delivery"],
    "arrive": ["ship", "delivery"],
    "postage": ["shipping"],
    "price": ["cost"],
    "cost": ["price"],
    "broken": ["damaged"],
    "contact": ["email"],
}


def stem(word: str) -> str:
    """Very light suffix stripping so 'refunds'/'refunded'/'refund' match."""

    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            word = word[: -len(suffix)]
            # 'shipping' -> 'shipp' -> 'ship'
            if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "lsz":
                word = word[:-1]
            break
    return word


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def expand_query(query: str) -> str:
    """Append synonym terms for any SYNONYMS key that appears in `query`."""

    padded = " " + " ".join(_TOKEN_RE.findall(query.lower())) + " "
    extra = [t for key, terms in SYNONYMS.items() if f" {key} " in padded for t in terms]
    return " ".join([query, *extra])


@dataclass
class Hit:
    chunk: Chunk
    score: float


class Retriever:
    """TF-IDF over chunks with cosine similarity ranking."""

    def __init__(self, chunks: list[Chunk], expand: bool = True):
        if not chunks:
            raise ValueError("Retriever needs at least one chunk to index.")
        self.chunks = chunks
        self.expand = expand
        # Index the section heading too: it's often the best summary of the chunk.
        self._doc_tokens = [tokenize(f"{c.heading}\n{c.text}") for c in chunks]
        self._idf = self._compute_idf(self._doc_tokens)
        self._vectors = [self._vectorize(toks) for toks in self._doc_tokens]
        self._norms = [math.sqrt(sum(w * w for w in v.values())) for v in self._vectors]

    def _compute_idf(self, docs: list[list[str]]) -> dict[str, float]:
        n = len(docs)
        df: Counter[str] = Counter()
        for toks in docs:
            for term in set(toks):
                df[term] += 1
        # Smoothed idf so a term in every doc still contributes a little.
        return {term: math.log((n + 1) / (freq + 1)) + 1.0 for term, freq in df.items()}

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        if not tokens:
            return {}
        tf = Counter(tokens)
        length = len(tokens)
        return {term: (count / length) * self._idf.get(term, 0.0) for term, count in tf.items()}

    def search(self, query: str, k: int = 4) -> list[Hit]:
        """Return the top-`k` chunks most similar to `query`, best first."""

        q_tokens = tokenize(expand_query(query) if self.expand else query)
        q_vec = self._vectorize(q_tokens)
        q_norm = math.sqrt(sum(w * w for w in q_vec.values()))
        if q_norm == 0:
            return []

        scored: list[Hit] = []
        for chunk, vec, norm in zip(self.chunks, self._vectors, self._norms):
            if norm == 0:
                continue
            # Cosine similarity, iterating the shorter vector.
            small, large = (q_vec, vec) if len(q_vec) < len(vec) else (vec, q_vec)
            dot = sum(w * large.get(term, 0.0) for term, w in small.items())
            if dot > 0:
                scored.append(Hit(chunk=chunk, score=dot / (q_norm * norm)))

        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]
