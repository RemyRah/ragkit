"""Turn retrieved chunks into a grounded, cited answer.

If an Anthropic API key is present, Claude synthesizes an answer constrained to
the retrieved context. If not, we degrade gracefully and return the ranked
context itself — so the demo is useful (and honest) with zero setup.
"""

from __future__ import annotations

import os
import textwrap

from .retriever import Hit

# Cheap by default — this is a demo, not a production judge. Override with
# RAGKIT_MODEL. See https://docs.anthropic.com for current model ids.
DEFAULT_MODEL = os.environ.get("RAGKIT_MODEL", "claude-haiku-4-5")

SYSTEM_PROMPT = textwrap.dedent(
    """
    You answer questions using ONLY the provided context passages. Rules:
    - If the answer is not in the context, say so plainly. Do not invent facts.
    - Cite the sources you used by their [n] marker inline.
    - Be concise and direct.
    """
).strip()


def _format_context(hits: list[Hit]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(f"[{i}] {hit.chunk.citation}\n{hit.chunk.text}")
    return "\n\n".join(blocks)


def answer_question(question: str, hits: list[Hit], model: str | None = None) -> str:
    """Compose an answer from the retrieved hits, using Claude when available."""

    if not hits:
        return "No relevant context found in the indexed documents."

    context = _format_context(hits)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        return (
            "No ANTHROPIC_API_KEY set — returning the retrieved context instead "
            "of a synthesized answer.\n\n"
            f"Question: {question}\n\nMost relevant passages:\n\n{context}"
        )

    try:
        import anthropic
    except ImportError:
        return (
            "The `anthropic` package is not installed (pip install anthropic).\n\n"
            f"Retrieved context:\n\n{context}"
        )

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model or DEFAULT_MODEL,
        max_tokens=700,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Context passages:\n\n{context}\n\nQuestion: {question}",
            }
        ],
    )
    parts = [block.text for block in resp.content if getattr(block, "type", None) == "text"]
    answer = "\n".join(parts).strip()

    sources = "\n".join(f"[{i}] {h.chunk.citation}" for i, h in enumerate(hits, start=1))
    return f"{answer}\n\nSources:\n{sources}"
