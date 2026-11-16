"""Claude answer path with a fake `anthropic` module (no network, no SDK needed).

A refused, truncated, or thinking-only reply must never be shown as an answer:
synthesize() returns (None, note) so callers fall back to the cited passages.
"""

import sys
import types
from pathlib import Path

import pytest

from ragkit import answer as A
from ragkit.ingest import chunk_documents, load_docs
from ragkit.retriever import Retriever

DOCS = Path(__file__).resolve().parent.parent / "sample_docs" / "shop"


def _block(type_, **kw):
    return types.SimpleNamespace(type=type_, **kw)


def _install_fake(monkeypatch, content, stop_reason="end_turn"):
    calls = []

    class Messages:
        def create(self, **kw):
            calls.append(kw)
            return types.SimpleNamespace(content=content, stop_reason=stop_reason)

    class Anthropic:
        def __init__(self, *a, **kw):
            self.messages = Messages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=Anthropic))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("RAGKIT_LLM_URL", raising=False)
    return calls


@pytest.fixture(scope="module")
def hits():
    return Retriever(chunk_documents(load_docs(DOCS))).search("can I get my money back?", k=3)


def test_default_model_is_opus_55_and_room_for_thinking(monkeypatch, hits):
    calls = _install_fake(monkeypatch, [_block("thinking", thinking=""),
                                        _block("text", text="Yes, within 30 days [1].")])
    text, note = A.synthesize("can I get my money back?", hits)
    assert text == "Yes, within 30 days [1]."   # first text block, not content[0]
    assert calls[0]["model"] == A.DEFAULT_MODEL
    assert calls[0]["max_tokens"] >= 4000
    assert "temperature" not in calls[0]


def test_default_model_constant():
    # RAGKIT_MODEL overrides at import time; unset, the default is Opus 5.5.
    import importlib, os
    old = os.environ.pop("RAGKIT_MODEL", None)
    try:
        assert importlib.reload(A).DEFAULT_MODEL == "claude-opus-5-5"
    finally:
        if old is not None:
            os.environ["RAGKIT_MODEL"] = old
        importlib.reload(A)


def test_model_env_override(monkeypatch):
    import importlib
    monkeypatch.setenv("RAGKIT_MODEL", "claude-sonnet-5-5")
    try:
        assert importlib.reload(A).DEFAULT_MODEL == "claude-sonnet-5-5"
    finally:
        monkeypatch.delenv("RAGKIT_MODEL")
        importlib.reload(A)


@pytest.mark.parametrize("content,stop", [
    ([_block("thinking", thinking="")], "end_turn"),
    ([], "end_turn"),
    ([_block("text", text="   ")], "end_turn"),
    ([_block("text", text="Yes, within 30 da")], "max_tokens"),
    ([_block("text", text="I can't help with that.")], "refusal"),
], ids=["thinking-only", "empty", "whitespace", "truncated", "refusal"])
def test_unusable_reply_falls_back_to_passages(monkeypatch, hits, content, stop):
    _install_fake(monkeypatch, content, stop)
    text, note = A.synthesize("can I get my money back?", hits)
    assert text is None
    assert "passages" in note
    out = A.answer_question("can I get my money back?", hits)
    assert "Most relevant passages" in out
    assert "returns-and-refunds" in out
