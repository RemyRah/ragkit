"""Query expansion + light stemming: customers don't use the policy's words."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragkit import chunk_documents, load_docs
from ragkit.retriever import Retriever, expand_query, stem

SHOP = Path(__file__).resolve().parents[1] / "sample_docs" / "shop"


def test_money_back_finds_refund_policy():
    retriever = Retriever(chunk_documents(load_docs(SHOP)))
    hits = retriever.search("can I get my money back?", k=3)
    assert hits
    assert "returns-and-refunds" in hits[0].chunk.source
    assert "refund" in hits[0].chunk.text.lower()


def test_expansion_can_be_turned_off():
    retriever = Retriever(chunk_documents(load_docs(SHOP)), expand=False)
    assert retriever.search("reimburse me", k=3) == []
    on = Retriever(chunk_documents(load_docs(SHOP))).search("reimburse me", k=3)
    assert "returns-and-refunds" in on[0].chunk.source


def test_expand_query_adds_synonyms():
    assert "refund" in expand_query("I want my money back")
    assert "ship" in expand_query("when will it be delivered")


def test_stem_merges_plurals_and_tenses():
    assert stem("refunds") == stem("refunded") == stem("refund")
    assert stem("shipping") == stem("ship")
    assert stem("is") == "is"  # short words untouched
