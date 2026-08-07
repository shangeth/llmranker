import pytest
from conftest import by_text

pytest.importorskip("langchain_core")

from langchain_core.documents import Document

from llmranker.integrations.langchain import LLMRankerCompressor
from llmranker.llm import LLMConfig
from llmranker.rankers.pointwise import PointwiseRanker


def test_compress_documents_reorders_by_ranker_score_and_returns_originals(fake_llm):
    docs = [
        Document(page_content="alpha"),
        Document(page_content="beta"),
        Document(page_content="gamma"),
    ]
    fake_llm.responses = by_text({"alpha": "3", "beta": "9", "gamma": "1"})

    compressor = LLMRankerCompressor(ranker=PointwiseRanker(LLMConfig(model="gpt-4o-mini")))
    result = compressor.compress_documents(docs, query="q")

    assert [d.page_content for d in result] == ["beta", "alpha", "gamma"]
    # Originals are returned, not copies -- callers may hold references.
    assert result[0] is docs[1]
    assert result[1] is docs[0]
    assert result[2] is docs[2]


def test_compress_documents_preserves_explicit_ids_and_metadata(fake_llm):
    docs = [
        Document(page_content="alpha", id="doc-a", metadata={"k": "a"}),
        Document(page_content="beta", id="doc-b", metadata={"k": "b"}),
    ]
    fake_llm.responses = by_text({"alpha": "1", "beta": "9"})

    compressor = LLMRankerCompressor(ranker=PointwiseRanker(LLMConfig(model="gpt-4o-mini")))
    result = compressor.compress_documents(docs, query="q")

    assert [d.id for d in result] == ["doc-b", "doc-a"]
    assert result[0].metadata == {"k": "b"}


def test_compress_documents_handles_unset_ids_via_positional_fallback(fake_llm):
    # Document.id is None unless explicitly set -- the common case for
    # documents coming out of a retriever.
    docs = [Document(page_content="alpha"), Document(page_content="beta")]
    assert docs[0].id is None and docs[1].id is None
    fake_llm.responses = by_text({"alpha": "1", "beta": "9"})

    compressor = LLMRankerCompressor(ranker=PointwiseRanker(LLMConfig(model="gpt-4o-mini")))
    result = compressor.compress_documents(docs, query="q")

    assert [d.page_content for d in result] == ["beta", "alpha"]


def test_compress_documents_empty_list(fake_llm):
    compressor = LLMRankerCompressor(ranker=PointwiseRanker(LLMConfig(model="gpt-4o-mini")))
    assert compressor.compress_documents([], query="q") == []
