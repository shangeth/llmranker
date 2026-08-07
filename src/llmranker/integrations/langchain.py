"""LangChain integration: use any `llmranker` ranker as a document compressor.

Optional dependency -- install with `pip install "llmranker[langchain]"`.
"""

from __future__ import annotations

from collections.abc import Sequence

try:
    from langchain_core.callbacks import Callbacks
    from langchain_core.documents import Document
    from langchain_core.documents.compressor import BaseDocumentCompressor
    from pydantic import ConfigDict
except ImportError as exc:  # pragma: no cover - depends on install extras
    raise ImportError(
        "llmranker.integrations.langchain needs langchain-core, an optional "
        'dependency. Install it with: pip install "llmranker[langchain]"'
    ) from exc

from ..types import Candidate, Ranker


class LLMRankerCompressor(BaseDocumentCompressor):
    """Wraps any `llmranker` ranker as a LangChain `BaseDocumentCompressor`,
    so it plugs into `ContextualCompressionRetriever` like Cohere's or
    RankGPT's rerankers do.

    Takes an already-constructed `Ranker` (any of this package's rankers,
    or a `CascadeRanker`/`RerankAPIRanker`) rather than an `LLMConfig`, the
    same "wraps a already-configured ranker" shape `CascadeRanker` uses --
    this class owns no ranking logic of its own, just the `Document` <->
    `Candidate` conversion.

        from langchain_core.documents import Document
        from llmranker import LLMConfig, SetwiseRanker
        from llmranker.integrations.langchain import LLMRankerCompressor

        compressor = LLMRankerCompressor(ranker=SetwiseRanker(LLMConfig(model="gpt-4o-mini")))
        compressed = compressor.compress_documents(docs, query="...")

    A `Document`'s `id` is often unset; when it is, a positional fallback
    id is used internally only to pair ranked results back to their
    original `Document` objects -- the originals are always returned
    unmodified, never copies.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ranker: Ranker

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        documents = list(documents)
        by_id: dict[str, Document] = {}
        candidates: list[Candidate] = []
        for i, doc in enumerate(documents):
            doc_id = doc.id if doc.id is not None else str(i)
            by_id[doc_id] = doc
            candidates.append(Candidate(id=doc_id, text=doc.page_content, metadata=doc.metadata))

        ranked = self.ranker.rank(query, candidates)
        return [by_id[c.id] for c in ranked]
