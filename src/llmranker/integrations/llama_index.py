"""LlamaIndex integration: use any `llmranker` ranker as a node postprocessor.

Optional dependency -- install with `pip install "llmranker[llama-index]"`.
"""

from __future__ import annotations

try:
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.core.schema import NodeWithScore, QueryBundle
    from pydantic import ConfigDict
except ImportError as exc:  # pragma: no cover - depends on install extras
    raise ImportError(
        "llmranker.integrations.llama_index needs llama-index-core, an "
        'optional dependency. Install it with: pip install "llmranker[llama-index]"'
    ) from exc

from ..types import Candidate, Ranker


class LLMRankerPostprocessor(BaseNodePostprocessor):
    """Wraps any `llmranker` ranker as a LlamaIndex `BaseNodePostprocessor`,
    so it plugs into a query engine's `node_postprocessors` the same way
    `LLMRerank`/`RankGPTRerank`/`CohereRerank` do.

    Takes an already-constructed `Ranker`, the same "wraps an
    already-configured ranker" shape `CascadeRanker` uses -- this class
    owns no ranking logic of its own, just the `NodeWithScore` <->
    `Candidate` conversion.

        from llmranker import LLMConfig, SetwiseRanker
        from llmranker.integrations.llama_index import LLMRankerPostprocessor

        postprocessor = LLMRankerPostprocessor(ranker=SetwiseRanker(LLMConfig(model="gpt-4o-mini")))
        query_engine = index.as_query_engine(node_postprocessors=[postprocessor])

    Each node's `.score` is overwritten with the ranker's own
    `Candidate.score` (whatever `ranker.score_kind` means for it -- see the
    README's "Two contracts every ranker honors" section), since LlamaIndex
    postprocessors are expected to report a relevance score, not just
    reorder.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ranker: Ranker

    @classmethod
    def class_name(cls) -> str:
        return "LLMRankerPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        if not nodes or query_bundle is None:
            return nodes

        by_id: dict[str, NodeWithScore] = {}
        candidates: list[Candidate] = []
        for nws in nodes:
            node_id = nws.node.node_id
            by_id[node_id] = nws
            candidates.append(
                Candidate(id=node_id, text=nws.node.get_content(), metadata=nws.node.metadata)
            )

        ranked = self.ranker.rank(query_bundle.query_str, candidates)

        result = []
        for c in ranked:
            nws = by_id[c.id]
            nws.score = c.score
            result.append(nws)
        return result
