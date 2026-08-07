from __future__ import annotations

import logging

from ..llm import LLMConfig, call_rerank
from ..types import Candidate

logger = logging.getLogger("llmranker")


class RerankAPIRanker:
    """Ranks candidates with a dedicated rerank model instead of by prompting
    a chat LLM: **one** request scores the whole candidate list.

    Every other ranker in this package prompts a general-purpose chat model
    and parses what it writes back. This one calls a purpose-trained
    relevance model (Cohere Rerank, Jina Reranker, Bedrock, Azure AI,
    Infinity, ...) through LiteLLM's rerank endpoint. The trade is the
    exact inverse of the prompting strategies:

      - **Cost/latency**: one request for the whole list, not `O(n)` or
        `O(n log n)` LLM calls. Orders of magnitude cheaper and faster.
      - **Capability**: it returns a similarity-style relevance score and
        nothing else. No reasoning, no multi-criteria breakdown, no
        explanation, and it handles compositional natural-language intent
        ("family friendly, near historic sites, *not* on the beach") far
        worse than a chat model reading the same text.

    So its natural home is the cheap first stage of a `CascadeRanker`,
    where "throw out the obvious junk" is all that's being asked of it:

        CascadeRanker(
            narrow=RerankAPIRanker(LLMConfig(model="cohere/rerank-v3.5")),
            refine=SetwiseRanker(LLMConfig(model="gpt-4o"), num_child=4),
            narrow_to=15,
        )

    Doesn't subclass `BaseRanker`: there's no prompt to build, no response
    text to parse, and none of `reasoning`/`num_samples`/
    `structured_output` mean anything for a model that isn't generating
    text. It satisfies the structural `llmranker.types.Ranker` contract
    instead, the same way `CascadeRanker` does, so it composes and
    benchmarks like any other ranker.

    **Cost reporting caveat**: rerank endpoints bill per *search unit*, not
    per token, and return no token counts. `total_prompt_tokens` and
    `total_completion_tokens` are therefore always 0, and
    `estimate_cost_usd()` returns `None` rather than `$0.00` — LiteLLM has
    no pricing table for rerank models, and reporting zero for a call that
    costs real money would be worse than reporting nothing.
    `total_search_units` carries the provider's own billing count when it
    reports one, and `total_calls` is always meaningful.

    `top_n`, if set, asks the provider to return only its top N (which is
    what `CascadeRanker`'s `narrow_to` slices to anyway); candidates the
    provider drops are appended after the scored ones in their original
    order with a score of `None`, so `rank()` never silently loses a
    candidate.
    """

    def __init__(
        self,
        config: LLMConfig,
        top_n: int | None = None,
        name: str | None = None,
    ):
        if top_n is not None and top_n < 1:
            raise ValueError("top_n must be >= 1")
        self.config = config
        self.top_n = top_n
        self.name = name or type(self).__name__
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_search_units = 0

    def estimate_cost_usd(self) -> float | None:
        """Always `None`: see the cost-reporting caveat on the class.

        `llmranker.benchmark.compare_rankers` calls this when a ranker
        defines it, in preference to its token-based estimate, so this
        ranker shows a blank cost cell instead of a misleading $0.00.
        """
        return None

    def _reset_stats(self) -> None:
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_search_units = 0

    def rank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        self._reset_stats()
        if not candidates:
            return []

        response = call_rerank(
            query=query,
            documents=[c.text for c in candidates],
            config=self.config,
            top_n=self.top_n,
        )
        self.total_calls += 1
        if response.search_units is not None:
            self.total_search_units += response.search_units

        ranked: list[Candidate] = []
        seen: set[int] = set()
        for result in response.results:
            if not 0 <= result.index < len(candidates):
                logger.warning(
                    "Rerank response referenced out-of-range index %d for %d candidates, skipping",
                    result.index,
                    len(candidates),
                )
                continue
            if result.index in seen:
                continue
            seen.add(result.index)
            source = candidates[result.index]
            ranked.append(
                Candidate(
                    id=source.id,
                    text=source.text,
                    score=result.relevance_score,
                    metadata=source.metadata,
                )
            )

        # Anything the provider didn't score (top_n truncation, or a
        # provider returning a short list) keeps its original relative
        # order behind the scored candidates, scored None to mark it as
        # "not ranked" rather than "ranked worst".
        ranked.extend(
            Candidate(id=c.id, text=c.text, score=None, metadata=c.metadata)
            for i, c in enumerate(candidates)
            if i not in seen
        )
        return ranked
