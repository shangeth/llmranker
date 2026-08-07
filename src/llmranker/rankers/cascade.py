from __future__ import annotations

from ..types import Candidate, Ranker


class CascadeRanker:
    """Cheap-then-expensive tiered ranking: a fast/cheap ranker narrows the
    field, a slower/pricier ranker does a thorough re-rank of just the
    survivors.

    This is the classic multi-stage ("telescoping") retrieval cascade,
    where each stage sees fewer candidates than the last. It is *not*
    FrugalGPT's LLM cascade (arXiv:2305.05176), which routes the same
    query through progressively stronger models and uses a confidence
    score to decide whether to stop early -- there, the expensive model
    is often never called at all. Here both stages always run; the
    saving comes from the second one seeing a shorter list. A
    confidence gate that skips `refine` outright is tracked in
    ROADMAP.md.

    Takes two already-constructed rankers rather than raw `LLMConfig`s, so
    each stage is configured exactly like it would be standalone (its own
    model, `reasoning`/`num_samples`/`structured_output`, strategy, etc.) —
    `CascadeRanker` only owns `narrow_to`, how many survive the first
    stage.

    Either stage only has to satisfy the structural `Ranker` protocol, not
    subclass `BaseRanker` — so the cheap stage can be a
    `RerankAPIRanker`, turning the whole narrowing step into a single
    rerank-endpoint request instead of one LLM call per candidate.

    Doesn't subclass `BaseRanker`: it has no single `LLMConfig` of its own,
    just two stages that each have theirs. It satisfies the same structural
    `llmranker.types.Ranker` contract instead (`name`, `config`,
    `total_calls`, `total_prompt_tokens`, `total_completion_tokens`,
    `rank()`), so it plugs into `llmranker.benchmark.compare_rankers`
    unchanged. `config`/the token and call totals all delegate to (or sum)
    the two wrapped rankers live, rather than tracking their own copy.
    """

    def __init__(
        self,
        narrow: Ranker,
        refine: Ranker,
        narrow_to: int,
        name: str | None = None,
    ):
        self.narrow = narrow
        self.refine = refine
        self.narrow_to = narrow_to
        self.name = name or f"Cascade({narrow.name}->{refine.name})"

    @property
    def config(self):
        """The `refine` stage's config: whichever model actually produces
        the final ranking."""
        return self.refine.config

    @property
    def score_kind(self) -> str:
        """The `refine` stage's, since it produced the returned scores."""
        return self.refine.score_kind

    @property
    def total_calls(self) -> int:
        return self.narrow.total_calls + self.refine.total_calls

    @property
    def total_prompt_tokens(self) -> int:
        return self.narrow.total_prompt_tokens + self.refine.total_prompt_tokens

    @property
    def total_completion_tokens(self) -> int:
        return self.narrow.total_completion_tokens + self.refine.total_completion_tokens

    def rank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        narrowed = self.narrow.rank(query, candidates)[: self.narrow_to]
        return self.refine.rank(query, narrowed)
