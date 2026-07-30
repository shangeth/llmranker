from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

from ..llm import LLMConfig, LLMResponse, call_llm
from ..types import Candidate


class BaseRanker(ABC):
    """Common bookkeeping shared by every ranking strategy.

    Subclasses implement `rank()`. `_call()` is the single choke point every
    subclass routes its LLM calls through, so usage stats (`total_calls`,
    `total_prompt_tokens`, `total_completion_tokens`) stay accurate no matter
    which strategy is used -- these feed `llmranker.benchmark.compare_rankers`.

    `max_concurrency` controls how many LLM calls `_call_many()` runs at
    once. It only affects strategies whose calls are independent of each
    other (PointwiseRanker, PairwiseRanker's "allpairs" method) -- see each
    class's docstring for whether it applies. Set to 1 to force fully
    sequential calls (e.g. to stay under a strict rate limit).

    `reasoning` asks the model to think step by step before giving its
    final answer (a prompting technique, not a switch to a dedicated
    reasoning model -- works with any chat model; use a reasoning-capable
    `LLMConfig.model` for that instead). It doesn't change how many calls
    are made, only prompt/completion content -- expect longer, more
    expensive completions. Off by default.
    """

    def __init__(
        self,
        config: LLMConfig,
        item_label: str = "item",
        system_prompt: str | None = None,
        name: str | None = None,
        max_concurrency: int = 5,
        reasoning: bool = False,
    ):
        self.config = config
        self.item_label = item_label
        self.system_prompt_override = system_prompt
        self.name = name or type(self).__name__
        self.max_concurrency = max_concurrency
        self.reasoning = reasoning
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def _reset_stats(self) -> None:
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def _call(self, messages: list[dict]) -> LLMResponse:
        self.total_calls += 1
        response = call_llm(messages, self.config)
        self.total_prompt_tokens += response.prompt_tokens
        self.total_completion_tokens += response.completion_tokens
        return response

    def _call_many(self, message_batches: list[list[dict]]) -> list[LLMResponse]:
        """Run independent LLM calls concurrently (up to `max_concurrency`),
        returning responses in the same order as `message_batches`.

        Only for strategies where calls don't depend on each other's
        results. Falls back to the plain sequential path when
        `max_concurrency <= 1` or there's nothing to parallelize.
        """
        if self.max_concurrency <= 1 or len(message_batches) <= 1:
            return [self._call(m) for m in message_batches]

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            responses = list(pool.map(lambda m: call_llm(m, self.config), message_batches))

        self.total_calls += len(responses)
        self.total_prompt_tokens += sum(r.prompt_tokens for r in responses)
        self.total_completion_tokens += sum(r.completion_tokens for r in responses)
        return responses

    @staticmethod
    def _finalize(top: list[Candidate], original: list[Candidate]) -> list[Candidate]:
        """Combine a ranked top-k with the untouched remainder of `original`,
        assigning a synthetic descending score (higher = more relevant).

        Shared by ranking strategies (pairwise, setwise) that only establish
        the order of a top-k subset via comparisons, not a full permutation.
        """
        top_ids = {c.id for c in top}
        rest = [c for c in original if c.id not in top_ids]
        ranked = list(top) + rest
        n = len(ranked)
        return [
            Candidate(id=c.id, text=c.text, score=float(n - i), metadata=c.metadata)
            for i, c in enumerate(ranked)
        ]

    @abstractmethod
    def rank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        """Rank `candidates` by relevance to `query`, most relevant first."""
        raise NotImplementedError
