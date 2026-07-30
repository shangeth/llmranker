from __future__ import annotations

from abc import ABC, abstractmethod

from ..llm import LLMConfig, LLMResponse, call_llm
from ..types import Candidate


class BaseRanker(ABC):
    """Common bookkeeping shared by every ranking strategy.

    Subclasses implement `rank()`. `_call()` is the single choke point every
    subclass routes its LLM calls through, so usage stats (`total_calls`,
    `total_prompt_tokens`, `total_completion_tokens`) stay accurate no matter
    which strategy is used -- these feed `llmranker.benchmark.compare_rankers`.
    """

    def __init__(
        self,
        config: LLMConfig,
        item_label: str = "item",
        system_prompt: str | None = None,
        name: str | None = None,
    ):
        self.config = config
        self.item_label = item_label
        self.system_prompt_override = system_prompt
        self.name = name or type(self).__name__
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
