from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

from ..llm import LLMConfig, LLMResponse, call_llm
from ..types import Candidate

logger = logging.getLogger("llmranker")


class BaseRanker(ABC):
    """Common bookkeeping shared by every ranking strategy.

    Subclasses implement `rank()`. `_call()` is the single choke point every
    subclass routes its LLM calls through, so usage stats (`total_calls`,
    `total_prompt_tokens`, `total_completion_tokens`) stay accurate no matter
    which strategy is used; these feed `llmranker.benchmark.compare_rankers`.

    `max_concurrency` controls how many LLM calls `_call_many()` runs at
    once. It only affects strategies whose calls are independent of each
    other (PointwiseRanker, PairwiseRanker's "allpairs" method); see each
    class's docstring for whether it applies. Set to 1 to force fully
    sequential calls (e.g. to stay under a strict rate limit).

    Three params control how hard the ranker works to get a reliable
    judgment: `reasoning` (chain-of-thought prompting), `num_samples`
    (repeat each judgment and aggregate instead of trusting one call), and
    `structured_output` (JSON-schema response instead of regex-parsed
    text). All off/1 by default. `reasoning` and `structured_output` can't
    both be enabled: reasoning needs free text ending in a final-answer
    marker, while structured_output needs the entire completion to be the
    JSON payload, leaving no room for reasoning text.
    """

    def __init__(
        self,
        config: LLMConfig,
        item_label: str = "item",
        system_prompt: str | None = None,
        name: str | None = None,
        max_concurrency: int = 5,
        reasoning: bool = False,
        num_samples: int = 1,
        structured_output: bool = False,
    ):
        if reasoning and structured_output:
            raise ValueError(
                "reasoning and structured_output can't both be enabled. "
                "Reasoning needs free text ending in a final-answer marker; "
                "structured_output needs the entire completion to be the "
                "JSON payload, leaving no room for reasoning text."
            )
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        self.config = config
        self.item_label = item_label
        self.system_prompt_override = system_prompt
        self.name = name or type(self).__name__
        self.max_concurrency = max_concurrency
        self.reasoning = reasoning
        self.num_samples = num_samples
        self.structured_output = structured_output
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self._warned_low_temperature = False

    def _reset_stats(self) -> None:
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def _warn_if_low_temperature(self) -> None:
        """Warn once when `num_samples > 1` can't possibly help.

        For strategies that send the *same* prompt every sample
        (PointwiseRanker, ListwiseRanker), `temperature=0` makes every
        repeat return an identical response, so the extra calls buy
        nothing. Strategies that reshuffle candidate positions between
        samples (PairwiseRanker, SetwiseRanker) send genuinely different
        prompts and do benefit at any temperature, so they don't call this.
        """
        if (
            self.num_samples > 1
            and self.config.temperature == 0.0
            and not self._warned_low_temperature
        ):
            logger.warning(
                "num_samples=%d but LLMConfig.temperature=0.0: this strategy "
                "sends an identical prompt for every sample, so the repeats "
                "return identical responses and the extra calls don't add "
                "anything. Set temperature > 0 for num_samples to help.",
                self.num_samples,
            )
            self._warned_low_temperature = True

    def _call(self, messages: list[dict], response_format: dict | None = None) -> LLMResponse:
        self.total_calls += 1
        response = call_llm(messages, self.config, response_format=response_format)
        self.total_prompt_tokens += response.prompt_tokens
        self.total_completion_tokens += response.completion_tokens
        return response

    def _call_many(
        self, message_batches: list[list[dict]], response_format: dict | None = None
    ) -> list[LLMResponse]:
        """Run independent LLM calls concurrently (up to `max_concurrency`),
        returning responses in the same order as `message_batches`.

        Only for strategies where calls don't depend on each other's
        results. Falls back to the plain sequential path when
        `max_concurrency <= 1` or there's nothing to parallelize.

        Usage stats are accumulated per-call inside each worker rather than
        in bulk afterwards, so that a batch where one call raises still
        reports what the calls that did succeed actually cost.
        """
        if self.max_concurrency <= 1 or len(message_batches) <= 1:
            return [self._call(m, response_format=response_format) for m in message_batches]

        stats_lock = threading.Lock()

        def run(messages: list[dict]) -> LLMResponse:
            response = call_llm(messages, self.config, response_format=response_format)
            with stats_lock:
                self.total_calls += 1
                self.total_prompt_tokens += response.prompt_tokens
                self.total_completion_tokens += response.completion_tokens
            return response

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            return list(pool.map(run, message_batches))

    @staticmethod
    def _finalize(top: list[Candidate], original: list[Candidate]) -> list[Candidate]:
        """Combine a ranked top-k with the untouched remainder of `original`,
        assigning a synthetic descending score (higher = more relevant).

        Shared by ranking strategies (pairwise, setwise) that only establish
        the order of a top-k subset via comparisons, not a full permutation.

        Membership is tracked by object identity rather than by `id` field:
        the strategies that call this only ever shuffle the very objects
        they were handed, and matching on the `id` field would drop *every*
        candidate sharing an id with a ranked one — silently returning
        fewer candidates than were passed in when a caller's list contains
        duplicate ids.
        """
        top_ids = {id(c) for c in top}
        rest = [c for c in original if id(c) not in top_ids]
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
