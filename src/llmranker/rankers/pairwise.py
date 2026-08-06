from __future__ import annotations

import logging
import random
import re

from .. import structured
from ..llm import LLMConfig
from ..prompts import extract_final_answer, pairwise_system_prompt, pairwise_user_prompt
from ..types import Candidate
from .base import BaseRanker

logger = logging.getLogger("llmranker")

_LABEL_RE = re.compile(r"\b([AB])\b")

_STRATEGIES = ("heapsort", "bubblesort", "allpairs")


class PairwiseRanker(BaseRanker):
    """Sorts candidates via repeated pairwise LLM comparisons ("which is more
    relevant, A or B?").

    `strategy`:
      - "heapsort": build a max-heap, pop the top `k`. O(n + k log n) comparisons,
        run sequentially: each comparison's outcome determines the next one,
        so `max_concurrency` has no effect here.
      - "bubblesort": k bubble passes, each moving the best remaining candidate
        to the front. O(n * k) comparisons, also sequential for the same reason.
      - "allpairs": every candidate compared against every other once, ranked
        by win count. O(n^2) comparisons, most robust to individual
        comparison noise, and the only strategy here where comparisons are
        independent of each other, so it's fully parallelized by
        `max_concurrency` (see `BaseRanker`).

    Only `score()` on `PointwiseRanker` produces an independently calibrated
    relevance score. Here the output `score` is a synthetic rank position
    (higher = more relevant) since comparisons only ever establish relative
    order, not an absolute value.

    `num_samples > 1`: LLMs have a documented bias toward whichever
    candidate is listed first (or second, model-dependent) in a pairwise
    prompt. Each sample randomly assigns the two candidates to slots A/B
    before asking, and the comparison is decided by majority vote across
    samples instead of trusting a single call; this cancels position bias
    as a side effect, since it's no longer systematically tied to a fixed
    slot. Costs `num_samples` calls per comparison instead of 1. `seed`
    makes the position randomization reproducible across a `rank()` call.
    """

    def __init__(
        self,
        config: LLMConfig,
        strategy: str = "heapsort",
        k: int | None = None,
        item_label: str = "item",
        system_prompt: str | None = None,
        name: str | None = None,
        max_concurrency: int = 5,
        reasoning: bool = False,
        num_samples: int = 1,
        structured_output: bool = False,
        seed: int | None = None,
    ):
        super().__init__(
            config,
            item_label,
            system_prompt,
            name,
            max_concurrency,
            reasoning,
            num_samples,
            structured_output,
        )
        if strategy not in _STRATEGIES:
            raise ValueError(f"Unknown strategy {strategy!r}, expected one of {_STRATEGIES}")
        self.strategy = strategy
        self.k = k
        self.seed = seed
        self._rng = random.Random(seed)

    def _build_compare_messages(self, query: str, a: Candidate, b: Candidate) -> list[dict]:
        system = self.system_prompt_override or pairwise_system_prompt(self.item_label)
        user = pairwise_user_prompt(
            query,
            [a, b],
            self.item_label,
            self.reasoning,
            self.structured_output,
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _response_format(self) -> dict | None:
        return structured.pairwise_schema() if self.structured_output else None

    def _parse_label(self, text: str, a: Candidate, b: Candidate) -> Candidate:
        if self.structured_output:
            choice = structured.parse_pairwise_json(text)
            if choice is not None:
                return a if choice == "A" else b
        text = extract_final_answer(text)
        match = _LABEL_RE.search(text.strip())
        if match is None:
            logger.warning("Could not parse an A/B label from output: %r", text)
            return a
        return a if match.group(1) == "A" else b

    def compare(self, query: str, a: Candidate, b: Candidate) -> Candidate:
        """Return whichever of `a`/`b` the LLM judges more relevant to `query`.

        See `num_samples` on the class for the randomized-position,
        majority-vote behavior when it's set above 1.
        """
        n = self.num_samples
        response_format = self._response_format()
        if n <= 1:
            forward = self._call(self._build_compare_messages(query, a, b), response_format)
            return self._parse_label(forward.text, a, b)

        orders: list[tuple[Candidate, Candidate]] = []
        batches = []
        for _ in range(n):
            if self._rng.random() < 0.5:
                orders.append((a, b))
                batches.append(self._build_compare_messages(query, a, b))
            else:
                orders.append((b, a))
                batches.append(self._build_compare_messages(query, b, a))
        responses = self._call_many(batches, response_format)
        votes = {a.id: 0, b.id: 0}
        for (x, y), r in zip(orders, responses):
            winner = self._parse_label(r.text, x, y)
            votes[winner.id] += 1
        return a if votes[a.id] >= votes[b.id] else b

    def _better(self, query: str, a: Candidate, b: Candidate) -> bool:
        return self.compare(query, a, b) is a

    def rank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        self._reset_stats()
        arr = list(candidates)
        k = len(arr) if self.k is None else min(self.k, len(arr))

        if self.strategy == "heapsort":
            top = self._heapsort(query, arr, k)
        elif self.strategy == "bubblesort":
            top = self._bubblesort(query, arr, k)
        else:
            top = self._allpairs(query, arr, k)

        return self._finalize(top, candidates)

    # -- heapsort ------------------------------------------------------

    def _sift_down(self, query: str, arr: list[Candidate], start: int, end: int) -> None:
        root = start
        while True:
            left = 2 * root + 1
            if left > end:
                return
            largest = root
            if self._better(query, arr[left], arr[largest]):
                largest = left
            right = left + 1
            if right <= end and self._better(query, arr[right], arr[largest]):
                largest = right
            if largest == root:
                return
            arr[root], arr[largest] = arr[largest], arr[root]
            root = largest

    def _heapsort(self, query: str, arr: list[Candidate], k: int) -> list[Candidate]:
        n = len(arr)
        for start in range((n - 2) // 2, -1, -1):
            self._sift_down(query, arr, start, n - 1)

        top: list[Candidate] = []
        end = n - 1
        while end >= 0 and len(top) < k:
            arr[0], arr[end] = arr[end], arr[0]
            top.append(arr[end])
            end -= 1
            if end >= 0:
                self._sift_down(query, arr, 0, end)
        return top

    # -- bubblesort ------------------------------------------------------

    def _bubblesort(self, query: str, arr: list[Candidate], k: int) -> list[Candidate]:
        n = len(arr)
        for i in range(k):
            for j in range(n - 1, i, -1):
                if self._better(query, arr[j], arr[j - 1]):
                    arr[j - 1], arr[j] = arr[j], arr[j - 1]
        return arr[:k]

    # -- allpairs ------------------------------------------------------

    def _allpairs(self, query: str, arr: list[Candidate], k: int) -> list[Candidate]:
        pairs = [(arr[i], arr[j]) for i in range(len(arr)) for j in range(i + 1, len(arr))]
        n = self.num_samples
        response_format = self._response_format()

        if n <= 1:
            batches = [self._build_compare_messages(query, a, b) for a, b in pairs]
            responses = self._call_many(batches, response_format)
            winners = [self._parse_label(r.text, a, b) for (a, b), r in zip(pairs, responses)]
        else:
            # Every pair's n samples are all mutually independent, so
            # dispatch them all together in one _call_many. This scales the
            # call count by n but not the wall-clock time under concurrency.
            orders: list[tuple[Candidate, Candidate]] = []
            batches = []
            for a, b in pairs:
                for _ in range(n):
                    if self._rng.random() < 0.5:
                        orders.append((a, b))
                        batches.append(self._build_compare_messages(query, a, b))
                    else:
                        orders.append((b, a))
                        batches.append(self._build_compare_messages(query, b, a))
            responses = self._call_many(batches, response_format)
            winners = []
            for i, (a, b) in enumerate(pairs):
                chunk_orders = orders[i * n : (i + 1) * n]
                chunk_responses = responses[i * n : (i + 1) * n]
                votes = {a.id: 0, b.id: 0}
                for (x, y), r in zip(chunk_orders, chunk_responses):
                    winner = self._parse_label(r.text, x, y)
                    votes[winner.id] += 1
                winners.append(a if votes[a.id] >= votes[b.id] else b)

        wins = {c.id: 0 for c in arr}
        for winner in winners:
            wins[winner.id] += 1
        ranked = sorted(arr, key=lambda c: wins[c.id], reverse=True)
        return ranked[:k]
