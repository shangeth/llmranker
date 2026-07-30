from __future__ import annotations

import logging
import re

from ..llm import LLMConfig
from ..prompts import extract_final_answer, pairwise_system_prompt, pairwise_user_prompt
from ..types import Candidate
from .base import BaseRanker

logger = logging.getLogger("llmranker")

_LABEL_RE = re.compile(r"\b([AB])\b")

_METHODS = ("heapsort", "bubblesort", "allpairs")


class PairwiseRanker(BaseRanker):
    """Sorts candidates via repeated pairwise LLM comparisons ("which is more
    relevant, A or B?").

    `method`:
      - "heapsort": build a max-heap, pop the top `k`. O(n + k log n) comparisons,
        run sequentially -- each comparison's outcome determines the next one,
        so `max_concurrency` has no effect here.
      - "bubblesort": k bubble passes, each moving the best remaining candidate
        to the front. O(n * k) comparisons, also sequential for the same reason.
      - "allpairs": every candidate compared against every other once, ranked
        by win count. O(n^2) comparisons -- most robust to individual
        comparison noise, and the only method here where comparisons are
        independent of each other, so it's fully parallelized by
        `max_concurrency` (see `BaseRanker`).

    Only `score()` on `PointwiseRanker` produces an independently calibrated
    relevance score. Here the output `score` is a synthetic rank position
    (higher = more relevant) since comparisons only ever establish relative
    order, not an absolute value.

    `debias_position`: LLMs have a documented bias toward whichever
    candidate is listed first (or second, model-dependent) in a pairwise
    prompt. When True, every comparison is run both ways (a-as-A/b-as-B and
    b-as-A/a-as-B); the result is only trusted when both agree, otherwise
    the comparison is treated as too ambiguous to trust and defaults to `a`
    (the original first argument). This roughly **doubles** the LLM calls
    for whichever comparisons it's applied to -- off by default.
    """

    def __init__(
        self,
        config: LLMConfig,
        method: str = "heapsort",
        k: int | None = None,
        item_label: str = "item",
        system_prompt: str | None = None,
        name: str | None = None,
        max_concurrency: int = 5,
        reasoning: bool = False,
        debias_position: bool = False,
    ):
        super().__init__(config, item_label, system_prompt, name, max_concurrency, reasoning)
        if method not in _METHODS:
            raise ValueError(f"Unknown method {method!r}, expected one of {_METHODS}")
        self.method = method
        self.k = k
        self.debias_position = debias_position

    def _build_compare_messages(self, query: str, a: Candidate, b: Candidate) -> list[dict]:
        system = self.system_prompt_override or pairwise_system_prompt(self.item_label)
        user = pairwise_user_prompt(query, [a, b], self.item_label, self.reasoning)
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _parse_label(self, text: str, a: Candidate, b: Candidate) -> Candidate:
        text = extract_final_answer(text)
        match = _LABEL_RE.search(text.strip())
        if match is None:
            logger.warning("Could not parse an A/B label from output: %r", text)
            return a
        return a if match.group(1) == "A" else b

    def compare(self, query: str, a: Candidate, b: Candidate) -> Candidate:
        """Return whichever of `a`/`b` the LLM judges more relevant to `query`.

        See `debias_position` on the class for the bidirectional-agreement
        behavior when enabled.
        """
        forward = self._call(self._build_compare_messages(query, a, b))
        forward_winner = self._parse_label(forward.text, a, b)
        if not self.debias_position:
            return forward_winner
        backward = self._call(self._build_compare_messages(query, b, a))
        backward_winner = self._parse_label(backward.text, b, a)
        return forward_winner if forward_winner is backward_winner else a

    def _better(self, query: str, a: Candidate, b: Candidate) -> bool:
        return self.compare(query, a, b) is a

    def rank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        self._reset_stats()
        arr = list(candidates)
        k = len(arr) if self.k is None else min(self.k, len(arr))

        if self.method == "heapsort":
            top = self._heapsort(query, arr, k)
        elif self.method == "bubblesort":
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

        if not self.debias_position:
            batches = [self._build_compare_messages(query, a, b) for a, b in pairs]
            responses = self._call_many(batches)
            winners = [self._parse_label(r.text, a, b) for (a, b), r in zip(pairs, responses)]
        else:
            # Forward and backward calls for every pair are all mutually
            # independent, so dispatch them together in one _call_many --
            # doubles the call count but not the wall-clock time under
            # concurrency.
            forward_batches = [self._build_compare_messages(query, a, b) for a, b in pairs]
            backward_batches = [self._build_compare_messages(query, b, a) for a, b in pairs]
            responses = self._call_many(forward_batches + backward_batches)
            forward_responses = responses[: len(pairs)]
            backward_responses = responses[len(pairs) :]
            winners = []
            for (a, b), fr, br in zip(pairs, forward_responses, backward_responses):
                fw = self._parse_label(fr.text, a, b)
                bw = self._parse_label(br.text, b, a)
                winners.append(fw if fw is bw else a)

        wins = {c.id: 0 for c in arr}
        for winner in winners:
            wins[winner.id] += 1
        ranked = sorted(arr, key=lambda c: wins[c.id], reverse=True)
        return ranked[:k]
