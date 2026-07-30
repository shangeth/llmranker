from __future__ import annotations

import logging
import re

from ..llm import LLMConfig
from ..prompts import setwise_system_prompt, setwise_user_prompt
from ..types import Candidate
from .base import BaseRanker

logger = logging.getLogger("llmranker")

_METHODS = ("heapsort", "bubblesort")


class SetwiseRanker(BaseRanker):
    """Sorts candidates via repeated n-way LLM comparisons ("which of these
    `num_child` items is the most relevant?").

    Compared to PairwiseRanker, grouping `num_child` candidates per LLM call
    cuts the number of calls roughly by a factor of `num_child`, at the cost
    of a longer prompt per call -- the core trade-off introduced by the
    Setwise paper (arXiv:2310.09497) this package is named after.

    `method`:
      - "heapsort": build a `num_child`-ary max-heap, pop the top `k`.
      - "bubblesort": k passes, each sliding the best remaining candidate to
        the front of the unranked region, `num_child` at a time.

    Both methods are sequential comparison sorts -- each comparison's
    outcome determines what gets compared next, so unlike PointwiseRanker
    or PairwiseRanker's "allpairs" method, `max_concurrency` has **no
    effect** here. It's accepted for constructor-signature consistency with
    the other rankers only.
    """

    def __init__(
        self,
        config: LLMConfig,
        num_child: int = 3,
        method: str = "heapsort",
        k: int | None = None,
        item_label: str = "item",
        system_prompt: str | None = None,
        name: str | None = None,
        max_concurrency: int = 5,
    ):
        super().__init__(config, item_label, system_prompt, name, max_concurrency)
        if method not in _METHODS:
            raise ValueError(f"Unknown method {method!r}, expected one of {_METHODS}")
        if num_child < 2:
            raise ValueError("num_child must be >= 2 (use PairwiseRanker for pairs)")
        if num_child > 25:
            raise ValueError(
                "num_child must be <= 25 (heapsort compares a node against its "
                "num_child children *and* itself, needing num_child+1 single-letter "
                "labels A-Z)"
            )
        self.num_child = num_child
        self.method = method
        self.k = k
        # Sized to num_child + 1: heapsort's sift-down compares a parent
        # against all of its children in one call, so the largest group any
        # single `compare()` call sees is num_child children + 1 parent.
        self.characters = [chr(ord("A") + i) for i in range(num_child + 1)]

    def compare(self, query: str, candidates: list[Candidate]) -> Candidate:
        """Return whichever of `candidates` the LLM judges most relevant to `query`."""
        labels = self.characters[: len(candidates)]
        label_re = re.compile(r"\b(" + "|".join(labels) + r")\b")
        system = self.system_prompt_override or setwise_system_prompt(self.item_label)
        user = setwise_user_prompt(query, candidates, labels, self.item_label)
        response = self._call(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        )
        match = label_re.search(response.text.strip())
        if match is None:
            logger.warning("Could not parse a label from output: %r", response.text)
            return candidates[0]
        return candidates[labels.index(match.group(1))]

    def rank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        self._reset_stats()
        arr = list(candidates)
        k = len(arr) if self.k is None else min(self.k, len(arr))

        if self.method == "heapsort":
            top = self._heapsort(query, arr, k)
        else:
            top = self._bubblesort(query, arr, k)

        return self._finalize(top, candidates)

    # -- heapsort ------------------------------------------------------

    def _sift_down(self, query: str, arr: list[Candidate], root: int, end: int) -> None:
        """Sift the element at `root` down a `num_child`-ary max-heap over
        indices [0, end]."""
        while True:
            first_child = self.num_child * root + 1
            if first_child > end:
                return
            last_child = min(first_child + self.num_child - 1, end)
            group_indices = [root] + list(range(first_child, last_child + 1))
            group = [arr[i] for i in group_indices]
            winner = self.compare(query, group)
            winner_pos = next(i for i, c in enumerate(group) if c is winner)
            winner_index = group_indices[winner_pos]
            if winner_index == root:
                return
            arr[root], arr[winner_index] = arr[winner_index], arr[root]
            root = winner_index

    def _heapsort(self, query: str, arr: list[Candidate], k: int) -> list[Candidate]:
        n = len(arr)
        for start in range((n - 2) // self.num_child, -1, -1):
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

    def _bubble_pass(self, query: str, arr: list[Candidate], start: int, end: int) -> None:
        """Slide `num_child`-sized windows leftward across arr[start:end+1],
        carrying the winner of each window forward, so the true best of the
        whole range ends up at `start`."""
        pos = max(start, end - self.num_child + 1)
        while True:
            window_end = min(pos + self.num_child, end + 1)
            group_indices = list(range(pos, window_end))
            group = [arr[i] for i in group_indices]
            winner = self.compare(query, group)
            winner_pos = next(i for i, c in enumerate(group) if c is winner)
            winner_index = group_indices[winner_pos]
            if winner_index != pos:
                arr[pos], arr[winner_index] = arr[winner_index], arr[pos]
            if pos == start:
                return
            pos = max(start, pos - (self.num_child - 1))

    def _bubblesort(self, query: str, arr: list[Candidate], k: int) -> list[Candidate]:
        n = len(arr)
        for i in range(k):
            self._bubble_pass(query, arr, i, n - 1)
        return arr[:k]
