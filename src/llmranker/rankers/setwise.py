from __future__ import annotations

import logging
import random
import re

from .. import structured
from ..llm import LLMConfig
from ..prompts import extract_final_answer, setwise_system_prompt, setwise_user_prompt
from ..types import Candidate
from .base import BaseRanker

logger = logging.getLogger("llmranker")

_STRATEGIES = ("heapsort", "bubblesort", "insertion")


class SetwiseRanker(BaseRanker):
    """Sorts candidates via repeated n-way LLM comparisons ("which of these
    `num_child` items is the most relevant?").

    Compared to PairwiseRanker, grouping `num_child` candidates per LLM call
    cuts the number of calls roughly by a factor of `num_child`, at the cost
    of a longer prompt per call: the core trade-off introduced by the
    Setwise paper (arXiv:2310.09497), which this class implements.

    `strategy`:
      - "heapsort": build a `num_child`-ary max-heap, pop the top `k`.
      - "bubblesort": k passes, each sliding the best remaining candidate to
        the front of the unranked region, `num_child` at a time.
      - "insertion": "Setwise Insertion" (Podolak et al., SIGIR'25,
        arXiv:2504.10509), a direct efficiency successor to the above two.
        Sorts just the first `k` candidates from `candidates`' *existing*
        order into a max-heap-sorted top-k, then walks the rest in chunks,
        comparing each chunk against the current worst-of-top-k (the
        "guard"): if the guard wins, the whole chunk is discarded in one
        call; otherwise the winner is binary-inserted into its correct
        position. This is much cheaper than heapsort/bubblesort when
        `candidates` already arrives in a reasonable order (e.g. from an
        upstream retriever/embedding search) since most chunks get
        discarded in a single call. It degrades toward more calls (not
        incorrect results) on an unordered input, so the benefit depends on
        `candidates` carrying a meaningful prior order.

    All strategies are sequential comparison sorts: each comparison's
    outcome determines what gets compared next, so unlike PointwiseRanker
    or PairwiseRanker's "allpairs" strategy, `max_concurrency` does **not**
    parallelize the comparisons. It is still used when `num_samples > 1`,
    where the repeated judgments of a *single* comparison are independent
    and dispatched together.

    `num_samples > 1`: each group comparison reshuffles the
    candidate-to-label assignment per sample and decides the winner by
    majority vote across samples instead of trusting a single call, which
    cancels position bias (e.g. a tendency to favor whichever candidate
    lands on label "A") as a side effect. Costs `num_samples` calls per
    comparison instead of 1. `seed` makes the shuffling reproducible across
    a `rank()` call.
    """

    score_kind = "rank_position"

    def __init__(
        self,
        config: LLMConfig,
        num_child: int = 3,
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
        if num_child < 2:
            raise ValueError("num_child must be >= 2 (use PairwiseRanker for pairs)")
        if num_child > 25:
            raise ValueError(
                "num_child must be <= 25 (heapsort compares a node against its "
                "num_child children *and* itself, needing num_child+1 single-letter "
                "labels A-Z)"
            )
        self.num_child = num_child
        self.strategy = strategy
        self.k = k
        self.seed = seed
        self._rng = random.Random(seed)
        # Sized to num_child + 1: heapsort's sift-down compares a parent
        # against all of its children in one call, so the largest group any
        # single `compare()` call sees is num_child children + 1 parent.
        self.characters = [chr(ord("A") + i) for i in range(num_child + 1)]

    def _build_group_messages(
        self, query: str, group: list[Candidate], labels: list[str]
    ) -> list[dict]:
        system = self.system_prompt_override or setwise_system_prompt(self.item_label)
        user = setwise_user_prompt(
            query,
            group,
            labels,
            self.item_label,
            self.reasoning,
            self.structured_output,
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _response_format(self, labels: list[str]) -> dict | None:
        return structured.setwise_schema(labels) if self.structured_output else None

    def _parse_choice(self, text: str, group: list[Candidate], labels: list[str]) -> Candidate:
        if self.structured_output:
            choice = structured.parse_setwise_json(text, labels)
            if choice is not None:
                return group[labels.index(choice)]
        label_re = re.compile(r"\b(" + "|".join(labels) + r")\b")
        text = extract_final_answer(text)
        match = label_re.search(text.strip())
        if match is None:
            logger.warning("Could not parse a label from output: %r", text)
            return group[0]
        return group[labels.index(match.group(1))]

    def compare(self, query: str, candidates: list[Candidate]) -> Candidate:
        """Return whichever of `candidates` the LLM judges most relevant to `query`.

        See `num_samples` on the class for the reshuffled,
        majority-vote behavior when it's set above 1.
        """
        labels = self.characters[: len(candidates)]
        response_format = self._response_format(labels)
        n = self.num_samples

        if n <= 1:
            messages = self._build_group_messages(query, candidates, labels)
            response = self._call(messages, response_format)
            return self._parse_choice(response.text, candidates, labels)

        batches = []
        sample_groups = []
        for _ in range(n):
            shuffled = list(candidates)
            self._rng.shuffle(shuffled)
            batches.append(self._build_group_messages(query, shuffled, labels))
            sample_groups.append(shuffled)
        responses = self._call_many(batches, response_format)

        # Votes are indexed by position, not keyed on the `id` field:
        # candidates sharing an id must be counted as distinct entrants.
        votes = [0] * len(candidates)
        for group, response in zip(sample_groups, responses):
            winner = self._parse_choice(response.text, group, labels)
            for i, c in enumerate(candidates):
                if c is winner:
                    votes[i] += 1
                    break
        best_count = max(votes)
        tied = [candidates[i] for i, v in enumerate(votes) if v == best_count]
        if len(tied) == 1:
            return tied[0]
        # Resolving a tie by original position would reinstate the position
        # bias that reshuffling across samples is meant to cancel, so the
        # same seeded RNG that drives the shuffling breaks the tie too.
        return self._rng.choice(tied)

    def rank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        self._reset_stats()
        arr = list(candidates)
        k = len(arr) if self.k is None else min(self.k, len(arr))

        if self.strategy == "heapsort":
            top = self._heapsort(query, arr, k)
        elif self.strategy == "bubblesort":
            top = self._bubblesort(query, arr, k)
        else:
            top = self._insertion(query, arr, k)

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
        """Slide a window leftward across arr[start:end+1], carrying the
        winner of each window forward, so the true best of the whole range
        ends up at `start`.

        The window holds `num_child + 1` candidates, matching heapsort's
        parent-plus-children group and the Setwise paper's reference
        implementation: one slot carries the running winner and `num_child`
        slots bring in new contenders. Stepping by `num_child` keeps that
        carried winner in the next window.
        """
        window = self.num_child + 1
        pos = max(start, end - window + 1)
        while True:
            group_indices = list(range(pos, min(pos + window, end + 1)))
            # A one-candidate window has nothing to decide; asking the model
            # which of a single item is best is a guaranteed-wasted call.
            if len(group_indices) >= 2:
                group = [arr[i] for i in group_indices]
                winner = self.compare(query, group)
                winner_pos = next(i for i, c in enumerate(group) if c is winner)
                winner_index = group_indices[winner_pos]
                if winner_index != pos:
                    arr[pos], arr[winner_index] = arr[winner_index], arr[pos]
            if pos == start:
                return
            pos = max(start, pos - self.num_child)

    def _bubblesort(self, query: str, arr: list[Candidate], k: int) -> list[Candidate]:
        n = len(arr)
        for i in range(k):
            # Nothing left to order once one candidate remains unplaced.
            if n - i < 2:
                break
            self._bubble_pass(query, arr, i, n - 1)
        return arr[:k]

    # -- insertion (Setwise Insertion, arXiv:2504.10509) ------------------

    def _binary_insert_position(
        self, query: str, sorted_list: list[Candidate], item: Candidate
    ) -> int:
        """Find where `item` belongs in `sorted_list` (best-to-worst) using
        O(log n) pairwise-style compare() calls."""
        lo, hi = 0, len(sorted_list)
        while lo < hi:
            mid = (lo + hi) // 2
            winner = self.compare(query, [item, sorted_list[mid]])
            if winner is item:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def _insertion(self, query: str, arr: list[Candidate], k: int) -> list[Candidate]:
        if k <= 0:
            return []
        s = self._heapsort(query, arr[:k], k)
        chunk_size = max(1, self.num_child - 1)
        remaining = arr[k:]
        for start in range(0, len(remaining), chunk_size):
            chunk = remaining[start : start + chunk_size]
            guard = s[-1]
            winner = self.compare(query, [guard] + chunk)
            if winner is guard:
                continue
            s = s[:-1]
            pos = self._binary_insert_position(query, s, winner)
            s.insert(pos, winner)
        return s
