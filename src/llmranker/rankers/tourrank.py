from __future__ import annotations

import math
import random
import re

from ..llm import LLMConfig
from ..prompts import extract_final_answer, tourrank_group_user_prompt, tourrank_system_prompt
from ..types import Candidate
from .base import BaseRanker


class TourRankRanker(BaseRanker):
    """Tournament-style ranking (Chen et al., WWW'25, arXiv:2406.11678).

    Candidates are split into small groups -- like a sports tournament's
    group stage -- and an LLM selects the top `advance_per_group` of each
    group to advance, earning +1 point each. Survivors are regrouped and
    the process repeats for `num_stages`; the whole thing is one
    *tournament*. `num_tournaments` independent tournaments are run (fresh
    random grouping/shuffling each time) and points are summed across all
    of them. A candidate's final `score` is that total point count -- how
    many stages, across how many tournament runs, it survived. This is a
    genuine calibrated score (like `PointwiseRanker.score()`), not a
    synthetic rank position.

    Groups are formed by **round-robin seeding** from the candidates'
    current order (spreads strong/weak candidates evenly across groups)
    and then **shuffled** within the group before prompting (mitigates
    position bias in the prompt itself). Combined with averaging over
    several independent tournament runs, TourRank is explicitly designed
    to be **robust to the initial order of `candidates`** -- unlike
    `ListwiseRanker`'s sliding window, which is quite sensitive to it.

    Groups within a stage are independent of each other, so they're
    dispatched via `_call_many()` and parallelized by `max_concurrency`.
    Stages within a tournament, and separate tournament runs, are
    sequential.

    `k`: optional, purely truncates the *output* to the top `k` by points.
    Unlike `PairwiseRanker`/`SetwiseRanker` heapsort, this does **not**
    reduce LLM calls -- every group in every stage of every tournament
    still runs, since the points system needs the full tournament to be
    meaningful.

    Ties in final points are broken by each candidate's original position
    in `candidates` (stable sort).
    """

    def __init__(
        self,
        config: LLMConfig,
        group_size: int = 4,
        advance_per_group: int = 2,
        num_stages: int = 2,
        num_tournaments: int = 3,
        k: int | None = None,
        item_label: str = "item",
        system_prompt: str | None = None,
        name: str | None = None,
        max_concurrency: int = 5,
        reasoning: bool = False,
        seed: int | None = None,
    ):
        super().__init__(config, item_label, system_prompt, name, max_concurrency, reasoning)
        if group_size < 2:
            raise ValueError("group_size must be >= 2")
        if advance_per_group >= group_size:
            raise ValueError("advance_per_group must be < group_size")
        if group_size > 26:
            raise ValueError("group_size must be <= 26 (single-letter labels A-Z)")
        self.group_size = group_size
        self.advance_per_group = advance_per_group
        self.num_stages = num_stages
        self.num_tournaments = num_tournaments
        self.k = k
        self.seed = seed
        self.characters = [chr(ord("A") + i) for i in range(group_size)]

    def _build_group_messages(self, query: str, group: list[Candidate]) -> list[dict]:
        labels = self.characters[: len(group)]
        system = self.system_prompt_override or tourrank_system_prompt(self.item_label)
        user = tourrank_group_user_prompt(
            query, group, labels, self.advance_per_group, self.item_label, self.reasoning
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _parse_group_selection(self, text: str, group: list[Candidate]) -> list[Candidate]:
        labels = self.characters[: len(group)]
        label_re = re.compile(r"\b(" + "|".join(labels) + r")\b")
        text = extract_final_answer(text)

        found_order: list[str] = []
        seen: set[str] = set()
        for match in label_re.finditer(text):
            label = match.group(1)
            if label not in seen:
                found_order.append(label)
                seen.add(label)

        selected_indices = [labels.index(label) for label in found_order[: self.advance_per_group]]
        if len(selected_indices) < self.advance_per_group:
            for i in range(len(group)):
                if i not in selected_indices:
                    selected_indices.append(i)
                    if len(selected_indices) == self.advance_per_group:
                        break
        return [group[i] for i in selected_indices]

    def select_group(self, query: str, group: list[Candidate]) -> list[Candidate]:
        """Return the `advance_per_group` most relevant of `group` (order
        within the returned list isn't meaningful, only membership is)."""
        response = self._call(self._build_group_messages(query, group))
        return self._parse_group_selection(response.text, group)

    def _make_groups(
        self, active: list[Candidate], rng: random.Random
    ) -> list[list[Candidate]]:
        num_groups = math.ceil(len(active) / self.group_size)
        groups: list[list[Candidate]] = [[] for _ in range(num_groups)]
        for i, candidate in enumerate(active):
            groups[i % num_groups].append(candidate)
        for group in groups:
            rng.shuffle(group)
        return groups

    def _run_tournament(
        self, query: str, candidates: list[Candidate], rng: random.Random
    ) -> dict[str, int]:
        active = list(candidates)
        points = {c.id: 0 for c in candidates}

        for _ in range(self.num_stages):
            if len(active) <= self.advance_per_group:
                break

            groups = self._make_groups(active, rng)
            trivial_groups = [g for g in groups if len(g) <= self.advance_per_group]
            real_groups = [g for g in groups if len(g) > self.advance_per_group]
            survivors: list[Candidate] = []

            for group in trivial_groups:
                survivors.extend(group)
                for c in group:
                    points[c.id] += 1

            if real_groups:
                batches = [self._build_group_messages(query, g) for g in real_groups]
                responses = self._call_many(batches)
                for group, response in zip(real_groups, responses):
                    selected = self._parse_group_selection(response.text, group)
                    survivors.extend(selected)
                    for c in selected:
                        points[c.id] += 1

            active = survivors

        return points

    def rank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        self._reset_stats()
        rng = random.Random(self.seed)
        total_points = {c.id: 0 for c in candidates}

        for _ in range(self.num_tournaments):
            tournament_points = self._run_tournament(query, candidates, rng)
            for cid, p in tournament_points.items():
                total_points[cid] += p

        ranked = sorted(candidates, key=lambda c: total_points[c.id], reverse=True)
        if self.k is not None:
            ranked = ranked[: self.k]
        return [
            Candidate(id=c.id, text=c.text, score=float(total_points[c.id]), metadata=c.metadata)
            for c in ranked
        ]
