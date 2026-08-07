from __future__ import annotations

import itertools
import logging
import math
import random
import re

from .. import structured
from ..llm import LLMConfig
from ..prompts import extract_final_answer, tourrank_group_user_prompt, tourrank_system_prompt
from ..types import Candidate, copy_metadata
from .base import BaseRanker

logger = logging.getLogger("llmranker")

# The paper's schedule for 100 candidates is 100 -> 50 -> 20 -> 10 -> 5 -> 2,
# i.e. each stage keeps this fraction of the *original* pool. Expressed as
# fractions so the same shape applies to any candidate-list length.
_PAPER_SCHEDULE_FRACTIONS = (1 / 2, 1 / 5, 1 / 10, 1 / 20, 1 / 50)


class TourRankRanker(BaseRanker):
    """Tournament-style ranking (Chen et al., WWW'25, arXiv:2406.11678).

    Candidates play through a series of elimination stages. In each stage
    the field is split into groups, an LLM picks the best of each group to
    advance, and **every survivor earns +1 point**. The whole sequence is
    one *tournament*; `num_tournaments` independent tournaments are run
    (fresh grouping and shuffling each time) and points are summed. A
    candidate's final `score` is that total: how many stages, across how
    many runs, it survived. That's a genuine calibrated quantity, not a
    synthetic rank position.

    `schedule` controls the elimination: a list of how many candidates
    survive each stage, largest first. The default follows the paper's
    100 -> 50 -> 20 -> 10 -> 5 -> 2 shape scaled to the actual candidate
    count, which is what makes the points meaningful — with too few stages
    almost every candidate ends on the same score and the final order
    collapses back onto the tie-break (i.e. onto input order), defeating
    the point of the method. Pass an explicit list to override, e.g.
    `schedule=[20, 5]` for a cheap two-stage run.

    `group_size` is the maximum number of candidates shown in a single
    prompt (the paper uses 20). Each stage forms `ceil(len(active) /
    group_size)` groups and advances `ceil(target / num_groups)` from each.

    Each tournament draws its own groups: the field is put in a canonical
    order, shuffled with the ranker's seeded RNG, dealt round-robin into
    groups, and shuffled again within each group before prompting. The
    consequence worth relying on is that **the points a candidate earns
    depend on the candidate set and the seed, not on the order you passed
    them in** -- unlike `ListwiseRanker`'s sliding window, which is quite
    sensitive to it. Candidates that survive exactly the same stages do
    tie on points, and that tie is broken by input order (see below), so
    a schedule with too few stages hands most of the ordering back to the
    tie-break; see `schedule` above.

    Groups within a stage are independent, so they're dispatched via
    `_call_many()` and parallelized by `max_concurrency`. Stages within a
    tournament, and separate tournaments, are sequential.

    Like every ranker here, `rank()` returns **every** candidate it was
    given, ordered; slice the result yourself for a top-k.

    Ties in final points are broken by each candidate's original position
    in `candidates` (stable sort).

    `num_samples` is ignored here: `num_tournaments` is TourRank's own
    repeated-sampling-and-aggregation mechanism, and stacking `num_samples`
    on top of it would just compound two sampling schemes in a way that's
    hard to reason about. A warning is logged if it's set above 1; use
    `num_tournaments` instead.
    """

    score_kind = "tournament_points"

    def __init__(
        self,
        config: LLMConfig,
        group_size: int = 20,
        schedule: list[int] | None = None,
        num_tournaments: int = 10,
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
        if group_size < 2:
            raise ValueError("group_size must be >= 2")
        if group_size > 26:
            raise ValueError("group_size must be <= 26 (single-letter labels A-Z)")
        if schedule is not None:
            if not schedule:
                raise ValueError("schedule must not be empty")
            if any(t < 1 for t in schedule):
                raise ValueError("schedule targets must all be >= 1")
            if any(b >= a for a, b in itertools.pairwise(schedule)):
                raise ValueError("schedule must be strictly decreasing, largest first")
        if num_tournaments < 1:
            raise ValueError("num_tournaments must be >= 1")
        self.group_size = group_size
        self.schedule = schedule
        self.num_tournaments = num_tournaments
        self.seed = seed
        self.characters = [chr(ord("A") + i) for i in range(group_size)]
        self._warned_num_samples = False

    def _resolve_schedule(self, n: int) -> list[int]:
        """Survivor counts per stage for a pool of `n` candidates.

        The paper's shape, scaled: each stage keeps a fixed fraction of the
        original pool. Duplicates and targets that wouldn't eliminate
        anyone are dropped, so short lists simply get fewer stages rather
        than a run of no-op rounds.
        """
        if self.schedule is not None:
            return [t for t in self.schedule if t < n]
        targets: list[int] = []
        for fraction in _PAPER_SCHEDULE_FRACTIONS:
            target = max(2, round(n * fraction))
            if target < n and (not targets or target < targets[-1]):
                targets.append(target)
        return targets

    def _build_group_messages(self, query: str, group: list[Candidate], advance: int) -> list[dict]:
        labels = self.characters[: len(group)]
        system = self.system_prompt_override or tourrank_system_prompt(self.item_label)
        user = tourrank_group_user_prompt(
            query,
            group,
            labels,
            advance,
            self.item_label,
            self.reasoning,
            self.structured_output,
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _response_format(self, group: list[Candidate], advance: int) -> dict | None:
        if not self.structured_output:
            return None
        labels = self.characters[: len(group)]
        return structured.tourrank_schema(labels, advance)

    def _parse_group_selection(
        self, text: str, group: list[Candidate], advance: int
    ) -> list[Candidate]:
        labels = self.characters[: len(group)]
        if self.structured_output:
            selected = structured.parse_tourrank_json(text, labels)
            if selected is not None:
                # De-duplicate before the length check: a model can satisfy
                # the schema's minItems/maxItems with a repeated label
                # (["A", "A"]), which would otherwise advance one candidate
                # twice -- awarding it more points than there were stages
                # and shrinking the survivor pool. Falling short after
                # de-duplication drops through to the text parser below,
                # which fills the remaining slots.
                deduped = list(dict.fromkeys(selected))
                if len(deduped) == advance:
                    return [group[labels.index(label)] for label in deduped]
        label_re = re.compile(r"\b(" + "|".join(labels) + r")\b")
        text = extract_final_answer(text)

        found_order: list[str] = []
        seen: set[str] = set()
        for match in label_re.finditer(text):
            label = match.group(1)
            if label not in seen:
                found_order.append(label)
                seen.add(label)

        selected_indices = [labels.index(label) for label in found_order[:advance]]
        if len(selected_indices) < advance:
            for i in range(len(group)):
                if i not in selected_indices:
                    selected_indices.append(i)
                    if len(selected_indices) == advance:
                        break
        return [group[i] for i in selected_indices]

    def select_group(self, query: str, group: list[Candidate], advance: int) -> list[Candidate]:
        """Return the `advance` most relevant of `group` (order within the
        returned list isn't meaningful, only membership is)."""
        messages = self._build_group_messages(query, group, advance)
        response = self._call(messages, self._response_format(group, advance))
        return self._parse_group_selection(response.text, group, advance)

    def _make_groups(self, active: list[Candidate], rng: random.Random) -> list[list[Candidate]]:
        """Split `active` into groups for one stage.

        The pool is shuffled before the round-robin deal, then each group is
        shuffled again before prompting. Both matter, for different reasons:

        - Shuffling the *pool* means each tournament draws different
          match-ups. Without it, group membership is a pure function of the
          candidates' input positions, so every tournament in the ensemble
          plays out identically and `num_tournaments` just multiplies every
          score by a constant. It also means the result no longer depends
          on the order `candidates` arrived in, which is the property
          TourRank is chosen for.
        - Shuffling *within* a group varies which candidate lands on which
          label, mitigating the model's positional preference in the prompt.
        """
        # Sorted by id before shuffling so the draw is a function of (seed,
        # set of candidates) rather than of the order they were passed in --
        # shuffling an already-differently-ordered list with the same seed
        # still yields a different permutation, which would leave the result
        # quietly dependent on input order.
        pool = sorted(active, key=lambda c: c.id)
        rng.shuffle(pool)
        num_groups = math.ceil(len(pool) / self.group_size)
        groups: list[list[Candidate]] = [[] for _ in range(num_groups)]
        for i, candidate in enumerate(pool):
            groups[i % num_groups].append(candidate)
        for group in groups:
            rng.shuffle(group)
        return groups

    def _run_stage(
        self, query: str, active: list[Candidate], target: int, rng: random.Random
    ) -> list[Candidate]:
        """Cut `active` down to roughly `target` survivors in one stage."""
        groups = self._make_groups(active, rng)
        advance = max(1, math.ceil(target / len(groups)))

        survivors: list[Candidate] = []
        # A group no larger than its advance quota has nothing to decide, so
        # it advances whole without spending a call on it.
        trivial = [g for g in groups if len(g) <= advance]
        contested = [g for g in groups if len(g) > advance]
        for group in trivial:
            survivors.extend(group)

        if contested:
            # Batched by group size because structured_output's schema
            # depends on the label count, and _call_many takes one
            # response_format for the whole batch.
            by_size: dict[int, list[list[Candidate]]] = {}
            for g in contested:
                by_size.setdefault(len(g), []).append(g)
            for groups_of_size in by_size.values():
                batches = [self._build_group_messages(query, g, advance) for g in groups_of_size]
                response_format = self._response_format(groups_of_size[0], advance)
                responses = self._call_many(batches, response_format)
                for group, response in zip(groups_of_size, responses):
                    survivors.extend(self._parse_group_selection(response.text, group, advance))

        return survivors

    def _run_tournament(
        self, query: str, candidates: list[Candidate], rng: random.Random
    ) -> dict[str, int]:
        active = list(candidates)
        points = {c.id: 0 for c in candidates}

        for target in self._resolve_schedule(len(candidates)):
            if len(active) <= target:
                continue
            active = self._run_stage(query, active, target, rng)
            for c in active:
                points[c.id] += 1

        return points

    def rank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        self._reset_stats()
        if self.num_samples > 1 and not self._warned_num_samples:
            logger.warning(
                "num_samples=%d is ignored on TourRankRanker: "
                "num_tournaments is its own repeated-sampling mechanism.",
                self.num_samples,
            )
            self._warned_num_samples = True
        rng = random.Random(self.seed)
        total_points = {c.id: 0 for c in candidates}

        for _ in range(self.num_tournaments):
            for cid, p in self._run_tournament(query, candidates, rng).items():
                total_points[cid] += p

        ranked = sorted(candidates, key=lambda c: total_points[c.id], reverse=True)
        return [
            Candidate(
                id=c.id,
                text=c.text,
                score=float(total_points[c.id]),
                metadata=copy_metadata(c.metadata),
            )
            for c in ranked
        ]
