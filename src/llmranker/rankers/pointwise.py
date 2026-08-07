from __future__ import annotations

import logging
import random
import re

from .. import criteria as criteria_module
from .. import structured
from ..llm import LLMConfig, LLMResponse
from ..prompts import (
    criteria_extraction_system_prompt,
    criteria_extraction_user_prompt,
    extract_final_answer,
    pointwise_batch_user_prompt,
    pointwise_multi_criteria_user_prompt,
    pointwise_system_prompt,
    pointwise_user_prompt,
)
from ..types import Candidate, copy_metadata
from .base import BaseRanker

logger = logging.getLogger("llmranker")

_NUMBER = r"-?\d+(?:\.\d+)?"
_SCORE_RE = re.compile(_NUMBER)
_BARE_SCORE_RE = re.compile(rf"^\s*({_NUMBER})\s*$")
# "score: 8", "score = 8", "Relevance score 8", "scores 8/10", "score of 9".
# The connector between the word and the number is deliberately narrow, so
# a "score" mentioned far from the actual number doesn't produce a match.
_LABELLED_SCORE_RE = re.compile(rf"scores?\b\s*(?:of|is|was)?\s*[:=]?\s*({_NUMBER})", re.IGNORECASE)


def _strip_denominators(text: str, max_score: float) -> str:
    """Remove '/10' and 'out of 10' style denominators before scanning for a
    number, so '9/10' reads as 9 rather than risking the 10."""
    for pattern in (
        rf"\s*/\s*{re.escape(str(int(max_score)))}\b",
        rf"\s*out\s+of\s+{re.escape(str(int(max_score)))}\b",
    ):
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text


class PointwiseRanker(BaseRanker):
    """Scores every candidate independently, then sorts by score.

    Cheapest strategy: exactly `len(candidates)` LLM calls (times
    `num_samples`, see below), no comparisons between candidates. Because
    each score is produced in isolation it is the weakest at capturing
    *relative* preference between similar candidates. Reach for
    pairwise/setwise/listwise when that matters more than cost.

    Every candidate's score is independent of every other's, so `rank()`
    dispatches all calls via `_call_many()` and is fully parallelized by
    `max_concurrency` (see `BaseRanker`).

    `num_samples > 1` repeats each candidate's scoring call and averages
    the results. This only helps when `LLMConfig.temperature > 0`: at the
    default `temperature=0.0` every repeat scores identically, so the
    extra calls are wasted; a warning is logged if you set `num_samples >
    1` without raising `temperature`.

    `batch_size > 1` scores several candidates per call instead of one,
    using Shuffled-Then-Batched self-consistency (arXiv:2505.12570): each
    of `num_samples` rounds reshuffles *all* candidates and re-splits them
    into `batch_size`-sized groups (so which candidates share a call, not
    just their order, changes every round), and a candidate's final score
    is the mean of its score across every round it appeared in. Because
    batch composition genuinely varies per round, `num_samples > 1` helps
    here even at `temperature=0.0` (no warning is logged for this case,
    matching pairwise/setwise). `rank()`-only: `score()` scores one
    candidate at a time and ignores `batch_size`. Not currently supported
    together with `criteria` (raises `ValueError` at construction).
    `seed` makes the per-round shuffling reproducible.

    `criteria`, optional, scores named sub-criteria separately instead of
    one holistic judgment, then combines them:
      - a dict of name -> positive number: weighted sum, normalized
        internally (weights need not sum to 1).
      - a dict of name -> "high"/"medium"/"low": priority-hierarchical.
        A higher tier mathematically dominates any possible combination of
        lower tiers (see `llmranker.criteria.resolve_weights`), not a
        blend, so a candidate can't compensate for failing a high-priority
        criterion by scoring well on lower-priority ones.
      - `"auto"`: the LLM extracts criteria from the query itself (one
        extra call per `rank()`, not per candidate) and they're combined
        with equal weight. Falls back to plain holistic scoring if
        extraction produces nothing parseable.
    Mixing weight types in one dict, non-positive weights, an empty dict,
    or an unrecognized `criteria` value all raise `ValueError` at
    construction time. Every criterion is scored on the same
    `min_score`/`max_score` range as holistic scoring, in exactly the same
    number of calls (`"auto"` adds exactly one extra call for the whole
    `rank()`, not per candidate) - see the README for cost details.
    `rank()`'s output candidates carry the per-criterion breakdown in
    `Candidate.metadata["criteria_scores"]` (merged with, not overwriting,
    any metadata already on the input candidate); `score()` keeps its
    plain `float` return and re-extracts on every call in `"auto"` mode,
    so prefer `rank()` over repeated `score()` calls when using it.
    """

    score_kind = "relevance"

    def __init__(
        self,
        config: LLMConfig,
        item_label: str = "item",
        system_prompt: str | None = None,
        min_score: float = 0,
        max_score: float = 10,
        name: str | None = None,
        max_concurrency: int = 5,
        reasoning: bool = False,
        num_samples: int = 1,
        structured_output: bool = False,
        criteria: dict[str, float] | dict[str, str] | str | None = None,
        batch_size: int = 1,
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
        self.min_score = min_score
        self.max_score = max_score

        if criteria is not None and criteria != "auto" and not isinstance(criteria, dict):
            raise ValueError('criteria must be a dict, "auto", or None')
        self.criteria = criteria
        self._criteria_weights: dict[str, float] | None = None
        if isinstance(criteria, dict):
            self._criteria_weights = criteria_module.resolve_weights(
                criteria, max_score - min_score
            )

        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if batch_size > 1 and criteria is not None:
            raise ValueError("batch_size > 1 is not currently supported together with criteria")
        self.batch_size = batch_size
        self.seed = seed
        self._rng = random.Random(seed)

    # -- holistic (single-score) path --------------------------------------

    def _build_messages(self, query: str, candidate: Candidate) -> list[dict]:
        system = self.system_prompt_override or pointwise_system_prompt(self.item_label)
        user = pointwise_user_prompt(
            query,
            candidate,
            self.item_label,
            self.reasoning,
            self.structured_output,
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _response_format(self) -> dict | None:
        return structured.pointwise_schema() if self.structured_output else None

    def _parse_score(self, text: str) -> float:
        """Pull a numeric score out of the model's response.

        The prompt asks for a bare number, so that's tried first. Failing
        that, an explicitly labelled score ("score: 8", "scores 8") is
        preferred over a naked scan, because the first number in a chatty
        response is often *not* the score (e.g. "Item 3 deserves a 9").
        Only if neither matches does it fall back to the first number in
        the text, with denominators stripped so "9/10" reads as 9, and a
        warning when the text is ambiguous enough that the guess could be
        wrong.
        """
        if self.structured_output:
            parsed = structured.parse_pointwise_json(text)
            if parsed is not None:
                return min(max(parsed, self.min_score), self.max_score)

        text = extract_final_answer(text)

        bare = _BARE_SCORE_RE.match(text)
        if bare is not None:
            return self._clamp(float(bare.group(1)))

        labelled = _LABELLED_SCORE_RE.search(text)
        if labelled is not None:
            return self._clamp(float(labelled.group(1)))

        stripped = _strip_denominators(text, self.max_score)
        numbers = _SCORE_RE.findall(stripped)
        if not numbers:
            logger.warning("Could not parse a score from output: %r", text)
            return self.min_score
        if len(numbers) > 1:
            logger.warning(
                "Ambiguous score output (%d numbers, none labelled): %r. "
                "Using the first (%s); consider structured_output=True.",
                len(numbers),
                text,
                numbers[0],
            )
        return self._clamp(float(numbers[0]))

    def _clamp(self, value: float) -> float:
        return min(max(value, self.min_score), self.max_score)

    # -- batched (STB self-consistency) path --------------------------------

    def _build_batch_messages(
        self, query: str, candidates: list[Candidate], labels: list[str]
    ) -> list[dict]:
        system = self.system_prompt_override or pointwise_system_prompt(self.item_label)
        user = pointwise_batch_user_prompt(
            query,
            candidates,
            labels,
            self.item_label,
            self.reasoning,
            self.structured_output,
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _parse_batch_scores(self, text: str, labels: list[str]) -> dict[str, float]:
        if self.structured_output:
            parsed = structured.parse_pointwise_batch_json(text, labels)
            if parsed is not None:
                scored = {label: self._clamp(parsed[label]) for label in parsed}
                missing = [label for label in labels if label not in scored]
                if missing:
                    logger.warning("Batch response missing labels %s; using min_score", missing)
                    for label in missing:
                        scored[label] = self.min_score
                return scored

        text = extract_final_answer(text)
        result: dict[str, float] = {}
        for label in labels:
            match = re.search(rf"\b{re.escape(label)}\b\s*[:=]?\s*({_NUMBER})", text)
            if match:
                result[label] = self._clamp(float(match.group(1)))
        missing = [label for label in labels if label not in result]
        if missing:
            logger.warning(
                "Could not parse scores for labels %s in batch response: %r", missing, text
            )
            for label in missing:
                result[label] = self.min_score
        return result

    def _rank_batched(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        batches: list[tuple[list[Candidate], list[str]]] = []
        for _ in range(self.num_samples):
            shuffled = list(candidates)
            self._rng.shuffle(shuffled)
            for start in range(0, len(shuffled), self.batch_size):
                group = shuffled[start : start + self.batch_size]
                labels = [chr(ord("A") + i) for i in range(len(group))]
                batches.append((group, labels))

        # A structured-output schema is sized to its batch (the label enum),
        # so batches of different sizes -- the last, shorter batch of an
        # uneven split -- can't share one response_format. `_call_many`
        # takes a single response_format for its whole dispatch, so group
        # batches by size and dispatch each size together instead of one
        # call per individual batch.
        by_size: dict[int, list[int]] = {}
        for i, (group, _labels) in enumerate(batches):
            by_size.setdefault(len(group), []).append(i)

        responses: list[LLMResponse | None] = [None] * len(batches)
        for size, indices in by_size.items():
            labels = [chr(ord("A") + i) for i in range(size)]
            response_format = (
                structured.pointwise_batch_schema(labels) if self.structured_output else None
            )
            message_batches = [
                self._build_batch_messages(query, batches[i][0], batches[i][1]) for i in indices
            ]
            for idx, resp in zip(indices, self._call_many(message_batches, response_format)):
                responses[idx] = resp

        totals = {c.id: 0.0 for c in candidates}
        counts = {c.id: 0 for c in candidates}
        for (group, labels), response in zip(batches, responses):
            assert response is not None
            scores = self._parse_batch_scores(response.text, labels)
            for label, candidate in zip(labels, group):
                totals[candidate.id] += scores[label]
                counts[candidate.id] += 1

        scored = [
            Candidate(
                id=c.id,
                text=c.text,
                score=totals[c.id] / counts[c.id] if counts[c.id] else self.min_score,
                metadata=copy_metadata(c.metadata),
            )
            for c in candidates
        ]
        scored.sort(key=lambda c: c.score or 0.0, reverse=True)
        return scored

    def _score_holistic(self, query: str, candidate: Candidate) -> float:
        n = self.num_samples
        response_format = self._response_format()
        if n <= 1:
            response = self._call(self._build_messages(query, candidate), response_format)
            return self._parse_score(response.text)
        self._warn_if_low_temperature()
        batches = [self._build_messages(query, candidate) for _ in range(n)]
        responses = self._call_many(batches, response_format)
        scores = [self._parse_score(r.text) for r in responses]
        return sum(scores) / len(scores)

    def _rank_holistic(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        n = self.num_samples
        response_format = self._response_format()

        if n <= 1:
            batches = [self._build_messages(query, c) for c in candidates]
            responses = self._call_many(batches, response_format)
            scored = [
                Candidate(
                    id=c.id,
                    text=c.text,
                    score=self._parse_score(r.text),
                    metadata=copy_metadata(c.metadata),
                )
                for c, r in zip(candidates, responses)
            ]
        else:
            self._warn_if_low_temperature()
            batches = [self._build_messages(query, c) for c in candidates for _ in range(n)]
            responses = self._call_many(batches, response_format)
            scored = []
            for i, c in enumerate(candidates):
                chunk = responses[i * n : (i + 1) * n]
                avg = sum(self._parse_score(r.text) for r in chunk) / n
                scored.append(
                    Candidate(id=c.id, text=c.text, score=avg, metadata=copy_metadata(c.metadata))
                )

        scored.sort(key=lambda c: c.score or 0.0, reverse=True)
        return scored

    # -- multi-criteria path ------------------------------------------------

    def _extract_criteria(self, query: str) -> list[str]:
        system = criteria_extraction_system_prompt(self.item_label)
        user = criteria_extraction_user_prompt(
            query, self.item_label, self.reasoning, self.structured_output
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        response_format = None
        if self.structured_output:
            response_format = structured.criteria_extraction_schema()
        response = self._call(messages, response_format)

        names = None
        if self.structured_output:
            names = structured.parse_criteria_extraction_json(response.text)
        if names is None:
            names = criteria_module.parse_extracted_criteria_text(response.text)
        return [n.strip() for n in names if n and n.strip()]

    def _get_criteria_names_and_weights(
        self, query: str
    ) -> tuple[list[str], dict[str, float], str]:
        if self.criteria == "auto":
            names = self._extract_criteria(query)
            if not names:
                logger.warning(
                    "Criteria extraction returned nothing parseable; "
                    "falling back to holistic scoring for this rank() call."
                )
                return [], {}, "auto"
            weights = criteria_module.resolve_weights(
                {n: 1.0 for n in names}, self.max_score - self.min_score
            )
            return names, weights, "auto"
        # Only reachable for the dict form: `criteria=None` short-circuits
        # before this method is called, and "auto" is handled above.
        assert isinstance(self.criteria, dict) and self._criteria_weights is not None
        return list(self.criteria.keys()), self._criteria_weights, "user"

    def _build_multi_criteria_messages(
        self, query: str, candidate: Candidate, names: list[str]
    ) -> list[dict]:
        system = self.system_prompt_override or pointwise_system_prompt(self.item_label)
        user = pointwise_multi_criteria_user_prompt(
            query, candidate, names, self.item_label, self.reasoning, self.structured_output
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _multi_criteria_response_format(self, names: list[str]) -> dict | None:
        if not self.structured_output:
            return None
        return structured.pointwise_multi_criteria_schema(names)

    def _parse_multi_criteria(self, text: str, names: list[str]) -> dict[str, float]:
        if self.structured_output:
            parsed = structured.parse_pointwise_multi_criteria_json(text, names)
            if parsed is not None:
                return {k: min(max(v, self.min_score), self.max_score) for k, v in parsed.items()}
        scores = criteria_module.parse_criteria_text(text, names, self.min_score)
        return {k: min(max(v, self.min_score), self.max_score) for k, v in scores.items()}

    def _score_criteria_for_candidate(
        self, query: str, candidate: Candidate, names: list[str]
    ) -> dict[str, float]:
        n = self.num_samples
        response_format = self._multi_criteria_response_format(names)
        if n <= 1:
            messages = self._build_multi_criteria_messages(query, candidate, names)
            response = self._call(messages, response_format)
            return self._parse_multi_criteria(response.text, names)
        self._warn_if_low_temperature()
        batches = [self._build_multi_criteria_messages(query, candidate, names) for _ in range(n)]
        responses = self._call_many(batches, response_format)
        parsed = [self._parse_multi_criteria(r.text, names) for r in responses]
        return {name: sum(p[name] for p in parsed) / n for name in names}

    def _score_multi_criteria(self, query: str, candidate: Candidate) -> float:
        names, weights, _source = self._get_criteria_names_and_weights(query)
        if not names:
            return self._score_holistic(query, candidate)
        criterion_scores = self._score_criteria_for_candidate(query, candidate, names)
        final = sum(weights[name] * criterion_scores[name] for name in names)
        return min(max(final, self.min_score), self.max_score)

    def _rank_multi_criteria(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        names, weights, source = self._get_criteria_names_and_weights(query)
        if not names:
            return self._rank_holistic(query, candidates)

        n = self.num_samples
        response_format = self._multi_criteria_response_format(names)

        if n <= 1:
            batches = [self._build_multi_criteria_messages(query, c, names) for c in candidates]
            responses = self._call_many(batches, response_format)
            per_candidate_scores = [self._parse_multi_criteria(r.text, names) for r in responses]
        else:
            self._warn_if_low_temperature()
            batches = [
                self._build_multi_criteria_messages(query, c, names)
                for c in candidates
                for _ in range(n)
            ]
            responses = self._call_many(batches, response_format)
            per_candidate_scores = []
            for i in range(len(candidates)):
                chunk = responses[i * n : (i + 1) * n]
                parsed = [self._parse_multi_criteria(r.text, names) for r in chunk]
                per_candidate_scores.append(
                    {name: sum(p[name] for p in parsed) / n for name in names}
                )

        scored = []
        for c, criterion_scores in zip(candidates, per_candidate_scores):
            final = sum(weights[name] * criterion_scores[name] for name in names)
            final = min(max(final, self.min_score), self.max_score)
            metadata = {
                **(c.metadata or {}),
                "criteria_scores": criterion_scores,
                "criteria_weights": weights,
                "criteria_source": source,
            }
            scored.append(Candidate(id=c.id, text=c.text, score=final, metadata=metadata))

        scored.sort(key=lambda c: c.score or 0.0, reverse=True)
        return scored

    # -- public API -----------------------------------------------------

    def score(self, query: str, candidate: Candidate) -> float:
        if self.criteria is None:
            return self._score_holistic(query, candidate)
        return self._score_multi_criteria(query, candidate)

    def rank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        self._reset_stats()
        if self.criteria is None:
            if self.batch_size > 1:
                return self._rank_batched(query, candidates)
            return self._rank_holistic(query, candidates)
        return self._rank_multi_criteria(query, candidates)
