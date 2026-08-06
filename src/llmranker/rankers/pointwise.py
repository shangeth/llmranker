from __future__ import annotations

import logging
import re

from .. import criteria as criteria_module
from .. import structured
from ..llm import LLMConfig
from ..prompts import (
    criteria_extraction_system_prompt,
    criteria_extraction_user_prompt,
    extract_final_answer,
    pointwise_multi_criteria_user_prompt,
    pointwise_system_prompt,
    pointwise_user_prompt,
)
from ..types import Candidate
from .base import BaseRanker

logger = logging.getLogger("llmranker")

_SCORE_RE = re.compile(r"-?\d+(?:\.\d+)?")


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
        self._warned_low_temperature = False

        if criteria is not None and criteria != "auto" and not isinstance(criteria, dict):
            raise ValueError('criteria must be a dict, "auto", or None')
        self.criteria = criteria
        self._criteria_weights: dict[str, float] | None = None
        if isinstance(criteria, dict):
            self._criteria_weights = criteria_module.resolve_weights(
                criteria, max_score - min_score
            )

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
        if self.structured_output:
            parsed = structured.parse_pointwise_json(text)
            if parsed is not None:
                return min(max(parsed, self.min_score), self.max_score)
        text = extract_final_answer(text)
        match = _SCORE_RE.search(text)
        if match is None:
            logger.warning("Could not parse a score from output: %r", text)
            return self.min_score
        value = float(match.group())
        return min(max(value, self.min_score), self.max_score)

    def _warn_if_low_temperature(self) -> None:
        low_temp = self.config.temperature == 0.0
        if self.num_samples > 1 and low_temp and not self._warned_low_temperature:
            logger.warning(
                "num_samples=%d but LLMConfig.temperature=0.0: repeated "
                "calls will return identical scores, so the extra calls don't "
                "add anything. Set temperature > 0 for num_samples to help.",
                self.num_samples,
            )
            self._warned_low_temperature = True

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
                    id=c.id, text=c.text, score=self._parse_score(r.text), metadata=c.metadata
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
                scored.append(Candidate(id=c.id, text=c.text, score=avg, metadata=c.metadata))

        scored.sort(key=lambda c: c.score, reverse=True)
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

        scored.sort(key=lambda c: c.score, reverse=True)
        return scored

    # -- public API -----------------------------------------------------

    def score(self, query: str, candidate: Candidate) -> float:
        if self.criteria is None:
            return self._score_holistic(query, candidate)
        return self._score_multi_criteria(query, candidate)

    def rank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        self._reset_stats()
        if self.criteria is None:
            return self._rank_holistic(query, candidates)
        return self._rank_multi_criteria(query, candidates)
