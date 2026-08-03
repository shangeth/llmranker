from __future__ import annotations

import logging
import re

from ..llm import LLMConfig
from ..prompts import extract_final_answer, pointwise_system_prompt, pointwise_user_prompt
from ..types import Candidate
from .base import BaseRanker

logger = logging.getLogger("llmranker")

_SCORE_RE = re.compile(r"-?\d+(?:\.\d+)?")


class PointwiseRanker(BaseRanker):
    """Scores every candidate independently, then sorts by score.

    Cheapest strategy: exactly `len(candidates)` LLM calls, no comparisons
    between candidates. Because each score is produced in isolation it is
    the weakest at capturing *relative* preference between similar
    candidates. Reach for pairwise/setwise/listwise when that matters more
    than cost.

    Every candidate's score is independent of every other's, so `rank()`
    dispatches all calls via `_call_many()` and is fully parallelized by
    `max_concurrency` (see `BaseRanker`).
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
    ):
        super().__init__(config, item_label, system_prompt, name, max_concurrency, reasoning)
        self.min_score = min_score
        self.max_score = max_score

    def _build_messages(self, query: str, candidate: Candidate) -> list[dict]:
        system = self.system_prompt_override or pointwise_system_prompt(self.item_label)
        user = pointwise_user_prompt(query, candidate, self.item_label, self.reasoning)
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _parse_score(self, text: str) -> float:
        text = extract_final_answer(text)
        match = _SCORE_RE.search(text)
        if match is None:
            logger.warning("Could not parse a score from output: %r", text)
            return self.min_score
        value = float(match.group())
        return min(max(value, self.min_score), self.max_score)

    def score(self, query: str, candidate: Candidate) -> float:
        response = self._call(self._build_messages(query, candidate))
        return self._parse_score(response.text)

    def rank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        self._reset_stats()
        batches = [self._build_messages(query, c) for c in candidates]
        responses = self._call_many(batches)
        scored = [
            Candidate(id=c.id, text=c.text, score=self._parse_score(r.text), metadata=c.metadata)
            for c, r in zip(candidates, responses)
        ]
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored
