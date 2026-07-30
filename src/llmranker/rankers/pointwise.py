from __future__ import annotations

import logging
import re

from ..llm import LLMConfig
from ..prompts import pointwise_system_prompt, pointwise_user_prompt
from ..types import Candidate
from .base import BaseRanker

logger = logging.getLogger("llmranker")

_SCORE_RE = re.compile(r"-?\d+(?:\.\d+)?")


class PointwiseRanker(BaseRanker):
    """Scores every candidate independently, then sorts by score.

    Cheapest strategy: exactly `len(candidates)` LLM calls, no comparisons
    between candidates. Because each score is produced in isolation it is
    the weakest at capturing *relative* preference between similar
    candidates -- reach for pairwise/setwise/listwise when that matters more
    than cost.
    """

    def __init__(
        self,
        config: LLMConfig,
        item_label: str = "item",
        system_prompt: str | None = None,
        min_score: float = 0,
        max_score: float = 10,
        name: str | None = None,
    ):
        super().__init__(config, item_label, system_prompt, name)
        self.min_score = min_score
        self.max_score = max_score

    def score(self, query: str, candidate: Candidate) -> float:
        system = self.system_prompt_override or pointwise_system_prompt(self.item_label)
        user = pointwise_user_prompt(query, candidate, self.item_label)
        response = self._call(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        )
        match = _SCORE_RE.search(response.text)
        if match is None:
            logger.warning("Could not parse a score from output: %r", response.text)
            return self.min_score
        value = float(match.group())
        return min(max(value, self.min_score), self.max_score)

    def rank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        self._reset_stats()
        scored = [
            Candidate(
                id=c.id,
                text=c.text,
                score=self.score(query, c),
                metadata=c.metadata,
            )
            for c in candidates
        ]
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored
