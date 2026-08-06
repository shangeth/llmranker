from __future__ import annotations

import time
from collections.abc import Sequence

import pandas as pd

from .llm import estimate_cost
from .metrics import RankingMetrics
from .types import Candidate, Ranker


def compare_rankers(
    rankers: list[Ranker],
    query: str,
    candidates: list[Candidate],
    true_ranking: Sequence[str],
    k: int | None = None,
) -> pd.DataFrame:
    """Run each ranker over the same (query, candidates) and report ranking
    quality, LLM usage, latency and estimated cost side by side.

    `true_ranking` is the ground-truth ordering of candidate ids, best to
    worst; see `RankingMetrics` for how it's used to score each ranker's
    predicted order.
    """
    metrics_calc = RankingMetrics()
    rows = []

    for ranker in rankers:
        start = time.perf_counter()
        result = ranker.rank(query, candidates)
        elapsed = time.perf_counter() - start

        predicted_ranking = [c.id for c in result]
        metrics = metrics_calc.get_metrics(true_ranking, predicted_ranking, k=k)

        prompt_tokens = getattr(ranker, "total_prompt_tokens", 0)
        completion_tokens = getattr(ranker, "total_completion_tokens", 0)
        calls = getattr(ranker, "total_calls", 0)
        cost = estimate_cost(ranker.config.model, prompt_tokens, completion_tokens)

        rows.append(
            {
                "ranker": ranker.name,
                "model": ranker.config.model,
                **metrics,
                "llm_calls": calls,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": cost,
                "time_s": elapsed,
            }
        )

    return pd.DataFrame(rows)
