from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .llm import estimate_cost
from .metrics import RankingMetrics
from .types import Candidate, Ranker

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


def _require_pandas() -> Any:
    try:
        import pandas
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ImportError(
            "compare_rankers() needs pandas, which is an optional dependency. "
            'Install it with: pip install "llmranker[benchmark]"'
        ) from exc
    return pandas


def compare_rankers(
    rankers: list[Ranker],
    query: str,
    candidates: list[Candidate],
    true_ranking: Sequence[str],
    k: int | None = None,
    relevance: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Run each ranker over the same (query, candidates) and report ranking
    quality, LLM usage, latency and estimated cost side by side.

    `true_ranking` is the ground-truth ordering of candidate ids, best to
    worst. `relevance`, optional, supplies graded judgments (id -> gain)
    for NDCG instead of deriving them from `true_ranking`'s order; see
    `RankingMetrics` for what each reported metric means.

    Requires pandas, which is an optional dependency
    (`pip install "llmranker[benchmark]"`).
    """
    pd = _require_pandas()
    metrics_calc = RankingMetrics()
    rows = []

    for ranker in rankers:
        start = time.perf_counter()
        result = ranker.rank(query, candidates)
        elapsed = time.perf_counter() - start

        predicted_ranking = [c.id for c in result]
        metrics = metrics_calc.get_metrics(
            true_ranking, predicted_ranking, k=k, relevance=relevance
        )

        prompt_tokens = getattr(ranker, "total_prompt_tokens", 0)
        completion_tokens = getattr(ranker, "total_completion_tokens", 0)
        calls = getattr(ranker, "total_calls", 0)
        # A ranker may report its own cost when the token-based estimate
        # doesn't apply to it -- e.g. RerankAPIRanker, which is billed per
        # search unit and returns no token counts, so the default estimate
        # would report a misleading $0.00 rather than "unknown".
        own_estimate = getattr(ranker, "estimate_cost_usd", None)
        if callable(own_estimate):
            cost = own_estimate()
        else:
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
