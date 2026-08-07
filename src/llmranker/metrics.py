from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from scipy.stats import ConstantInputWarning, kendalltau, spearmanr


class RankingMetrics:
    """Compares a predicted ranking against a ground truth, both expressed as
    ordered sequences of candidate ids (best-to-worst).

    Two ways to supply ground truth:

    - **An order only** (the default). Relevance is derived from position in
      `true_ranking`, linearly decreasing from `len(true_ranking)` (best)
      down to 1 (worst). Use this when what you have is a preference
      *order* rather than graded judgments.
    - **Graded relevance** (`relevance=`, a mapping of id -> gain). Use this
      when you have real judgments, e.g. TREC-style 0-3 grades or Amazon
      ESCI's Exact/Substitute/Complement/Irrelevant mapped to numbers.
      `true_ranking` is then only used by the rank-correlation metrics;
      NDCG's ideal ordering is computed from the grades themselves, so it
      doesn't matter whether `true_ranking` happens to be sorted by them.

    Metric names mean what they say, which is worth spelling out because
    two of them are easy to misread:

    - `ndcg`: standard NDCG, optionally truncated at `k`. Always in [0, 1].
    - `reciprocal_rank`: the reciprocal rank of the *single* best item
      (`true_ranking[0]`) in the predicted order. This is **not** MRR: MRR
      is a mean over many queries against a set of relevant items, whereas
      this is one query against one target. Aggregate it yourself across
      queries if you want MRR.
    - `rank_mae`: mean absolute *rank displacement* — the average of
      |true position - predicted position| over the judged items. It is an
      error in positions, not in scores, so its scale is "places moved",
      not the 0-10 range `PointwiseRanker` uses.
    - `spearman` / `kendall_tau`: rank correlations, in [-1, 1]. Both are
      mathematically **undefined** for fewer than two judged items (and
      for a constant ranking), and `float("nan")` is returned in that case
      rather than a made-up number. Guard for it if you average across
      queries.
    """

    @staticmethod
    def _positions(ranking: Sequence[str]) -> dict[str, int]:
        return {item_id: pos for pos, item_id in enumerate(ranking)}

    @staticmethod
    def _order_relevance(true_ranking: Sequence[str]) -> dict[str, float]:
        """Linear gains derived from position, for the order-only case."""
        n = len(true_ranking)
        return {item_id: float(n - pos) for pos, item_id in enumerate(true_ranking)}

    @classmethod
    def ndcg(
        cls,
        true_ranking: Sequence[str],
        predicted_ranking: Sequence[str],
        k: int | None = None,
        relevance: dict[str, float] | None = None,
    ) -> float:
        if relevance is None:
            relevance = cls._order_relevance(true_ranking)

        def dcg(ranking: Sequence[str]) -> float:
            ranking = ranking[:k] if k is not None else ranking
            return sum(
                relevance.get(item_id, 0.0) / np.log2(i + 2) for i, item_id in enumerate(ranking)
            )

        # The ideal ordering is the judged items sorted by gain, *not*
        # `true_ranking` as given: when a caller supplies graded relevance,
        # nothing guarantees their ordering is sorted by those grades, and
        # using it directly can produce a "normalized" score above 1.
        ideal_ranking = sorted(relevance, key=lambda item_id: relevance[item_id], reverse=True)
        ideal = dcg(ideal_ranking)
        return float(dcg(predicted_ranking) / ideal) if ideal > 0 else 0.0

    @staticmethod
    def reciprocal_rank(true_ranking: Sequence[str], predicted_ranking: Sequence[str]) -> float:
        """Reciprocal rank of the single most relevant item (`true_ranking[0]`)
        within the predicted order. See the class docstring on why this is
        not MRR."""
        if not true_ranking:
            return 0.0
        target = true_ranking[0]
        for i, item_id in enumerate(predicted_ranking):
            if item_id == target:
                return 1.0 / (i + 1)
        return 0.0

    @classmethod
    def rank_mae(cls, true_ranking: Sequence[str], predicted_ranking: Sequence[str]) -> float:
        """Mean absolute rank displacement over the judged items. Items
        missing from the prediction are treated as sitting just past its
        end."""
        true_pos = cls._positions(true_ranking)
        pred_pos = cls._positions(predicted_ranking)
        if not true_pos:
            return 0.0
        missing = len(true_pos)
        return float(
            np.mean(
                [abs(pos - pred_pos.get(item_id, missing)) for item_id, pos in true_pos.items()]
            )
        )

    @classmethod
    def _aligned_positions(
        cls, true_ranking: Sequence[str], predicted_ranking: Sequence[str]
    ) -> tuple[list[int], list[int]]:
        true_pos = cls._positions(true_ranking)
        pred_pos = cls._positions(predicted_ranking)
        ids = list(true_pos)
        return (
            [true_pos[i] for i in ids],
            [pred_pos.get(i, len(ids)) for i in ids],
        )

    @staticmethod
    def _correlation(fn, x: list[int], y: list[int]) -> float:
        """Run a scipy correlation, returning NaN for the degenerate inputs
        where it is undefined instead of leaking a scipy warning."""
        if len(x) < 2:
            return float("nan")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConstantInputWarning)
            stat, _ = fn(x, y)
        return float(stat)

    @classmethod
    def spearman_corr(cls, true_ranking: Sequence[str], predicted_ranking: Sequence[str]) -> float:
        return cls._correlation(spearmanr, *cls._aligned_positions(true_ranking, predicted_ranking))

    @classmethod
    def kendall_tau(cls, true_ranking: Sequence[str], predicted_ranking: Sequence[str]) -> float:
        return cls._correlation(
            kendalltau, *cls._aligned_positions(true_ranking, predicted_ranking)
        )

    def get_metrics(
        self,
        true_ranking: Sequence[str],
        predicted_ranking: Sequence[str],
        k: int | None = None,
        relevance: dict[str, float] | None = None,
    ) -> dict[str, float]:
        return {
            "ndcg": self.ndcg(true_ranking, predicted_ranking, k=k, relevance=relevance),
            "reciprocal_rank": self.reciprocal_rank(true_ranking, predicted_ranking),
            "rank_mae": self.rank_mae(true_ranking, predicted_ranking),
            "spearman": self.spearman_corr(true_ranking, predicted_ranking),
            "kendall_tau": self.kendall_tau(true_ranking, predicted_ranking),
        }
