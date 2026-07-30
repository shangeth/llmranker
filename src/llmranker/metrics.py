from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error


class RankingMetrics:
    """Compares a predicted ranking against a ground-truth ranking, both
    expressed as ordered sequences of candidate ids (best-to-worst).

    Relevance for NDCG is derived from position in `true_ranking` --
    linearly decreasing from `len(true_ranking)` (best) down to 1 (worst) --
    since ground truth here is typically a preference *order* rather than
    graded relevance scores. Pass `relevance` explicitly if you have real
    relevance judgments (id -> score).
    """

    @staticmethod
    def _positions(ranking: Sequence[str]) -> dict[str, int]:
        return {item_id: pos for pos, item_id in enumerate(ranking)}

    @staticmethod
    def ndcg(
        true_ranking: Sequence[str],
        predicted_ranking: Sequence[str],
        k: int | None = None,
        relevance: dict[str, float] | None = None,
    ) -> float:
        if relevance is None:
            n = len(true_ranking)
            relevance = {item_id: n - pos for pos, item_id in enumerate(true_ranking)}

        def dcg(ranking: Sequence[str]) -> float:
            ranking = ranking[:k] if k is not None else ranking
            return sum(
                relevance.get(item_id, 0) / np.log2(i + 2)
                for i, item_id in enumerate(ranking)
            )

        ideal = dcg(true_ranking)
        actual = dcg(predicted_ranking)
        return actual / ideal if ideal > 0 else 0.0

    @staticmethod
    def mrr(true_ranking: Sequence[str], predicted_ranking: Sequence[str]) -> float:
        """Reciprocal rank of the single most relevant item (true_ranking[0])
        within the predicted order."""
        if not true_ranking:
            return 0.0
        target = true_ranking[0]
        for i, item_id in enumerate(predicted_ranking):
            if item_id == target:
                return 1.0 / (i + 1)
        return 0.0

    @classmethod
    def mae(cls, true_ranking: Sequence[str], predicted_ranking: Sequence[str]) -> float:
        true_pos = cls._positions(true_ranking)
        pred_pos = cls._positions(predicted_ranking)
        ids = list(true_pos)
        return mean_absolute_error(
            [true_pos[i] for i in ids], [pred_pos.get(i, len(ids)) for i in ids]
        )

    @classmethod
    def spearman_corr(cls, true_ranking: Sequence[str], predicted_ranking: Sequence[str]) -> float:
        true_pos = cls._positions(true_ranking)
        pred_pos = cls._positions(predicted_ranking)
        ids = list(true_pos)
        corr, _ = spearmanr(
            [true_pos[i] for i in ids], [pred_pos.get(i, len(ids)) for i in ids]
        )
        return corr

    @classmethod
    def kendall_tau(cls, true_ranking: Sequence[str], predicted_ranking: Sequence[str]) -> float:
        true_pos = cls._positions(true_ranking)
        pred_pos = cls._positions(predicted_ranking)
        ids = list(true_pos)
        tau, _ = kendalltau(
            [true_pos[i] for i in ids], [pred_pos.get(i, len(ids)) for i in ids]
        )
        return tau

    def get_metrics(
        self,
        true_ranking: Sequence[str],
        predicted_ranking: Sequence[str],
        k: int | None = None,
    ) -> dict[str, float]:
        return {
            "ndcg": self.ndcg(true_ranking, predicted_ranking, k=k),
            "mrr": self.mrr(true_ranking, predicted_ranking),
            "mae": self.mae(true_ranking, predicted_ranking),
            "spearman": self.spearman_corr(true_ranking, predicted_ranking),
            "kendall_tau": self.kendall_tau(true_ranking, predicted_ranking),
        }
