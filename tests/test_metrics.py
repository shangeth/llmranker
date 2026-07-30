import pytest

from llmranker.metrics import RankingMetrics


@pytest.fixture
def metrics():
    return RankingMetrics()


def test_perfect_prediction_scores_maximally(metrics):
    true_ranking = ["a", "b", "c", "d"]
    result = metrics.get_metrics(true_ranking, true_ranking)

    assert result["ndcg"] == pytest.approx(1.0)
    assert result["mrr"] == pytest.approx(1.0)
    assert result["mae"] == pytest.approx(0.0)
    assert result["spearman"] == pytest.approx(1.0)
    assert result["kendall_tau"] == pytest.approx(1.0)


def test_reversed_prediction_scores_poorly(metrics):
    true_ranking = ["a", "b", "c", "d"]
    reversed_ranking = ["d", "c", "b", "a"]
    result = metrics.get_metrics(true_ranking, reversed_ranking)

    assert 0 < result["ndcg"] < 1.0
    assert result["mrr"] == pytest.approx(1 / 4)  # "a" ends up last
    assert result["spearman"] == pytest.approx(-1.0)
    assert result["kendall_tau"] == pytest.approx(-1.0)
    assert result["mae"] > 0


def test_ndcg_respects_k(metrics):
    true_ranking = ["a", "b", "c", "d"]
    # "a" (the single most relevant) correctly placed first -> ndcg@1 is perfect
    predicted = ["a", "d", "c", "b"]
    assert metrics.ndcg(true_ranking, predicted, k=1) == pytest.approx(1.0)
    assert metrics.ndcg(true_ranking, predicted, k=None) < 1.0


def test_mrr_zero_when_top_item_missing(metrics):
    true_ranking = ["a", "b", "c"]
    predicted = ["b", "c"]  # "a" never shows up
    assert metrics.mrr(true_ranking, predicted) == 0.0
