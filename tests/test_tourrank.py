import random
import re

import pytest

from llmranker.llm import LLMConfig
from llmranker.rankers.tourrank import TourRankRanker
from llmranker.types import Candidate

_ENTRY_RE = re.compile(r"Item ([A-Z]): <candidate>([^<]*)</candidate>")


def _ground_truth_responder(rank_of):
    """Fake LLM: returns every label in true-best-first order.

    The parser takes the first `advance` labels it finds, so this responder
    works for any per-stage advance count without needing to know it -- which
    matters now that the schedule varies the count between stages.
    """

    def fn(messages):
        entries = _ENTRY_RE.findall(messages[-1]["content"])
        entries.sort(key=lambda e: rank_of[e[1]])
        return ", ".join(label for label, _ in entries)

    return fn


def _shuffled_candidates(n=16, seed=7):
    ranks = list(range(1, n + 1))
    random.Random(seed).shuffle(ranks)
    return [Candidate(id=str(r), text=f"item-{r}") for r in ranks]


def test_tourrank_best_and_worst_candidates_get_extreme_points(fake_llm):
    """The true best wins every group it is ever placed in, so it survives
    every stage of every tournament; the true worst never survives one."""
    candidates = _shuffled_candidates(n=16)
    rank_of = {c.text: int(c.id) for c in candidates}

    ranker = TourRankRanker(
        LLMConfig(model="gpt-4o-mini"), group_size=4, num_tournaments=3, seed=42
    )
    fake_llm.responses = _ground_truth_responder(rank_of)
    result = ranker.rank("query", candidates)

    scores = {c.id: c.score for c in result}
    stages = len(ranker._resolve_schedule(len(candidates)))
    assert scores["1"] == ranker.num_tournaments * stages
    assert scores["16"] == 0
    # The final stage leaves two survivors on equal points, so the very
    # top is a tie the tournament cannot break -- see
    # test_result_is_stable_under_input_shuffling for what is guaranteed.
    assert result[0].score == scores["1"]
    assert {c.id for c in result} == {c.id for c in candidates}


def test_tourrank_returns_every_candidate_ordered(fake_llm):
    candidates = _shuffled_candidates(n=16)
    rank_of = {c.text: int(c.id) for c in candidates}

    ranker = TourRankRanker(LLMConfig(model="gpt-4o-mini"), group_size=4, num_tournaments=2, seed=1)
    fake_llm.responses = _ground_truth_responder(rank_of)
    result = ranker.rank("query", candidates)

    assert len(result) == len(candidates)
    assert "1" in {c.id for c in result[:2]}
    assert [c.score for c in result] == sorted((c.score for c in result), reverse=True)


def test_default_schedule_follows_the_papers_shape(fake_llm):
    """The paper eliminates 100 -> 50 -> 20 -> 10 -> 5 -> 2. Enough stages to
    separate candidates on points is what makes the score meaningful."""
    ranker = TourRankRanker(LLMConfig(model="gpt-4o-mini"))
    assert ranker._resolve_schedule(100) == [50, 20, 10, 5, 2]
    # Short lists get fewer stages rather than a run of no-op rounds.
    assert ranker._resolve_schedule(4) == [2]
    assert ranker._resolve_schedule(2) == []


def test_result_is_stable_under_input_shuffling(fake_llm):
    """The headline claim, stated precisely: the *points* a candidate earns
    do not depend on the order the candidates arrived in.

    Two things had to be fixed for this to hold. Group membership used to be
    a pure function of input position, so every tournament in the ensemble
    played out identically (num_tournaments only scaled the scores) and the
    draw tracked input order. And the schedule was too short to separate
    candidates onto distinct point levels, so most of the ranking was decided
    by the tie-break rather than by the tournament.

    Ordering *within* a point bucket still follows input order, by design --
    candidates that survived exactly the same stages are genuinely
    indistinguishable to the tournament, and preserving the caller's order
    keeps any upstream retriever ranking as the tie-break.
    """
    rank_of = {f"item-{i}": i for i in range(1, 41)}
    points_seen, top_bucket_seen = set(), set()
    for shuffle_seed in range(5):
        candidates = [Candidate(id=str(i), text=f"item-{i}") for i in range(1, 41)]
        random.Random(shuffle_seed).shuffle(candidates)
        fake_llm.responses = _ground_truth_responder(rank_of)
        ranker = TourRankRanker(
            LLMConfig(model="gpt-4o-mini"), group_size=20, num_tournaments=3, seed=0
        )
        result = ranker.rank("query", candidates)
        points_seen.add(tuple(sorted((c.id, c.score) for c in result)))
        best = result[0].score
        top_bucket_seen.add(frozenset(c.id for c in result if c.score == best))

    assert len(points_seen) == 1, "points changed with input order"
    assert len(top_bucket_seen) == 1, "top point bucket changed with input order"


def test_tournaments_are_not_all_identical(fake_llm):
    """The ensemble has to actually ensemble something. Group membership was
    derived from input position with only a within-group shuffle, so every
    tournament produced the same points and num_tournaments was a no-op
    multiplier."""
    rank_of = {f"item-{i}": i for i in range(1, 41)}
    candidates = [Candidate(id=str(i), text=f"item-{i}") for i in range(1, 41)]
    fake_llm.responses = _ground_truth_responder(rank_of)
    ranker = TourRankRanker(LLMConfig(model="gpt-4o-mini"), group_size=20, seed=0)

    rng = random.Random(0)
    groupings = {
        tuple(tuple(sorted(c.id for c in g)) for g in ranker._make_groups(candidates, rng))
        for _ in range(3)
    }
    assert len(groupings) == 3, "every tournament drew the same groups"


def test_score_granularity_separates_candidates(fake_llm):
    """Three distinct score values across 100 candidates leaves most of the
    ordering to the tie-break; the schedule exists to avoid that."""
    rank_of = {f"item-{i}": i for i in range(100)}
    candidates = [Candidate(id=str(i), text=f"item-{i}") for i in range(100)]
    fake_llm.responses = _ground_truth_responder(rank_of)

    ranker = TourRankRanker(LLMConfig(model="gpt-4o-mini"), num_tournaments=1, seed=0)
    result = ranker.rank("query", candidates)

    assert len({c.score for c in result}) >= 5
    assert sum(1 for c in result if c.score == result[0].score) <= 5


def test_select_group_returns_advance_count_candidates(fake_llm):
    fake_llm.responses = ["A, C"]
    group = [Candidate(id=str(i), text=f"hotel {i}") for i in range(4)]
    ranker = TourRankRanker(LLMConfig(model="gpt-4o-mini"), group_size=4)

    selected = ranker.select_group("query", group, advance=2)
    assert {c.id for c in selected} == {"0", "2"}


def test_parse_group_selection_pads_shortfall_with_remaining_group_members():
    ranker = TourRankRanker(LLMConfig(model="gpt-4o-mini"), group_size=4)
    group = [Candidate(id=str(i), text=f"hotel {i}") for i in range(4)]

    # Only one valid label found ("A"); the other slot must be padded from
    # the group's remaining members rather than crashing or returning fewer.
    selected = ranker._parse_group_selection("A", group, advance=2)
    assert len(selected) == 2
    assert selected[0].id == "0"


def test_parse_group_selection_falls_back_entirely_on_unparseable_output():
    ranker = TourRankRanker(LLMConfig(model="gpt-4o-mini"), group_size=4)
    group = [Candidate(id=str(i), text=f"hotel {i}") for i in range(4)]

    selected = ranker._parse_group_selection("I cannot decide.", group, advance=2)
    assert [c.id for c in selected] == ["0", "1"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"group_size": 1},
        {"group_size": 27},
        {"schedule": []},
        {"schedule": [5, 10]},  # not strictly decreasing
        {"schedule": [5, 0]},
        {"num_tournaments": 0},
    ],
)
def test_tourrank_rejects_invalid_config(kwargs):
    with pytest.raises(ValueError):
        TourRankRanker(LLMConfig(model="gpt-4o-mini"), **kwargs)


def test_explicit_schedule_overrides_the_default(fake_llm):
    rank_of = {f"item-{i}": i for i in range(20)}
    candidates = [Candidate(id=str(i), text=f"item-{i}") for i in range(20)]
    fake_llm.responses = _ground_truth_responder(rank_of)

    ranker = TourRankRanker(
        LLMConfig(model="gpt-4o-mini"), group_size=5, schedule=[10, 4], num_tournaments=1, seed=0
    )
    result = ranker.rank("query", candidates)

    # Two stages -> scores in {0, 1, 2}.
    assert {c.score for c in result} <= {0.0, 1.0, 2.0}
    assert result[0].score == 2.0


def test_tourrank_num_samples_is_ignored_with_warning(fake_llm, caplog):
    candidates = _shuffled_candidates(n=4)
    rank_of = {c.text: int(c.id) for c in candidates}
    fake_llm.responses = _ground_truth_responder(rank_of)

    ranker = TourRankRanker(
        LLMConfig(model="gpt-4o-mini"),
        group_size=4,
        num_tournaments=1,
        num_samples=3,
        seed=1,
    )
    with caplog.at_level("WARNING"):
        ranker.rank("query", candidates)

    assert any("num_samples" in r.message for r in caplog.records)


def test_tourrank_structured_output_parses_json_selection(fake_llm):
    fake_llm.responses = ['{"selected": ["A", "C"]}']
    group = [Candidate(id=str(i), text=f"hotel {i}") for i in range(4)]
    ranker = TourRankRanker(LLMConfig(model="gpt-4o-mini"), group_size=4, structured_output=True)

    selected = ranker.select_group("query", group, advance=2)
    assert {c.id for c in selected} == {"0", "2"}
    assert fake_llm.calls[0]["response_format"]["type"] == "json_schema"


def test_tourrank_structured_output_falls_back_to_regex_on_malformed_json(fake_llm):
    fake_llm.responses = ["not json, going with A and C"]
    group = [Candidate(id=str(i), text=f"hotel {i}") for i in range(4)]
    ranker = TourRankRanker(LLMConfig(model="gpt-4o-mini"), group_size=4, structured_output=True)

    selected = ranker.select_group("query", group, advance=2)
    assert {c.id for c in selected} == {"0", "2"}
