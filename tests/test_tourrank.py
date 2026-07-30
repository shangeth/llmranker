import random
import re

import pytest

from llmranker.llm import LLMConfig
from llmranker.rankers.tourrank import TourRankRanker
from llmranker.types import Candidate

_ENTRY_RE = re.compile(r'Item ([A-Z]): "([^"]*)"')


def _ground_truth_responder(rank_of, advance_count):
    """Fake LLM: selects the `advance_count` candidates in the group with the
    lowest rank_of value (true best), comma-separated."""

    def fn(messages):
        user_content = messages[-1]["content"]
        entries = _ENTRY_RE.findall(user_content)
        entries.sort(key=lambda e: rank_of[e[1]])
        return ", ".join(label for label, _ in entries[:advance_count])

    return fn


def _shuffled_candidates(n=16, seed=7):
    ranks = list(range(1, n + 1))
    random.Random(seed).shuffle(ranks)
    return [Candidate(id=str(r), text=f"item-{r}") for r in ranks]


def test_tourrank_best_and_worst_candidates_get_extreme_points(fake_llm):
    # n=16 divides evenly by group_size=4 at every stage (16 -> 8 -> 4), so
    # there are never any "trivial" auto-advancing groups to muddy the
    # analysis: the true best candidate is *guaranteed* to win every real
    # group it's ever placed in (it's better than any possible groupmate),
    # and the true worst is *guaranteed* to never win one. Several
    # candidates can tie at the max (advance_per_group=2 lets more than one
    # candidate per group survive a stage), so we check the extremes
    # directly rather than assuming a unique #1.
    candidates = _shuffled_candidates(n=16)
    rank_of = {c.text: int(c.id) for c in candidates}

    ranker = TourRankRanker(
        LLMConfig(model="gpt-4o-mini"),
        group_size=4,
        advance_per_group=2,
        num_stages=2,
        num_tournaments=3,
        seed=42,
    )
    fake_llm.responses = _ground_truth_responder(rank_of, ranker.advance_per_group)
    result = ranker.rank("query", candidates)

    scores = {c.id: c.score for c in result}
    max_possible = ranker.num_tournaments * ranker.num_stages
    assert scores["1"] == max_possible
    assert scores["16"] == 0
    assert result[0].score == max_possible
    assert {c.id for c in result} == {c.id for c in candidates}


def test_tourrank_k_truncates_output_only(fake_llm):
    candidates = _shuffled_candidates(n=16)
    rank_of = {c.text: int(c.id) for c in candidates}

    ranker = TourRankRanker(
        LLMConfig(model="gpt-4o-mini"),
        group_size=4,
        advance_per_group=2,
        num_stages=2,
        num_tournaments=2,
        k=3,
        seed=1,
    )
    fake_llm.responses = _ground_truth_responder(rank_of, ranker.advance_per_group)
    result = ranker.rank("query", candidates)

    assert len(result) == 3
    assert "1" in {c.id for c in result}  # true best always makes the cut
    assert [c.score for c in result] == sorted((c.score for c in result), reverse=True)


def test_select_group_returns_advance_count_candidates(fake_llm):
    fake_llm.responses = ["A, C"]
    group = [Candidate(id=str(i), text=f"hotel {i}") for i in range(4)]
    ranker = TourRankRanker(LLMConfig(model="gpt-4o-mini"), group_size=4, advance_per_group=2)

    selected = ranker.select_group("query", group)
    assert {c.id for c in selected} == {"0", "2"}


def test_parse_group_selection_pads_shortfall_with_remaining_group_members():
    ranker = TourRankRanker(LLMConfig(model="gpt-4o-mini"), group_size=4, advance_per_group=2)
    group = [Candidate(id=str(i), text=f"hotel {i}") for i in range(4)]

    # Only one valid label found ("A"); the other slot must be padded from
    # the group's remaining members rather than crashing or returning fewer.
    selected = ranker._parse_group_selection("A", group)
    assert len(selected) == 2
    assert selected[0].id == "0"


def test_parse_group_selection_falls_back_entirely_on_unparseable_output():
    ranker = TourRankRanker(LLMConfig(model="gpt-4o-mini"), group_size=4, advance_per_group=2)
    group = [Candidate(id=str(i), text=f"hotel {i}") for i in range(4)]

    selected = ranker._parse_group_selection("I cannot decide.", group)
    assert [c.id for c in selected] == ["0", "1"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"group_size": 1},
        {"group_size": 27},
        {"advance_per_group": 4, "group_size": 4},
        {"advance_per_group": 5, "group_size": 4},
    ],
)
def test_tourrank_rejects_invalid_config(kwargs):
    with pytest.raises(ValueError):
        TourRankRanker(LLMConfig(model="gpt-4o-mini"), **kwargs)
