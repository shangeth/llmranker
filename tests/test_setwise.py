import re

import pytest

from llmranker.llm import LLMConfig
from llmranker.prompts import FINAL_ANSWER_MARKER
from llmranker.rankers.setwise import SetwiseRanker
from llmranker.types import Candidate

_ENTRY_RE = re.compile(r'Item ([A-Z]): "([^"]*)"')


def _ground_truth_responder(rank_of):
    """Fake LLM: picks whichever candidate in the set has the lowest rank_of value."""

    def fn(messages):
        user_content = messages[-1]["content"]
        entries = _ENTRY_RE.findall(user_content)
        best_label, _ = min(entries, key=lambda e: rank_of[e[1]])
        return best_label

    return fn


def _shuffled_candidates():
    return [Candidate(id=str(r), text=f"item-{r}") for r in [7, 3, 1, 6, 2, 5, 4]]


@pytest.mark.parametrize("method", ["heapsort", "bubblesort", "insertion"])
def test_setwise_finds_top_k_in_true_order(fake_llm, method):
    candidates = _shuffled_candidates()
    rank_of = {c.text: int(c.id) for c in candidates}
    fake_llm.responses = _ground_truth_responder(rank_of)

    ranker = SetwiseRanker(LLMConfig(model="gpt-4o-mini"), num_child=3, method=method, k=3)
    result = ranker.rank("query", candidates)

    assert [c.id for c in result[:3]] == ["1", "2", "3"]
    assert len(result) == 7
    assert {c.id for c in result} == {c.id for c in candidates}


def test_setwise_insertion_costs_fewer_calls_than_heapsort_on_ordered_input(fake_llm):
    # A near-sorted prior order (a couple of items out of place) is exactly
    # the case Setwise Insertion is designed for: most later chunks should
    # be discarded against the guard in a single call each, rather than
    # heapsort's full tree of comparisons.
    ranks = [1, 2, 4, 3, 5, 7, 6, 8, 9, 10]
    candidates = [Candidate(id=str(r), text=f"item-{r}") for r in ranks]
    rank_of = {c.text: int(c.id) for c in candidates}
    fake_llm.responses = _ground_truth_responder(rank_of)

    insertion = SetwiseRanker(LLMConfig(model="gpt-4o-mini"), num_child=3, method="insertion", k=5)
    result = insertion.rank("query", candidates)
    assert [c.id for c in result[:5]] == ["1", "2", "3", "4", "5"]

    fake_llm.responses = _ground_truth_responder(rank_of)
    heapsort = SetwiseRanker(LLMConfig(model="gpt-4o-mini"), num_child=3, method="heapsort", k=5)
    heapsort.rank("query", candidates)

    assert insertion.total_calls < heapsort.total_calls


def test_setwise_reasoning_ignores_stray_label_before_final_answer(fake_llm):
    # Without marker-aware parsing, the naive first-match regex would latch
    # onto the stray "A" in "Item A has decent reviews" and return the
    # wrong candidate, never reaching the correct "C" after FINAL ANSWER.
    text = (
        "Item A has decent reviews, but comparing all options, Item C stands "
        f"out as most relevant.\n\n{FINAL_ANSWER_MARKER} C"
    )
    fake_llm.responses = [text]
    group = [Candidate(id=str(i), text=f"hotel {i}") for i in range(4)]
    ranker = SetwiseRanker(LLMConfig(model="gpt-4o-mini"), num_child=4, reasoning=True)

    winner = ranker.compare("query", group)
    assert winner is group[2]


def test_binary_insert_position_finds_correct_slot(fake_llm):
    # sorted_list is best-to-worst by true rank; item (rank 3) belongs at
    # index 2, between rank-2 and rank-4.
    sorted_list = [Candidate(id=str(r), text=f"item-{r}") for r in [1, 2, 4, 5]]
    item = Candidate(id="3", text="item-3")
    rank_of = {c.text: int(c.id) for c in [*sorted_list, item]}
    fake_llm.responses = _ground_truth_responder(rank_of)

    ranker = SetwiseRanker(LLMConfig(model="gpt-4o-mini"), num_child=3)
    pos = ranker._binary_insert_position("query", sorted_list, item)
    assert pos == 2


def test_setwise_default_k_ranks_everything(fake_llm):
    candidates = _shuffled_candidates()
    rank_of = {c.text: int(c.id) for c in candidates}
    fake_llm.responses = _ground_truth_responder(rank_of)

    ranker = SetwiseRanker(LLMConfig(model="gpt-4o-mini"), num_child=3, method="heapsort")
    result = ranker.rank("query", candidates)

    assert [c.id for c in result] == ["1", "2", "3", "4", "5", "6", "7"]


def test_setwise_unparseable_output_defaults_to_first_candidate(fake_llm):
    fake_llm.responses = ["none of these seem right"]
    group = [Candidate(id=str(i), text=f"hotel {i}") for i in range(4)]
    ranker = SetwiseRanker(LLMConfig(model="gpt-4o-mini"), num_child=4)

    winner = ranker.compare("query", group)
    assert winner is group[0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "quicksort"},
        {"num_child": 1},
        {"num_child": 27},
    ],
)
def test_setwise_rejects_invalid_config(kwargs):
    with pytest.raises(ValueError):
        SetwiseRanker(LLMConfig(model="gpt-4o-mini"), **kwargs)
