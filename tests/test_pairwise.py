import re

import pytest

from llmranker.llm import LLMConfig
from llmranker.prompts import FINAL_ANSWER_MARKER
from llmranker.rankers.pairwise import PairwiseRanker
from llmranker.types import Candidate


def _ground_truth_responder(rank_of):
    """Fake LLM: prefers whichever candidate has the lower `rank_of` value."""

    def fn(messages):
        user_content = messages[-1]["content"]
        a_text = re.search(r'Item A: "([^"]*)"', user_content).group(1)
        b_text = re.search(r'Item B: "([^"]*)"', user_content).group(1)
        return "A" if rank_of[a_text] < rank_of[b_text] else "B"

    return fn


def _shuffled_candidates():
    # id == true rank (as a string); text encodes it too so the fake LLM can read it.
    return [Candidate(id=str(r), text=f"item-{r}") for r in [5, 3, 1, 4, 2]]


@pytest.mark.parametrize("method", ["heapsort", "bubblesort", "allpairs"])
def test_pairwise_finds_top_k_in_true_order(fake_llm, method):
    candidates = _shuffled_candidates()
    rank_of = {c.text: int(c.id) for c in candidates}
    fake_llm.responses = _ground_truth_responder(rank_of)

    ranker = PairwiseRanker(LLMConfig(model="gpt-4o-mini"), method=method, k=3)
    result = ranker.rank("query", candidates)

    assert [c.id for c in result[:3]] == ["1", "2", "3"]
    assert len(result) == 5
    assert {c.id for c in result} == {c.id for c in candidates}
    assert result[0].score > result[1].score > result[2].score


def test_pairwise_default_k_ranks_everything(fake_llm):
    candidates = _shuffled_candidates()
    rank_of = {c.text: int(c.id) for c in candidates}
    fake_llm.responses = _ground_truth_responder(rank_of)

    ranker = PairwiseRanker(LLMConfig(model="gpt-4o-mini"), method="allpairs")
    result = ranker.rank("query", candidates)

    assert [c.id for c in result] == ["1", "2", "3", "4", "5"]


def test_pairwise_unparseable_output_defaults_to_first_candidate(fake_llm):
    fake_llm.responses = ["I cannot decide between these two options."]
    a = Candidate(id="a", text="hotel a")
    b = Candidate(id="b", text="hotel b")
    ranker = PairwiseRanker(LLMConfig(model="gpt-4o-mini"))

    winner = ranker.compare("query", a, b)
    assert winner is a


def test_pairwise_rejects_unknown_method():
    with pytest.raises(ValueError):
        PairwiseRanker(LLMConfig(model="gpt-4o-mini"), method="quicksort")


@pytest.mark.parametrize("max_concurrency", [1, 5])
def test_pairwise_allpairs_concurrency_matches_sequential(fake_llm, max_concurrency):
    candidates = _shuffled_candidates()
    rank_of = {c.text: int(c.id) for c in candidates}
    fake_llm.responses = _ground_truth_responder(rank_of)

    ranker = PairwiseRanker(
        LLMConfig(model="gpt-4o-mini"), method="allpairs", max_concurrency=max_concurrency
    )
    result = ranker.rank("query", candidates)

    assert [c.id for c in result] == ["1", "2", "3", "4", "5"]
    assert ranker.total_calls == 10  # n*(n-1)/2 for n=5, same regardless of concurrency


def test_pairwise_debias_position_matches_unbiased_ground_truth(fake_llm):
    candidates = _shuffled_candidates()
    rank_of = {c.text: int(c.id) for c in candidates}
    fake_llm.responses = _ground_truth_responder(rank_of)

    ranker = PairwiseRanker(
        LLMConfig(model="gpt-4o-mini"), method="allpairs", debias_position=True
    )
    result = ranker.rank("query", candidates)

    assert [c.id for c in result] == ["1", "2", "3", "4", "5"]
    assert ranker.total_calls == 20  # doubled: forward + backward for every pair


def _always_b_responder(messages):
    """A pure position-bias fake: always picks whatever's labeled 'B',
    regardless of which candidate that is or what the query says."""
    return "B"


def test_pairwise_debias_position_prevents_confidently_wrong_bias(fake_llm):
    # `candidates` is NOT in true-relevance order -- it's just some input order.
    candidates = _shuffled_candidates()

    fake_llm.responses = _always_b_responder
    biased = PairwiseRanker(LLMConfig(model="gpt-4o-mini"), method="allpairs")
    biased_result = biased.rank("query", candidates)
    # Undebiased: a model that always answers "B" makes the later-indexed
    # candidate in every pair "win", which structurally reverses whatever
    # order the candidates were passed in -- a confidently wrong result
    # driven entirely by position, not content.
    assert [c.id for c in biased_result] == [c.id for c in reversed(candidates)]

    fake_llm.responses = _always_b_responder
    debiased = PairwiseRanker(
        LLMConfig(model="gpt-4o-mini"), method="allpairs", debias_position=True
    )
    debiased_result = debiased.rank("query", candidates)
    # Debiasing detects the forward/backward disagreement on every single
    # pair (this adversary never actually reflects content) and falls back
    # to the original order instead of confidently reversing it.
    assert [c.id for c in debiased_result] == [c.id for c in candidates]
    assert debiased.total_calls == 2 * biased.total_calls


def _misleading_reasoning_responder(messages):
    return (
        "Item A looks great at first glance, but on closer inspection Item B "
        f"is the better match for the query.\n\n{FINAL_ANSWER_MARKER} B"
    )


def test_pairwise_reasoning_ignores_stray_label_before_final_answer(fake_llm):
    # Without marker-aware parsing, the naive first-match regex would latch
    # onto the stray "A" in "Item A looks great..." and return the wrong
    # candidate, never reaching the correct "B" after FINAL ANSWER.
    fake_llm.responses = [_misleading_reasoning_responder(None)]
    a = Candidate(id="a", text="hotel a")
    b = Candidate(id="b", text="hotel b")
    ranker = PairwiseRanker(LLMConfig(model="gpt-4o-mini"), reasoning=True)

    winner = ranker.compare("query", a, b)
    assert winner is b
