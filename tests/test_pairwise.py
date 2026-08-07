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
        a_text = re.search(r"Item A: <candidate>([^<]*)</candidate>", user_content).group(1)
        b_text = re.search(r"Item B: <candidate>([^<]*)</candidate>", user_content).group(1)
        return "A" if rank_of[a_text] < rank_of[b_text] else "B"

    return fn


def _shuffled_candidates():
    # id == true rank (as a string); text encodes it too so the fake LLM can read it.
    return [Candidate(id=str(r), text=f"item-{r}") for r in [5, 3, 1, 4, 2]]


@pytest.mark.parametrize("strategy", ["heapsort", "bubblesort", "allpairs"])
def test_pairwise_finds_top_k_in_true_order(fake_llm, strategy):
    candidates = _shuffled_candidates()
    rank_of = {c.text: int(c.id) for c in candidates}
    fake_llm.responses = _ground_truth_responder(rank_of)

    ranker = PairwiseRanker(LLMConfig(model="gpt-4o-mini"), strategy=strategy, k=3)
    result = ranker.rank("query", candidates)

    assert [c.id for c in result[:3]] == ["1", "2", "3"]
    assert len(result) == 5
    assert {c.id for c in result} == {c.id for c in candidates}
    assert result[0].score > result[1].score > result[2].score


def test_pairwise_default_k_ranks_everything(fake_llm):
    candidates = _shuffled_candidates()
    rank_of = {c.text: int(c.id) for c in candidates}
    fake_llm.responses = _ground_truth_responder(rank_of)

    ranker = PairwiseRanker(LLMConfig(model="gpt-4o-mini"), strategy="allpairs")
    result = ranker.rank("query", candidates)

    assert [c.id for c in result] == ["1", "2", "3", "4", "5"]


def test_pairwise_unparseable_output_defaults_to_first_candidate(fake_llm):
    fake_llm.responses = ["I cannot decide between these two options."]
    a = Candidate(id="a", text="hotel a")
    b = Candidate(id="b", text="hotel b")
    ranker = PairwiseRanker(LLMConfig(model="gpt-4o-mini"))

    winner = ranker.compare("query", a, b)
    assert winner is a


def test_pairwise_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        PairwiseRanker(LLMConfig(model="gpt-4o-mini"), strategy="quicksort")


@pytest.mark.parametrize("max_concurrency", [1, 5])
def test_pairwise_allpairs_concurrency_matches_sequential(fake_llm, max_concurrency):
    candidates = _shuffled_candidates()
    rank_of = {c.text: int(c.id) for c in candidates}
    fake_llm.responses = _ground_truth_responder(rank_of)

    ranker = PairwiseRanker(
        LLMConfig(model="gpt-4o-mini"), strategy="allpairs", max_concurrency=max_concurrency
    )
    result = ranker.rank("query", candidates)

    assert [c.id for c in result] == ["1", "2", "3", "4", "5"]
    # PRP-Allpair asks every pair in both orders: 2 * n*(n-1)/2 for n=5.
    assert ranker.total_calls == 20


def test_pairwise_num_samples_matches_unbiased_ground_truth(fake_llm):
    candidates = _shuffled_candidates()
    rank_of = {c.text: int(c.id) for c in candidates}
    fake_llm.responses = _ground_truth_responder(rank_of)

    ranker = PairwiseRanker(
        LLMConfig(model="gpt-4o-mini"),
        strategy="allpairs",
        num_samples=3,
        seed=1,
    )
    result = ranker.rank("query", candidates)

    assert [c.id for c in result] == ["1", "2", "3", "4", "5"]
    # Both orders per pair, num_samples times each: 2 * n*(n-1)/2 * 3.
    assert ranker.total_calls == 60


def _always_b_responder(messages):
    """A pure position-bias fake: always picks whatever's labeled 'B',
    regardless of which candidate that is or what the query says."""
    return "B"


def test_pairwise_num_samples_cancels_position_bias_statistically(fake_llm):
    # An always-picks-B adversary deterministically prefers whichever
    # candidate a single call happens to put in slot B. With several
    # samples and randomized position assignment per sample, the winner
    # across many independent pairs should no longer be systematically
    # tied to which argument was passed first (unlike a single, undebiased
    # call, which would always resolve to "whichever candidate is b").
    fake_llm.responses = _always_b_responder
    ranker = PairwiseRanker(LLMConfig(model="gpt-4o-mini"), num_samples=9, seed=7)

    first_argument_wins = 0
    trials = 30
    for i in range(trials):
        a = Candidate(id=f"a{i}", text=f"hotel a{i}")
        b = Candidate(id=f"b{i}", text=f"hotel b{i}")
        if ranker.compare("query", a, b) is a:
            first_argument_wins += 1

    # A single-sample call would make this 0/30 (b always wins); randomized
    # position + majority vote should land well away from that extreme.
    assert 5 <= first_argument_wins <= 25


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


def test_pairwise_structured_output_parses_json_choice(fake_llm):
    fake_llm.responses = ['{"choice": "B"}']
    a = Candidate(id="a", text="hotel a")
    b = Candidate(id="b", text="hotel b")
    ranker = PairwiseRanker(LLMConfig(model="gpt-4o-mini"), structured_output=True)

    winner = ranker.compare("query", a, b)
    assert winner is b
    assert fake_llm.calls[0]["response_format"]["type"] == "json_schema"


def test_pairwise_structured_output_falls_back_to_regex_on_malformed_json(fake_llm):
    fake_llm.responses = ["not json, but I'll go with B"]
    a = Candidate(id="a", text="hotel a")
    b = Candidate(id="b", text="hotel b")
    ranker = PairwiseRanker(LLMConfig(model="gpt-4o-mini"), structured_output=True)

    winner = ranker.compare("query", a, b)
    assert winner is b


def test_pairwise_reasoning_and_structured_output_conflict():
    with pytest.raises(ValueError):
        PairwiseRanker(LLMConfig(model="gpt-4o-mini"), reasoning=True, structured_output=True)
