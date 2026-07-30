import re

import pytest

from llmranker.llm import LLMConfig
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
