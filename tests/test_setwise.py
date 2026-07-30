import re

import pytest

from llmranker.llm import LLMConfig
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


@pytest.mark.parametrize("method", ["heapsort", "bubblesort"])
def test_setwise_finds_top_k_in_true_order(fake_llm, method):
    candidates = _shuffled_candidates()
    rank_of = {c.text: int(c.id) for c in candidates}
    fake_llm.responses = _ground_truth_responder(rank_of)

    ranker = SetwiseRanker(LLMConfig(model="gpt-4o-mini"), num_child=3, method=method, k=3)
    result = ranker.rank("query", candidates)

    assert [c.id for c in result[:3]] == ["1", "2", "3"]
    assert len(result) == 7
    assert {c.id for c in result} == {c.id for c in candidates}


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
