import re

import pytest

from llmranker.llm import LLMConfig
from llmranker.rankers.pointwise import PointwiseRanker
from llmranker.types import Candidate


def _score_responder(score_of):
    """Fake LLM: returns the score for whichever candidate text is in the prompt.

    Unlike a fixed response list, this is safe under concurrent dispatch --
    it doesn't matter which thread's call reaches the fake first, each call
    is answered based on its own content.
    """

    def fn(messages):
        text = re.search(r'Item: "([^"]*)"', messages[-1]["content"]).group(1)
        return str(score_of[text])

    return fn


def test_pointwise_sorts_by_score(fake_llm):
    fake_llm.responses = ["2", "9", "5"]
    candidates = [
        Candidate(id="a", text="budget hostel"),
        Candidate(id="b", text="luxury family resort"),
        Candidate(id="c", text="business hotel"),
    ]
    ranker = PointwiseRanker(LLMConfig(model="gpt-4o-mini"))
    result = ranker.rank("family friendly hotel", candidates)

    assert [c.id for c in result] == ["b", "c", "a"]
    assert result[0].score == 9
    assert ranker.total_calls == 3
    assert ranker.total_prompt_tokens == 30
    assert ranker.total_completion_tokens == 15


def test_pointwise_clamps_and_handles_unparseable_output(fake_llm):
    fake_llm.responses = ["15", "not a number", "-3"]
    candidates = [Candidate(id=str(i), text=f"item {i}") for i in range(3)]
    ranker = PointwiseRanker(LLMConfig(model="gpt-4o-mini"))
    result = ranker.rank("query", candidates)

    scores = {c.id: c.score for c in result}
    assert scores["0"] == 10  # clamped to max_score
    assert scores["1"] == 0  # unparseable -> min_score
    assert scores["2"] == 0  # clamped to min_score


def test_pointwise_preserves_candidate_text_and_metadata(fake_llm):
    fake_llm.responses = ["7"]
    candidate = Candidate(id="x", text="a nice hotel", metadata={"price": 100})
    ranker = PointwiseRanker(LLMConfig(model="gpt-4o-mini"))
    result = ranker.rank("query", [candidate])

    assert result[0].text == "a nice hotel"
    assert result[0].metadata == {"price": 100}


@pytest.mark.parametrize("max_concurrency", [1, 5])
def test_pointwise_concurrency_matches_sequential(fake_llm, max_concurrency):
    candidates = [
        Candidate(id="a", text="budget hostel"),
        Candidate(id="b", text="luxury family resort"),
        Candidate(id="c", text="business hotel"),
    ]
    score_of = {"budget hostel": 2, "luxury family resort": 9, "business hotel": 5}
    fake_llm.responses = _score_responder(score_of)

    ranker = PointwiseRanker(LLMConfig(model="gpt-4o-mini"), max_concurrency=max_concurrency)
    result = ranker.rank("family friendly hotel", candidates)

    assert [c.id for c in result] == ["b", "c", "a"]
    assert result[0].score == 9
    assert ranker.total_calls == 3
    assert ranker.total_prompt_tokens == 30
    assert ranker.total_completion_tokens == 15
