from llmranker.llm import LLMConfig
from llmranker.rankers.pointwise import PointwiseRanker
from llmranker.types import Candidate


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
