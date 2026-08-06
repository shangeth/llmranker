import re

import pytest

from llmranker.llm import LLMConfig
from llmranker.prompts import FINAL_ANSWER_MARKER
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


def test_pointwise_reasoning_ignores_stray_numbers_before_final_answer(fake_llm):
    # Without marker-aware parsing, the naive regex would latch onto the "3"
    # in "3 miles" and return the wrong score, never reaching the real "8".
    text = (
        "This hotel is within 3 miles of downtown and has 2 pools, which "
        f"suggests strong appeal.\n\n{FINAL_ANSWER_MARKER} 8"
    )
    fake_llm.responses = [text]
    candidate = Candidate(id="x", text="a nice hotel")
    ranker = PointwiseRanker(LLMConfig(model="gpt-4o-mini"), reasoning=True)

    assert ranker.score("query", candidate) == 8


def test_pointwise_num_samples_averages_scores(fake_llm):
    fake_llm.responses = ["4", "6", "8"]
    candidate = Candidate(id="x", text="a nice hotel")
    ranker = PointwiseRanker(
        LLMConfig(model="gpt-4o-mini", temperature=0.7),
        num_samples=3,
    )

    assert ranker.score("query", candidate) == 6
    assert ranker.total_calls == 3


def test_pointwise_num_samples_warns_at_zero_temperature(fake_llm, caplog):
    fake_llm.responses = ["5", "5"]
    candidate = Candidate(id="x", text="a nice hotel")
    ranker = PointwiseRanker(LLMConfig(model="gpt-4o-mini"), num_samples=2)

    with caplog.at_level("WARNING"):
        ranker.score("query", candidate)

    assert any("temperature" in r.message for r in caplog.records)


def test_pointwise_structured_output_parses_json_score(fake_llm):
    fake_llm.responses = ['{"score": 7}']
    candidate = Candidate(id="x", text="a nice hotel")
    ranker = PointwiseRanker(
        LLMConfig(model="gpt-4o-mini"), structured_output=True
    )

    assert ranker.score("query", candidate) == 7
    assert fake_llm.calls[0]["response_format"]["type"] == "json_schema"


def test_pointwise_structured_output_falls_back_to_regex_on_malformed_json(fake_llm):
    fake_llm.responses = ["not json, just the number 6"]
    candidate = Candidate(id="x", text="a nice hotel")
    ranker = PointwiseRanker(
        LLMConfig(model="gpt-4o-mini"), structured_output=True
    )

    assert ranker.score("query", candidate) == 6


# --- multi-criteria scoring --------------------------------------------------


def _multi_criteria_responder(scores_by_candidate_text):
    """Fake LLM: returns 'name=score, ...' for whichever candidate text is
    in the prompt, keyed by that candidate's exact text."""

    def fn(messages):
        content = messages[-1]["content"]
        for text, scores in scores_by_candidate_text.items():
            if f'"{text}"' in content:
                return ", ".join(f"{k}={v}" for k, v in scores.items())
        raise AssertionError(f"no matching candidate text in prompt: {content!r}")

    return fn


def test_pointwise_criteria_weighted_sum_combines_scores(fake_llm):
    candidates = [Candidate(id="a", text="hotel a"), Candidate(id="b", text="hotel b")]
    fake_llm.responses = _multi_criteria_responder(
        {
            "hotel a": {"price_fit": 8, "location_fit": 6},
            "hotel b": {"price_fit": 9, "location_fit": 9},
        }
    )
    ranker = PointwiseRanker(
        LLMConfig(model="gpt-4o-mini"), criteria={"price_fit": 0.5, "location_fit": 0.5}
    )
    result = ranker.rank("query", candidates)

    scores = {c.id: c.score for c in result}
    assert scores["a"] == pytest.approx(7.0)  # 0.5*8 + 0.5*6
    assert scores["b"] == pytest.approx(9.0)  # 0.5*9 + 0.5*9
    assert [c.id for c in result] == ["b", "a"]
    assert ranker.total_calls == 2  # one call per candidate, same cost as holistic


def test_pointwise_criteria_priority_hierarchical_dominates(fake_llm):
    # "a" wins the high-tier criterion by only 1 point but loses badly on
    # medium/low; it should still rank first.
    candidates = [Candidate(id="a", text="hotel a"), Candidate(id="b", text="hotel b")]
    fake_llm.responses = _multi_criteria_responder(
        {
            "hotel a": {"family_friendly": 8, "price_fit": 0, "location_fit": 0},
            "hotel b": {"family_friendly": 7, "price_fit": 10, "location_fit": 10},
        }
    )
    ranker = PointwiseRanker(
        LLMConfig(model="gpt-4o-mini"),
        criteria={"family_friendly": "high", "price_fit": "medium", "location_fit": "low"},
    )
    result = ranker.rank("family friendly, cheap, central", candidates)

    assert result[0].id == "a"


def test_pointwise_criteria_metadata_merges_with_existing(fake_llm):
    fake_llm.responses = ["price_fit=7, location_fit=9"]
    candidate = Candidate(id="x", text="hotel x", metadata={"price": 100})
    ranker = PointwiseRanker(
        LLMConfig(model="gpt-4o-mini"), criteria={"price_fit": 0.5, "location_fit": 0.5}
    )
    result = ranker.rank("query", [candidate])

    assert result[0].metadata["price"] == 100
    assert result[0].metadata["criteria_scores"] == {"price_fit": 7.0, "location_fit": 9.0}
    assert result[0].metadata["criteria_weights"] == {"price_fit": 0.5, "location_fit": 0.5}
    assert result[0].metadata["criteria_source"] == "user"


def test_pointwise_criteria_reasoning_ignores_stray_numbers_before_final_answer(fake_llm):
    text = (
        "This hotel has 3 pools and is 2 miles away, but overall strong.\n\n"
        f"{FINAL_ANSWER_MARKER} price_fit=9, location_fit=8"
    )
    fake_llm.responses = [text]
    candidate = Candidate(id="x", text="a nice hotel")
    ranker = PointwiseRanker(
        LLMConfig(model="gpt-4o-mini"),
        criteria={"price_fit": 0.5, "location_fit": 0.5},
        reasoning=True,
    )

    assert ranker.score("query", candidate) == pytest.approx(8.5)


def test_pointwise_criteria_num_samples_averages_per_criterion_before_combining(fake_llm):
    fake_llm.responses = [
        "price_fit=4, location_fit=6",
        "price_fit=6, location_fit=8",
        "price_fit=8, location_fit=10",
    ]
    candidate = Candidate(id="x", text="a nice hotel")
    ranker = PointwiseRanker(
        LLMConfig(model="gpt-4o-mini", temperature=0.7),
        criteria={"price_fit": 0.5, "location_fit": 0.5},
        num_samples=3,
    )

    # price_fit avg=(4+6+8)/3=6, location_fit avg=(6+8+10)/3=8, combined=7
    assert ranker.score("query", candidate) == pytest.approx(7.0)
    assert ranker.total_calls == 3


def test_pointwise_criteria_structured_output_parses_json(fake_llm):
    fake_llm.responses = ['{"price_fit": 7, "location_fit": 9}']
    candidate = Candidate(id="x", text="a nice hotel")
    ranker = PointwiseRanker(
        LLMConfig(model="gpt-4o-mini"),
        criteria={"price_fit": 0.5, "location_fit": 0.5},
        structured_output=True,
    )

    assert ranker.score("query", candidate) == pytest.approx(8.0)
    assert fake_llm.calls[0]["response_format"]["type"] == "json_schema"


def test_pointwise_criteria_structured_output_falls_back_to_regex_on_malformed_json(fake_llm):
    fake_llm.responses = ["not json, price_fit=7, location_fit=9"]
    candidate = Candidate(id="x", text="a nice hotel")
    ranker = PointwiseRanker(
        LLMConfig(model="gpt-4o-mini"),
        criteria={"price_fit": 0.5, "location_fit": 0.5},
        structured_output=True,
    )

    assert ranker.score("query", candidate) == pytest.approx(8.0)


def test_pointwise_criteria_auto_extracts_then_scores(fake_llm):
    candidates = [Candidate(id="a", text="hotel a"), Candidate(id="b", text="hotel b")]
    fake_llm.responses = ["budget, location", "budget=8, location=6", "budget=4, location=9"]
    ranker = PointwiseRanker(LLMConfig(model="gpt-4o-mini"), criteria="auto")

    result = ranker.rank("affordable and central", candidates)

    assert ranker.total_calls == 3  # 1 extraction + 2 candidates
    assert result[0].metadata["criteria_source"] == "auto"
    assert set(result[0].metadata["criteria_scores"]) == {"budget", "location"}


def test_pointwise_criteria_auto_falls_back_to_holistic_on_extraction_failure(fake_llm, caplog):
    candidates = [Candidate(id="a", text="hotel a"), Candidate(id="b", text="hotel b")]
    fake_llm.responses = ["", "2", "9"]  # extraction unparseable -> holistic fallback
    ranker = PointwiseRanker(LLMConfig(model="gpt-4o-mini"), criteria="auto")

    with caplog.at_level("WARNING"):
        result = ranker.rank("query", candidates)

    assert [c.id for c in result] == ["b", "a"]  # holistic: b=9 > a=2
    assert "criteria_scores" not in (result[0].metadata or {})
    assert any("extraction" in r.message.lower() for r in caplog.records)


@pytest.mark.parametrize(
    "criteria",
    [
        {"a": 1, "b": "high"},  # mixed types
        {"a": 0},  # non-positive weight
        {"a": "urgent"},  # unknown priority string
        {},  # empty
        ["a", "b"],  # wrong type entirely
        "not-auto",  # invalid string sentinel
    ],
)
def test_pointwise_criteria_rejects_invalid_config(criteria):
    with pytest.raises(ValueError):
        PointwiseRanker(LLMConfig(model="gpt-4o-mini"), criteria=criteria)


# rank() dispatches num_samples>1 through its own batching/averaging code,
# written separately from score()'s -- every num_samples test above only
# calls score(), so these exercise the rank()-specific paths directly.


def test_pointwise_rank_num_samples_averages_scores_holistic(fake_llm):
    candidates = [Candidate(id="a", text="hotel a"), Candidate(id="b", text="hotel b")]
    # batches are ordered [a-sample1, a-sample2, b-sample1, b-sample2]
    fake_llm.responses = ["4", "6", "2", "8"]
    ranker = PointwiseRanker(
        LLMConfig(model="gpt-4o-mini", temperature=0.7), num_samples=2, max_concurrency=1
    )

    result = ranker.rank("query", candidates)

    scores = {c.id: c.score for c in result}
    assert scores["a"] == pytest.approx(5.0)  # (4+6)/2
    assert scores["b"] == pytest.approx(5.0)  # (2+8)/2
    assert ranker.total_calls == 4


def test_pointwise_rank_criteria_num_samples_averages_before_combining(fake_llm):
    candidates = [Candidate(id="a", text="hotel a"), Candidate(id="b", text="hotel b")]
    fake_llm.responses = [
        "price_fit=4, location_fit=6",  # a sample 1
        "price_fit=6, location_fit=8",  # a sample 2
        "price_fit=2, location_fit=2",  # b sample 1
        "price_fit=8, location_fit=8",  # b sample 2
    ]
    ranker = PointwiseRanker(
        LLMConfig(model="gpt-4o-mini", temperature=0.7),
        criteria={"price_fit": 0.5, "location_fit": 0.5},
        num_samples=2,
        max_concurrency=1,
    )

    result = ranker.rank("query", candidates)

    scores = {c.id: c.score for c in result}
    assert scores["a"] == pytest.approx(6.0)  # price avg=5, location avg=7 -> 6.0
    assert scores["b"] == pytest.approx(5.0)  # price avg=5, location avg=5 -> 5.0
    assert ranker.total_calls == 4


def test_pointwise_score_criteria_auto_falls_back_to_holistic_on_extraction_failure(fake_llm):
    fake_llm.responses = ["", "7"]  # extraction unparseable -> holistic fallback
    candidate = Candidate(id="x", text="a nice hotel")
    ranker = PointwiseRanker(LLMConfig(model="gpt-4o-mini"), criteria="auto")

    assert ranker.score("query", candidate) == 7


def test_pointwise_criteria_auto_with_structured_output(fake_llm):
    candidates = [Candidate(id="a", text="hotel a"), Candidate(id="b", text="hotel b")]
    fake_llm.responses = [
        '{"criteria": ["budget", "location"]}',
        '{"budget": 8, "location": 6}',
        '{"budget": 4, "location": 9}',
    ]
    ranker = PointwiseRanker(
        LLMConfig(model="gpt-4o-mini"), criteria="auto", structured_output=True
    )

    result = ranker.rank("affordable and central", candidates)

    assert result[0].id == "a"
    assert result[0].score == pytest.approx(7.0)  # 0.5*8 + 0.5*6
    assert ranker.total_calls == 3
    assert all(c["response_format"]["type"] == "json_schema" for c in fake_llm.calls)
