import pytest

from llmranker.llm import LLMConfig
from llmranker.rankers.pointwise import PointwiseRanker


def test_reasoning_and_structured_output_defaults_are_off():
    ranker = PointwiseRanker(LLMConfig(model="gpt-4o-mini"))
    assert ranker.reasoning is False
    assert ranker.num_samples == 1
    assert ranker.structured_output is False


def test_rejects_reasoning_and_structured_output_together():
    with pytest.raises(ValueError):
        PointwiseRanker(LLMConfig(model="gpt-4o-mini"), reasoning=True, structured_output=True)


def test_rejects_num_samples_below_one():
    with pytest.raises(ValueError):
        PointwiseRanker(LLMConfig(model="gpt-4o-mini"), num_samples=0)
