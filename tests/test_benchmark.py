from llmranker.benchmark import compare_rankers
from llmranker.llm import LLMConfig
from llmranker.rankers.cascade import CascadeRanker
from llmranker.rankers.pointwise import PointwiseRanker
from llmranker.rankers.setwise import SetwiseRanker
from llmranker.types import Candidate


def test_compare_rankers_accepts_a_cascade_ranker(fake_llm):
    candidates = [Candidate(id=str(i), text=f"item-{i}") for i in range(4)]
    fake_llm.responses = lambda messages: "2"  # constant score/label, content doesn't matter here

    plain = PointwiseRanker(LLMConfig(model="gpt-4o-mini"), name="plain")
    cascade = CascadeRanker(
        PointwiseRanker(LLMConfig(model="gpt-4o-mini")),
        SetwiseRanker(LLMConfig(model="gpt-4o-mini"), num_child=3),
        narrow_to=3,
    )

    df = compare_rankers([plain, cascade], "query", candidates, true_ranking=["0", "1", "2", "3"])

    assert set(df["ranker"]) == {"plain", cascade.name}
    assert (df["llm_calls"] > 0).all()
