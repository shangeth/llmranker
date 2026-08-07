import re

import pytest

from llmranker.benchmark import compare_rankers
from llmranker.llm import LLMConfig
from llmranker.rankers.cascade import CascadeRanker
from llmranker.rankers.rerank_api import RerankAPIRanker
from llmranker.rankers.setwise import SetwiseRanker
from llmranker.types import Candidate, Ranker


def _candidates(n=4):
    return [Candidate(id=str(i), text=f"item-{i}") for i in range(n)]


def test_ranks_by_relevance_score_in_one_call(fake_rerank):
    fake_rerank.results = [
        {"index": 2, "relevance_score": 0.9},
        {"index": 0, "relevance_score": 0.5},
        {"index": 3, "relevance_score": 0.4},
        {"index": 1, "relevance_score": 0.1},
    ]
    ranker = RerankAPIRanker(LLMConfig(model="cohere/rerank-v3.5"))

    result = ranker.rank("query", _candidates())

    assert [c.id for c in result] == ["2", "0", "3", "1"]
    assert [c.score for c in result] == [0.9, 0.5, 0.4, 0.1]
    # The whole point: one request for the entire list, regardless of n.
    assert ranker.total_calls == 1


def test_sends_candidate_text_as_documents(fake_rerank):
    fake_rerank.results = [{"index": 0, "relevance_score": 1.0}]
    ranker = RerankAPIRanker(LLMConfig(model="cohere/rerank-v3.5"), top_n=1)

    ranker.rank("my query", _candidates(3))

    call = fake_rerank.calls[0]
    assert call["model"] == "cohere/rerank-v3.5"
    assert call["query"] == "my query"
    assert call["documents"] == ["item-0", "item-1", "item-2"]
    assert call["top_n"] == 1


def test_metadata_is_carried_through(fake_rerank):
    fake_rerank.results = [{"index": 0, "relevance_score": 0.7}]
    candidates = [Candidate(id="a", text="x", metadata={"url": "http://example.com"})]

    result = RerankAPIRanker(LLMConfig(model="cohere/rerank-v3.5")).rank("q", candidates)

    assert result[0].metadata == {"url": "http://example.com"}


def test_unscored_candidates_are_appended_with_none_score(fake_rerank):
    """top_n truncation must not silently drop candidates: the ones the
    provider didn't score come back after the scored ones, in their
    original order, marked score=None rather than ranked worst."""
    fake_rerank.results = [
        {"index": 3, "relevance_score": 0.8},
        {"index": 1, "relevance_score": 0.6},
    ]
    ranker = RerankAPIRanker(LLMConfig(model="cohere/rerank-v3.5"), top_n=2)

    result = ranker.rank("query", _candidates(4))

    assert [c.id for c in result] == ["3", "1", "0", "2"]
    assert [c.score for c in result] == [0.8, 0.6, None, None]


def test_out_of_range_and_duplicate_indices_are_skipped(fake_rerank):
    fake_rerank.results = [
        {"index": 1, "relevance_score": 0.9},
        {"index": 99, "relevance_score": 0.8},  # out of range
        {"index": 1, "relevance_score": 0.7},  # duplicate
        {"index": 0, "relevance_score": 0.6},
    ]
    ranker = RerankAPIRanker(LLMConfig(model="cohere/rerank-v3.5"))

    result = ranker.rank("query", _candidates(2))

    assert [c.id for c in result] == ["1", "0"]
    assert [c.score for c in result] == [0.9, 0.6]


def test_empty_candidates_makes_no_call(fake_rerank):
    ranker = RerankAPIRanker(LLMConfig(model="cohere/rerank-v3.5"))

    assert ranker.rank("query", []) == []
    assert fake_rerank.calls == []
    assert ranker.total_calls == 0


def test_search_units_are_recorded_when_reported(fake_rerank):
    fake_rerank.results = [{"index": 0, "relevance_score": 1.0}]
    fake_rerank.search_units = 3
    ranker = RerankAPIRanker(LLMConfig(model="cohere/rerank-v3.5"))

    ranker.rank("query", _candidates(1))

    assert ranker.total_search_units == 3
    # No prompt and no completion, so token accounting stays at zero.
    assert ranker.total_prompt_tokens == 0
    assert ranker.total_completion_tokens == 0


def test_search_units_absent_leaves_counter_at_zero(fake_rerank):
    fake_rerank.results = [{"index": 0, "relevance_score": 1.0}]
    fake_rerank.search_units = None
    ranker = RerankAPIRanker(LLMConfig(model="cohere/rerank-v3.5"))

    ranker.rank("query", _candidates(1))

    assert ranker.total_search_units == 0


def test_stats_reset_between_rank_calls(fake_rerank):
    fake_rerank.results = [{"index": 0, "relevance_score": 1.0}]
    fake_rerank.search_units = 2
    ranker = RerankAPIRanker(LLMConfig(model="cohere/rerank-v3.5"))

    ranker.rank("query", _candidates(1))
    ranker.rank("query", _candidates(1))

    assert ranker.total_calls == 1
    assert ranker.total_search_units == 2


def test_satisfies_ranker_protocol():
    assert isinstance(RerankAPIRanker(LLMConfig(model="cohere/rerank-v3.5")), Ranker)


def test_rejects_non_positive_top_n():
    with pytest.raises(ValueError, match="top_n must be >= 1"):
        RerankAPIRanker(LLMConfig(model="cohere/rerank-v3.5"), top_n=0)


def test_cascade_uses_rerank_api_as_the_cheap_narrow_stage(fake_rerank, fake_llm):
    """The headline composition: one rerank request narrows the field, then
    a setwise LLM ranker carefully reorders only the survivors."""
    candidates = _candidates(5)
    # Rerank ranks by descending id; setwise then picks the lowest id in a
    # group, a deliberately opposite criterion, so the assertion can tell
    # the refine stage actually re-decided rather than inheriting.
    fake_rerank.results = [{"index": i, "relevance_score": float(i)} for i in range(4, -1, -1)]

    def pick_lowest_id(messages):
        entries = re.findall(
            r"Item ([A-Z]): <candidate>item-(\d+)</candidate>", messages[-1]["content"]
        )
        best_label, _ = min(entries, key=lambda e: int(e[1]))
        return best_label

    fake_llm.responses = pick_lowest_id

    cascade = CascadeRanker(
        narrow=RerankAPIRanker(LLMConfig(model="cohere/rerank-v3.5")),
        refine=SetwiseRanker(LLMConfig(model="gpt-4o-mini"), num_child=3),
        narrow_to=3,
    )
    result = cascade.rank("query", candidates)

    assert [c.id for c in result] == ["2", "3", "4"]
    # One rerank request + however many LLM calls setwise needed.
    assert cascade.total_calls == 1 + len(fake_llm.calls)


def test_compare_rankers_reports_blank_cost_not_zero(fake_rerank):
    """Rerank endpoints bill per search unit and report no tokens, so the
    token-based estimate would say $0.00 for a call that costs real money.
    compare_rankers must defer to the ranker's own estimate instead."""
    fake_rerank.results = [
        {"index": 0, "relevance_score": 0.9},
        {"index": 1, "relevance_score": 0.1},
    ]
    ranker = RerankAPIRanker(LLMConfig(model="cohere/rerank-v3.5"))

    report = compare_rankers([ranker], "query", _candidates(2), true_ranking=["0", "1"])

    assert report.loc[0, "llm_calls"] == 1
    assert report["cost_usd"].isna().all()
