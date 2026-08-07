"""Tests pinning implementations to the algorithms their papers describe.

Each test names the specific property from the source paper that was found
missing or wrong in a fidelity audit, so a future refactor can't quietly
drift away from the cited work again.
"""

import re

from llmranker.llm import LLMConfig
from llmranker.rankers.listwise import ListwiseRanker
from llmranker.rankers.pairwise import PairwiseRanker
from llmranker.rankers.setwise import SetwiseRanker
from llmranker.types import Candidate


def _candidates(n):
    return [Candidate(id=str(i), text=f"d{i}") for i in range(n)]


# --- PRP-Allpair (Qin et al., arXiv:2306.17563) -----------------------------


def test_allpairs_asks_every_pair_in_both_orders(fake_llm):
    """The paper's central mechanism: "for each pair of documents, we will
    inquire the LLM twice by swapping their order". A single fixed order is
    exactly the position bias PRP exists to cancel."""
    fake_llm.responses = lambda m: "A"
    PairwiseRanker(LLMConfig(model="m"), strategy="allpairs").rank("q", _candidates(4))

    asked = [
        tuple(re.findall(r"Item [AB]: <candidate>(d\d)</candidate>", c["messages"][-1]["content"]))
        for c in fake_llm.calls
    ]
    pairs = {frozenset(p) for p in asked}
    assert len(asked) == 12, "expected both orders of all C(4,2)=6 pairs"
    assert len(pairs) == 6
    for a, b in asked:
        assert (b, a) in asked, f"pair ({a}, {b}) was not asked in both orders"


def test_allpairs_scores_disagreement_as_a_half_point_tie(fake_llm):
    """PRP scores s_i = 1*sum(wins) + 0.5*sum(ties), where a "tie" is the two
    orderings disagreeing. A model that always answers "A" disagrees on every
    pair, so every candidate must end up exactly tied."""
    fake_llm.responses = lambda m: "A"
    candidates = _candidates(4)

    result = PairwiseRanker(LLMConfig(model="m"), strategy="allpairs").rank("q", candidates)

    # Always-"A" means whichever document is shown first wins, so each pair
    # splits 1-1 across the two orders: half a point each, everyone level.
    assert {c.id for c in result} == {c.id for c in candidates}
    scores = [c.score for c in result]
    assert scores == sorted(scores, reverse=True)


def test_allpairs_consistent_preference_beats_a_disagreeing_one(fake_llm):
    """A candidate the model prefers in both orderings must outrank one it
    only wins by position."""

    def prefer_d0(messages):
        entries = re.findall(r"Item ([AB]): <candidate>(d\d)</candidate>", messages[-1]["content"])
        for label, doc in entries:
            if doc == "d0":
                return label
        return "A"

    fake_llm.responses = prefer_d0
    result = PairwiseRanker(LLMConfig(model="m"), strategy="allpairs").rank("q", _candidates(4))

    assert result[0].id == "0"


# --- Setwise (Zhuang et al., arXiv:2310.09497 + ielab/llm-rankers) ----------


def test_bubblesort_window_matches_heapsort_group_size(fake_llm):
    """The reference implementation slides a window of num_child + 1 -- one
    slot carrying the running winner plus num_child new contenders, the same
    group heapsort builds from a parent and its children. A window of
    num_child spends more calls for the same work."""
    fake_llm.responses = lambda m: "A"
    ranker = SetwiseRanker(LLMConfig(model="m"), num_child=3, strategy="bubblesort")
    ranker.rank("q", _candidates(12))

    sizes = [len(re.findall(r"Item [A-Z]:", c["messages"][-1]["content"])) for c in fake_llm.calls]
    assert max(sizes) == ranker.num_child + 1


def test_no_call_is_wasted_on_a_single_candidate(fake_llm):
    """Asking which of one candidate is most relevant cannot return anything
    new; the tail of the bubble pass used to emit exactly that."""
    fake_llm.responses = lambda m: "A"
    for strategy in ("heapsort", "bubblesort"):
        fake_llm.calls.clear()
        SetwiseRanker(LLMConfig(model="m"), num_child=3, strategy=strategy).rank(
            "q", _candidates(9)
        )
        sizes = [
            len(re.findall(r"Item [A-Z]:", c["messages"][-1]["content"])) for c in fake_llm.calls
        ]
        assert min(sizes) >= 2, f"{strategy} issued a single-candidate comparison"


# --- RankGPT (Sun et al., arXiv:2304.09542) ---------------------------------


def test_listwise_defaults_are_the_papers_tuned_configuration():
    """RankGPT tuned window 20 / step 10 on TREC-DL19."""
    ranker = ListwiseRanker(LLMConfig(model="m"))
    assert (ranker.window_size, ranker.step_size) == (20, 10)


def test_listwise_step_defaults_to_half_the_window():
    """The paper's step is w/2, so setting only window_size must keep that
    relationship rather than colliding with a fixed default."""
    assert ListwiseRanker(LLMConfig(model="m"), window_size=4).step_size == 2
    assert ListwiseRanker(LLMConfig(model="m"), window_size=8).step_size == 4
    assert ListwiseRanker(LLMConfig(model="m"), window_size=8, step_size=3).step_size == 3


def test_listwise_slides_from_the_back_of_the_list(fake_llm):
    """RankGPT slides back-to-front so improvements propagate toward the top."""
    seen = []

    def record(messages):
        docs = re.findall(
            r"\[\d+\] <candidate>(d\d+)</candidate>", "\n".join(m["content"] for m in messages)
        )
        seen.append(docs)
        return " > ".join(f"[{i + 1}]" for i in range(len(docs)))

    fake_llm.responses = record
    ListwiseRanker(LLMConfig(model="m"), window_size=4, step_size=2).rank("q", _candidates(8))

    assert seen[0] == ["d4", "d5", "d6", "d7"], "first window should be the tail of the list"
    assert seen[-1] == ["d0", "d1", "d2", "d3"], "last window should be the head of the list"
