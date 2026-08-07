"""Regression tests for issues found in a review of the existing rankers.

Each test here pins down a specific defect that was reproduced before being
fixed, so the fix can't silently regress.
"""

import logging

import pytest

from llmranker.llm import LLMConfig
from llmranker.rankers.listwise import ListwiseRanker
from llmranker.rankers.pairwise import PairwiseRanker
from llmranker.rankers.pointwise import PointwiseRanker
from llmranker.rankers.setwise import SetwiseRanker
from llmranker.rankers.tourrank import TourRankRanker
from llmranker.types import Candidate

# --- duplicate candidate ids must not lose candidates ----------------------


@pytest.mark.parametrize("strategy", ["heapsort", "bubblesort", "allpairs"])
def test_pairwise_keeps_every_candidate_when_ids_are_duplicated(fake_llm, strategy):
    """_finalize used to exclude the remainder by `id` field, so a candidate
    sharing an id with a ranked one vanished from the output entirely."""
    fake_llm.responses = lambda m: "A"
    duplicated = [
        Candidate(id="x", text="first"),
        Candidate(id="y", text="second"),
        Candidate(id="x", text="third"),
    ]

    result = PairwiseRanker(LLMConfig(model="gpt-4o-mini"), strategy=strategy, k=1).rank(
        "q", duplicated
    )

    assert len(result) == 3
    assert sorted(c.text for c in result) == ["first", "second", "third"]


def test_setwise_keeps_every_candidate_when_ids_are_duplicated(fake_llm):
    fake_llm.responses = lambda m: "A"
    duplicated = [
        Candidate(id="x", text="first"),
        Candidate(id="y", text="second"),
        Candidate(id="x", text="third"),
    ]

    result = SetwiseRanker(LLMConfig(model="gpt-4o-mini"), num_child=2, k=1).rank("q", duplicated)

    assert len(result) == 3
    assert sorted(c.text for c in result) == ["first", "second", "third"]


def test_pairwise_allpairs_counts_duplicate_ids_separately(fake_llm):
    """Win counts keyed on the `id` field merged two distinct candidates
    into one bucket; they're keyed on identity now."""

    # Whichever candidate has the longer text wins, so ordering is decidable
    # even though two candidates share an id.
    def prefer_longer(messages):
        import re

        entries = re.findall(r'Item ([AB]): "([^"]*)"', messages[-1]["content"])
        return max(entries, key=lambda e: len(e[1]))[0]

    fake_llm.responses = prefer_longer
    duplicated = [
        Candidate(id="x", text="short"),
        Candidate(id="x", text="a much longer description"),
        Candidate(id="y", text="mid length"),
    ]

    result = PairwiseRanker(LLMConfig(model="gpt-4o-mini"), strategy="allpairs").rank(
        "q", duplicated
    )

    assert [c.text for c in result] == ["a much longer description", "mid length", "short"]


# --- tie-breaks must not reinstate position bias ---------------------------


def test_pairwise_split_vote_does_not_always_favor_the_first_argument(fake_llm):
    """With an even num_samples a 50/50 split is common; it used to resolve
    to whichever candidate was passed as `a`, which is precisely the
    position bias num_samples exists to cancel."""
    state = {"n": 0}

    def alternate(messages):
        state["n"] += 1
        return "A" if state["n"] % 2 else "B"

    fake_llm.responses = alternate
    a, b = Candidate(id="a", text="a"), Candidate(id="b", text="b")

    ranker = PairwiseRanker(LLMConfig(model="gpt-4o-mini", temperature=1.0), num_samples=4, seed=0)
    winners = {ranker.compare("q", a, b).id for _ in range(30)}

    assert winners == {"a", "b"}, "split votes always resolved to the same slot"


def test_setwise_tied_vote_does_not_always_favor_original_order(fake_llm):
    state = {"n": 0}

    def cycle(messages):
        state["n"] += 1
        return ["A", "B", "C"][state["n"] % 3]

    fake_llm.responses = cycle
    group = [Candidate(id=c, text=c) for c in "pqr"]

    ranker = SetwiseRanker(
        LLMConfig(model="gpt-4o-mini", temperature=1.0), num_child=3, num_samples=3, seed=0
    )
    winners = {ranker.compare("q", group).id for _ in range(30)}

    assert len(winners) > 1, "three-way ties always resolved to the same position"


# --- pointwise score parsing ----------------------------------------------


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("8", 8.0),
        ("  8  ", 8.0),
        ("Score: 8", 8.0),
        ("score = 7", 7.0),
        ("This item scores 8 out of 10", 8.0),
        # Denominators are stripped, so the 10 can't be mistaken for the score.
        ("9/10", 9.0),
        # A labelled score wins over an earlier unlabelled number, which is
        # the case that used to be parsed as 3.
        ("Item 3 has a score of 9", 9.0),
        ("Relevance score: 4 because it only partly matches", 4.0),
        # Justification after a leading bare score.
        ("8 - matches most of the query", 8.0),
    ],
)
def test_pointwise_parses_score_from_chatty_output(fake_llm, output, expected):
    fake_llm.responses = lambda m: output

    score = PointwiseRanker(LLMConfig(model="gpt-4o-mini")).score("q", Candidate(id="1", text="t"))

    assert score == expected


@pytest.mark.parametrize(
    "output",
    [
        "somewhere between 4 and 6, hard to say",
        # Genuinely undecidable without a label: 3 or 9? The heuristic picks
        # the first, but the point of this test is that it no longer does so
        # *silently* -- the old parser returned 3 with no signal at all.
        "Item 3 deserves a 9",
    ],
)
def test_pointwise_warns_when_score_output_is_ambiguous(fake_llm, caplog, output):
    fake_llm.responses = lambda m: output

    with caplog.at_level(logging.WARNING, logger="llmranker"):
        PointwiseRanker(LLMConfig(model="gpt-4o-mini")).score("q", Candidate(id="1", text="t"))

    assert "Ambiguous score output" in caplog.text


# --- num_samples that cannot help must say so ------------------------------


def test_listwise_warns_when_num_samples_cannot_help(fake_llm, caplog):
    """Listwise sends an identical prompt every sample, so at temperature=0
    the repeats are pure waste. Pointwise warned about this; listwise
    silently burned the calls."""
    fake_llm.responses = lambda m: "[2] > [1] > [3] > [4]"
    candidates = [Candidate(id=str(i), text=f"item-{i}") for i in range(4)]

    with caplog.at_level(logging.WARNING, logger="llmranker"):
        ListwiseRanker(LLMConfig(model="gpt-4o-mini"), window_size=4, num_samples=5).rank(
            "q", candidates
        )

    assert "temperature=0.0" in caplog.text


def test_listwise_does_not_warn_when_temperature_is_raised(fake_llm, caplog):
    fake_llm.responses = lambda m: "[2] > [1] > [3] > [4]"
    candidates = [Candidate(id=str(i), text=f"item-{i}") for i in range(4)]

    with caplog.at_level(logging.WARNING, logger="llmranker"):
        ListwiseRanker(
            LLMConfig(model="gpt-4o-mini", temperature=0.7), window_size=4, num_samples=5
        ).rank("q", candidates)

    assert "temperature=0.0" not in caplog.text


# --- tourrank must not advance the same candidate twice --------------------


def test_tourrank_structured_output_rejects_duplicate_labels(fake_llm):
    """A model can satisfy the schema's minItems/maxItems with a repeated
    label; that used to advance one candidate twice, scoring it more points
    than there were stages."""
    fake_llm.responses = lambda m: '{"selected": ["A", "A"]}'
    candidates = [Candidate(id=str(i), text=f"item-{i}") for i in range(4)]

    ranker = TourRankRanker(
        LLMConfig(model="gpt-4o-mini"),
        group_size=4,
        advance_per_group=2,
        num_stages=1,
        num_tournaments=1,
        structured_output=True,
        seed=0,
    )
    result = ranker.rank("q", candidates)

    scores = [c.score for c in result]
    assert max(scores) <= 1.0, "a candidate scored more points than there were stages"
    assert sum(1 for s in scores if s == 1.0) == 2, "exactly two candidates should advance"


# --- usage stats survive a failing call in a concurrent batch --------------


def test_call_many_records_stats_for_calls_that_succeeded(fake_llm):
    """Stats used to be summed only after the whole batch returned, so one
    raising call discarded the accounting for every call that worked."""
    state = {"n": 0}

    def fail_on_third(messages):
        state["n"] += 1
        if state["n"] == 3:
            raise RuntimeError("boom")
        return "5"

    fake_llm.responses = fail_on_third
    candidates = [Candidate(id=str(i), text=f"item-{i}") for i in range(6)]

    ranker = PointwiseRanker(LLMConfig(model="gpt-4o-mini"), max_concurrency=2)
    with pytest.raises(RuntimeError):
        ranker.rank("q", candidates)

    assert ranker.total_calls > 0
    assert ranker.total_prompt_tokens == 10 * ranker.total_calls
