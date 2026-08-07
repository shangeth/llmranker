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
        schedule=[2],
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


# --- metrics: NDCG must stay normalized, and graded relevance must reach it ---


def test_ndcg_never_exceeds_one_with_graded_relevance():
    """IDCG used to be dcg(true_ranking), which is only ideal if the caller
    happened to sort true_ranking by their own grades. When they didn't, the
    'normalized' score came out above 1."""
    from llmranker.metrics import RankingMetrics

    relevance = {"a": 3.0, "b": 1.0, "c": 2.0}
    m = RankingMetrics()
    # every ordering of true_ranking, including ones not sorted by grade
    for true in (["b", "c", "a"], ["a", "c", "b"], ["c", "a", "b"]):
        for pred in (["a", "c", "b"], ["b", "a", "c"], ["c", "b", "a"]):
            score = m.ndcg(true, pred, relevance=relevance)
            assert 0.0 <= score <= 1.0, f"ndcg={score} for true={true} pred={pred}"
    # the best possible prediction scores exactly 1
    assert m.ndcg(["b", "c", "a"], ["a", "c", "b"], relevance=relevance) == pytest.approx(1.0)


def test_graded_relevance_reaches_get_metrics_and_compare_rankers(fake_llm):
    """`relevance` was documented but unreachable: get_metrics didn't accept
    it, and compare_rankers goes through get_metrics."""
    from llmranker.benchmark import compare_rankers
    from llmranker.metrics import RankingMetrics

    relevance = {"a": 3.0, "b": 0.0, "c": 1.0}
    graded = RankingMetrics().get_metrics(["a", "b", "c"], ["b", "a", "c"], relevance=relevance)
    ordered = RankingMetrics().get_metrics(["a", "b", "c"], ["b", "a", "c"])
    assert graded["ndcg"] != ordered["ndcg"], "relevance= had no effect"

    fake_llm.responses = lambda m: "5"
    report = compare_rankers(
        [PointwiseRanker(LLMConfig(model="gpt-4o-mini"))],
        "q",
        [Candidate(id=i, text=i) for i in ("a", "b", "c")],
        true_ranking=["a", "b", "c"],
        relevance=relevance,
    )
    assert 0.0 <= report.loc[0, "ndcg"] <= 1.0


def test_degenerate_correlations_are_nan_without_scipy_warnings(recwarn):
    from llmranker.metrics import RankingMetrics

    result = RankingMetrics().get_metrics(["a"], ["a"])
    assert result["spearman"] != result["spearman"]  # NaN
    assert result["kendall_tau"] != result["kendall_tau"]
    assert not [w for w in recwarn if "onstant" in str(w.message)]


# --- output contracts -------------------------------------------------------


def test_every_ranker_returns_all_candidates(fake_llm):
    """`k`/`top_n` cap sorting effort; they never drop candidates. TourRank
    used to truncate, making the same param name mean something different."""
    from llmranker.rankers.tourrank import TourRankRanker

    candidates = [Candidate(id=str(i), text=f"item-{i}") for i in range(6)]
    rankers = [
        PointwiseRanker(LLMConfig(model="m")),
        PairwiseRanker(LLMConfig(model="m"), k=2),
        SetwiseRanker(LLMConfig(model="m"), num_child=3, k=2),
        ListwiseRanker(LLMConfig(model="m"), window_size=3),
        TourRankRanker(LLMConfig(model="m"), num_tournaments=1),
    ]
    for ranker in rankers:
        fake_llm.responses = lambda m: "5 A B [1] > [2] > [3]"
        assert len(ranker.rank("q", candidates)) == 6, ranker.name


def test_cascade_is_the_documented_exception_to_that_rule(fake_llm):
    """CascadeRanker returns narrow_to candidates, not all of them --
    discarding what the cheap stage rejected is the point of it. The README
    states the invariant and this exception together; this pins the pair."""
    from llmranker.rankers.cascade import CascadeRanker

    fake_llm.responses = lambda m: "5 A B"
    candidates = [Candidate(id=str(i), text=f"item-{i}") for i in range(6)]
    cascade = CascadeRanker(
        narrow=PointwiseRanker(LLMConfig(model="m")),
        refine=SetwiseRanker(LLMConfig(model="m"), num_child=3),
        narrow_to=3,
    )

    assert len(cascade.rank("q", candidates)) == 3


def test_score_kind_declares_how_to_read_score():
    from llmranker.rankers.cascade import CascadeRanker
    from llmranker.rankers.rerank_api import RerankAPIRanker
    from llmranker.rankers.tourrank import TourRankRanker

    assert PointwiseRanker(LLMConfig(model="m")).score_kind == "relevance"
    assert PairwiseRanker(LLMConfig(model="m")).score_kind == "rank_position"
    assert SetwiseRanker(LLMConfig(model="m")).score_kind == "rank_position"
    assert ListwiseRanker(LLMConfig(model="m")).score_kind == "rank_position"
    assert TourRankRanker(LLMConfig(model="m")).score_kind == "tournament_points"
    assert RerankAPIRanker(LLMConfig(model="cohere/rerank-v3.5")).score_kind == "provider_relevance"
    # a cascade reports whichever stage actually produced the scores
    cascade = CascadeRanker(
        narrow=PointwiseRanker(LLMConfig(model="m")),
        refine=SetwiseRanker(LLMConfig(model="m")),
        narrow_to=2,
    )
    assert cascade.score_kind == "rank_position"


def test_output_metadata_does_not_alias_the_input(fake_llm):
    """Output candidates used to share the caller's metadata dict, so
    mutating one silently mutated the other."""
    fake_llm.responses = lambda m: "5"
    original = {"url": "http://example.com"}
    candidates = [Candidate(id="1", text="t", metadata=original)]

    result = PointwiseRanker(LLMConfig(model="m")).rank("q", candidates)

    assert result[0].metadata == original
    assert result[0].metadata is not original
    result[0].metadata["url"] = "mutated"
    assert original["url"] == "http://example.com"


def test_importing_llmranker_does_not_mutate_litellm_globals():
    """A library must not reconfigure a shared third-party module for the
    whole host process at import time."""
    import subprocess
    import sys

    probe = (
        "import litellm; before = litellm.suppress_debug_info; "
        "import llmranker; print(before == litellm.suppress_debug_info)"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=False)
    assert out.stdout.strip() == "True", out.stderr
