import re

import pytest

from llmranker.llm import LLMConfig
from llmranker.prompts import FINAL_ANSWER_MARKER
from llmranker.rankers.listwise import ListwiseRanker
from llmranker.types import Candidate

_ITEM_RE = re.compile(r"^\[(\d+)\] <candidate>(.*)</candidate>$")


def _ground_truth_responder(rank_of):
    """Fake LLM: reads the numbered items out of the conversation and returns
    them in true-rank order, formatted as '[k] > [j] > ...'."""

    def fn(messages):
        entries = []
        for m in messages:
            if m["role"] == "user":
                match = _ITEM_RE.match(m["content"])
                if match:
                    entries.append((int(match.group(1)), match.group(2)))
        entries.sort(key=lambda e: rank_of[e[1]])
        return " > ".join(f"[{r}]" for r, _ in entries)

    return fn


def test_listwise_compare_reorders_single_window(fake_llm):
    window = [
        Candidate(id="x", text="rank-3"),
        Candidate(id="y", text="rank-1"),
        Candidate(id="z", text="rank-2"),
    ]
    rank_of = {"rank-3": 3, "rank-1": 1, "rank-2": 2}
    fake_llm.responses = _ground_truth_responder(rank_of)

    ranker = ListwiseRanker(LLMConfig(model="gpt-4o-mini"), window_size=3, step_size=1)
    reordered = ranker.compare("query", window)

    assert [c.id for c in reordered] == ["y", "z", "x"]


def test_listwise_reasoning_ignores_stray_numbers_before_final_answer(fake_llm):
    # Without marker-aware parsing, the naive digit-scan would pick up the
    # stray "3" and "2" from the reasoning text before the real ranking,
    # producing a corrupted order ([2,1,0] instead of the correct [2,0,1]).
    window = [Candidate(id=str(i), text=f"item-{i}") for i in range(3)]
    text = (
        "There are 3 relevant factors and 2 secondary ones to consider "
        f"here.\n\n{FINAL_ANSWER_MARKER} [3] > [1] > [2]"
    )
    fake_llm.responses = [text]

    ranker = ListwiseRanker(LLMConfig(model="gpt-4o-mini"), window_size=3, reasoning=True)
    reordered = ranker.compare("query", window)

    assert [c.id for c in reordered] == ["2", "0", "1"]


def test_listwise_converges_to_true_order_with_repeats(fake_llm):
    candidates = [Candidate(id=str(r), text=f"item-{r}") for r in [6, 3, 1, 5, 2, 4]]
    rank_of = {c.text: int(c.id) for c in candidates}
    fake_llm.responses = _ground_truth_responder(rank_of)

    ranker = ListwiseRanker(
        LLMConfig(model="gpt-4o-mini"), window_size=4, step_size=2, num_repeat=3
    )
    result = ranker.rank("query", candidates)

    assert [c.id for c in result] == ["1", "2", "3", "4", "5", "6"]
    assert result[0].score > result[-1].score


@pytest.mark.parametrize(
    "text,n,expected",
    [
        ("[3] > [1] > [2]", 3, [2, 0, 1]),
        ("[2] > [2] > [1]", 2, [1, 0]),  # duplicate [2] dropped on repeat
        ("[5] > [1]", 3, [0, 1, 2]),  # out-of-range [5] dropped, [2] appended
        ("no numbers here", 2, [0, 1]),  # nothing parseable -> original order
    ],
)
def test_parse_permutation_handles_malformed_output(text, n, expected):
    ranker = ListwiseRanker(LLMConfig(model="gpt-4o-mini"))
    assert ranker._parse_permutation(text, n) == expected


def test_listwise_rejects_invalid_config():
    with pytest.raises(ValueError):
        ListwiseRanker(LLMConfig(model="gpt-4o-mini"), window_size=4, step_size=5)
    with pytest.raises(ValueError):
        ListwiseRanker(LLMConfig(model="gpt-4o-mini"), window_size=1)


def test_listwise_num_samples_borda_counts_across_samples(fake_llm):
    # Permutation self-consistency shuffles the window per sample, so a
    # fixed positional response list no longer means anything -- use the
    # content-based ground-truth responder instead, which computes the
    # correct order regardless of the position each candidate happens to
    # land in for a given sample.
    window = [
        Candidate(id="p", text="item-p"),
        Candidate(id="q", text="item-q"),
        Candidate(id="r", text="item-r"),
    ]
    rank_of = {"item-p": 1, "item-q": 2, "item-r": 3}
    fake_llm.responses = _ground_truth_responder(rank_of)

    ranker = ListwiseRanker(LLMConfig(model="gpt-4o-mini"), window_size=3, num_samples=3, seed=1)
    reordered = ranker.compare("query", window)

    assert [c.id for c in reordered] == ["p", "q", "r"]
    assert ranker.total_calls == 3

    # Prove shuffling actually happened, not just that Borda math still
    # works: each sample's messages should show the candidates in a
    # different order than at least one other sample.
    orders = [
        tuple(
            _ITEM_RE.match(m["content"]).group(2)
            for m in call["messages"]
            if _ITEM_RE.match(m["content"])
        )
        for call in fake_llm.calls
    ]
    assert len(set(orders)) > 1, "expected different candidate orders across samples"


def test_listwise_seed_reproducible(fake_llm):
    window = [Candidate(id=str(i), text=f"item-{i}") for i in range(5)]
    rank_of = {c.text: i for i, c in enumerate(window)}

    fake_llm.responses = _ground_truth_responder(rank_of)
    ranker_a = ListwiseRanker(LLMConfig(model="gpt-4o-mini"), window_size=5, num_samples=4, seed=7)
    ranker_a.compare("query", window)
    calls_a = [c["messages"] for c in fake_llm.calls]

    fake_llm.calls.clear()
    fake_llm.responses = _ground_truth_responder(rank_of)
    ranker_b = ListwiseRanker(LLMConfig(model="gpt-4o-mini"), window_size=5, num_samples=4, seed=7)
    ranker_b.compare("query", window)
    calls_b = [c["messages"] for c in fake_llm.calls]

    assert calls_a == calls_b


def test_listwise_num_samples_no_low_temperature_warning(fake_llm, caplog):
    window = [Candidate(id=str(i), text=f"item-{i}") for i in range(3)]
    rank_of = {c.text: i for i, c in enumerate(window)}
    fake_llm.responses = _ground_truth_responder(rank_of)

    ranker = ListwiseRanker(
        LLMConfig(model="gpt-4o-mini", temperature=0.0), window_size=3, num_samples=3
    )
    ranker.compare("query", window)

    assert "num_samples" not in caplog.text


def test_listwise_structured_output_parses_json_ranking(fake_llm):
    window = [Candidate(id=str(i), text=f"item-{i}") for i in range(3)]
    fake_llm.responses = ['{"ranking": [3, 1, 2]}']
    ranker = ListwiseRanker(
        LLMConfig(model="gpt-4o-mini"),
        window_size=3,
        structured_output=True,
    )

    reordered = ranker.compare("query", window)
    assert [c.id for c in reordered] == ["2", "0", "1"]
    assert fake_llm.calls[0]["response_format"]["type"] == "json_schema"


def test_listwise_structured_output_falls_back_to_regex_on_malformed_json(fake_llm):
    window = [Candidate(id=str(i), text=f"item-{i}") for i in range(3)]
    fake_llm.responses = ["not json but [3] > [1] > [2]"]
    ranker = ListwiseRanker(
        LLMConfig(model="gpt-4o-mini"),
        window_size=3,
        structured_output=True,
    )

    reordered = ranker.compare("query", window)
    assert [c.id for c in reordered] == ["2", "0", "1"]


def test_insert_rank_score_key_appends_score_to_prompt(fake_llm):
    window = [
        Candidate(id="1", text="item-1", metadata={"bm25": 12.3}),
        Candidate(id="2", text="item-2", metadata={"bm25": 4.5}),
    ]
    fake_llm.responses = lambda m: "[1] > [2]"

    ranker = ListwiseRanker(
        LLMConfig(model="gpt-4o-mini"), window_size=2, insert_rank_score_key="bm25"
    )
    ranker.compare("query", window)

    content = "\n".join(m["content"] for m in fake_llm.calls[0]["messages"])
    assert "<candidate>item-1</candidate> (bm25: 12.3)" in content
    assert "<candidate>item-2</candidate> (bm25: 4.5)" in content


def test_insert_rank_score_key_missing_warns_once(fake_llm, caplog):
    # One candidate never has the key; sliding the window across two steps
    # means it's considered twice, so the warning must still only fire once.
    candidates = [
        Candidate(id="1", text="item-1", metadata={"bm25": 9.0}),
        Candidate(id="2", text="item-2"),  # missing the key
        Candidate(id="3", text="item-3", metadata={"bm25": 3.0}),
    ]
    fake_llm.responses = lambda m: "[1] > [2]"

    ranker = ListwiseRanker(
        LLMConfig(model="gpt-4o-mini"),
        window_size=2,
        step_size=1,
        insert_rank_score_key="bm25",
    )
    ranker.rank("query", candidates)

    warnings = [r for r in caplog.records if "insert_rank_score_key" in r.getMessage()]
    assert len(warnings) == 1

    all_content = "\n".join(m["content"] for call in fake_llm.calls for m in call["messages"])
    # The candidate missing the key never gets a score suffix.
    assert "item-2</candidate> (" not in all_content


def test_insert_rank_score_key_default_none_is_unchanged(fake_llm):
    window = [
        Candidate(id="1", text="item-1", metadata={"bm25": 9.0}),
        Candidate(id="2", text="item-2"),
    ]
    fake_llm.responses = lambda m: "[1] > [2]"

    ranker = ListwiseRanker(LLMConfig(model="gpt-4o-mini"), window_size=2)
    ranker.compare("query", window)

    content = "\n".join(m["content"] for m in fake_llm.calls[0]["messages"])
    assert "bm25" not in content
    assert "<candidate>item-1</candidate>" in content
    assert "<candidate>item-2</candidate>" in content
