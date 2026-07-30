import re

import pytest

from llmranker.llm import LLMConfig
from llmranker.prompts import FINAL_ANSWER_MARKER
from llmranker.rankers.listwise import ListwiseRanker
from llmranker.types import Candidate

_ITEM_RE = re.compile(r"^\[(\d+)\] (.*)$")


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
    window = [Candidate(id="x", text="rank-3"), Candidate(id="y", text="rank-1"), Candidate(id="z", text="rank-2")]
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
