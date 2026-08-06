import re

from llmranker.llm import LLMConfig
from llmranker.rankers.cascade import CascadeRanker
from llmranker.rankers.pointwise import PointwiseRanker
from llmranker.rankers.setwise import SetwiseRanker
from llmranker.types import Candidate


def _narrow_prefers_high_id_refine_prefers_low_id(messages):
    """A single fake covering both stages' prompt shapes: pointwise scores
    a candidate by its numeric id (so narrowing keeps the *highest* ids),
    setwise then picks the *lowest* id in a group (a deliberately opposite
    criterion, so the test can tell whether refine actually re-decided
    the order rather than just inheriting narrow's).
    """
    content = messages[-1]["content"]
    pointwise_match = re.search(r'Item: "item-(\d+)"', content)
    if pointwise_match:
        return pointwise_match.group(1)
    entries = re.findall(r'Item ([A-Z]): "item-(\d+)"', content)
    best_label, _ = min(entries, key=lambda e: int(e[1]))
    return best_label


def test_cascade_narrows_then_refines_with_only_survivors(fake_llm):
    candidates = [Candidate(id=str(i), text=f"item-{i}") for i in range(5)]
    fake_llm.responses = _narrow_prefers_high_id_refine_prefers_low_id

    narrow = PointwiseRanker(LLMConfig(model="gpt-4o-mini"))
    refine = SetwiseRanker(LLMConfig(model="gpt-4o-mini"), num_child=3)
    cascade = CascadeRanker(narrow, refine, narrow_to=3)

    result = cascade.rank("query", candidates)

    # Narrow keeps the top 3 by (highest) id -- {4, 3, 2} -- then refine,
    # using the opposite (lowest-id-wins) criterion, reorders just those
    # three ascending. Candidates 0 and 1 never reach refine at all.
    assert [c.id for c in result] == ["2", "3", "4"]


def test_cascade_totals_sum_both_stages(fake_llm):
    candidates = [Candidate(id=str(i), text=f"item-{i}") for i in range(5)]
    fake_llm.responses = _narrow_prefers_high_id_refine_prefers_low_id

    narrow = PointwiseRanker(LLMConfig(model="gpt-4o-mini"))
    refine = SetwiseRanker(LLMConfig(model="gpt-4o-mini"), num_child=3)
    cascade = CascadeRanker(narrow, refine, narrow_to=3)
    cascade.rank("query", candidates)

    assert cascade.total_calls == narrow.total_calls + refine.total_calls
    assert cascade.total_calls > narrow.total_calls  # refine made at least one call too
    assert cascade.total_prompt_tokens == 10 * cascade.total_calls  # FakeResponse default
    assert cascade.total_completion_tokens == 5 * cascade.total_calls


def test_cascade_config_delegates_to_refine_stage():
    narrow = PointwiseRanker(LLMConfig(model="gpt-4o-mini"))
    refine = SetwiseRanker(LLMConfig(model="claude-3-5-sonnet-20241022"), num_child=3)
    cascade = CascadeRanker(narrow, refine, narrow_to=3)

    assert cascade.config is refine.config
    assert cascade.config.model == "claude-3-5-sonnet-20241022"


def test_cascade_default_name_composes_stage_names():
    narrow = PointwiseRanker(LLMConfig(model="gpt-4o-mini"))
    refine = SetwiseRanker(LLMConfig(model="gpt-4o-mini"), num_child=3)
    cascade = CascadeRanker(narrow, refine, narrow_to=3)

    assert cascade.name == "Cascade(PointwiseRanker->SetwiseRanker)"
