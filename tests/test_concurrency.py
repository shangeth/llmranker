import time

from llmranker.llm import LLMConfig
from llmranker.rankers.pointwise import PointwiseRanker
from llmranker.types import Candidate


def test_call_many_preserves_order_despite_out_of_order_completion(fake_llm):
    """_call_many() must return responses in input order even when worker
    threads finish in a different order; callers zip() the result back up
    against their original candidate/pair list and rely on that.
    """
    candidates = [Candidate(id=str(i), text=f"candidate-{i}") for i in range(5)]
    texts_to_id = {c.text: c.id for c in candidates}
    # Candidate 0 sleeps longest, candidate 4 shortest: if _call_many
    # returned responses in completion order rather than input order, this
    # would come back scrambled.
    delays = {"0": 0.05, "1": 0.04, "2": 0.03, "3": 0.02, "4": 0.01}

    def responder(messages):
        text = next(
            t for t in texts_to_id if f"<candidate>{t}</candidate>" in messages[-1]["content"]
        )
        cid = texts_to_id[text]
        time.sleep(delays[cid])
        return cid

    fake_llm.responses = responder

    ranker = PointwiseRanker(LLMConfig(model="gpt-4o-mini"), max_concurrency=5)
    batches = [ranker._build_messages("q", c) for c in candidates]
    responses = ranker._call_many(batches)

    assert [r.text for r in responses] == ["0", "1", "2", "3", "4"]


def test_call_many_sequential_fallback_for_max_concurrency_one(fake_llm):
    candidates = [Candidate(id=str(i), text=f"candidate-{i}") for i in range(3)]
    fake_llm.responses = ["a", "b", "c"]

    ranker = PointwiseRanker(LLMConfig(model="gpt-4o-mini"), max_concurrency=1)
    batches = [ranker._build_messages("q", c) for c in candidates]
    responses = ranker._call_many(batches)

    assert [r.text for r in responses] == ["a", "b", "c"]
    assert ranker.total_calls == 3
