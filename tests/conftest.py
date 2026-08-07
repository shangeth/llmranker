from __future__ import annotations

import threading

import litellm
import pytest


class FakeResponse(dict):
    """Mimics the subset of litellm's ModelResponse that llmranker.llm.call_llm reads."""

    def __init__(self, content: str, prompt_tokens: int = 10, completion_tokens: int = 5):
        super().__init__(
            choices=[{"message": {"content": content}}],
            usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        )


class FakeLLM:
    """Monkeypatched stand-in for litellm.completion.

    Set `.responses` to a list (consumed in call order) or a callable
    `(messages, **kwargs) -> str`. Every call is recorded in `.calls`.

    **List mode is consumed in *execution* order, not input order.** For a
    ranker that dispatches concurrently (PointwiseRanker, PairwiseRanker's
    "allpairs", anything with num_samples > 1), which prompt receives which
    response then depends on thread scheduling, so a list is only safe when
    the assertions don't care about the pairing -- or with
    `max_concurrency=1`. Use `by_text()` below when they do. The package
    itself pairs correctly regardless; `_call_many` returns responses in
    input order (see test_concurrency.py).

    Thread-safe: rankers with max_concurrency > 1 call this from multiple
    ThreadPoolExecutor worker threads, so list-mode's index increment (and
    the call log append) are guarded by a lock. The callable mode is safe
    as long as the callable itself is a pure function of `messages`, which
    is how the ground-truth responders in these tests are written.
    """

    def __init__(self):
        self.responses = []
        self.calls = []
        self._index = 0
        self._lock = threading.Lock()

    def __call__(self, model=None, messages=None, **kwargs):
        with self._lock:
            self.calls.append({"model": model, "messages": messages, **kwargs})
            if callable(self.responses):
                content = self.responses(messages)
            else:
                content = self.responses[self._index]
                self._index += 1
        return FakeResponse(content)


def by_text(mapping):
    """Responder that selects a response by matching a substring of the prompt.

    Deterministic under concurrency, unlike a positional list: the pairing
    follows the prompt's content rather than the order threads happen to
    reach the fake.
    """

    def responder(messages):
        content = messages[-1]["content"]
        for needle, response in mapping.items():
            if needle in content:
                return response
        raise AssertionError(f"no fake response registered for prompt: {content[:160]!r}")

    return responder


@pytest.fixture
def fake_llm(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(litellm, "completion", fake)
    return fake


class FakeRerankResponse(dict):
    """Mimics the subset of litellm's RerankResponse that llmranker.llm.call_rerank
    reads. A dict rather than a pydantic model on purpose: `call_rerank` has to
    cope with both shapes across providers/LiteLLM versions, and the dict branch
    is the one that would otherwise go untested."""

    def __init__(self, results: list[dict], search_units: int | None = None):
        meta = {"billed_units": {"search_units": search_units}} if search_units is not None else {}
        super().__init__(id="fake-rerank", results=results, meta=meta)


class FakeRerank:
    """Monkeypatched stand-in for litellm.rerank.

    Set `.results` to a list of `{"index": int, "relevance_score": float}`
    dicts, or to a callable `(query, documents) -> list[dict]`. Every call is
    recorded in `.calls`. `.search_units` is reported back in the response
    meta when set.
    """

    def __init__(self):
        self.results = []
        self.calls = []
        self.search_units = None

    def __call__(self, model=None, query=None, documents=None, **kwargs):
        self.calls.append({"model": model, "query": query, "documents": documents, **kwargs})
        results = self.results(query, documents) if callable(self.results) else self.results
        return FakeRerankResponse(results, search_units=self.search_units)


@pytest.fixture
def fake_rerank(monkeypatch):
    fake = FakeRerank()
    monkeypatch.setattr(litellm, "rerank", fake)
    return fake
