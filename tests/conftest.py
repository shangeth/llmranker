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


@pytest.fixture
def fake_llm(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(litellm, "completion", fake)
    return fake
