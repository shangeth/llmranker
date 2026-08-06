from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .llm import LLMConfig


@dataclass
class Candidate:
    """A single item to be ranked against a query.

    Domain-agnostic on purpose: `id` and `text` are all any ranker needs,
    `metadata` is there for callers who want to carry extra fields (price,
    url, category, ...) through the ranking call without the library caring
    about them.
    """

    id: str
    text: str
    score: float | None = None
    metadata: dict[str, Any] | None = None


@runtime_checkable
class Ranker(Protocol):
    """The structural contract `llmranker.benchmark.compare_rankers` (and
    anything else generic over rankers) relies on. Every `BaseRanker`
    subclass satisfies this already; `CascadeRanker` satisfies it without
    subclassing `BaseRanker`, since it wraps two rankers rather than owning
    a single `LLMConfig` itself.
    """

    name: str
    config: LLMConfig
    total_calls: int
    total_prompt_tokens: int
    total_completion_tokens: int

    def rank(self, query: str, candidates: list[Candidate]) -> list[Candidate]: ...
