from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
