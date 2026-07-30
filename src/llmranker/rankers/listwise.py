from __future__ import annotations

import logging
import re

from ..llm import LLMConfig, truncate_to_tokens
from ..prompts import extract_final_answer, listwise_post_prompt, listwise_prefix_messages
from ..types import Candidate
from .base import BaseRanker

logger = logging.getLogger("llmranker")

_DIGIT_RE = re.compile(r"\d+")


class ListwiseRanker(BaseRanker):
    """Reranks by asking the LLM to output a full permutation of a sliding
    window of candidates at once (RankGPT-style), sliding the window across
    the list and optionally repeating.

    One LLM call handles `window_size` candidates at a time -- far fewer
    calls than pairwise/setwise for long lists, at the cost of asking the
    model to reason about more candidates per turn (accuracy tends to
    degrade as `window_size` grows).

    `window_size`: how many candidates are shown to the LLM per call.
    `step_size`: how far the window slides toward the front each step
        (step_size < window_size means adjacent windows overlap, which is
        what lets a candidate's rank improve across multiple windows).
    `num_repeat`: how many full passes over the list to make; each pass
        starts from the current (already improved) order.
    `max_tokens_per_candidate`: if set, truncates each candidate's text to
        this many tokens before including it in the prompt (useful for long
        documents or small-context models). Off by default.

    Every window's call depends on the previous window's output, so this
    strategy is inherently sequential -- `max_concurrency` has **no
    effect** here. It's accepted for constructor-signature consistency with
    the other rankers only.
    """

    def __init__(
        self,
        config: LLMConfig,
        window_size: int = 4,
        step_size: int = 2,
        num_repeat: int = 1,
        max_tokens_per_candidate: int | None = None,
        item_label: str = "item",
        system_prompt: str | None = None,
        name: str | None = None,
        max_concurrency: int = 5,
        reasoning: bool = False,
    ):
        super().__init__(config, item_label, system_prompt, name, max_concurrency, reasoning)
        if step_size > window_size:
            raise ValueError("step_size must be <= window_size")
        if window_size < 2:
            raise ValueError("window_size must be >= 2")
        self.window_size = window_size
        self.step_size = step_size
        self.num_repeat = num_repeat
        self.max_tokens_per_candidate = max_tokens_per_candidate

    def _build_messages(self, query: str, window: list[Candidate]) -> list[dict]:
        messages = listwise_prefix_messages(query, len(window), self.item_label)
        if self.system_prompt_override:
            messages[0] = {"role": "system", "content": self.system_prompt_override}

        for rank, candidate in enumerate(window, start=1):
            text = candidate.text
            if self.max_tokens_per_candidate is not None:
                text = truncate_to_tokens(
                    text, self.config.model, self.max_tokens_per_candidate
                )
            messages.append({"role": "user", "content": f"[{rank}] {text}"})
            messages.append(
                {"role": "assistant", "content": f"Received {self.item_label} [{rank}]."}
            )
        messages.append(
            {
                "role": "user",
                "content": listwise_post_prompt(
                    query, len(window), self.item_label, self.reasoning
                ),
            }
        )
        return messages

    def _parse_permutation(self, text: str, n: int) -> list[int]:
        """Parse a ranking string like '[3] > [1] > [2]' into a 0-indexed
        order. Out-of-range or duplicate identifiers are dropped; any
        candidate the model didn't mention is appended in its original
        position so every candidate always ends up somewhere in the output.
        """
        text = extract_final_answer(text)
        found = [int(d) - 1 for d in _DIGIT_RE.findall(text)]
        seen = set()
        order: list[int] = []
        for idx in found:
            if 0 <= idx < n and idx not in seen:
                order.append(idx)
                seen.add(idx)
        for idx in range(n):
            if idx not in seen:
                order.append(idx)
        return order

    def compare(self, query: str, window: list[Candidate]) -> list[Candidate]:
        """Return `window` reordered most-to-least relevant to `query`."""
        response = self._call(self._build_messages(query, window))
        order = self._parse_permutation(response.text, len(window))
        return [window[i] for i in order]

    def rank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        self._reset_stats()
        arr = list(candidates)
        n = len(arr)

        for _ in range(self.num_repeat):
            end = n
            start = max(0, end - self.window_size)
            while True:
                window = arr[start:end]
                if len(window) >= 2:
                    arr[start:end] = self.compare(query, window)
                if start == 0:
                    break
                end = max(0, end - self.step_size)
                start = max(0, end - self.window_size)

        return [
            Candidate(id=c.id, text=c.text, score=float(n - i), metadata=c.metadata)
            for i, c in enumerate(arr)
        ]
