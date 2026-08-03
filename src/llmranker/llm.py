from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import litellm
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger("llmranker")

litellm.suppress_debug_info = True

# Transient, worth retrying. Auth/bad-request errors are not in this list on
# purpose: retrying a request that will never succeed just burns quota.
RETRYABLE_EXCEPTIONS = (
    litellm.exceptions.RateLimitError,
    litellm.exceptions.APIConnectionError,
    litellm.exceptions.Timeout,
    litellm.exceptions.ServiceUnavailableError,
    litellm.exceptions.InternalServerError,
)


@dataclass
class LLMConfig:
    """Configuration for calling an LLM through LiteLLM.

    `model` follows LiteLLM's model-string convention, e.g. "gpt-4o-mini",
    "gemini/gemini-1.5-flash", "claude-3-5-sonnet-20241022", "ollama/llama3".
    See https://docs.litellm.ai/docs/providers for the full list.
    """

    model: str
    api_key: str | None = None
    api_base: str | None = None
    temperature: float = 0.0
    timeout: float = 30.0
    max_retries: int = 3
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int


def call_llm(messages: list[dict[str, str]], config: LLMConfig) -> LLMResponse:
    """Call the configured LLM via LiteLLM, retrying on transient errors.

    Retries with exponential backoff up to `config.max_retries` times on
    rate limits / connection / timeout / server errors, then re-raises.
    Non-retryable errors (auth, bad request, ...) propagate immediately.
    """

    @retry(
        reraise=True,
        stop=stop_after_attempt(max(config.max_retries, 1)),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        before_sleep=lambda retry_state: logger.warning(
            "LLM call failed (attempt %d/%d): %s, retrying",
            retry_state.attempt_number,
            config.max_retries,
            retry_state.outcome.exception(),
        ),
    )
    def _call() -> LLMResponse:
        response = litellm.completion(
            model=config.model,
            messages=messages,
            temperature=config.temperature,
            timeout=config.timeout,
            api_key=config.api_key,
            api_base=config.api_base,
            **config.extra_kwargs,
        )
        usage = response.get("usage") or {}
        return LLMResponse(
            text=response["choices"][0]["message"]["content"] or "",
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        )

    return _call()


def truncate_to_tokens(text: str, model: str, max_tokens: int) -> str:
    """Truncate `text` to at most `max_tokens` tokens for `model`.

    Uses LiteLLM's provider-aware token counter so this works the same way
    regardless of which provider `model` points at. Falls back to a rough
    word-based truncation if LiteLLM can't resolve a tokenizer for it.
    """
    try:
        token_count = litellm.token_counter(model=model, text=text)
        if token_count <= max_tokens:
            return text
        words = text.split()
        if not words:
            return text
        ratio = max_tokens / token_count
        approx_words = max(1, int(len(words) * ratio))
        truncated = " ".join(words[:approx_words])
        while litellm.token_counter(model=model, text=truncated) > max_tokens and approx_words > 1:
            approx_words -= 1
            truncated = " ".join(words[:approx_words])
        return truncated
    except Exception:
        logger.debug("Falling back to word-based truncation for model=%s", model)
        return " ".join(text.split()[:max_tokens])


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """Estimate USD cost for the given token usage.

    Returns None if LiteLLM has no pricing data for `model` (e.g. a local
    Ollama model) rather than guessing.
    """
    try:
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        return prompt_cost + completion_cost
    except Exception:
        logger.debug("No pricing data available for model=%s", model)
        return None
