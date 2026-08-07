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


@dataclass
class RerankResult:
    """One scored document from a rerank-endpoint response.

    `index` refers back into the `documents` list that was sent, which is
    how the caller maps a score onto the candidate it came from.
    """

    index: int
    relevance_score: float


@dataclass
class RerankResponse:
    """A rerank endpoint's reply, normalized the way `LLMResponse`
    normalizes a chat completion.

    `search_units` is what rerank providers actually bill on (roughly: one
    per query per batch of documents), reported when the provider returns
    it and `None` otherwise. There are no prompt/completion token counts
    here because there is no prompt and no completion.
    """

    results: list[RerankResult]
    search_units: int | None = None


def call_llm(
    messages: list[dict[str, str]],
    config: LLMConfig,
    response_format: dict[str, Any] | None = None,
) -> LLMResponse:
    """Call the configured LLM via LiteLLM, retrying on transient errors.

    Retries with exponential backoff up to `config.max_retries` times on
    rate limits / connection / timeout / server errors, then re-raises.
    Non-retryable errors (auth, bad request, ...) propagate immediately.

    `response_format`, when given, is LiteLLM's normalized structured-output
    schema (see `llmranker.structured.json_schema_format`); passed straight
    through to `litellm.completion`.
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
        kwargs = dict(config.extra_kwargs)
        if response_format is not None:
            kwargs["response_format"] = response_format
        response = litellm.completion(
            model=config.model,
            messages=messages,
            temperature=config.temperature,
            timeout=config.timeout,
            api_key=config.api_key,
            api_base=config.api_base,
            **kwargs,
        )
        usage = response.get("usage") or {}
        return LLMResponse(
            text=response["choices"][0]["message"]["content"] or "",
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        )

    return _call()


def call_rerank(
    query: str,
    documents: list[str],
    config: LLMConfig,
    top_n: int | None = None,
) -> RerankResponse:
    """Call a dedicated rerank endpoint via LiteLLM, retrying on transient errors.

    This is the cross-encoder/rerank-model counterpart to `call_llm`: a
    purpose-trained relevance model scores every document against `query`
    in a single request, rather than a chat model being prompted to reason
    about them. LiteLLM normalizes these behind the Cohere rerank format,
    so `config.model` follows the same provider-prefixed convention as
    everywhere else ("cohere/rerank-v3.5", "jina_ai/jina-reranker-v2-...",
    "bedrock/...", "azure_ai/...", "infinity/...").

    Retry behavior, and which errors are considered retryable, are shared
    with `call_llm` — see `RETRYABLE_EXCEPTIONS`. `config.temperature` is
    ignored: there is no sampling involved in scoring.
    """

    @retry(
        reraise=True,
        stop=stop_after_attempt(max(config.max_retries, 1)),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        before_sleep=lambda retry_state: logger.warning(
            "Rerank call failed (attempt %d/%d): %s, retrying",
            retry_state.attempt_number,
            config.max_retries,
            retry_state.outcome.exception(),
        ),
    )
    def _call() -> RerankResponse:
        response = litellm.rerank(
            model=config.model,
            query=query,
            documents=documents,
            top_n=top_n,
            api_key=config.api_key,
            api_base=config.api_base,
            timeout=config.timeout,
            **config.extra_kwargs,
        )
        raw_results = _get(response, "results") or []
        results = [
            RerankResult(
                index=int(_get(r, "index")),
                relevance_score=float(_get(r, "relevance_score")),
            )
            for r in raw_results
        ]
        return RerankResponse(results=results, search_units=_search_units(response))

    return _call()


def _get(obj: Any, key: str) -> Any:
    """Read `key` off a LiteLLM response object that may be a pydantic model
    or a plain dict, depending on provider and LiteLLM version."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _search_units(response: Any) -> int | None:
    """Pull `meta.billed_units.search_units` out of a rerank response, if the
    provider reported it. Absent for several providers, so never assume it."""
    billed = _get(_get(response, "meta"), "billed_units")
    units = _get(billed, "search_units")
    return int(units) if units is not None else None


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
