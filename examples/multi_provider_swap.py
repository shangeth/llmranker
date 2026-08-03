"""Same query, same candidates, same ranker code; only the model string
changes. This is the entire point of building llmranker on top of LiteLLM:
one interface, any provider.

Run whichever providers you have API keys for, e.g.:
    export OPENAI_API_KEY=...
    export GEMINI_API_KEY=...
    export ANTHROPIC_API_KEY=...
    python multi_provider_swap.py
"""

from llmranker import Candidate, LLMConfig, SetwiseRanker

# Each entry is a LiteLLM model string (see https://docs.litellm.ai/docs/providers)
MODELS_TO_TRY = [
    "gpt-4o-mini",  # OpenAI
    "gemini/gemini-1.5-flash",  # Google Gemini
    "claude-3-5-sonnet-20241022",  # Anthropic
    # "ollama/llama3",            # local model via Ollama, no API key needed
]

CANDIDATES = [
    Candidate(
        id="a", text="A budget-friendly hostel in the city center, walking distance to museums."
    ),
    Candidate(id="b", text="A five-star beachfront resort with an adults-only pool and spa."),
    Candidate(
        id="c", text="A family-run guesthouse near the old town, kid-friendly, no beach access."
    ),
]

QUERY = "affordable place to stay near historical sites, good for families"

if __name__ == "__main__":
    for model in MODELS_TO_TRY:
        ranker = SetwiseRanker(LLMConfig(model=model), num_child=3, item_label="hotel")
        try:
            result = ranker.rank(QUERY, CANDIDATES)
        except Exception as exc:  # missing API key for this provider, etc.
            print(f"{model}: skipped ({exc})")
            continue
        print(f"{model}: {[c.id for c in result]}")
