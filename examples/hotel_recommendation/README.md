# Hotel recommendation (flagship example)

This is the running example for the whole `llmranker` README, kept here as a
standalone, runnable script.

## Why hotel recommendation

It's a good stress test for LLM-based ranking specifically because:

- **Relevance is subjective and compositional.** "Family friendly, near
  historic sites, not on the beach" is a conjunction of soft preferences a
  keyword search can't express and an embedding search tends to blur
  together (a listing that's family-friendly *and* beachfront will often
  out-score one that's family-friendly and historic, because "beach" and
  "family" co-occur more often in training data than "historic" and
  "family" do).
- **No labeled training data.** Nobody has click/purchase logs for a brand
  new inventory, a new market, or a one-off internal tool. Zero-shot LLM
  ranking needs none.
- **Candidate sets are small.** A search page shows dozens of hotels, not
  millions: exactly the regime where `O(n)` to `O(n log n)` LLM calls per
  query is affordable.

## Run it

```bash
pip install llmranker
export OPENAI_API_KEY=...   # or GEMINI_API_KEY / ANTHROPIC_API_KEY / ...
python run.py
```

`run.py` does two things:

1. **Quickstart**: ranks 7 hotels against a natural-language query with
   `SetwiseRanker` and prints the order plus LLM call/token counts.
2. **Strategy comparison**: runs `PointwiseRanker`, `PairwiseRanker`,
   `SetwiseRanker`, `ListwiseRanker`, and `TourRankRanker` on the same query
   and reports ranking quality (NDCG, MRR, Spearman, Kendall's Tau against a
   hand-labeled ground truth), LLM calls, tokens, estimated cost, and
   latency side by side via `llmranker.compare_rankers`.

Swap `MODEL` at the top of `run.py` to any [LiteLLM model string](https://docs.litellm.ai/docs/providers)
to run the same comparison against Gemini, Claude, a local Ollama model, etc.
with no other code changes; see `examples/multi_provider_swap.py`.
