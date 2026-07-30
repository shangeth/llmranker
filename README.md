# llmranker

**LLM-based rerankers for any provider.** Pointwise, pairwise, listwise, and
setwise ranking strategies implemented on top of [LiteLLM](https://github.com/BerriAI/litellm),
so the same code runs against OpenAI, Gemini, Anthropic, Azure, Bedrock,
local Ollama models, or any of the 100+ providers LiteLLM supports -- just
change a model string.

[![PyPI](https://img.shields.io/pypi/v/llmranker.svg)](https://pypi.org/project/llmranker/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/shangeth/llmranker/actions/workflows/ci.yml/badge.svg)](https://github.com/shangeth/llmranker/actions/workflows/ci.yml)

```python
from llmranker import Candidate, LLMConfig, SetwiseRanker

ranker = SetwiseRanker(LLMConfig(model="gpt-4o-mini"), num_child=4, k=5)

candidates = [
    Candidate(id="1", text="A budget hostel in the city center."),
    Candidate(id="2", text="A five-star beachfront resort with a spa."),
    Candidate(id="3", text="A family-run guesthouse near the old town, kid-friendly."),
]

result = ranker.rank(query="affordable, family friendly, near historical sites", candidates=candidates)
print([c.id for c in result])  # ['3', '1', '2']
```

## What this is

Before you fine-tune a ranking model or build an embedding index, you can
often just *ask* an LLM which candidate is more relevant to a query. This
package implements the four ways of doing that described in
[**"A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking
with Large Language Models"**](https://arxiv.org/abs/2310.09497) (Zhuang et
al., 2023):

| Strategy | How it works | LLM calls | Notes |
|---|---|---|---|
| **Pointwise** | Score each candidate independently (0-10) | `O(n)` | Cheapest, but ignores relative preference between candidates |
| **Pairwise** | Repeatedly ask "A or B?", sort via heapsort/bubblesort/allpairs | `O(n log n)` to `O(n²)` | Simple, robust comparisons |
| **Setwise** | Ask "which of these `k` is best?", sort via `k`-ary heapsort/bubblesort | `O(n log n / log k)` | Fewer calls than pairwise for the same sort, longer prompts |
| **Listwise** | Ask the LLM to output a full ranking of a sliding window at once | `O(n / step)` | Fewest calls, but degrades as window size grows |

All four are zero-shot: no training data, no fine-tuning, no embeddings.
You give it a query and a list of candidates, it gives you a ranked list.

## Why LLM-based ranking

- **No training data.** New inventory, a new market, or a one-off internal
  tool rarely comes with click/purchase logs to train a ranker on.
- **Captures compositional, natural-language preference.** "Family
  friendly, near historic sites, not on the beach" is a conjunction of soft
  constraints that keyword search can't express and embedding search tends
  to blur together.
- **Cheap at the scale that matters for reranking.** You're not ranking
  your whole catalog with an LLM -- you're reranking the top-k (dozens, not
  millions) that a cheap first-pass retrieval already narrowed down.

See [`examples/hotel_recommendation/`](examples/hotel_recommendation/) for
the full worked example this README's numbers come from, and
[`examples/`](examples/) for RAG document reranking, product search, and
multi-provider comparisons.

## Install

```bash
pip install llmranker
```

Set whichever provider's API key you're using as an environment variable --
LiteLLM reads the standard ones automatically (`OPENAI_API_KEY`,
`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, ...). See [LiteLLM's provider
docs](https://docs.litellm.ai/docs/providers) for the full list, including
self-hosted/local options that need no key at all.

## Quickstart

```python
from llmranker import Candidate, LLMConfig, PairwiseRanker

ranker = PairwiseRanker(LLMConfig(model="gpt-4o-mini"), method="heapsort", k=10)

candidates = [Candidate(id=str(i), text=doc) for i, doc in enumerate(my_documents)]
result = ranker.rank(query="my search query", candidates=candidates)

for c in result:
    print(c.id, c.score)
```

## Swap providers in one line

Every ranker takes an `LLMConfig`, whose `model` field is a [LiteLLM model
string](https://docs.litellm.ai/docs/providers). Nothing else about your
code changes:

```python
LLMConfig(model="gpt-4o-mini")                     # OpenAI
LLMConfig(model="gemini/gemini-1.5-flash")         # Google Gemini
LLMConfig(model="claude-3-5-sonnet-20241022")      # Anthropic
LLMConfig(model="azure/my-deployment-name")        # Azure OpenAI
LLMConfig(model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0")  # AWS Bedrock
LLMConfig(model="ollama/llama3")                   # local, via Ollama
```

See [`examples/multi_provider_swap.py`](examples/multi_provider_swap.py).

## Choosing a strategy

Rough guidance, in order of what to reach for first:

- Start with **setwise** (`num_child=4-8`, `method="heapsort"`) -- the best
  cost/quality tradeoff for most use cases, and what this package is named
  after.
- If you want the simplest possible mental model (and don't mind more LLM
  calls), use **pairwise**.
- If latency matters more than call count and your candidate list is small
  (fits in one window), use **listwise**.
- Use **pointwise** when you need a standalone relevance score per
  candidate (e.g. for thresholding "is this even relevant at all") rather
  than just a ranking, or when `n` is large and you can't afford
  comparisons at all.

## Use case: hotel recommendation

The flagship example lives in
[`examples/hotel_recommendation/`](examples/hotel_recommendation/). It
reranks 7 hotels against natural-language guest preferences like *"family
friendly hotel with kids, close to historical places, not right on the
beach"* -- exactly the kind of compositional, subjective query that trips
up keyword and embedding search but an LLM reading full descriptions
handles naturally.

```bash
cd examples/hotel_recommendation
python run.py
```

It runs all four strategies against the same query and candidates and
prints a side-by-side comparison of ranking quality, LLM calls, tokens,
estimated cost, and latency using `llmranker.compare_rankers`.

## More use cases

- [`examples/rag_document_reranking.py`](examples/rag_document_reranking.py)
  -- rerank RAG retrieval results before they go into a prompt, so context
  budget goes to passages that actually answer the question instead of
  merely-related near-duplicates.
- [`examples/product_search_reranking.py`](examples/product_search_reranking.py)
  -- e-commerce search reranking against multi-constraint natural-language
  intent (price, fit, use case).
- Other good fits: job/candidate matching, content and media
  recommendation, support ticket triage, lead scoring -- anywhere you have
  a short list of candidates and a query or profile to rank them against.

## Customizing for your domain

Every ranker accepts an `item_label` (used in the default prompts -- "hotel",
"product", "document", ...) and an optional `system_prompt` override if you
want full control over the wording:

```python
ranker = SetwiseRanker(
    LLMConfig(model="gpt-4o-mini"),
    item_label="job candidate",
    system_prompt="You are a hiring assistant ranking candidates against a job description...",
)
```

## Evaluation & benchmarking

```python
from llmranker import RankingMetrics, compare_rankers

metrics = RankingMetrics()
metrics.get_metrics(true_ranking=["b", "a", "c"], predicted_ranking=["a", "b", "c"])
# {'ndcg': ..., 'mrr': ..., 'mae': ..., 'spearman': ..., 'kendall_tau': ...}

report = compare_rankers([ranker_a, ranker_b], query, candidates, true_ranking)
# pandas DataFrame: ranking quality + LLM calls/tokens/cost/latency, side by side
```

`true_ranking` is a ground-truth ordering of candidate ids (best to worst),
if you have one -- e.g. from human labels or a held-out click log.

## API reference

| Module | Contents |
|---|---|
| `llmranker.types` | `Candidate(id, text, score, metadata)` |
| `llmranker.llm` | `LLMConfig`, `call_llm`, `truncate_to_tokens`, `estimate_cost` |
| `llmranker.rankers` | `PointwiseRanker`, `PairwiseRanker`, `SetwiseRanker`, `ListwiseRanker` |
| `llmranker.metrics` | `RankingMetrics` (NDCG, MRR, MAE, Spearman, Kendall's Tau) |
| `llmranker.benchmark` | `compare_rankers` |

Every ranker implements `rank(query, candidates) -> list[Candidate]` and
tracks `total_calls` / `total_prompt_tokens` / `total_completion_tokens`
after each call.

## Contributing

Issues and PRs welcome. Run tests with:

```bash
pip install -e ".[dev]"
pytest
```

Tests run entirely offline against a fake LiteLLM backend -- no API key
needed to contribute.

## Citation

If you use this package, please cite the paper it implements:

```bibtex
@article{zhuang2023setwise,
  title={A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models},
  author={Zhuang, Shengyao and Zhuang, Honglei and Koopman, Bevan and Zuccon, Guido},
  journal={arXiv preprint arXiv:2310.09497},
  year={2023}
}
```

## License

MIT -- see [LICENSE](LICENSE).
