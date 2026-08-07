# Changelog

## Unreleased

### Added

- `llmranker.integrations.langchain.LLMRankerCompressor`: wraps any
  `llmranker` ranker as a LangChain `BaseDocumentCompressor`, so it plugs
  into `ContextualCompressionRetriever` like Cohere's or RankGPT's
  rerankers do. Optional dependency: `pip install "llmranker[langchain]"`.
- `llmranker.integrations.llama_index.LLMRankerPostprocessor`: the same
  idea for LlamaIndex's `BaseNodePostprocessor`, usable in a query engine's
  `node_postprocessors`. Optional dependency:
  `pip install "llmranker[llama-index]"`.
- Neither adds a required dependency or is imported by `llmranker/__init__.py`
  — `import llmranker` is unaffected whether or not either extra is
  installed.

## 0.3.0

### Breaking

- **Metric keys renamed** in `RankingMetrics.get_metrics()` (and therefore
  in `compare_rankers`' DataFrame): `mrr` -> `reciprocal_rank`, `mae` ->
  `rank_mae`. Neither was what its name implied — `mrr` is the reciprocal
  rank of one item for one query, not a mean over queries against a
  relevance set, and `rank_mae` is mean absolute *rank displacement*, not
  error on a score. The methods were renamed to match.
- **`TourRankRanker(k=...)` removed.** It truncated the output, while `k`
  on `PairwiseRanker`/`SetwiseRanker` caps sorting effort and never drops
  candidates — the same name meaning two different things. `rank()` now
  returns every candidate on every ranker; slice the result for a top-k.
- **`TourRankRanker(advance_per_group=..., num_stages=...)` replaced by
  `schedule=[...]`**, and `TourRankRanker.select_group()` now takes an
  explicit `advance` argument, since the advance count varies per stage.
  See the paper-fidelity fixes under Fixed below.
- **pandas is no longer a required dependency.** It's only used by
  `compare_rankers`, which now raises a clear `ImportError` pointing at
  `pip install "llmranker[benchmark]"`.
- **Minimum Python is now 3.10** (3.9 is end-of-life); 3.13 added to the
  supported set.
- **Minimum litellm raised to 1.45.0**, the first release containing
  `litellm.rerank`. The previously declared floor of 1.40.0 was never
  installable with this package — it predates both `rerank` and
  `litellm.exceptions.InternalServerError`, so importing `llmranker` at
  all raised `AttributeError`. Caught by the new minimum-dependency CI
  job on its first run.

### Added

- `score_kind` on every ranker (`relevance`, `rank_position`,
  `tournament_points`, `provider_relevance`), and on the `Ranker`
  protocol, so callers can tell what `Candidate.score` actually means
  instead of inferring it from which class they used.
- `relevance=` on `RankingMetrics.get_metrics()` and `compare_rankers()`,
  for graded judgments (TREC-style grades, Amazon ESCI labels) rather
  than a preference order. It existed on `ndcg()` and the docstring told
  people to use it, but no aggregate entry point accepted or forwarded it.
- A "Security" section in the README documenting that candidate text
  reaches the model as instructions and that nothing here sanitizes it.
- CI now checks formatting, runs mypy against the shipped annotations,
  enforces a 90% coverage floor, tests against the *minimum* declared
  dependency versions, byte-compiles `examples/` and `scripts/` so a public
  API change can't land without updating them, and builds/installs the
  wheel to catch packaging regressions.
- `scripts/validate_live.py`: runs every strategy against a real provider on
  a ranking task with an obvious right answer, reporting ordering, call
  counts and any unparseable response. The test suite is fully offline by
  design, so nothing in it demonstrates the package works against an actual
  model; this is that check. Defaults to an OpenRouter free model, takes any
  LiteLLM model string via `--model`, and has a request budget guard. The
  `rerank_api` phase needs a rerank endpoint rather than a chat model and is
  skipped unless `COHERE_API_KEY` is set (`--rerank-model` to change it).
- README documents `CascadeRanker` as the one deliberate exception to
  "`rank()` returns every candidate you passed in" -- it returns
  `narrow_to`, since discarding what the cheap stage rejected is the point
  of it. The blanket claim added earlier in this release was wrong.
- README notes two behaviors confirmed against a live model: LiteLLM's
  stderr chatter and the one-liner to silence it, and that small models
  saturate `PointwiseRanker`'s scale (a live run scored five candidates
  `[10, 9, 0, 0, 0]`, and tied scores fall back to input order).
- `RerankAPIRanker`: ranks with a dedicated rerank model (Cohere, Jina,
  Bedrock, Azure AI, Infinity) via LiteLLM's rerank endpoint — one request
  scores the whole candidate list, instead of prompting a chat model and
  parsing its output. No new dependency; LiteLLM already ships it.
  Designed as the cheap `narrow` stage of a `CascadeRanker`. It bills per
  search unit rather than per token, so it reports `total_search_units`,
  leaves the token counters at 0, and reports its cost as unknown rather
  than `$0.00`.
- `llmranker.llm.call_rerank`, plus the `RerankResult` / `RerankResponse`
  dataclasses it returns: the rerank-endpoint counterpart to `call_llm`,
  sharing its retry/backoff behavior.
- `py.typed` marker (PEP 561), so the package's existing annotations are
  visible to type checkers in downstream projects.

### Changed

- **`max_concurrency` documentation was wrong.** The README and the
  pairwise/setwise/listwise docstrings said it "genuinely does nothing"
  for the sequential strategies. It does: with `num_samples > 1` the
  repeated judgments of a single comparison are dispatched together, on
  every strategy. Anyone sizing it against a rate limit was misinformed.
- `estimate_cost`'s docstring claimed it returns `None` for a local Ollama
  model. It returns `0.0` — LiteLLM resolves those and prices them at
  zero. `None` means *unknown*, `0.0` means *free*.
- `call_llm`'s docstring said it retries "up to `max_retries` times";
  `max_retries` is the total *attempt* count (3 = one call plus two
  retries).
- scikit-learn dropped as a dependency (~41 MB, pulled in for a single
  `mean_absolute_error` call now done with numpy).
- `compare_rankers` now defers to a ranker's own `estimate_cost_usd()`
  when it defines one, falling back to the token-based estimate
  otherwise. This keeps per-token pricing from being applied to rankers
  that aren't billed that way.
- `CascadeRanker`'s `narrow`/`refine` params are annotated as the
  structural `Ranker` protocol rather than `BaseRanker`, which is what
  they always accepted in practice — needed so a `RerankAPIRanker` can be
  a stage.
- `ListwiseRanker` now warns when `num_samples > 1` is combined with
  `temperature=0.0`, matching `PointwiseRanker`. Unlike pairwise/setwise,
  listwise sends an identical prompt for every sample, so at temperature 0
  the repeats were provably wasted spend with no signal to the user.

### Fixed

An audit of each strategy against its source paper and reference
implementation turned up the first six divergences below.

- **TourRank's ensemble did nothing.** Group membership was a pure function
  of each candidate's input position, with only a within-group shuffle, so
  every tournament played out identically and `num_tournaments` just
  multiplied every score by a constant. Each tournament now draws its own
  groups from a canonically-ordered, seeded shuffle — which also makes the
  points a function of the candidate *set* and the seed rather than of the
  order they were passed in, the property the method is chosen for.
- **TourRank's defaults defeated the method.** `num_stages=2` with a fixed
  advance ratio produced 3 distinct score values across 100 candidates (26
  tied at the top), leaving most of the ranking to the tie-break. Replaced
  `advance_per_group`/`num_stages` with `schedule`, a list of per-stage
  survivor counts defaulting to the paper's 100→50→20→10→5→2 shape scaled
  to the candidate count. `group_size` now defaults to 20 and
  `num_tournaments` to 10, both the paper's values.
- **`PairwiseRanker(strategy="allpairs")` was not PRP-Allpair.** It asked
  each pair once in a fixed order — the position bias PRP exists to cancel.
  It now asks both orders and scores by the share of calls favouring each
  side, which reproduces the paper's `1·wins + 0.5·ties` exactly at
  `num_samples=1`. This doubles the call count for that strategy.
- **Setwise bubblesort used the wrong window.** It compared `num_child`
  candidates where the paper's reference implementation uses
  `num_child + 1` (a slot for the running winner plus `num_child`
  contenders, the same group heapsort builds). It also emitted
  single-candidate comparisons at the tail of each pass — an LLM call that
  cannot return anything.
- **`ListwiseRanker` defaults were not RankGPT's.** `window_size` was 4 with
  `step_size` 2; the paper tuned 20/10 on TREC-DL19, so the old defaults
  cost ~5× the calls in a configuration the paper never evaluated.
  `step_size` now defaults to `window_size // 2` rather than a fixed
  number, preserving the paper's relationship when only the window is set.
- **`CascadeRanker` mis-cited FrugalGPT.** FrugalGPT's LLM cascade routes
  one query through progressively stronger models behind a confidence gate,
  so the expensive model is often never called. This is a multi-stage
  retrieval cascade where both stages always run. Citation corrected; a
  confidence gate is tracked in ROADMAP.md.
- **NDCG could exceed 1.0.** The ideal DCG was computed from
  `true_ranking`'s order, which is only ideal when that order happens to
  be sorted by the caller's `relevance` grades. It's now computed from the
  grades themselves. (Measured 1.27 before the fix on a realistic input.)
- Three tests paired fake LLM responses to candidates positionally while
  the ranker under test dispatched them concurrently, so the pairing
  depended on thread scheduling. They passed on Python 3.10-3.12 and
  failed on 3.13, which schedules differently. `tests/conftest.py` gains
  `by_text()`, which keys responses on prompt content instead. The
  package was never affected -- `_call_many` returns responses in input
  order, which `test_concurrency.py` already covers.
- `spearman` / `kendall_tau` returned a bare `nan` plus an unexplained
  scipy `ConstantInputWarning` for fewer than two judged items. They still
  return `nan` — the statistics are genuinely undefined there — but the
  warning is suppressed and the behavior is documented.
- Output candidates no longer share the caller's `metadata` dict, so
  mutating a result no longer mutates the input.
- `llmranker` no longer sets `litellm.suppress_debug_info` process-wide at
  import time. Configuring a shared third-party module on behalf of the
  whole host application isn't a library's call to make.
- **Candidates with duplicate `id` values were silently dropped.**
  `PairwiseRanker` and `SetwiseRanker` excluded the unranked remainder by
  `id` field, so a candidate sharing an id with a ranked one disappeared
  from the result — `rank()` returned fewer candidates than it was given.
  Membership is now tracked by object identity. `PairwiseRanker`'s
  win-counting and both rankers' `num_samples` vote-counting had the same
  id-collision flaw and are identity/index-based now too.
- **`TourRankRanker` could advance the same candidate twice** with
  `structured_output=True`: a response like `{"selected": ["A", "A"]}`
  satisfies the schema's item count, so one candidate scored more points
  than there were stages while the survivor pool shrank. Selections are
  de-duplicated before the count check.
- **`PointwiseRanker` mis-parsed scores from chatty responses.** It took
  the first number in the text, so "Item 3 has a score of 9" scored 3.
  Parsing now prefers a bare number, then an explicitly labelled score,
  and strips "/10" and "out of 10" denominators before falling back to a
  scan — which now also warns when the output is genuinely ambiguous
  instead of silently returning a wrong score.
- **Tie-breaks reinstated the position bias `num_samples` exists to
  cancel.** A split pairwise vote resolved to whichever candidate was
  passed first, and a tied setwise vote to whichever came first in the
  input, so the same candidate set could rank differently purely by input
  order. Both now break ties with the ranker's seeded RNG.
- `_call_many` accumulated usage stats only after a whole concurrent batch
  returned, so one failing call discarded the accounting for every call
  that had succeeded. Stats are now recorded per call as it completes.

## 0.2.0

### Added

- `reasoning`, `num_samples`, and `structured_output` params on every
  ranker, controlling how hard it works to get a reliable judgment:
  chain-of-thought prompting, repeated-sampling self-consistency, and
  LiteLLM JSON-schema output (with a regex fallback if a model still
  returns malformed JSON) respectively.
- `CascadeRanker`: composes a cheap ranker that narrows a candidate list
  with an expensive one that re-ranks just the survivors.
- `PointwiseRanker(criteria=...)`: multi-criteria scoring. Named
  sub-criteria combined either by weighted sum, by priority tier
  (`"high"`/`"medium"`/`"low"`, where a higher tier mathematically
  dominates any combination of lower tiers), or `"auto"` (the LLM extracts
  the criteria from the query itself). `rank()`'s output candidates carry
  the per-criterion breakdown in `Candidate.metadata["criteria_scores"]`.
- `llmranker.types.Ranker`: a structural protocol so composite rankers
  like `CascadeRanker` work with `compare_rankers` without subclassing
  `BaseRanker`.

### Changed

- **Breaking**: `PairwiseRanker`/`SetwiseRanker`'s `method=` constructor
  param renamed to `strategy=`.
- **Breaking**: `PairwiseRanker`'s `debias_position` flag removed,
  superseded by `num_samples` (randomized-position majority vote, which
  generalizes the old bidirectional-check behavior to any sample count
  instead of a fixed 2).

## 0.1.0

Initial release: `PointwiseRanker`, `PairwiseRanker`
(heapsort/bubblesort/allpairs), `SetwiseRanker`
(heapsort/bubblesort/insertion), `ListwiseRanker`, and `TourRankRanker`,
plus an optional `reasoning` mode, built on top of LiteLLM.
