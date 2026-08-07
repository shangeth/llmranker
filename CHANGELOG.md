# Changelog

## Unreleased

### Added

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
