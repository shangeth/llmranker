# Changelog

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
