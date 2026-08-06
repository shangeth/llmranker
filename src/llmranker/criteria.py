from __future__ import annotations

import logging
import re

from .prompts import extract_final_answer

logger = logging.getLogger("llmranker")

_PRIORITY_TIERS = ("high", "medium", "low")
_TIER_RANK = {"high": 2, "medium": 1, "low": 0}

_SPLIT_RE = re.compile(r"[,;\n]")


def resolve_weights(
    criteria: dict[str, float] | dict[str, str], score_range: float
) -> dict[str, float]:
    """Turn a user-supplied `criteria` dict into normalized combination
    weights (summing to 1).

    Two input shapes, dispatched by value type:
      - all numeric: treated as relative weights, normalized by their sum.
      - all in {"high", "medium", "low"}: converted to place-value weights
        so a higher tier mathematically dominates any possible combination
        of lower tiers, then normalized. `base = int(score_range) *
        len(criteria) + 1` is deliberately generous: it guarantees
        domination holds no matter how many criteria end up sharing a
        lower tier, not just in the common case of one-per-tier.

        Worked example: 3 criteria, score_range=10, one each of
        high/medium/low. base = 10*3+1 = 31. Raw weights: high=31**2=961,
        medium=31**1=31, low=31**0=1 -> normalized ~= 0.968, 0.0312,
        0.0010. The maximum possible combined swing from medium+low
        (0.0312*10 + 0.0010*10 ~= 0.322) is smaller than even a 1-point
        difference on the high-tier criterion (0.968*1 = 0.968), so high
        always wins regardless of how medium/low score.

    Mixing the two shapes, non-positive numeric weights, unknown priority
    strings, and an empty dict all raise `ValueError`.
    """
    if not criteria:
        raise ValueError("criteria must not be empty")

    values = list(criteria.values())
    is_numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)
    is_priority = all(v in _PRIORITY_TIERS for v in values)

    if is_numeric:
        if any(v <= 0 for v in values):
            raise ValueError("criteria weights must all be > 0")
        total = sum(values)
        return {k: v / total for k, v in criteria.items()}

    if is_priority:
        base = int(score_range) * len(criteria) + 1
        raw = {k: base ** _TIER_RANK[v] for k, v in criteria.items()}
        total = sum(raw.values())
        return {k: w / total for k, w in raw.items()}

    raise ValueError(
        "criteria values must be all numeric weights or all priority labels "
        f"{_PRIORITY_TIERS}, not a mix of the two"
    )


def parse_criteria_text(text: str, names: list[str], min_score: float) -> dict[str, float]:
    """Free-text fallback: parse 'name=score' / 'name: score' pairs out of
    `text` for each of `names`. Any name that doesn't match defaults to
    `min_score` with a warning, the same graceful-degradation convention
    `PointwiseRanker._parse_score` already uses for the single-score case,
    applied per-field instead of to one scalar.
    """
    text = extract_final_answer(text)
    scores: dict[str, float] = {}
    for name in names:
        match = re.search(rf"\b{re.escape(name)}\b\s*[:=]\s*(-?\d+(?:\.\d+)?)", text)
        if match is None:
            logger.warning("Could not parse a score for criterion %r from output: %r", name, text)
            scores[name] = min_score
        else:
            scores[name] = float(match.group(1))
    return scores


def parse_extracted_criteria_text(text: str) -> list[str]:
    """Free-text fallback for auto-mode extraction: split on commas,
    semicolons, or newlines, strip whitespace, drop empties."""
    text = extract_final_answer(text)
    return [part.strip() for part in _SPLIT_RE.split(text) if part.strip()]
