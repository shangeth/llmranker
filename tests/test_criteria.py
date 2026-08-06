import pytest

from llmranker.criteria import parse_criteria_text, parse_extracted_criteria_text, resolve_weights


def test_numeric_weights_normalize_to_sum_one():
    weights = resolve_weights({"price_fit": 0.5, "location_fit": 0.3, "family_friendly": 0.2}, 10)
    assert weights == {"price_fit": 0.5, "location_fit": 0.3, "family_friendly": 0.2}
    assert sum(weights.values()) == pytest.approx(1.0)


def test_numeric_weights_need_not_sum_to_one():
    weights = resolve_weights({"a": 2, "b": 2}, 10)
    assert weights == {"a": 0.5, "b": 0.5}


def test_priority_tiers_place_value_domination():
    # Worked example from the design: 3 criteria, score_range=10, one each
    # of high/medium/low. base = 10*3+1 = 31.
    weights = resolve_weights(
        {"family_friendly": "high", "price_fit": "medium", "location_fit": "low"}, 10
    )
    base = 31
    total = base**2 + base + 1
    assert weights["family_friendly"] == pytest.approx(base**2 / total)
    assert weights["price_fit"] == pytest.approx(base / total)
    assert weights["location_fit"] == pytest.approx(1 / total)

    # A single 1-point difference on the high-tier criterion outweighs the
    # maximum possible combined swing of every lower-tier criterion.
    high_swing = 1 * weights["family_friendly"]
    max_lower_swing = 10 * (weights["price_fit"] + weights["location_fit"])
    assert high_swing > max_lower_swing


def test_priority_domination_holds_with_multiple_criteria_per_tier():
    # Two mediums and two lows sharing a tier shouldn't be able to gang up
    # and outweigh a single high-tier criterion.
    weights = resolve_weights(
        {
            "a": "high",
            "b": "medium",
            "c": "medium",
            "d": "low",
            "e": "low",
        },
        10,
    )
    high_swing = 1 * weights["a"]
    max_lower_swing = 10 * (weights["b"] + weights["c"] + weights["d"] + weights["e"])
    assert high_swing > max_lower_swing


def test_mixing_weight_types_raises():
    with pytest.raises(ValueError):
        resolve_weights({"a": 1, "b": "high"}, 10)


def test_non_positive_weight_raises():
    with pytest.raises(ValueError):
        resolve_weights({"a": 0, "b": 1}, 10)
    with pytest.raises(ValueError):
        resolve_weights({"a": -1, "b": 1}, 10)


def test_empty_criteria_raises():
    with pytest.raises(ValueError):
        resolve_weights({}, 10)


def test_unknown_priority_string_raises():
    with pytest.raises(ValueError):
        resolve_weights({"a": "urgent"}, 10)


def test_parse_criteria_text_parses_name_value_pairs():
    scores = parse_criteria_text("price_fit=7, location_fit=9", ["price_fit", "location_fit"], 0)
    assert scores == {"price_fit": 7.0, "location_fit": 9.0}


def test_parse_criteria_text_defaults_missing_criterion_to_min_score():
    scores = parse_criteria_text("price_fit=7", ["price_fit", "location_fit"], 2)
    assert scores == {"price_fit": 7.0, "location_fit": 2.0}


def test_parse_extracted_criteria_text_splits_and_strips():
    names = parse_extracted_criteria_text("budget,  location ,\nfamily friendly")
    assert names == ["budget", "location", "family friendly"]


def test_parse_extracted_criteria_text_drops_empties():
    names = parse_extracted_criteria_text("budget,, location,")
    assert names == ["budget", "location"]
