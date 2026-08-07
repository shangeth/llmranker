from __future__ import annotations

import json
from typing import Any


def json_schema_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Build a LiteLLM/OpenAI-style `response_format` value for a given
    JSON schema. LiteLLM normalizes this across the providers that support
    structured output (see https://docs.litellm.ai/docs/completion/json_mode).
    """
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "schema": schema, "strict": True},
    }


def _load(text: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


# --- pointwise -------------------------------------------------------------


def pointwise_schema() -> dict[str, Any]:
    return json_schema_format(
        "pointwise_score",
        {
            "type": "object",
            "properties": {"score": {"type": "number"}},
            "required": ["score"],
            "additionalProperties": False,
        },
    )


def parse_pointwise_json(text: str) -> float | None:
    obj = _load(text)
    if obj is None or not isinstance(obj.get("score"), (int, float)):
        return None
    return float(obj["score"])


def pointwise_multi_criteria_schema(names: list[str]) -> dict[str, Any]:
    return json_schema_format(
        "pointwise_multi_criteria_score",
        {
            "type": "object",
            "properties": {name: {"type": "number"} for name in names},
            "required": list(names),
            "additionalProperties": False,
        },
    )


def parse_pointwise_multi_criteria_json(text: str, names: list[str]) -> dict[str, float] | None:
    obj = _load(text)
    if obj is None:
        return None
    scores = {}
    for name in names:
        value = obj.get(name)
        if not isinstance(value, (int, float)):
            return None
        scores[name] = float(value)
    return scores


def pointwise_batch_schema(labels: list[str]) -> dict[str, Any]:
    return json_schema_format(
        "pointwise_batch_scores",
        {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string", "enum": labels},
                            "score": {"type": "number"},
                        },
                        "required": ["label", "score"],
                        "additionalProperties": False,
                    },
                    "minItems": len(labels),
                    "maxItems": len(labels),
                }
            },
            "required": ["scores"],
            "additionalProperties": False,
        },
    )


def parse_pointwise_batch_json(text: str, labels: list[str]) -> dict[str, float] | None:
    obj = _load(text)
    scores = obj.get("scores") if obj else None
    if not isinstance(scores, list):
        return None
    out: dict[str, float] = {}
    for entry in scores:
        if not isinstance(entry, dict):
            continue
        label, value = entry.get("label"), entry.get("score")
        if label in labels and isinstance(value, (int, float)):
            out[label] = float(value)  # duplicate label: last-write-wins
    return out or None


def criteria_extraction_schema() -> dict[str, Any]:
    return json_schema_format(
        "criteria_extraction",
        {
            "type": "object",
            "properties": {"criteria": {"type": "array", "items": {"type": "string"}}},
            "required": ["criteria"],
            "additionalProperties": False,
        },
    )


def parse_criteria_extraction_json(text: str) -> list[str] | None:
    obj = _load(text)
    criteria = obj.get("criteria") if obj else None
    if not isinstance(criteria, list) or not all(isinstance(c, str) for c in criteria):
        return None
    return criteria


# --- pairwise ----------------------------------------------------------------


def pairwise_schema() -> dict[str, Any]:
    return json_schema_format(
        "pairwise_choice",
        {
            "type": "object",
            "properties": {"choice": {"type": "string", "enum": ["A", "B"]}},
            "required": ["choice"],
            "additionalProperties": False,
        },
    )


def parse_pairwise_json(text: str) -> str | None:
    obj = _load(text)
    choice = obj.get("choice") if obj else None
    return choice if choice in ("A", "B") else None


# --- setwise -------------------------------------------------------------------


def setwise_schema(labels: list[str]) -> dict[str, Any]:
    return json_schema_format(
        "setwise_choice",
        {
            "type": "object",
            "properties": {"choice": {"type": "string", "enum": labels}},
            "required": ["choice"],
            "additionalProperties": False,
        },
    )


def parse_setwise_json(text: str, labels: list[str]) -> str | None:
    obj = _load(text)
    choice = obj.get("choice") if obj else None
    return choice if choice in labels else None


# --- listwise --------------------------------------------------------------------


def listwise_schema() -> dict[str, Any]:
    return json_schema_format(
        "listwise_ranking",
        {
            "type": "object",
            "properties": {"ranking": {"type": "array", "items": {"type": "integer"}}},
            "required": ["ranking"],
            "additionalProperties": False,
        },
    )


def parse_listwise_json(text: str) -> list[int] | None:
    obj = _load(text)
    ranking = obj.get("ranking") if obj else None
    if not isinstance(ranking, list) or not all(isinstance(i, int) for i in ranking):
        return None
    return ranking


# --- tourrank ----------------------------------------------------------------------


def tourrank_schema(labels: list[str], advance_count: int) -> dict[str, Any]:
    return json_schema_format(
        "tourrank_selection",
        {
            "type": "object",
            "properties": {
                "selected": {
                    "type": "array",
                    "items": {"type": "string", "enum": labels},
                    "minItems": advance_count,
                    "maxItems": advance_count,
                }
            },
            "required": ["selected"],
            "additionalProperties": False,
        },
    )


def parse_tourrank_json(text: str, labels: list[str]) -> list[str] | None:
    obj = _load(text)
    selected = obj.get("selected") if obj else None
    if not isinstance(selected, list) or not all(s in labels for s in selected):
        return None
    return selected
