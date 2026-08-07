"""Smoke-test every ranker against a real LLM provider.

The test suite runs entirely offline against a fake LiteLLM backend, which
is what keeps contribution frictionless -- but it means nothing in it proves
the package works against an actual model. This script closes that gap: it
runs each strategy end to end on a small hand-built ranking task where the
right answer is obvious, and reports whether the output is sane, how many
calls it took, and whether any response failed to parse.

It is deliberately not part of `pytest`: it needs a key, it costs requests,
and it is a judgement call rather than a pass/fail assertion.

Usage:

    python scripts/validate_live.py                       # default model
    python scripts/validate_live.py --model gpt-4o-mini
    python scripts/validate_live.py --budget 20 --only pointwise setwise

Any LiteLLM model string works. The default targets OpenRouter's free tier,
which allows **50 requests per day and 20 per minute** -- a full sweep costs
around 45, so plan before running. OpenRouter does not expose the free-model
request counter anywhere, so that daily cap has to be tracked by hand;
`--check-budget` reports account tier and dollar spend (always $0 on free
models) without spending a request, and every run prints what it spent.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.request
from pathlib import Path

# Default targets OpenRouter's free tier. This particular model is one of
# the few free ones advertising `response_format` support, so structured
# output is exercisable on it; see https://openrouter.ai/api/v1/models.
DEFAULT_MODEL = "openrouter/google/gemma-4-26b-a4b-it:free"

# RerankAPIRanker talks to a rerank endpoint, not a chat model, so it needs
# its own provider. Skipped unless COHERE_API_KEY is set. Cohere's trial key
# allows 1,000 calls/month and 10 rerank requests/minute -- and one call
# scores the whole candidate list, so this phase is cheap.
DEFAULT_RERANK_MODEL = "cohere/rerank-v3.5"

QUERY = "quiet family hotel walking distance to museums, not on the beach"
HOTELS = [
    ("h1", "Beachfront party resort with a nightclub and swim-up bar. Adults only."),
    (
        "h2",
        (
            "Family guesthouse in the old town, 5 min walk to the history museum. "
            "Quiet street, kids stay free."
        ),
    ),
    ("h3", "Business hotel by the airport. Soundproofed rooms, no family facilities."),
    ("h4", "Central apartment next to the art museum quarter, family rooms, quiet courtyard."),
    ("h5", "Budget hostel with shared dorms, 40 min from the city centre by bus."),
]
# h2 and h4 are the only ones that satisfy every constraint; h1 violates all
# of them. Anything that puts {h2, h4} on top and h1 last is behaving.
BEST = {"h2", "h4"}
WORST = "h1"


def load_dotenv() -> None:
    """Read the repo's .env, if present, without overriding a real environ."""
    for candidate in (Path.cwd(), *Path.cwd().parents[:3]):
        path = candidate / ".env"
        if path.is_file():
            for line in path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
            return


def openrouter_budget() -> str:
    """Remaining free-tier usage, queried without spending a request."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return "OPENROUTER_API_KEY not set"
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.load(response)["data"]
    except Exception as exc:  # reporting only; any failure here is informational
        return f"could not read key info: {exc}"
    tier = "free" if data.get("is_free_tier") else "paid"
    # `usage_daily` is dollars spent, not a request count -- free models cost
    # $0, so it stays at 0 however many you make. OpenRouter does not expose
    # the free-model request counter, so the daily cap has to be tracked by
    # hand; this run reports what it spent at the end.
    return (
        f"tier={tier} spend_today=${data.get('usage_daily', '?')} "
        f"(free-model cap is 50 requests/day and 20/min, not reported here)"
    )


def build_rankers(model: str, rerank_model: str | None):
    from llmranker import (
        ListwiseRanker,
        LLMConfig,
        PairwiseRanker,
        PointwiseRanker,
        RerankAPIRanker,
        SetwiseRanker,
        TourRankRanker,
    )

    def config(**kwargs):
        return LLMConfig(model=model, timeout=120, **kwargs)

    # (name, ranker, candidate count, rough call cost) -- the O(n^2) and
    # multi-pass strategies get a shorter list so a full sweep still fits in
    # a free-tier day.
    phases = [
        ("pointwise", PointwiseRanker(config(), item_label="hotel", max_concurrency=3), 5, 5),
        ("setwise", SetwiseRanker(config(), num_child=3, item_label="hotel"), 5, 6),
        ("listwise", ListwiseRanker(config(), item_label="hotel"), 5, 1),
        ("pairwise", PairwiseRanker(config(), item_label="hotel"), 5, 12),
        (
            "allpairs",
            PairwiseRanker(config(), strategy="allpairs", item_label="hotel", max_concurrency=3),
            3,
            6,
        ),
        (
            "tourrank",
            TourRankRanker(
                config(),
                group_size=5,
                schedule=[3, 2],
                num_tournaments=2,
                item_label="hotel",
                seed=0,
            ),
            5,
            4,
        ),
        (
            "structured_output",
            SetwiseRanker(config(), num_child=3, item_label="hotel", structured_output=True),
            3,
            2,
        ),
        (
            "reasoning",
            PointwiseRanker(
                config(extra_kwargs={"max_tokens": 500}),
                item_label="hotel",
                reasoning=True,
                max_concurrency=2,
            ),
            3,
            3,
        ),
        (
            "criteria_auto",
            PointwiseRanker(config(), item_label="hotel", criteria="auto", max_concurrency=2),
            3,
            4,
        ),
    ]
    if rerank_model and os.environ.get("COHERE_API_KEY"):
        phases.append(
            (
                "rerank_api",
                RerankAPIRanker(LLMConfig(model=rerank_model, timeout=60)),
                5,
                1,
            )
        )
    return phases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LiteLLM model string")
    parser.add_argument(
        "--rerank-model",
        default=DEFAULT_RERANK_MODEL,
        help="rerank endpoint for RerankAPIRanker; skipped without COHERE_API_KEY",
    )
    parser.add_argument("--budget", type=int, default=45, help="max requests to spend")
    parser.add_argument("--only", nargs="*", help="run only these phases")
    parser.add_argument(
        "--pause", type=float, default=4.0, help="seconds between phases (rate limiting)"
    )
    parser.add_argument(
        "--check-budget", action="store_true", help="report OpenRouter usage and exit"
    )
    args = parser.parse_args()

    load_dotenv()
    if args.check_budget:
        print(openrouter_budget())
        return 0

    import litellm

    # LiteLLM prints provider chatter to stderr. The library deliberately
    # does not set this for you (it is process-wide state), so a script that
    # wants quiet output opts in itself.
    litellm.suppress_debug_info = True

    from llmranker import Candidate

    warnings: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record):
            warnings.append(record.getMessage())

    logger = logging.getLogger("llmranker")
    logger.addHandler(Capture())
    logger.setLevel(logging.WARNING)

    print(f"model:  {args.model}")
    if args.model.startswith("openrouter/"):
        print(f"budget: {openrouter_budget()}")
    print()

    spent = 0
    failures = 0
    for name, ranker, n, cost in build_rankers(args.model, args.rerank_model):
        if args.only and name not in args.only:
            continue
        if spent + cost > args.budget:
            print(f"  {name:18} SKIPPED (would exceed --budget {args.budget})")
            continue

        candidates = [Candidate(id=i, text=t) for i, t in HOTELS[:n]]
        seen = len(warnings)
        started = time.perf_counter()
        try:
            result = ranker.rank(QUERY, candidates)
        except Exception as exc:  # a failed phase is the finding, not a crash
            print(f"  {name:18} ERROR {type(exc).__name__}: {str(exc)[:100]}")
            failures += 1
            continue

        spent += ranker.total_calls
        order = [c.id for c in result]
        # Only judge the constraints the candidate subset can actually express.
        expected_best = BEST & {c.id for c in candidates}
        ok = set(order[: len(expected_best)]) == expected_best
        if WORST in order:
            ok = ok and order[-1] == WORST
        failures += not ok
        new_warnings = len(warnings) - seen
        print(
            f"  {name:18} {' > '.join(order):24} calls={ranker.total_calls:2} "
            f"{time.perf_counter() - started:5.1f}s  {'ok' if ok else 'CHECK'}"
            f"{f'  warnings={new_warnings}' if new_warnings else ''}"
        )
        if name in ("pointwise", "reasoning"):
            print(f"  {'':18} scores={[c.score for c in result]}")
        if name == "rerank_api":
            print(
                f"  {'':18} search_units={ranker.total_search_units} "
                f"cost={ranker.estimate_cost_usd()} (None = billed per search unit, "
                f"not per token)"
            )
        if name == "criteria_auto" and result[0].metadata:
            print(f"  {'':18} criteria={list(result[0].metadata['criteria_scores'])}")
        time.sleep(args.pause)

    print(f"\nrequests spent: {spent}")
    if warnings:
        print("\nwarnings (parse failures and retries):")
        for message in dict.fromkeys(warnings):
            print(f"  - {message[:140]}")
    if failures:
        print(f"\n{failures} phase(s) need a look -- 'CHECK' means the ordering looked wrong.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
