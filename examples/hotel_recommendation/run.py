"""Flagship example: rerank hotels by natural-language guest preference.

Why hotels? Ranking hotel search results is a natural fit for LLM-based
rerankers: relevance is subjective and preference-driven ("family
friendly", "walkable to historic sites", "not touristy") in ways plain
keyword or embedding search struggles to capture, candidate sets are small
enough (dozens, not millions) that per-item LLM calls are cheap, and there's
usually no labeled data to fine-tune a ranking model on.

Run:
    export OPENAI_API_KEY=...      # or GEMINI_API_KEY / ANTHROPIC_API_KEY / ...
    python run.py
"""

from llmranker import (
    Candidate,
    ListwiseRanker,
    LLMConfig,
    PairwiseRanker,
    PointwiseRanker,
    SetwiseRanker,
    TourRankRanker,
    compare_rankers,
)

# Swap this one string to change provider, e.g. "gemini/gemini-1.5-flash",
# "claude-3-5-sonnet-20241022", "ollama/llama3". See examples/multi_provider_swap.py.
MODEL = "gpt-4o-mini"

HOTELS = [
    (
        "cozy-boutique",
        (
            "A cozy boutique hotel perfect for couples seeking a romantic getaway. "
            "Nestled in a quiet neighborhood, it offers a peaceful ambiance and is "
            "within walking distance of charming cafes and boutique shops. The "
            "hotel's intimate size and personalized service ensure a memorable "
            "experience."
        ),
    ),
    (
        "lively-beachfront",
        (
            "A lively and vibrant hotel suitable for friends and groups looking for "
            "a fun-filled vacation. Situated right on the beach with bustling "
            "nightlife, it offers swimming pools, bars, and restaurants. The "
            "hotel's lively atmosphere and affordable prices make it a popular "
            "choice for young travelers."
        ),
    ),
    (
        "luxury-family-resort",
        (
            "A luxurious and upscale resort perfect for families seeking a relaxing "
            "vacation. Set amidst lush gardens with stunning ocean views, it "
            "provides a serene environment, spacious family rooms, a children's "
            "play area, and a spa."
        ),
    ),
    (
        "budget-hostel",
        (
            "A budget-friendly hotel ideal for backpackers and travelers on a tight "
            "budget. Located centrally with easy access to public transportation, "
            "it offers basic amenities and comfortable, no-frills accommodations."
        ),
    ),
    (
        "historic-charm",
        (
            "A historic and charming hotel perfect for couples and history "
            "enthusiasts. Situated in a historic district, it offers elegant "
            "architecture, antique furnishings, and an on-site restaurant for a "
            "romantic, timeless atmosphere."
        ),
    ),
    (
        "family-history-inn",
        (
            "A welcoming family-run inn in the heart of the old town, a short walk "
            "from museums, ruins, and historic landmarks. Family rooms, a kids' "
            "menu, and a courtyard for children to play, all away from the beach "
            "crowds."
        ),
    ),
    (
        "business-central",
        (
            "A modern and stylish hotel ideal for solo travelers or business "
            "professionals. Conveniently located in the city center with easy "
            "access to business districts, transit, and attractions."
        ),
    ),
]

CANDIDATES = [Candidate(id=hid, text=text) for hid, text in HOTELS]

QUERY = "family friendly hotel with kids, close to historical places, not right on the beach"

# Ground truth for this query (most to least relevant), only used below to
# demonstrate llmranker.metrics / compare_rankers. You won't have this in
# production; it's here purely to make the example measurable.
TRUE_RANKING = [
    "family-history-inn",
    "luxury-family-resort",
    "historic-charm",
    "cozy-boutique",
    "business-central",
    "budget-hostel",
    "lively-beachfront",
]


def quickstart() -> None:
    ranker = SetwiseRanker(
        LLMConfig(model=MODEL), num_child=4, strategy="heapsort", k=5, item_label="hotel"
    )
    result = ranker.rank(QUERY, CANDIDATES)

    print("Setwise ranking:")
    for i, c in enumerate(result, start=1):
        print(f"  {i}. {c.id}")
    total_tokens = ranker.total_prompt_tokens + ranker.total_completion_tokens
    print(f"  ({ranker.total_calls} LLM calls, {total_tokens} tokens)\n")


def compare_all_strategies() -> None:
    rankers = [
        PointwiseRanker(LLMConfig(model=MODEL), item_label="hotel", name="pointwise"),
        PairwiseRanker(
            LLMConfig(model=MODEL), strategy="heapsort", item_label="hotel", name="pairwise-heapsort"
        ),
        SetwiseRanker(
            LLMConfig(model=MODEL),
            num_child=4,
            strategy="heapsort",
            item_label="hotel",
            name="setwise-heapsort",
        ),
        ListwiseRanker(
            LLMConfig(model=MODEL),
            window_size=4,
            step_size=2,
            num_repeat=2,
            item_label="hotel",
            name="listwise",
        ),
        TourRankRanker(
            LLMConfig(model=MODEL),
            group_size=4,
            # 7 hotels: two elimination stages, cheap enough for an example.
            schedule=[4, 2],
            num_tournaments=3,
            item_label="hotel",
            name="tourrank",
        ),
    ]
    report = compare_rankers(rankers, QUERY, CANDIDATES, TRUE_RANKING)
    print("Comparison across strategies:")
    print(report.to_string(index=False))


if __name__ == "__main__":
    quickstart()
    compare_all_strategies()
