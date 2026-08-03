"""Rerank e-commerce search results by natural-language shopping intent.

Keyword/embedding search over "waterproof running shoes for wide feet under
$100" will happily return non-waterproof shoes that mention "running" a lot,
or waterproof boots that aren't for running. An LLM reading the full intent
against each product description can catch constraints (price ceiling,
width, use case) that a bag-of-words match misses.

Run:
    export OPENAI_API_KEY=...
    python product_search_reranking.py
"""

from llmranker import Candidate, LLMConfig, SetwiseRanker

QUERY = "waterproof running shoes for wide feet, under $100"

PRODUCTS = [
    (
        "p1",
        "TrailGrip Wide Trail Runner - waterproof membrane, wide-fit last, $89.99. Built for muddy trail conditions.",
    ),
    (
        "p2",
        "CityStride Runner - breathable mesh running shoe, standard fit, $74.99. Not water resistant.",
    ),
    (
        "p3",
        "AquaBoot Hiking Boot - fully waterproof, wide fit available, $134.99. Built for hiking, not running.",
    ),
    ("p4", "SpeedLite Racer - ultra-light racing flat, narrow fit, $110.00, no water resistance."),
    ("p5", "AllWeather Wide Runner - waterproof, extra-wide fit, road/trail hybrid sole, $95.00."),
    (
        "p6",
        "ComfortWalk Wide - waterproof walking shoe (not designed for running), wide fit, $65.00.",
    ),
]

candidates = [Candidate(id=pid, text=text) for pid, text in PRODUCTS]

ranker = SetwiseRanker(
    LLMConfig(model="gpt-4o-mini"),
    num_child=3,
    method="heapsort",
    k=6,
    item_label="product",
)

if __name__ == "__main__":
    result = ranker.rank(QUERY, candidates)
    print("Ranked products:")
    for i, c in enumerate(result, start=1):
        print(f"  {i}. [{c.id}] {c.text[:70]}...")
