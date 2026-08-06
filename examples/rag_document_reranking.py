"""Rerank RAG retrieval results before stuffing them into a prompt.

Vector search over-recalls: a top-k similarity search reliably returns
*related* chunks, but "related" and "actually answers the question" aren't
the same thing. Reranking the retrieved chunks with an LLM before building
the final context window is a cheap way to push the genuinely relevant
passages to the front and drop the rest, instead of spending context budget
on near-duplicates that merely share vocabulary with the query.

Run:
    export OPENAI_API_KEY=...
    python rag_document_reranking.py
"""

from llmranker import Candidate, LLMConfig, PairwiseRanker

QUERY = "What's the cancellation policy if I need to reschedule within 24 hours of check-in?"

# Pretend these came back from a vector store as the top-6 nearest neighbors
# for the query above, all "about" hotel bookings, only a couple actually
# answer the question.
RETRIEVED_CHUNKS = [
    (
        "chunk-1",
        "Guests can modify room preferences (bed type, floor, view) free of charge up to 48 hours before arrival by contacting the front desk.",
    ),
    (
        "chunk-2",
        "Cancellations made more than 7 days before check-in receive a full refund. Cancellations within 24 hours of check-in are non-refundable, but may be rescheduled once at no fee within the same calendar year.",
    ),
    (
        "chunk-3",
        "Our loyalty program awards 1 point per dollar spent, redeemable for free nights, room upgrades, and late checkout.",
    ),
    (
        "chunk-4",
        "Check-in begins at 3:00 PM and check-out is at 11:00 AM; early check-in and late check-out are subject to availability.",
    ),
    (
        "chunk-5",
        "For bookings made through a third-party site, cancellation and rescheduling policies are governed by that site's terms, not ours.",
    ),
    (
        "chunk-6",
        "Pets under 25 lbs are welcome for a one-time $75 cleaning fee; service animals are always welcome at no charge.",
    ),
]

candidates = [Candidate(id=cid, text=text) for cid, text in RETRIEVED_CHUNKS]

ranker = PairwiseRanker(
    LLMConfig(model="gpt-4o-mini"),
    strategy="heapsort",
    k=3,
    item_label="document",
)

if __name__ == "__main__":
    reranked = ranker.rank(QUERY, candidates)
    print("Top 3 chunks to actually put in the RAG prompt:")
    for i, c in enumerate(reranked[:3], start=1):
        print(f"  {i}. [{c.id}] {c.text[:80]}...")
