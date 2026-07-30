from __future__ import annotations

from .types import Candidate

DEFAULT_ITEM_LABEL = "item"


# --- pointwise -----------------------------------------------------------


def pointwise_system_prompt(item_label: str = DEFAULT_ITEM_LABEL) -> str:
    return (
        f"You are an intelligent assistant that rates how relevant a {item_label} "
        f"is to a user's query on a scale from 0 (not relevant at all) to 10 "
        f"(perfectly relevant)."
    )


def pointwise_user_prompt(
    query: str, candidate: Candidate, item_label: str = DEFAULT_ITEM_LABEL
) -> str:
    return (
        f'Query: "{query}"\n\n'
        f'{item_label.capitalize()}: "{candidate.text}"\n\n'
        f"On a scale from 0 to 10, how relevant is this {item_label} to the query? "
        f"Output only the integer score, nothing else."
    )


# --- pairwise --------------------------------------------------------------


def pairwise_system_prompt(item_label: str = DEFAULT_ITEM_LABEL) -> str:
    return (
        f"You are an intelligent assistant specialized in selecting the most "
        f"relevant {item_label} from a pair of {item_label}s based on their "
        f"relevance to a user's query."
    )


def pairwise_user_prompt(
    query: str, candidates: list[Candidate], item_label: str = DEFAULT_ITEM_LABEL
) -> str:
    a, b = candidates[0], candidates[1]
    label = item_label.capitalize()
    return (
        f'Given a query "{query}", which of the following two {item_label}s is '
        f"more relevant to the query?\n\n"
        f'{label} A: "{a.text}"\n\n'
        f'{label} B: "{b.text}"\n\n'
        f"Output only the label of the more relevant {item_label}, 'A' or 'B'. "
        f"You must choose exactly one, do not output anything else."
    )


# --- setwise -----------------------------------------------------------------


def setwise_system_prompt(item_label: str = DEFAULT_ITEM_LABEL) -> str:
    return (
        f"You are an intelligent assistant specialized in selecting the most "
        f"relevant {item_label} from a set of {item_label}s based on their "
        f"relevance to a user's query."
    )


def setwise_user_prompt(
    query: str,
    candidates: list[Candidate],
    characters: list[str],
    item_label: str = DEFAULT_ITEM_LABEL,
) -> str:
    label = item_label.capitalize()
    body = "\n\n".join(
        f'{label} {characters[i]}: "{c.text}"' for i, c in enumerate(candidates)
    )
    return (
        f'Given a query "{query}", which of the following {item_label}s is the '
        f"most relevant one to the query?\n\n{body}\n\n"
        f"Output only the label of the single most relevant {item_label}, "
        f"e.g. 'A' or 'D'. You must choose exactly one, do not choose multiple or none."
    )


# --- listwise ------------------------------------------------------------------


def listwise_system_prompt(item_label: str = DEFAULT_ITEM_LABEL) -> str:
    return (
        f"You are an intelligent assistant specialized in ranking {item_label}s "
        f"by their relevance to a user's query. You must rank every {item_label} "
        f"provided, do not omit any."
    )


def listwise_prefix_messages(
    query: str, num: int, item_label: str = DEFAULT_ITEM_LABEL
) -> list[dict]:
    return [
        {"role": "system", "content": listwise_system_prompt(item_label)},
        {
            "role": "user",
            "content": (
                f"I will provide you with {num} {item_label}s, each indicated by a "
                f"numerical identifier []. Rank the {item_label}s based on their "
                f"relevance to the query: {query}."
            ),
        },
        {"role": "assistant", "content": f"Okay, please provide the {item_label}s."},
    ]


def listwise_post_prompt(
    query: str, num: int, item_label: str = DEFAULT_ITEM_LABEL
) -> str:
    return (
        f"Query: {query}.\n"
        f"Rank the {num} {item_label}s above based on their relevance to the query. "
        f"The {item_label}s should be listed in descending order using identifiers. "
        f"The most relevant {item_label} should be listed first. The output format "
        f"should be [] > [], e.g. [1] > [2]. Only respond with the ranking, do not "
        f"say any word or explain."
    )
