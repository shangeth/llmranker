from __future__ import annotations

from .types import Candidate

DEFAULT_ITEM_LABEL = "item"

FINAL_ANSWER_MARKER = "FINAL ANSWER:"


def reasoning_suffix(answer_hint: str) -> str:
    """Instruction appended to a prompt when reasoning=True: think first,
    then give a clearly delimited final answer so it can be parsed reliably
    out of a longer response instead of picking up stray tokens from the
    reasoning text itself."""
    return (
        f"\n\nFirst, briefly reason about this. Then, on a new final line, "
        f"write '{FINAL_ANSWER_MARKER} {answer_hint}' with nothing else on "
        f"that line."
    )


def extract_final_answer(text: str) -> str:
    """If a `FINAL ANSWER:` marker is present, return only the text after it
    (last occurrence, in case the marker text appears earlier by
    coincidence); otherwise return `text` unchanged. Safe to call
    unconditionally: a no-op when the marker isn't present, which also
    guards against a model volunteering explanatory text even when
    reasoning wasn't requested.
    """
    idx = text.upper().rfind(FINAL_ANSWER_MARKER)
    if idx == -1:
        return text
    return text[idx + len(FINAL_ANSWER_MARKER) :]


# --- pointwise -----------------------------------------------------------


def pointwise_system_prompt(item_label: str = DEFAULT_ITEM_LABEL) -> str:
    return (
        f"You are an intelligent assistant that rates how relevant a {item_label} "
        f"is to a user's query on a scale from 0 (not relevant at all) to 10 "
        f"(perfectly relevant)."
    )


def pointwise_user_prompt(
    query: str,
    candidate: Candidate,
    item_label: str = DEFAULT_ITEM_LABEL,
    reasoning: bool = False,
    structured_output: bool = False,
) -> str:
    prompt = (
        f'Query: "{query}"\n\n'
        f'{item_label.capitalize()}: "{candidate.text}"\n\n'
        f"On a scale from 0 to 10, how relevant is this {item_label} to the query?"
    )
    if reasoning:
        return prompt + reasoning_suffix("<integer score 0-10>")
    if structured_output:
        return prompt + " Respond with the score as JSON."
    return prompt + " Output only the integer score, nothing else."


def pointwise_multi_criteria_user_prompt(
    query: str,
    candidate: Candidate,
    names: list[str],
    item_label: str = DEFAULT_ITEM_LABEL,
    reasoning: bool = False,
    structured_output: bool = False,
) -> str:
    criteria_list = ", ".join(names)
    prompt = (
        f'Query: "{query}"\n\n'
        f'{item_label.capitalize()}: "{candidate.text}"\n\n'
        f"Rate this {item_label} from 0 to 10 on each of the following "
        f"criteria, independently of the others: {criteria_list}."
    )
    if reasoning:
        hint = ", ".join(f"{name}=<score>" for name in names)
        return prompt + reasoning_suffix(hint)
    if structured_output:
        return prompt + " Respond with a JSON object mapping each criterion name to its score."
    example = ", ".join(f"{name}=<score>" for name in names)
    return prompt + f" Output only '{example}', comma-separated, nothing else."


# --- criteria extraction ----------------------------------------------------


def criteria_extraction_system_prompt(item_label: str = DEFAULT_ITEM_LABEL) -> str:
    return (
        f"You are an intelligent assistant that identifies the distinct "
        f"relevance criteria a user's query expresses about a {item_label}."
    )


def criteria_extraction_user_prompt(
    query: str,
    item_label: str = DEFAULT_ITEM_LABEL,
    reasoning: bool = False,
    structured_output: bool = False,
) -> str:
    prompt = (
        f'Given the query "{query}", extract the distinct relevance criteria '
        f"it expresses about a {item_label} (e.g. price, location, a "
        f"specific feature), as short names (2-4 words each)."
    )
    if reasoning:
        return prompt + reasoning_suffix("<comma-separated criteria names>")
    if structured_output:
        return prompt + " Respond with the criteria names as a JSON array of strings."
    return prompt + " Output only the criteria names, comma-separated, nothing else."


# --- pairwise --------------------------------------------------------------


def pairwise_system_prompt(item_label: str = DEFAULT_ITEM_LABEL) -> str:
    return (
        f"You are an intelligent assistant specialized in selecting the most "
        f"relevant {item_label} from a pair of {item_label}s based on their "
        f"relevance to a user's query."
    )


def pairwise_user_prompt(
    query: str,
    candidates: list[Candidate],
    item_label: str = DEFAULT_ITEM_LABEL,
    reasoning: bool = False,
    structured_output: bool = False,
) -> str:
    a, b = candidates[0], candidates[1]
    label = item_label.capitalize()
    prompt = (
        f'Given a query "{query}", which of the following two {item_label}s is '
        f"more relevant to the query?\n\n"
        f'{label} A: "{a.text}"\n\n'
        f'{label} B: "{b.text}"'
    )
    if reasoning:
        return prompt + reasoning_suffix("A or B")
    if structured_output:
        return prompt + " Respond with the chosen label as JSON."
    return prompt + (
        f"\n\nOutput only the label of the more relevant {item_label}, 'A' or 'B'. "
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
    reasoning: bool = False,
    structured_output: bool = False,
) -> str:
    label = item_label.capitalize()
    body = "\n\n".join(f'{label} {characters[i]}: "{c.text}"' for i, c in enumerate(candidates))
    prompt = (
        f'Given a query "{query}", which of the following {item_label}s is the '
        f"most relevant one to the query?\n\n{body}"
    )
    if reasoning:
        return prompt + reasoning_suffix("<label letter>")
    if structured_output:
        return prompt + " Respond with the chosen label as JSON."
    return prompt + (
        f"\n\nOutput only the label of the single most relevant {item_label}, "
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
    query: str,
    num: int,
    item_label: str = DEFAULT_ITEM_LABEL,
    reasoning: bool = False,
    structured_output: bool = False,
) -> str:
    prompt = (
        f"Query: {query}.\n"
        f"Rank the {num} {item_label}s above based on their relevance to the query. "
        f"The {item_label}s should be listed in descending order using identifiers. "
        f"The most relevant {item_label} should be listed first. The output format "
        f"should be [] > [], e.g. [1] > [2]."
    )
    if reasoning:
        return prompt + reasoning_suffix("[] > [] > ...")
    if structured_output:
        return prompt + " Respond with the ranking (a list of identifiers) as JSON."
    return prompt + " Only respond with the ranking, do not say any word or explain."


# --- tourrank ------------------------------------------------------------------


def tourrank_system_prompt(item_label: str = DEFAULT_ITEM_LABEL) -> str:
    return (
        f"You are an intelligent assistant specialized in selecting the most "
        f"relevant {item_label}s from a group based on their relevance to a "
        f"user's query."
    )


def tourrank_group_user_prompt(
    query: str,
    candidates: list[Candidate],
    characters: list[str],
    advance_count: int,
    item_label: str = DEFAULT_ITEM_LABEL,
    reasoning: bool = False,
    structured_output: bool = False,
) -> str:
    label = item_label.capitalize()
    body = "\n\n".join(f'{label} {characters[i]}: "{c.text}"' for i, c in enumerate(candidates))
    prompt = (
        f'Given a query "{query}", select the {advance_count} most relevant '
        f"{item_label}s from the following {len(candidates)}:\n\n{body}"
    )
    if reasoning:
        return prompt + reasoning_suffix(f"<{advance_count} comma-separated labels>")
    if structured_output:
        return prompt + f" Respond with the {advance_count} selected labels as JSON."
    return prompt + (
        f"\n\nOutput only the labels of the {advance_count} most relevant "
        f"{item_label}s, separated by commas, e.g. 'A, C'. Select exactly "
        f"{advance_count}, no more and no fewer."
    )
