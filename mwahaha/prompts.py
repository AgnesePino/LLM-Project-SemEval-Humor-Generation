from __future__ import annotations

from .schema import TaskInput


SYSTEM_GENERATOR = (
    "You are writing entries for a humor-generation competition. "
    "Return only the final joke. No explanation, no labels, no markdown."
)

SYSTEM_JUDGE = (
    "You are judging a humor-generation competition. Return only valid JSON. "
    "Prefer concise, surprising, specific jokes that satisfy the input. "
    "Penalize generic jokes, explanations, missing required words, and offensive content."
)

GENERATION_STYLES = [
    "wordplay",
    "reversal",
    "absurd literalism",
    "deadpan understatement",
    "topical punchline",
    "incongruity",
]

REFINE_STYLES = [
    "newsroom deadpan",
    "absurd consequence",
    "headline misread",
    "satirical tag",
    "specific analogy",
    "entity-based punchline",
]


def build_generation_prompt(item: TaskInput, style: str) -> str:
    if item.kind == "word_inclusion":
        return (
            "/no_think\n"
            f"Write one short joke in English that naturally includes these required words verbatim: "
            f"`{item.word1}`, `{item.word2}`.\n"
            f"Comedic mechanism: {style}.\n"
            "The punchline must depend on both words. Return only the joke."
        )
    return (
        "/no_think\n"
        f"Write one short joke in English inspired by this news headline: {item.headline}\n"
        f"Comedic mechanism: {style}.\n"
        "Keep it clearly related to the headline, but do not summarize it. Return only the joke."
    )


def build_refine_prompt(item: TaskInput, style: str, incumbent: str) -> str:
    if item.kind == "word_inclusion":
        return (
            "/no_think\n"
            f"Write a better short joke in English that naturally includes these required words verbatim: "
            f"`{item.word1}`, `{item.word2}`.\n"
            f"Current joke to beat: {incumbent}\n"
            f"Comedic mechanism: {style}.\n"
            "Make the punchline depend on both words. Use one sentence if possible. "
            "Avoid starting with 'I tried', 'I asked', or 'I told'. Avoid 'turns out'. "
            "Avoid dialogue and unnecessary quotation marks. Return only the joke."
        )
    return (
        "/no_think\n"
        f"Write a better short joke in English inspired by this news headline: {item.headline}\n"
        f"Current joke to beat: {incumbent}\n"
        f"Comedic mechanism: {style}.\n"
        "Keep it clearly tied to a specific entity, event, or phrase from the headline. "
        "Do not summarize the headline. Use one sentence if possible, ideally under 25 words. "
        "Avoid starting with 'I tried', 'I asked', or 'I told'. Avoid 'turns out'. "
        "Avoid dialogue and unnecessary quotation marks. Return only the joke."
    )


def build_score_prompt(item: TaskInput, candidate: str) -> str:
    if item.kind == "word_inclusion":
        context = f"Required words: {item.word1}, {item.word2}"
    else:
        context = f"Headline: {item.headline}"
    return (
        "/no_think\n"
        f"{context}\n\n"
        f"Candidate joke: {candidate}\n\n"
        "Score this candidate from 0 to 10 for a human pairwise humor arena. "
        "Consider constraint compliance, brevity, surprise, specificity, natural English, and funniness. "
        'Return JSON exactly like {"score": 7.5, "reason": "short reason"}'
    )
