from __future__ import annotations

import json

SYSTEM_PROMPT = (
    "You are a sharp English comedy writer for a humor generation benchmark. "
    "Return exactly one short joke or punchline. Do not add explanations, labels, greetings, "
    "alternatives, or meta-commentary."
)

# Esempi Few-Shot per guidare Llama sull'output atteso ed evitare formati generici
FEW_SHOT_HEADLINE = (
    "Examples of ideal outputs:\n"
    "Headline: Scientists teach robots to laugh at awkward office meetings\n"
    "Punchline: The project sounded great until the HR department hired a vacuum cleaner as a manager.\n"
    "---\n"
    "Headline: Local bakery wins award for the world's heaviest loaf of bread\n"
    "Punchline: Witnesses say the baker didn't win a trophy, he just lifted the bread and everyone gave up.\n"
    "---"
)

FEW_SHOT_WORD_PAIR = (
    "Examples of ideal outputs:\n"
    "Required word 1: umbrella | Required word 2: lasagna\n"
    "Punchline: I tried using a lasagna to protect myself from the rain, but I just ended up covered in cheese and disappointment.\n"
    "---\n"
    "Required word 1: hammer | Required word 2: pumpkin\n"
    "Punchline: I tried to carve a jack-o'-lantern with a rusty hammer, but it just wasn't smashing.\n"
    "---"
)


def build_generation_prompt(item: dict[str, str], contexts: list[str] | None = None) -> str:
    context_block = ""
    if contexts:
        bullets = "\n".join(f"- {ctx}" for ctx in contexts)
        context_block = f"\nOptional background facts from retrieval. Use them only if helpful; do not quote them:\n{bullets}\n"
    
    # Allineato il controllo alla chiave 'headline'
    if item["input_type"] == "headline":
        headline_text = item.get("headline", "")
        return (
            f"{SYSTEM_PROMPT}\n{context_block}\n"
            f"{FEW_SHOT_HEADLINE}\n\n"
            "Task: Write one brief joke in English related to this headline.\n"
            f"Headline: {headline_text}\n\n"
            "Style: make it feel like a late-night punchline or satirical caption. Use a specific comic twist. "
            "Do not repeat the full headline. Do not write a generic 'Why did...' template unless it is genuinely apt.\n"
            "Rules: 8-25 words, one line, no preface, no 'Sure', no 'Here is a joke', no explanation, one joke only.\n"
            "Punchline:"
        )
        
    return (
        f"{SYSTEM_PROMPT}\n{context_block}\n"
        f"{FEW_SHOT_WORD_PAIR}\n\n"
        "Task: Write one brief joke in English that naturally includes both required words.\n"
        f"Required word 1: {item.get('word1', '-')}\n"
        f"Required word 2: {item.get('word2', '-')}\n\n"
        "Style: make the two words collide in an unexpected but coherent situation. Prefer a compact one-line punchline.\n"
        "Rules: 8-25 words, include both words exactly or as clear inflected forms, no preface, no explanation, one joke only.\n"
        "Punchline:"
    )


def build_judge_prompt(item: dict[str, str], joke_a: str, joke_b: str) -> str:
    headline_text = item.get("headline", "")
    task_input = headline_text if item["input_type"] == "headline" else f"{item.get('word1', '-')} / {item.get('word2', '-')}"
    
    schema = {
        "winner": "A or B or tie",
        "reason": "short explanation",
        "scores": {
            "humor": {"A": "1-5", "B": "1-5"},
            "relevance": {"A": "1-5", "B": "1-5"},
            "constraint_satisfaction": {"A": "1-5", "B": "1-5"},
            "originality": {"A": "1-5", "B": "1-5"},
            "fluency": {"A": "1-5", "B": "1-5"},
        },
    }
    return (
        "You are a blind evaluator for an English humor generation task. "
        "You do not know which model wrote either joke. Compare Joke A and Joke B.\\n\\n"
        f"Input type: {item['input_type']}\n"
        f"Task input: {task_input}\n\n"
        f"Joke A: {joke_a}\n"
        f"Joke B: {joke_b}\n\n"
        "Evaluate humor, relevance, constraint satisfaction, originality, and fluency. "
        "Return only valid JSON with this schema:\n"
        f"{json.dumps(schema, indent=2)}"
    )


def build_preference_prompt(item: dict[str, str]) -> str:
    return build_generation_prompt(item)