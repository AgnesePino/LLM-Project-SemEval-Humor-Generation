from humor_gen.prompts import build_generation_prompt, build_semantic_judge_prompt


def test_headline_prompt_is_compact_and_ends_with_anchor() -> None:
    prompt = build_generation_prompt(
        {"input_type": "headline", "headline": "Robots learn to laugh at meetings"},
        contexts=["Robots can classify laughter patterns."],
    )

    assert "### RELEVANT BACKGROUND FACTS" in prompt
    assert "### STYLE & RULES" in prompt
    assert "Examples of ideal outputs" not in prompt
    assert prompt.endswith("Punchline:")


def test_word_pair_prompt_preserves_words_and_contexts() -> None:
    prompt = build_generation_prompt(
        {"input_type": "word_pair", "word1": "umbrella", "word2": "lasagna"},
        contexts=["An umbrella protects against rain."],
    )

    assert "Required word 1: umbrella" in prompt
    assert "Required word 2: lasagna" in prompt
    assert "An umbrella protects against rain." in prompt
    assert prompt.endswith("Joke:")


def test_semantic_judge_prompt_uses_zero_based_candidates_and_json_schema() -> None:
    prompt = build_semantic_judge_prompt(
        {"input_type": "headline", "headline": "Robots learn office humor"},
        ["Candidate one.", "Candidate two."],
        contexts=["Robots can classify laughter."],
    )

    assert "[0] Candidate one." in prompt
    assert "[1] Candidate two." in prompt
    assert '"winner_index"' in prompt
    assert prompt.count('"overall"') == 2
    assert "do not provide private chain-of-thought" in prompt
