def build_prompt(item):
    task_type = item.get("type")

    if task_type == "word_inclusion":
        return (
            "Write exactly one original, concise joke in English. "
            f"The joke must include BOTH required words: {item['word1']} and {item['word2']}. "
            "Do not explain the joke. Do not add labels such as 'Joke:'."
        )

    if task_type == "news_headline":
        return (
            "Write exactly one original, concise joke or punchline in English inspired by this news headline: "
            f"{item['headline']}. Do not explain the joke. Do not add labels such as 'Joke:'."
        )

    raise ValueError(f"Unsupported item type: {task_type}")
