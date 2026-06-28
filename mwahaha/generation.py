from __future__ import annotations

import re

from .clients import LLMClient
from .prompts import GENERATION_STYLES, REFINE_STYLES, SYSTEM_GENERATOR, build_generation_prompt, build_refine_prompt
from .schema import Candidate, TaskInput
from .validation import clean_text, validate_candidate


def generate_candidates(client: LLMClient, item: TaskInput, variants_per_style: int, base_seed: int) -> list[Candidate]:
    candidates: list[Candidate] = []
    for style_index, style in enumerate(GENERATION_STYLES):
        for variant in range(variants_per_style):
            seed = base_seed + style_index * 100 + variant
            temperature = [0.80, 0.90, 0.98, 1.05][variant % 4]
            prompt = build_generation_prompt(item, style)
            raw = client.chat(
                [
                    {"role": "system", "content": SYSTEM_GENERATOR},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=140,
                seed=seed,
            )
            text = clean_text(raw)
            valid, reason = validate_candidate(item, text)
            candidates.append(Candidate(text, style, seed, temperature, valid=valid, invalid_reason=reason))
    return dedupe_candidates(candidates)


def generate_candidate_pool_for_item(
    client: LLMClient,
    item: TaskInput,
    variants_per_entry: int,
    base_seed: int,
    source_model: str,
    source_backend: str,
    source_base_url: str,
    start_variant: int = 0,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for variant in range(start_variant, variants_per_entry):
        style = GENERATION_STYLES[variant % len(GENERATION_STYLES)]
        seed = base_seed + variant
        temperature = [0.82, 0.92, 1.02, 1.08][variant % 4]
        prompt = build_generation_prompt(item, style)
        raw = client.chat(
            [
                {"role": "system", "content": SYSTEM_GENERATOR},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=140,
            seed=seed,
        )
        text = clean_text(raw)
        valid, reason = validate_candidate(item, text)
        candidates.append(
            Candidate(
                text,
                style,
                seed,
                temperature,
                source_model=source_model,
                source_backend=source_backend,
                source_base_url=source_base_url,
                valid=valid,
                invalid_reason=reason,
            )
        )
    return candidates


def generate_refine_candidates(
    client: LLMClient,
    item: TaskInput,
    incumbent: str,
    variants_per_style: int,
    base_seed: int,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for style_index, style in enumerate(REFINE_STYLES):
        for variant in range(variants_per_style):
            seed = base_seed + style_index * 100 + variant
            temperature = [0.82, 0.92, 1.02, 1.08][variant % 4]
            prompt = build_refine_prompt(item, style, incumbent)
            raw = client.chat(
                [
                    {"role": "system", "content": SYSTEM_GENERATOR},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=110,
                seed=seed,
            )
            text = clean_text(raw)
            valid, reason = validate_candidate(item, text)
            candidates.append(Candidate(text, style, seed, temperature, valid=valid, invalid_reason=reason))
    return dedupe_candidates(candidates)


def dedupe_candidates(candidates: list[Candidate]) -> list[Candidate]:
    seen: set[str] = set()
    out: list[Candidate] = []
    for candidate in candidates:
        key = normalize_for_dedupe(candidate.text)
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def normalize_for_dedupe(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def generate_baseline(client: LLMClient, item: TaskInput, seed: int) -> str:
    if item.kind == "word_inclusion":
        prompt = (
            "/no_think\n"
            "Create a joke based on the following elements:\n\n"
            f"Words to include: {{{item.word1}, {item.word2}}}\n\n"
            "Please craft a joke that incorporates all the specified elements. "
            "The joke should be concise, creative, and genuinely funny. "
            "All required words must appear somewhere in the joke. "
            "Only return the joke and nothing else."
        )
    else:
        prompt = (
            "/no_think\n"
            "Create a joke based on this title of a news article:\n\n"
            f'"{item.headline}"\n\n'
            "The joke should be concise, creative and genuinely funny. "
            "Only return the joke and nothing else."
        )
    raw = client.chat(
        [{"role": "system", "content": SYSTEM_GENERATOR}, {"role": "user", "content": prompt}],
        temperature=0.9,
        max_tokens=140,
        seed=seed,
    )
    return clean_text(raw)
