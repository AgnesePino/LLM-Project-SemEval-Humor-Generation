from __future__ import annotations

import hashlib
import logging
from typing import Any

# Importiamo le funzioni aggiornate dai tuoi moduli
from humor_gen.prompts import build_generation_prompt, build_judge_prompt
from humor_gen.validate import clean_joke

LOGGER = logging.getLogger(__name__)


class MockGenerator:
    def __init__(self, model_key: str):
        self.model_key = model_key

    def generate_joke(
        self,
        item: dict[str, str],
        contexts: list[str] | None = None,
        generation_overrides: dict[str, Any] | None = None,
    ) -> str:
        if item["input_type"] == "word_pair":
            return (
                f"My {item.get('word1', 'word1')} hired a {item.get('word2', 'word2')} as a life coach; "
                "now every crisis comes with garnish and confidence."
            )
            
        headline_text = item.get("headline", "")
        keyword = _headline_keyword(headline_text)
        return f"{headline_text} sounds serious until the {keyword} department hires a punchline consultant."

    def generate_text(
        self,
        prompt: str,
        generation_overrides: dict[str, Any] | None = None,
    ) -> str:
        return (
            '{"winner_index": 0, "reasoning": "Mock judge selected the first valid candidate.", '
            '"scores": [{"index": 0, "overall": 3.0}]}'
        )

    def judge(
        self,
        item: dict[str, str],
        joke_a: str,
        joke_b: str,
        generation_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        score_a = _mock_score(item, joke_a)
        score_b = _mock_score(item, joke_b)
        if abs(score_a - score_b) <= 1:
            winner = "tie"
        else:
            winner = "A" if score_a > score_b else "B"
        return {
            "winner": winner,
            "reason": "Mock judge prefers the joke with stronger constraint coverage and compact wording.",
            "scores": {
                metric: {"A": max(1, min(5, score_a)), "B": max(1, min(5, score_b))}
                for metric in ["humor", "relevance", "constraint_satisfaction", "originality", "fluency"]
            },
        }


class HFGenerator:
    def __init__(self, model_cfg: dict[str, Any], generation_cfg: dict[str, Any]):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        model_id = model_cfg["hf_id"]
        gen = generation_cfg.get("generation", generation_cfg)
        quantization_config = None
        if gen.get("load_in_4bit", True):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        LOGGER.info("Loading %s", model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
        )
        self.model.eval()
        self.generation = gen

    def generate_joke(
        self,
        item: dict[str, str],
        contexts: list[str] | None = None,
        generation_overrides: dict[str, Any] | None = None,
    ) -> str:
        prompt = build_generation_prompt(item, contexts)
        text = self._generate_text(prompt, generation_overrides=generation_overrides)
        return clean_joke(text)

    def generate_text(
        self,
        prompt: str,
        generation_overrides: dict[str, Any] | None = None,
    ) -> str:
        """Generate raw text for structured tasks such as semantic reranking."""
        return self._generate_text(prompt, generation_overrides=generation_overrides)

    def judge(
        self,
        item: dict[str, str],
        joke_a: str,
        joke_b: str,
        generation_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        import json
        import re

        prompt = build_judge_prompt(item, joke_a, joke_b)
        text = self._generate_text(prompt, generation_overrides=generation_overrides)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise ValueError(f"Judge did not return JSON: {text[:300]}")
            return json.loads(match.group(0))

    def _generate_text(
        self,
        prompt: str,
        generation_overrides: dict[str, Any] | None = None,
    ) -> str:
        import torch
    
        generation_overrides = generation_overrides or {}
    
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids = input_ids.to(self.model.device)
    
        do_sample = generation_overrides.get(
            "do_sample",
            self.generation.get("do_sample", True),
        )
    
        generate_kwargs = {
            "do_sample": do_sample,
            "max_new_tokens": generation_overrides.get(
                "max_new_tokens",
                self.generation.get("max_new_tokens", 60),
            ),
            "repetition_penalty": generation_overrides.get(
                "repetition_penalty",
                self.generation.get("repetition_penalty", 1.12),
            ),
            "pad_token_id": self.tokenizer.pad_token_id,
        }
    
        if do_sample:
            generate_kwargs["temperature"] = generation_overrides.get(
                "temperature",
                self.generation.get("temperature", 0.4),
            )
            generate_kwargs["top_p"] = generation_overrides.get(
                "top_p",
                self.generation.get("top_p", 0.9),
            )
    
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                **generate_kwargs,
            )
    
        generated_ids = output_ids[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()


def get_runner(model_cfg: dict[str, Any], generation_cfg: dict[str, Any], mock: bool):
    if mock:
        return MockGenerator(model_cfg["key"])
    return HFGenerator(model_cfg, generation_cfg)


def _headline_keyword(headline: str) -> str:
    words = [word.strip(".,:;!?").casefold() for word in headline.split() if len(word.strip(".,:;!?")) > 4]
    return words[0] if words else "headline"


def _mock_score(item: dict[str, str], joke: str) -> int:
    # Uscita sicura tramite .get() per scongiurare KeyError su ID mancanti nel mock
    record_id = item.get("id") or item.get("ID", "000")
    digest = int(hashlib.md5((record_id + joke).encode("utf-8")).hexdigest(), 16)
    compact_bonus = 1 if len(joke.split()) <= 28 else 0
    return 2 + digest % 3 + compact_bonus
