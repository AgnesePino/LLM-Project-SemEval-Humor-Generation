import argparse
import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from humor_generation.evaluation.validators import validate_candidate
from humor_generation.generation.prompts import build_prompt
from humor_generation.utils.io import load_jsonl


MODEL_PRESETS = {
    "llama": "meta-llama/Llama-3.2-3B-Instruct",
    "qwen": "Qwen/Qwen2.5-3B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}


def clean_generation(text):
    text = text.strip()
    text = re.sub(r"^(joke|answer|response)\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip().strip('"')


def load_model(model_id, load_in_4bit=True):
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
    )
    model.eval()
    return model, tokenizer


def generate_one(model, tokenizer, item, temperature, top_p, max_new_tokens, retries):
    prompt = build_prompt(item)
    messages = [{"role": "user", "content": prompt}]

    best = ""
    for _ in range(retries):
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                repetition_penalty=1.12,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_ids = output_ids[0][input_ids.shape[-1] :]
        candidate = clean_generation(tokenizer.decode(generated_ids, skip_special_tokens=True))
        best = candidate
        if validate_candidate(candidate, item).is_valid:
            return candidate

    return best


def already_processed(output_path):
    ids = set()
    if not os.path.exists(output_path):
        return ids

    with open(output_path, encoding="utf-8") as output_file:
        for line in output_file:
            if line.strip():
                ids.add(json.loads(line)["id"])
    return ids


def run(args):
    model_id = args.model_id or MODEL_PRESETS[args.model]
    model, tokenizer = load_model(model_id, load_in_4bit=not args.no_4bit)
    processed_ids = already_processed(args.output)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.output, "a", encoding="utf-8") as output_file:
        for item in load_jsonl(args.input):
            if item["id"] in processed_ids:
                continue

            joke = generate_one(
                model=model,
                tokenizer=tokenizer,
                item=item,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                retries=args.retries,
            )
            output_file.write(json.dumps({"id": item["id"], "joke": joke}, ensure_ascii=False) + "\n")
            output_file.flush()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate jokes with a HuggingFace model on Colab/GPU.")
    parser.add_argument("--input", required=True, help="Internal JSONL input file.")
    parser.add_argument("--output", required=True, help="JSONL output file.")
    parser.add_argument("--model", choices=sorted(MODEL_PRESETS), default="llama")
    parser.add_argument("--model-id", default=None, help="Override the preset with any HuggingFace model id.")
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization.")
    return parser.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
