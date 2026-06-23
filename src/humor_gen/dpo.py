from __future__ import annotations

from pathlib import Path
from typing import Any

from humor_gen.utils import read_jsonl, require_gpu_for_real_run, require_hf_token, resolve_model_config


def run_dpo_training(config: dict[str, Any], mock: bool = False) -> None:
    """Run optional DPO/LoRA training. Intended for Colab GPU runtimes."""
    if mock:
        output_dir = Path(config.get("output_dir", "checkpoints/mock_dpo"))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "MOCK_CHECKPOINT.txt").write_text("Mock DPO checkpoint placeholder.\n", encoding="utf-8")
        return
    require_gpu_for_real_run(mock=False)
    try:
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
        from trl import DPOTrainer
        import torch
    except ImportError as exc:
        raise RuntimeError("DPO requires Colab dependencies. Install requirements-colab.txt.") from exc
    model_key = config.get("base_model", "llama")
    model_cfg = resolve_model_config(model_key, config.get("models_config", "configs/models.yaml"))
    require_hf_token(model_cfg, mock=False)
    preferences = read_jsonl(config.get("preferences_path", "data/final/preferences.jsonl"))
    if not preferences:
        raise ValueError("Preference dataset is empty; run build_preferences first.")
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["hf_id"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["hf_id"],
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quant,
    )
    lora_cfg = config.get("lora", {})
    peft_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get("target_modules"),
        task_type="CAUSAL_LM",
    )
    train_cfg = config.get("training", {})
    args = TrainingArguments(
        output_dir=config.get("output_dir", "checkpoints/llama_dpo_lora"),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=train_cfg.get("learning_rate", 5e-5),
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        logging_steps=train_cfg.get("logging_steps", 5),
        save_strategy="epoch",
        report_to=[],
    )
    dataset = Dataset.from_list(
        [{"prompt": row["prompt"], "chosen": row["chosen"], "rejected": row["rejected"]} for row in preferences]
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        beta=train_cfg.get("beta", 0.1),
        max_prompt_length=train_cfg.get("max_prompt_length", 512),
        max_length=train_cfg.get("max_length", 768),
    )
    trainer.train()
    trainer.save_model(config.get("output_dir", "checkpoints/llama_dpo_lora"))
