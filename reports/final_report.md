# Final Report - SemEval 2026 MWAHAHA Subtask A

## 1. Task Overview

Describe the humor generation task, input types, constraints and English-only scope.

## 2. Models

Summarize Llama 3.1 8B Instruct, Qwen2.5 7B Instruct and Mistral 7B Instruct v0.3, including 4-bit/Unsloth usage.

## 3. Pipeline

Explain baseline generation, validation, RAG variants, blind LLM-as-a-judge tournament, preference construction and DPO/Best-of-N.

## 4. Experimental Setup

Document dataset version, Colab GPU, generation parameters, RAG `k`, judge mapping and mock/local checks.

## 5. Results

Add tables from:

- `reports/figures/generation_summary.csv`
- `reports/figures/win_rate_summary.csv`

Include figures (`.png` when Matplotlib is available, `.svg` fallback otherwise):

- `reports/figures/win_rate_per_model`
- `reports/figures/constraint_satisfaction`
- `reports/figures/invalid_rate_by_method`

## 6. Error Analysis

Discuss invalid outputs, missing words, low relevance, long jokes and judge disagreements.

## 7. Discussion

Compare baseline vs RAG and, if available, baseline vs Best-of-N or DPO.

## 8. Limitations

Mention automatic English/relevance validation limitations, judge bias, GPU budget and dataset size.

## 9. Conclusion

Summarize main findings and future work.
