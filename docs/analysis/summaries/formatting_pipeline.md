# Formatting Pipeline

Last updated: 2026-03-17

End-to-end pipeline for training models to produce formatting-aware markdown (bold, italic, strikethrough) from document images.

---

## Current Results

| Model | Stage | F1 | Bold F1 | Italic F1 | Strike F1 | olmOCR |
|-------|-------|----|:-------:|:---------:|:---------:|:------:|
| `Q3-2B` base | - | 0.000 | 0.000 | 0.000 | 0.000 | 56.1 |
| `ocr_sft` | OCR SFT | 0.000 | 0.000 | 0.000 | 0.000 | 67.1 |
| `fmt_sft` | + Fmt SFT | 0.766 | 0.698 | 0.445 | 0.218 | 66.3 |
| `fmt_grpo` | + GRPO | 0.622 | 0.563 | 0.283 | 0.000 | - |
| Qwen3.5-9B pipeline | all stages | **INVALID** | - | - | - | - |

All Qwen3.5-9B bench results are contaminated by `</think>` leakage. Re-runs pending (`model_profiles.py` fix added 2026-03-17).

**GRPO regression**: F1 drops 0.766 → 0.622, strikethrough collapses to 0. Diagnostic experiments and fix plan in `fmt_grpo_regression.md`.

---

## Pipeline

```
Qwen3-VL-2B                                  Qwen3.5-9B
    |                                              |
    v  OCR SFT (5K olmocr-mix)                     v  OCR SFT (5K olmocr-mix)
ocr_sft ✓ (olmOCR=67.1)                        ocr_lora_qwen35-9b ✓ (bench invalid)
    |                                              |
    v  Fmt SFT (4,750 pages, 5 epochs)             v  Fmt SFT (4,750 pages, 3 epochs)
fmt_sft ✓ (F1=0.766)                           fmt_lora_qwen35-9b ✓ (bench invalid)
    |                                              |
    v  GRPO (formatting reward)                    v  GRPO (edit_sim + fmt_pertype)
fmt_grpo ✓ (F1=0.622, regressed)               fmt_grpo_qwen35-9b ✓ (bench invalid)
```

The 9B pipeline serves as the teacher for KD into Qwen3.5-2B/0.8B students. Plan in `ocr_sft-rl_large-to-small-kd.md`.

---

## Active Problems

### GRPO regression

GRPO regresses formatting F1 after SFT in both model sizes. The 2B is the confirmed case (-19% F1, strikethrough → 0 spans). Three mechanisms identified:

1. `edit_similarity` competes with formatting - dropping markup improves edit distance
2. Reward sparsity for rare types (strikethrough at 15% of GT spans)
3. Recall collapse - GRPO learns "omit when uncertain"

E1-E4 diagnostic experiments target the 2B. The winning reward config will be applied to the 9B GRPO re-run. Until then, `fmt_lora_qwen35-9b` (SFT-only) is the preferred KD teacher candidate.

### 9B bench re-runs needed

`model_profiles.py` applies `enable_thinking: False` for all Qwen3.5 models. Validated on 0.8B and 2B base models. Fine-tuned 9B checkpoints (`ocr_lora_qwen35-9b`, `fmt_lora_qwen35-9b`, `fmt_grpo_qwen35-9b`) are queued for re-run.

Commands (run after `pt_qwen35-9b` confirms clean output):

```bash
sbatch scripts/slurm/run_bench.sh formatting --checkpoint outputs/ocr_lora_qwen35-9b
sbatch scripts/slurm/run_bench.sh formatting --checkpoint outputs/fmt_lora_qwen35-9b
sbatch scripts/slurm/run_bench.sh formatting --checkpoint outputs/fmt_grpo_qwen35-9b
```

---

## Infrastructure

| Script | Purpose |
|--------|---------|
| `scripts/benchmarks/formatting_bench.py` | Per-type F1 (bold/italic/strike), span counts, 250 eval pages |
| `scripts/benchmarks/omnidocbench_bench.py` | OmniDocBench eval (TEDS, CDM, reading order) |
| `scripts/benchmarks/olmocr_bench.py` | olmOCR-Bench eval (7 categories, 7,019 tests) |
| `scripts/slurm/run_bench.sh` | SLURM wrapper with auto-resubmit, supports `--shard` |
| `scripts/data/formatting_generator.py` | Synthetic formatting data generation (Playwright) |

Bench output dirs on cluster (`/scratch/ravi.ka/Forge/outputs/bench/`):

| Dir | Contents |
|-----|----------|
| `formatting/pt_qwen35-0.8b/` | Clean PT baseline (F1=0.258) |
| `formatting/pt_qwen35-2b/` | Clean PT baseline (F1=0.277) |
| `formatting/archive/` | Contaminated 9B runs (fmt_lora, fmt_grpo, ocr_lora, pt_qwen35-9b) |
| `olmocr/archive/` | Contaminated Qwen3.5 olmOCR runs |

---

## Data

| Dataset | Location | Size | Used for |
|---------|----------|------|---------|
| `formatting` | `data/ocr/formatting/` | 4,750 train + 250 eval | SFT + GRPO |
| `formatting-rl` | `data/ocr/formatting-rl/` | same structure | 9B GRPO |
| `olmocr-mix-full` | `data/ocr/olmocr-mix-full/` | 5K (capped) | OCR SFT |
| OmniDocBench | `data/ocr/omnidocbench/` | 1,355 pages | Eval |
| olmOCR-Bench | `data/olmocr-bench/` | 7,019 tests | Eval |

Formatting data per category (10 categories x 475 train, 25 eval): general prose, tables, academic, financial, legal, technical, books, slides, news, forms. Strikethrough is heaviest in technical and financial categories.

---

## IDP Leaderboard

Target: [IDP Leaderboard](https://idp-leaderboard.org/) - unweighted mean of olmOCR-Bench + OmniDocBench + IDP Core Bench.

| Rank | Model | Overall | olmOCR | OmniDoc | IDP Core |
|------|-------|:-------:|:------:|:-------:|:--------:|
| 1 | Nanonets OCR2+ | 81.8 | 82.2 | 89.5 | 73.8 |
| 2 | Gemini-3-Pro | 81.4 | 73.5 | 88.8 | 81.8 |
| 3 | Claude Sonnet 4.6 | 80.8 | 74.4 | 86.9 | 81.2 |
| 6 | GPT-5.2 | 79.2 | 72.2 | 88.0 | 77.4 |
| 10 | Ministral-8B | 69.3 | 57.8 | 78.3 | 71.7 |
| 11 | GLM-OCR | 63.6 | 66.7 | 69.2 | 54.9 |

Our position: `ocr_sft` at 67.1% olmOCR sits between GPT-5.2 and GLM-OCR. No small open-source models (2-3B) on the leaderboard yet - the goal is to get a sub-3B model competitive with the mid-tier API models.
