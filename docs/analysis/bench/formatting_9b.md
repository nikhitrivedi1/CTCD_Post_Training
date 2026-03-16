# Formatting Bench: 9B and 2B Pipeline Comparison

Bench date: 2026-03-15

## Training Configs

All values from saved `outputs/{run}/config.yaml`.

### SFT runs

| | fmt_sft (2B) | fmt_lora_qwen35-9b |
|---|---|---|
| Model | Qwen3-VL-2B-Instruct | Qwen3.5-9B |
| Family | Qwen3-VL | Qwen3.5 |
| data_dir | `formatting` | `formatting` |
| Epochs | **5** | **3** |
| LR | **5e-6** | **2e-6** |
| Batch x GradAcc | 2 x 16 = 32 | 1 x 32 = 32 |
| LoRA r | 16 | 16 |
| Adapter base | `ocr_sft` | `ocr_lora_qwen35-9b` |

Same dataset, same effective batch size. But the 2B got 67% more epochs (5 vs 3) and 2.5x higher LR.

### GRPO runs

| | fmt_grpo (2B) | fmt_grpo_qwen35-9b |
|---|---|---|
| Model | Qwen3-VL-2B-Instruct | Qwen3.5-9B |
| data_dir | **`formatting`** | **`formatting-rl`** |
| Epochs | 1 | 1 |
| LR | **1e-6** | **2e-6** |
| Batch x GradAcc | 1 x 32 = 32 | 2 x 8 = 16 |
| LoRA r | 16 | 16 |
| Reward | `[edit_similarity, inline_formatting]` | `{edit_similarity: 0.25, inline_formatting_pertype: 1.0}` |
| Beta (KL penalty) | **0.0** | **0.1** |
| Temperature | **1.3** | **1.0** |
| num_generations | 4 | 4 |

The two GRPO runs are not directly comparable. They differ in dataset, reward format, beta, temperature, effective batch size, and LR. The 2B used the old reward list format (equal weight), the 9B used the weighted dict format.

### Dataset info

| Dataset | Path | Size | Used by |
|---------|------|------|---------|
| formatting | `data/ocr/formatting/` | 5000 pages (10 cats x 500), 5% eval | SFT (both), GRPO (2B only) |
| formatting-rl | `data/ocr/formatting-rl/` | same structure | GRPO (9B only) |
| olmocr-mix-full | `data/ocr/olmocr-mix-full/` | 5000 (capped) | OCR grounding (both) |

## Scores

### Qwen3.5-9B (3 stages)

| Model | Stage | P | R | F1 |
|-------|-------|---|---|-----|
| `ocr_lora_qwen35-9b` | OCR only | 0.263 | 0.393 | 0.295 |
| `fmt_lora_qwen35-9b` | + fmt SFT | 0.346 | 0.563 | **0.405** |
| `fmt_grpo_qwen35-9b` | + fmt GRPO | 0.321 | 0.496 | 0.365 |

### Qwen3-VL-2B (for comparison)

| Model | Stage | P | R | F1 |
|-------|-------|---|---|-----|
| `ocr_sft` | OCR only | 0.000 | 0.000 | 0.000 |
| `fmt_sft` | + fmt SFT | 0.837 | 0.721 | **0.766** |
| `fmt_grpo` | + fmt GRPO | 0.726 | 0.575 | 0.622 |

### Per-type breakdown (9B)

| Model | Bold F1 | Italic F1 | Strike F1 |
|-------|---------|-----------|-----------|
| `ocr_lora_qwen35-9b` | 0.259 | 0.167 | 0.073 |
| `fmt_lora_qwen35-9b` | 0.364 | 0.248 | **0.222** |
| `fmt_grpo_qwen35-9b` | 0.327 | 0.209 | 0.100 |

### Per-type breakdown (2B)

| Model | Bold F1 | Italic F1 | Strike F1 |
|-------|---------|-----------|-----------|
| `ocr_sft` | 0.000 | 0.000 | 0.000 |
| `fmt_sft` | 0.698 | 0.445 | 0.218 |
| `fmt_grpo` | 0.563 | 0.283 | 0.000 |

### Span counts (9B)

| Model | Bold (gt=2070) | Italic (gt=951) | Strike (gt=756) |
|-------|----------------|-----------------|-----------------|
| `ocr_lora_qwen35-9b` | 3652 | 883 | 74 |
| `fmt_lora_qwen35-9b` | 4095 | 1214 | 279 |
| `fmt_grpo_qwen35-9b` | 4060 | 1116 | 104 |

### Span counts (2B)

| Model | Bold (gt=2070) | Italic (gt=951) | Strike (gt=756) |
|-------|----------------|-----------------|-----------------|
| `ocr_sft` | 0 | 0 | 0 |
| `fmt_sft` | 2048 | 826 | 233 |
| `fmt_grpo` | 2107 | 577 | 0 |

## Analysis

### GRPO regresses formatting in both model sizes

Both sizes regress after GRPO, but the runs are not cleanly comparable:

| Size | SFT F1 | GRPO F1 | Delta | GRPO dataset | GRPO reward format |
|------|--------|---------|-------|--------------|--------------------|
| 9B | 0.405 | 0.365 | -0.040 (-10%) | `formatting-rl` | weighted dict |
| 2B | 0.766 | 0.622 | -0.144 (-19%) | `formatting` (same as SFT) | equal-weight list |

The 2B regression is a pure GRPO effect (same data, same examples, just RL on top). The 9B regression has multiple confounds: different dataset (`formatting-rl`), different reward format, different beta (0.1 vs 0.0), different temperature (1.0 vs 1.3).

### Strikethrough is the primary casualty

GRPO nearly eliminates strikethrough detection:

| Size | SFT strike pred | GRPO strike pred | SFT strike F1 | GRPO strike F1 |
|------|-----------------|------------------|----------------|-----------------|
| 9B | 279 | 104 | 0.222 | 0.100 |
| 2B | 233 | 0 | 0.218 | 0.000 |

The 2B GRPO model produces zero strikethrough spans. The 9B GRPO model still tries (104 preds vs 756 gt) but with low precision.

### Why does GRPO hurt formatting?

The 9B GRPO reward was `edit_similarity * 0.25 + inline_formatting_pertype * 1.0`. The 2B used an older equal-weight list format. Despite formatting having higher weight in the 9B config, the regression still happens. Likely causes:

1. **edit_similarity competes with formatting.** The edit_similarity reward favors plain text output that closely matches the reference. Formatting markup (bold/italic/strike markers) changes the edit distance. The model learns to drop formatting to improve edit_similarity, even though formatting_pertype rewards it.

2. **Strikethrough is rare and hard.** Only 756 gt spans vs 2070 bold and 951 italic. GRPO's on-policy sampling means the model rarely generates strikethrough, gets little reward signal for it, and gradually stops trying. A form of reward sparsity.

3. **Recall drops more than precision.** Both models show recall dropping more than precision after GRPO (9B: R 0.563->0.496, P 0.346->0.321). The model becomes more conservative, predicting fewer spans overall. GRPO may be learning "when in doubt, omit formatting" because false positives are penalized by both rewards while false negatives only hurt formatting_pertype.

4. **9B overpredicts bold across all stages.** Bold preds are ~2x gt (4060-4095 vs 2070) regardless of training stage. SFT and GRPO both fail to fix this, so bold precision stays low (~0.25-0.28). The formatting_pertype reward doesn't distinguish precision vs recall, so GRPO has no signal to reduce false positive bold spans.

### 9B vs 2B: why is 9B so much worse at formatting?

The 2B (Qwen3-VL-2B) SFT hits 0.766 F1, while the 9B (Qwen3.5-9B) only reaches 0.405. This gap is large but has multiple potential causes:

1. **Different model families.** Qwen3-VL-2B vs Qwen3.5-9B are different architectures with different vision encoders and tokenizers. The 2B's vision encoder may handle formatting markup better.

2. **Bold overprediction.** The 9B predicts ~2x too many bold spans (4095 vs 2070 gt), tanking bold precision to 0.280. The 2B keeps bold predictions close to gt (2048 vs 2070), giving 0.724 precision. This single difference accounts for a large portion of the gap.

3. **Training differences.** The 2B SFT got 5 epochs at 5e-6 LR, the 9B got 3 epochs at 2e-6 LR. The 9B saw 40% fewer gradient updates at a lower learning rate. It may simply be underfit on the formatting task.

4. **LoRA capacity.** Both use r=16 LoRA. The 9B model has more parameters but the same LoRA rank, so the adapter is proportionally smaller relative to the base model. May need higher rank for the 9B.

These confounds make it impossible to attribute the gap to model capacity alone. The KD student (Qwen3.5-2B) is from the same family as the 9B, so P0 will tell us whether it's a family-level issue or specific to the 9B config.

### Implications for KD

- **T2 (SFT-only) may be a better teacher than T1 (GRPO)** for formatting KD. T2 has higher F1 (0.405 vs 0.365) and significantly better strikethrough (0.222 vs 0.100). The P1 experiment comparing them is critical.

- **The 2B vs 9B comparison is confounded.** Different families, different epochs, different LR. The Qwen3-VL-2B SFT (0.766) outperforms all Qwen3.5-9B variants, but we cannot conclude that a smaller model is inherently better at formatting. The KD student (Qwen3.5-2B) may score closer to its 9B sibling than to the VL-2B.

- **P0 will resolve the key unknowns.** If Qwen3.5-2B SFT scores low (like the 9B), KD from the 9B teacher has room to help. If it scores high (like VL-2B), the 9B teacher may not have useful knowledge to transfer.

- **If KD from 9B doesn't help, consider self-distillation or fixing the 9B first** (address bold overprediction, try more epochs, higher LoRA rank).
