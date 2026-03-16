# KD: Formatting Distillation

## Goal

Distill formatting knowledge from a strong teacher into a smaller student model. The primary teacher (Qwen3.5-9B GRPO) has been through OCR grounding -> formatting SFT -> formatting GRPO and achieves strong formatting performance. Can we transfer this to a 2B model via KD?

## Students

| ID | Model | Init | Notes |
|----|-------|------|-------|
| S1 | `Qwen/Qwen3.5-2B` | Base (no adapter) | Cold start, isolates KD signal |
| S2 | `Qwen/Qwen3.5-2B` | OCR-grounded (`ocr_lora_qwen35-2b`) | Warm start, has OCR foundation |
| S3 | `Qwen/Qwen3-VL-2B-Instruct` | OCR-grounded (`ocr_sft`) | Old family, cross-family KD |

S1 and S2 are the same model with different initialization. S3 is a different architecture (Qwen3-VL vs Qwen3.5), testing whether KD works across families.

## Teachers

| ID | Model | Checkpoint | Training | Notes |
|----|-------|------------|----------|-------|
| T0 | None | - | - | SFT baseline (no KD) |
| T1 | `Qwen/Qwen3.5-9B` | `fmt_grpo_qwen35-9b` | OCR + fmt SFT + fmt GRPO | Best formatting, primary |
| T2 | `Qwen/Qwen3.5-9B` | `fmt_lora_qwen35-9b` | OCR + fmt SFT (no GRPO) | Isolates GRPO contribution |
| T3 | `Qwen/Qwen3.5-27B` | `fmt_lora_qwen35-27b` | OCR + fmt SFT | Larger teacher, more capacity |

T0 is the no-teacher baseline for each student. Every student gets a T0 run so KD gains are measured fairly.

## Datasets

| Dataset | Path | What it teaches | Use for KD? |
|---------|------|----------------|-------------|
| Formatting SFT | `data/ocr/formatting/` | Inline formatting (bold, italic, strikethrough) | Yes (primary) |
| Formatting RL | `data/ocr/formatting-rl/` | Same domain, RL-quality samples | Maybe (P2) |
| olmocr-mix | `data/ocr/olmocr-mix-full/` | Broad OCR grounding | Maybe (P2) |

Primary dataset: **formatting SFT**. The teacher's GRPO-learned soft labels on this data are the value-add over plain SFT.

## KD Strategy

**Response KL (logit distillation)** is the starting point:
- No projectors or layer mapping needed
- Works across different architectures/vocab sizes (trainer handles alignment)
- Proven in existing olmocr KD config (T=2.0, alpha=0.3)
- Loss: `(1 - alpha) * CE + alpha * T^2 * KL(teacher_soft || student_soft)`

Feature/attention distillation deferred to P2 (requires projectors for dim mismatch).

## Knobs

| Knob | Values | What it tests |
|------|--------|---------------|
| alpha | 0.3, 0.5, 0.7 | Teacher vs ground-truth balance |
| temperature | 2.0, 4.0 | Softness of teacher distribution |
| loss_type | response_kl, response_kl + feature_cosine | Logit-only vs intermediate transfer |
| dataset | formatting, olmocr-mix | Domain-specific vs broad data |
| LR | 2e-6, 5e-6 | Standard vs higher for smaller model |

## Experiment Grid

| Priority | Student | Teacher | Knobs | Question |
|----------|---------|---------|-------|----------|
| P0 | S1 (2B cold) | T0 (none) | - | SFT baseline |
| P0 | S1 (2B cold) | T1 (9B GRPO) | alpha/temp sweep | Does KD work? |
| P1 | S2 (2B OCR-init) | T0 (none) | - | OCR-init SFT baseline |
| P1 | S2 (2B OCR-init) | T1 (9B GRPO) | best P0 knobs | Does student init matter? |
| P1 | S1 (2B cold) | T2 (9B SFT-only) | best P0 knobs | Does GRPO in teacher matter? |
| P2 | S1 (2B cold) | T3 (27B) | best P0 knobs | Does a bigger teacher help? |
| P2 | S3 (VL-2B) | T1 (9B GRPO) | best P0 knobs | Cross-family KD viable? |
| P2 | best S x T | best knobs | + feature_cosine | Intermediate transfer? |
| P2 | best S x T | best knobs | olmocr-mix dataset | Broader data? |

### P0 - Does it work? (5 runs)

Validate the pipeline end-to-end. S1 x T0 is the baseline, S1 x T1 with alpha/temp sweep tests KD.

| Run | Student | Teacher | Alpha | Temp |
|-----|---------|---------|-------|------|
| `fmt_lora_qwen35-2b` | S1 | T0 (none) | - | - |
| `kd_fmt_2b_response-kl` | S1 | T1 (9B GRPO) | 0.5 | 2.0 |
| `kd_fmt_2b_a03` | S1 | T1 (9B GRPO) | 0.3 | 2.0 |
| `kd_fmt_2b_a07` | S1 | T1 (9B GRPO) | 0.7 | 2.0 |
| `kd_fmt_2b_t4` | S1 | T1 (9B GRPO) | 0.5 | 4.0 |

Success criteria: at least one KD run beats the SFT baseline on formatting accuracy.

### P1 - Student init and teacher quality (3 runs)

Using the best alpha/temp from P0.

| Run | Student | Teacher | Question |
|-----|---------|---------|----------|
| `fmt_lora_qwen35-2b_ocr-init` | S2 (OCR-init) | T0 (none) | OCR-init baseline |
| `kd_fmt_2b_ocr-init` | S2 (OCR-init) | T1 (9B GRPO) | Does student init matter? |
| `kd_fmt_2b_sft-teacher` | S1 | T2 (9B SFT-only) | Does GRPO in teacher matter? |

### P2 - Advanced (4 runs)

| Run | What | Question |
|-----|------|----------|
| `kd_fmt_2b_27b-teacher` | S1 x T3 (27B) | Does a bigger teacher help? |
| `kd_fmt_vl2b` | S3 (VL-2B) x T1 | Cross-family KD viable? |
| `kd_fmt_2b_feature` | best + feature_cosine | Does intermediate transfer help? |
| `kd_fmt_2b_olmocr-data` | best on olmocr-mix | Does broader data help? |

## Config Structure

```
configs/ocr/formatting/qwen35-2b/
  base.yaml              # Qwen3.5-2B model + training params (LR=2e-6)
  00_sft.yaml            # P0 baseline: plain SFT
  kd/
    base.yaml            # KD base, inherits qwen35-2b base
    qwen35-9b/           # Teacher: Qwen3.5-9B
      base.yaml          # Sets teacher model + freeze
      01_response-kl.yaml  # P0: response_kl, alpha=0.5, T=2.0
      02_a03.yaml          # P1: alpha=0.3
      03_a07.yaml          # P1: alpha=0.7
      04_t4.yaml           # P1: T=4.0
```

Teacher subdirectory (`kd/qwen35-9b/`) isolates all configs for a given teacher. Adding a second teacher (e.g. 27B) means adding `kd/qwen35-27b/` with its own base + variants.

## Bench Plan

All runs benched on:
- Formatting (same as SFT bench)
- olmOCR (broader OCR quality check)

Compare against:
- 2B SFT baseline (no teacher)
- 2B GRPO (existing RL pipeline, no KD)
- 9B GRPO teacher (upper bound)
