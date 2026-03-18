# Forge: Task Log

Date written: 2026-03-17
Branch at time of writing: `Model-Profiles-plus-Cleanup`

This log covers the full arc of work done in the Forge project - every experiment, dataset, code change, and decision from start to current state - with rationale built into each section.

---

## Phase 0: Classification KD Baseline (late 2025 - Feb 2026)

### What was done

The project started as a study of knowledge distillation (KD) for text classification using BERT/DistilBERT on GLUE tasks (MNLI, QQP, SST-2, CoLA). The goal was to understand when KD helps, when it hurts, and why - as a systematic prerequisite before applying KD to harder tasks.

Four ablations were run across two architectures (BERT-base teacher, DistilBERT student):

| Config | Description | MNLI | QQP (F1) | SST-2 | CoLA (MCC) |
|--------|-------------|:----:|:--------:|:-----:|:----------:|
| 00 | CE-only (student baseline) | 82.19 | 90.26 | 90.94 | 53.80 |
| 01 | Frozen base teacher | 81.58 | 90.44 | 91.51 | 53.39 |
| 02 | Frozen SFT teacher | 79.82 | 55.89 | 90.94 | 51.45 |
| 03 | Co-train base teacher | 82.62 | 90.61 | 91.51 | 51.73 |
| 04 | Co-train SFT teacher | 82.69 | 90.46 | 91.28 | 49.67 |

### Key findings

**Frozen SFT teacher can be catastrophic.** Config 02 on QQP: 90.26% → 55.89% (-34.4%). The SFT teacher (91.6% accurate) was wrong with 82.1% average confidence on its errors. Those high-confidence wrong predictions created irrecoverable gradient conflict: CE said one thing, KL said the opposite. Only 8% of predictions were wrong, but they were wrong loudly. Co-training with the SFT teacher (config 04) avoids this because the teacher updates during training and reduces its confidence on hard examples.

**CoLA hurts even with co-training.** The SFT teacher barely beats CE-only (53.4% vs 53.8% MCC). KD from a teacher that has nothing to offer just adds noise and hurts (-4.1%).

**Teacher underperformance in co-training.** In config 03 (co-train base), the teacher BERT-base reached only 73.25% while the student DistilBERT hit 82.62% - a 9.4pp gap where the student surpassed its teacher. The hyperparameters were tuned for DistilBERT, not BERT. Config 04 confirmed: SFT teacher degraded from 84.57% to 81.66% under student-tuned HPs.

### Why this mattered

The classification KD study produced a prerequisite framework: data quantity, class balance, task difficulty, and teacher quality all determine whether KD helps. This informed how to approach KD in the OCR domain, where similar tradeoffs exist.

Separately: the project was renamed from DistilKit → Forge, and scope expanded from KD-only to SFT + KD + RL.

---

## Phase 1: OCR Foundation (Jan - early Mar 2026)

### OCR infrastructure

Built the OCR training and evaluation stack:
- `src/inference/engine.py` - InferenceEngine for VLM inference, adapter merging
- `scripts/benchmarks/olmocr_bench.py` - olmOCR-Bench eval (7,019 tests, 7 categories)
- SLURM scripts with auto-resubmit on SIGUSR1 (`run_training.sh`, `run_bench.sh`)
- GRPO infrastructure in `forge.py`, `src/training/rewards.py`

**Rationale for olmOCR-Bench as primary eval**: 7,019 verifiable binary tests across 7 document categories. No learned judge or approximate scoring - just pass/fail on text present/absent/order/math. KaTeX DOM comparison for math (not string matching). Directly comparable to the olmOCR 2 paper.

### CE-only fine-tuning (ablations_00 / ocr_sft)

Trained Qwen3-VL-2B on 5K olmocr-mix samples with CE loss only. Config: LoRA r=16, LR=5e-5, 3 epochs, effective batch 32.

| Model | olmOCR Overall | vs base |
|-------|:-------------:|:-------:|
| Qwen3-VL-2B base | 56.1% | - |
| ocr_sft (ablations_00) | 67.1% | +11.0pp |

+11.0pp from 5K samples. Biggest gains: headers_footers (+33.3pp, the model learned to strip metadata), old_scans_math (+15.3pp), table_tests (+10.7pp). Remaining gap to olmOCR-7B (77.3%) = 10.2pp. CE captured 52% of the base-to-7B gap.

**Key insight**: 5K samples on the 2B beats Qwen3-VL-8B base (65.6%) and Qwen2.5-VL-7B base (65.9%) - training signal outweighs 4x the raw parameters.

### Qwen3.5 baselines

Ran olmOCR-Bench on the newly-released Qwen3.5 family (0.8B, 2B, 4B, 9B). Found non-monotonic scaling:

| Model | Overall | Pattern |
|-------|:-------:|---------|
| Qwen3.5-0.8B | 53.3% | - |
| Qwen3.5-2B | 59.9% | peak |
| Qwen3.5-4B | 49.9% | **-10.0pp regression** |
| Qwen3.5-9B | 52.7% | partial recovery |

The 2B→4B regression is broad: long_tiny_text -21.9pp, multi_column -9.9pp. The 2B and 4B share the same ViT depth (24 layers), isolating the regression to the LM side. Likely cause: Qwen3.5's mixed attention architecture (linear + full) trades visual quality for efficiency, and the tradeoff worsens with scale. Qwen3.5-2B is competitive with Qwen3-VL-2B overall (+3.8pp), but beyond 2B, Qwen3-VL pulls ahead by 11-14pp.

### GRPO on synthmix (synthmix_grpo_01)

First GRPO experiment: trained with unit_test rewards on olmOCR2-synthmix (2,186 pages), starting from ablations_00.

| Model | olmOCR Overall |
|-------|:-------------:|
| ablations_00 (CE-only) | 67.1% |
| synthmix_grpo_01 (GRPO) | 66.7% |

Result: essentially flat (-0.4pp). GRPO on extraction did not improve over CE. The synthmix unit tests provide good signal for extraction tasks but the model is already near the ceiling for what 5K-sample SFT can do on this bench. The remaining 10.2pp gap requires either more data, a larger model, or KD from a stronger teacher.

---

## Phase 2: OmniDocBench Evaluation (Mar 2026)

### What was built

Built `scripts/benchmarks/omnidocbench_bench.py`: evaluates models on OmniDocBench's 1,352 pages across four dimensions (text edit distance, table cell F1, formula CDM, reading order LCS). Sharded execution, resume-capable, same InferenceEngine pattern as olmOCR bench.

**Rationale**: OmniDocBench covers formatting quality (tables, formulas, inline markup) - dimensions that olmOCR-Bench doesn't test at all. We need it to know whether formatting training improves or hurts document structure quality.

### First results

| Model | text | table | formula | reading_order | overall |
|-------|:----:|:-----:|:-------:|:-------------:|:-------:|
| ocr_sft | 67.0 | 2.0 | 2.1 | 92.3 | 23.7 |
| fmt_sft | 69.4 | 55.3 | 29.5 | 93.0 | 51.4 |

The near-zero table/formula scores for ocr_sft aren't a perception failure - the olmocr-mix GT (GPT-4o labels) strips formatting, so the model learned plain text output. The backbone already perceives tables and formulas in images; it just needed training data that asked for structured output.

---

## Phase 3: Formatting Dataset and SFT (Mar 2026)

### Synthetic data generation

Built `scripts/data/formatting_generator.py`: generates synthetic document pages with inline formatting. Pipeline: markdown text with bold/italic/strikethrough → HTML+CSS → Playwright render → PNG. Generates both the image and the ground truth markdown.

10 categories, each targeting a different formatting profile:

| Category | Heavy type | Rationale |
|----------|------------|-----------|
| text, academic, books | italic | Emphasis, citations, titles |
| financial, technical | strikethrough | Deprecated values, corrections |
| slides, news | bold | Headlines, key terms |
| tables, forms | mixed | Structure |
| legal | light formatting | Legal text conventions |

500 samples per category, 475 train / 25 eval = 4,750 train + 250 eval total. Stored as `data/ocr/formatting/` - this becomes the primary training dataset for all formatting SFT stages (both 2B and 9B).

**Why synthetic and not real data**: Real documents with formatting annotations at this scale don't exist in open datasets. OmniDocBench has 1,355 annotated pages but the annotations are JSON, not markdown GT. Generating synthetic data gives us exactly the input-output pairs we need (image → formatted markdown) at any scale with controllable type distribution.

**Strikethrough is the rarest type** (~15% of GT spans, vs 55% bold, 30% italic). This imbalance later caused problems in GRPO.

### Formatting-RL dataset (data/ocr/formatting-rl/)

A second pass of the same generator, same 10-category structure, separate random sample. Stored as `data/ocr/formatting-rl/`.

**Rationale**: Created for the 9B GRPO stage specifically. The 2B GRPO (v1) trained RL on the same `formatting` data used for SFT - meaning the policy had already been trained on those exact examples via CE. Reusing SFT data for RL can inflate reward estimates and bias the advantage computation, since the model's SFT priors are tuned to that exact distribution. A fresh draw from the same generator gives the RL stage a distribution it hasn't memorized. Using separate data for SFT and GRPO is also cleaner experimentally: if the 9B GRPO regresses, it's harder to attribute to data overlap.

Note: this also means the 2B and 9B GRPO runs are not directly comparable - one trained on `formatting`, the other on `formatting-rl`. This is one of the confounds documented in `formatting_9b.md`.

### Formatting SFT (fmt_sft)

Trained on the 4,750 formatting pages, starting from ocr_sft. Config: LoRA r=16, LR=5e-6, 5 epochs, effective batch 32.

**Results:**

| Model | F1 | Bold F1 | Italic F1 | Strike F1 | olmOCR | OmniDoc |
|-------|:--:|:-------:|:---------:|:---------:|:------:|:-------:|
| ocr_sft | 0.000 | 0.000 | 0.000 | 0.000 | 67.1% | 23.7 |
| fmt_sft | 0.766 | 0.698 | 0.445 | 0.218 | 66.3% | 51.4 |

Formatting went from zero to 0.766 F1 in one training stage. OmniDoc nearly doubled (+27.7pp). olmOCR regressed -0.8pp (the model is slightly worse at suppressing headers/footers and raw text extraction). The tradeoff is acceptable: the model gained structured output capability while keeping 98.8% of its extraction accuracy.

**Span counts vs GT** (GT: bold=2,070, italic=951, strike=756):
- Bold: 2,048 predicted - nearly calibrated
- Italic: 826 predicted - 87% recall by count
- Strike: 233 predicted - 31% recall by count (the weak link)

Recall is the bottleneck overall (0.721 vs precision 0.837). The model is conservative: when it does predict formatting, it's usually right, but it misses many spans, especially strikethrough.

---

## Phase 4: Formatting GRPO (Mar 2026)

### GRPO v1 (fmt_grpo) - regression

Ran GRPO starting from fmt_sft on the same 4,750 formatting pages. Rewards: `inline_formatting` (pooled across types) + `edit_similarity`, equal weights. beta=0.0, temperature=1.3.

**Results:**

| Model | F1 | Bold F1 | Italic F1 | Strike F1 |
|-------|:--:|:-------:|:---------:|:---------:|
| fmt_sft | 0.766 | 0.698 | 0.445 | 0.218 |
| fmt_grpo | 0.622 | 0.563 | 0.283 | 0.000 |

-0.144 F1 regression. Strikethrough collapsed to zero. Three root causes identified and confirmed from training logs:

**1. Pooled reward hid type imbalance.** Bold is 55% of GT spans. A pooled F1 reward maximizes by focusing on bold. The model over-produced bold, ignored italic, and stopped predicting strikethrough entirely. Bold span count: 2,048 → 2,107 (more); italic: 826 → 577 (-30%); strikethrough: 233 → 0.

**2. beta=0.0 caused entropy collapse.** Without KL regularization, the policy became near-deterministic: entropy fell from ~1.0 to 0.059. GRPO shed the diversity that SFT had learned. With no constraint pulling back toward the reference distribution, the model converged on "bold everything, ignore the rest."

**3. edit_similarity competed with formatting.** The model found it easier to improve edit distance (text accuracy) than formatting. Adding italic or strikethrough markers risks edit_similarity losses when spans don't match GT text exactly. The reward pushed the model away from speculative formatting even though formatting was weighted 3x higher.

**The three causes formed a feedback loop**: pooled reward makes bold the highest-return strategy → beta=0.0 lets the policy collapse onto that strategy → edit_similarity reinforces the conservative equilibrium.

### GRPO v2 design (01_grpo.yaml) - stalled

Designed and started training a v2 config addressing all three root causes:

| Parameter | v1 | v2 | Fix |
|-----------|----|----|-----|
| Reward | pooled `inline_formatting` | per-type `inline_formatting_pertype` | Each type contributes equally regardless of span count |
| edit_sim weight | 1.0 | 0.25 | Reduced to a light guardrail (1:4 ratio) |
| beta | 0.0 | 0.1 | Strong KL regularization prevents collapse |
| temperature | 1.3 | 1.0 | Was compensating for beta=0, no longer needed |

Training stalled at step 270/594: entropy already at 0.048, formatting reward barely moving (0.304 → 0.317). The 2B model may lack the capacity to explore diverse formatting strategies through RL, even with the improved reward. This motivated the "RL on large, distill into small" approach.

---

## Phase 5: KD Strategy Shift (Mar 2026)

### Motivation

Two GRPO runs on the 2B had failed. The model either regressed (v1) or stalled (v2). DeepSeek R1 provided a template: train RL on a large model first, then distill into small via SFT on teacher outputs. Small models can't discover complex behaviors through RL exploration, but they can learn them through imitation once a larger model has figured them out.

**For formatting specifically**: maintaining bold, italic, and strikethrough simultaneously requires the model to track multiple pattern types without dropping any. Under RL reward pressure, 2B collapses to the dominant type (bold). A larger model has more representational headroom.

### Model selection

Chose Qwen3.5-9B as the teacher for two reasons:
- Newest generation (Feb-Mar 2026), natively multimodal (no separate VL line)
- Fits on 1x H200 for SFT, 2x H200 for GRPO

Chose Qwen3.5-2B as the primary student:
- Same architecture family as the 9B teacher - representations, tokenizer, and attention mechanism match
- Natively multimodal
- Better OCR baseline than Qwen3-VL-2B at this scale (+3.8pp olmOCR before any training)

### Chain orchestrator

Built `scripts/slurm/run_chain.sh`: SLURM job chaining with a YAML registry file. Submits bench and scoring jobs automatically after training completes, without manual sbatch calls. Handles queue-full states with retry logic, passes env vars between stages (CHECKPOINT, MODEL, OUTPUT, SHARD).

**Rationale**: The 9B pipeline has 3 training stages + bench jobs after each stage. Without automation, each requires manual monitoring and resubmission. The chain file defines the full pipeline declaratively.

---

## Phase 6: Qwen3.5-9B Training Pipeline (Mar 2026)

### Stage 1: OCR SFT (ocr_lora_qwen35-9b)

Trained Qwen3.5-9B on 5K olmocr-mix samples. Same structure as ocr_sft but for the larger model. Config: LoRA r=16, LR adapted for 9B.

Status: **Done (training)**. Bench results: **invalid** (see Phase 7).

### Stage 2: Formatting SFT (fmt_lora_qwen35-9b)

Trained on 4,750 formatting pages starting from ocr_lora_qwen35-9b. Config: LoRA r=16, 3 epochs, LR=2e-6, effective batch 32.

Status: **Done (training)**. Bench results: **invalid** (see Phase 7).

The 9B got 3 epochs at 2e-6 LR vs the 2B's 5 epochs at 5e-6 LR. Whether the 9B underfit on formatting will only be clear once valid bench results are available.

### Stage 3: Formatting GRPO (fmt_grpo_qwen35-9b)

Trained with improved reward config: `{edit_similarity: 0.25, inline_formatting_pertype: 1.0}`, beta=0.1, temp=1.0. Used `formatting-rl` dataset (see Phase 3 for rationale - separate draw from the same generator, not seen during SFT).

Status: **Done (training)**. Bench results: **invalid** (see Phase 7).

Span count data from the contaminated bench runs still tells a partial story: strikethrough at 104 predicted vs 279 from SFT - the same regression pattern as the 2B, just less severe (beta=0.1 vs 0.0 probably helped). Whether the improved reward config (edit_sim reduced, per-type reward) produces valid improvements over SFT will only be confirmed after re-runs.

---

## Phase 7: `</think>` Contamination Discovery and Fix (2026-03-17)

### What went wrong

All three 9B bench jobs (run on 2026-03-14) produced invalid results. Generated markdown files contained thinking content and a bare `</think>` tag before the actual document content:

```
[several lines of thinking content]
</think>

**actual** OCR output here
```

Contamination confirmed by grepping generated files: ocr_lora_qwen35-9b (187/250), fmt_lora_qwen35-9b (186/250), fmt_grpo_qwen35-9b (172/250).

### Root cause

Qwen3.5 tokenizers register `<think>` as a special token. `skip_special_tokens=True` strips it. But `</think>` is plain text and survives in the decoded output. The `enable_thinking: False` chat template flag (which controls what the template prepends to the assistant turn) was not in place when the 2026-03-14 jobs ran.

**Why `</think>` stripping in engine.py was rejected**: An early fix post-hoc stripped `</think>` from generated strings. This was dropped because: (1) fragile - any `</think>` in legitimate document content would be stripped, (2) treats a symptom rather than the cause. The model should never generate thinking in the first place.

### Fix: model_profiles.py

Built `src/inference/model_profiles.py`: matches the model name from `config.yaml` against `MODEL_PROFILES = {"Qwen3.5": {"enable_thinking": False}}`. Applies `chat_template_kwargs: {enable_thinking: False}` during inference, suppressing the thinking prefix via the chat template. This operates at generation setup time, not post-hoc.

**Validated** on pt_qwen35-0.8b (F1=0.258) and pt_qwen35-2b (F1=0.277) - both clean, no `</think>` tags. Fine-tuned 9B checkpoints haven't been re-run yet; we don't know whether they generate thinking anyway despite the flag (model-level behavior vs template-level hint).

### Cluster cleanup

- Moved all contaminated bench dirs to `archive/` subdirs in both `formatting/` and `olmocr/`
- Renamed new clean runs from `Qwen3.5-0.8B/20260317-1116Z/` → `pt_qwen35-0.8b/` (flat, no timestamp, consistent with naming convention)

---

## Phase 8: Analysis Doc Updates and Cleanup (2026-03-17)

### Docs marked invalid

- `formatting_9b.md`: top-level INVALID notice + strikethrough on all 9B table values. 2B rows untouched.
- `olmocr.md`: NOTE blockquote above results table + struck-through Q3.5-* column headers and all cell values.
- `fmt_grpo_regression.md`: struck-through 9B rows in problem table and reference baseline.

### Docs updated with current state

- **`ocr_sft-rl_large-to-small-kd.md`**: Added Status (2026-03-17) section: phase completion table, pt_qwen35-0.8b/2b clean baselines. Updated Student Model section: Qwen3.5-2B promoted to primary target (same family as 9B teacher), with two-stage student pipeline diagram.

- **`fmt_grpo_regression.md`**: Added Connection to KD Pipeline section. E1-E4 experiments on 2B determine which 9B checkpoint becomes the teacher:
  - GRPO recovers → re-run `fmt_grpo_qwen35-9b` with winning config, use as Phase 3 teacher
  - GRPO still regresses → use `fmt_lora_qwen35-9b` (SFT-only) as teacher

### Docs renamed / replaced

- `fmt_grpo_reward_experiments.md` → `fmt_grpo_regression.md` (name reflects purpose: diagnosing and fixing a regression)
- `formatting_bench_plan.md` (600-line 2026-03-06 planning doc with stale state) → `formatting_pipeline.md` (100-line current-state doc: results table, pipeline diagram, active problems, infra reference, data table, IDP Leaderboard)

---

## Current State (2026-03-17)

### Results

| Model | Stage | F1 | olmOCR | OmniDoc |
|-------|-------|----|:------:|:-------:|
| Qwen3-VL-2B base | - | 0.000 | 56.1% | - |
| ocr_sft | OCR SFT | 0.000 | 67.1% | 23.7 |
| fmt_sft | + Fmt SFT | 0.766 | 66.3% | 51.4 |
| fmt_grpo | + GRPO | 0.622 | - | - |
| Qwen3.5-9B pipeline | all stages | **INVALID** | - | - |
| pt_qwen35-0.8b | base | 0.258 | - | - |
| pt_qwen35-2b | base | 0.277 | - | - |

### Active problems

1. **GRPO regression**: GRPO consistently regresses formatting F1 post-SFT. Diagnosed (3 root causes). E1-E4 diagnostic experiments in `fmt_grpo_regression.md` are the next step. E1 (remove edit_sim entirely) is P0.

2. **9B bench re-runs needed**: model_profiles.py fix in place. Validated on base models. Fine-tuned checkpoints queued; run after pt_qwen35-9b comes back clean.

### Next steps (in priority order)

1. **E1-E4** on 2B (Qwen3-VL): fast iteration, 30 min per run on H200
2. **pt_qwen35-9b bench** (job 5140618): confirm base 9B is clean
3. **Fine-tuned 9B bench re-runs**: after pt_qwen35-9b confirms clean
4. **Qwen3.5-2B student pipeline**: `ocr_sft_qwen35-2b` can start now, independent of E1-E4 and 9B re-runs
5. **Teacher decision** (after E1-E4 + valid 9B bench): pick teacher, generate outputs, run KD

---

## Infrastructure Built

| Component | Location | Purpose |
|-----------|----------|---------|
| Training (SFT + GRPO) | `forge.py`, `src/training/` | End-to-end training with adapter merging |
| InferenceEngine | `src/inference/engine.py` | VLM inference, handles LoRA adapters |
| Model profiles | `src/inference/model_profiles.py` | Per-family generation settings (e.g., `enable_thinking: False`) |
| Reward functions | `src/training/rewards.py` | edit_similarity, inline_formatting, inline_formatting_pertype, unit_test, table_teds, formula_cdm, reading_order |
| RL dataset loaders | `src/data/rl_datasets.py` | formatting, synthmix, omnidocbench |
| olmOCR bench | `scripts/benchmarks/olmocr_bench.py` | 7,019 binary tests, KaTeX math scoring |
| OmniDocBench | `scripts/benchmarks/omnidocbench_bench.py` | text/table/formula/reading_order |
| Formatting bench | `scripts/benchmarks/formatting_bench.py` | Per-type F1 + span counts |
| Formatting generator | `scripts/data/formatting_generator.py` | Synthetic doc pages with inline formatting |
| SLURM training | `scripts/slurm/run_training.sh` | Auto-resubmit on SIGUSR1 |
| SLURM bench | `scripts/slurm/run_bench.sh` | Bench with sharding + auto-resubmit |
| Chain orchestrator | `scripts/slurm/run_chain.sh` | YAML-defined multi-stage job pipelines |
| Model lineage | `run_metadata.json` | Tracks base_model + adapter chain per run |
