# Formatting GRPO: Regression Diagnosis and Fix

Date: 2026-03-16

## Problem

GRPO consistently regresses formatting quality after SFT in both model sizes:

| Size | SFT F1 | GRPO F1 | Delta |
|------|--------|---------|-------|
| 2B (Qwen3-VL) | 0.766 | 0.622 | -0.144 (-19%) |
| 9B (Qwen3.5) | ~~0.405~~ | ~~0.365~~ | ~~-0.040 (-10%)~~ - **INVALID** |

Strikethrough collapses entirely in the 2B (233 → 0 predicted spans). The 9B numbers are contaminated by `</think>` leakage and cannot be used. The 2B regression is the confirmed signal; E1-E4 target that.

## Root Causes to Isolate

Three distinct mechanisms are conflated in the current setup:

1. **edit_similarity competes with formatting.** Dropping markup improves edit distance to the reference. The model can simultaneously satisfy edit_similarity while violating formatting_pertype. Even with formatting weighted 4x higher (9B config: `{edit_sim: 0.25, fmt_pertype: 1.0}`), the regression persists.

2. **Reward sparsity for rare types.** Strikethrough is only 15% of GT spans (756 vs 2070 bold, 951 italic). On-policy GRPO rarely samples strikethrough, so the reward function has almost no gradient for it. The model gradually stops producing it.

3. **Recall collapse.** GRPO learns "omit when uncertain" because false positives hurt both rewards (edit_sim + formatting), while false negatives only hurt formatting_pertype. The result is conservative predictions with recall falling more than precision.

## Experiments

Run on 2B (faster to train), 500-example stratified subset, 1 epoch each. Run sequentially - each informs the next.

### E1: Remove edit_similarity (P0)

**Config**: `reward: {fmt_pertype: 1.0}` only. Everything else identical to current 2B GRPO.

**Tests**: Is edit_sim the primary driver of regression?

If formatting F1 recovers, edit_sim is the culprit and all subsequent experiments should exclude it. If regression persists, the problem is in the reward signal itself or the training dynamics.

---

### E2: Decompose P/R in the reward (P0, run after E1)

**Config**: Replace scalar F1 reward with recall-weighted per-type reward: `{recall: 1.0, precision: 0.5}`. Use the best reward structure from E1.

**Tests**: Is the recall collapse caused by symmetric F1 penalizing false positives equally to false negatives?

Track bold/italic/strike span counts every 50 steps. Span count recovery (predicted approaching GT count) confirms the hypothesis within the first 100 steps.

---

### E3: Upsample strikethrough in RL data (P1)

**Config**: Build 500-example subset with 33% each bold/italic/strike (natural distribution is ~15% strike). Use the best reward from E1/E2.

**Tests**: Is strikethrough collapse caused by reward sparsity from rare examples, or by a more fundamental reward failure?

If strike F1 recovers with forced exposure, the fix for the full run is rebalancing `formatting-rl`. If it doesn't recover despite exposure, the reward function itself doesn't provide sufficient learning signal for strikethrough.

---

### E4: KL beta sweep (P1)

**Configs**: beta = {0.0, 0.05, 0.1} using the best reward from E1/E2.

**Tests**: Does KL regularization prevent collapse by anchoring to the SFT checkpoint?

The 2B used beta=0.0 (no constraint), allowing unconstrained drift. Small beta may preserve SFT's formatting behavior while still improving on the reward signal. Compare final F1 and per-type span counts across beta values.

---

## Monitoring Protocol

Do not rely on end-of-epoch evaluation. Log every 50 steps:

| Metric | Why |
|--------|-----|
| Span survival rate per type: `predicted / gt` | Detects strikethrough collapse early (usually within 200 steps) |
| Per-type F1 (bold, italic, strike separately) | Aggregates mask which type is failing |
| Individual reward component values | Reveals which signal the model is actually maximizing |
| KL divergence from SFT | Tracks how far GRPO is drifting from the formatting-capable checkpoint |

**Early stop condition**: if strike predicted spans drop to 0 at any checkpoint, kill the run. That's a diagnostic signal, not a reason to keep training.

## Dataset Setup

For all experiments: 500-example stratified sample from `formatting`, with at minimum 100 strikethrough examples. Keeps each run under 30 minutes on H200.

For E3 specifically: build a rebalanced subset with strikethrough oversampled to 33% of examples (vs natural ~15%).

## Eval Bench: Stratified Mode (Default)

The default bench runs all 250 eval pages (~20 min on H200). For fast diagnostic iteration, pass `--max_pages 60` to get a stratified 60-page sample (seed 42, deterministic, ~5-8 min).

```bash
# Full bench (default) - use for final comparison
python scripts/benchmarks/formatting_bench.py --checkpoint outputs/<run>

# Stratified sample - use during diagnostic experiments
python scripts/benchmarks/formatting_bench.py --checkpoint outputs/<run> --max_pages 60
```

**Comparability note**: Stratified scores will diverge slightly from full-bench scores (different page mix). Only compare like-for-like: stratified vs stratified, full vs full. Use `--max_pages 60` for E1-E4 iteration; run without it once you have a winning config.

## Experiment Ordering

```
E1 (edit_sim ablation)
  |
  +-- if regression fixed --> E2, E3, E4 can run in parallel on edit_sim-free reward
  |
  +-- if regression persists --> E2 (recall decomposition) is next priority
```

## Success Criteria

A reward configuration "works" if:
- Overall F1 does not regress from SFT baseline (0.766 for 2B)
- Strike predicted span count stays above 100 (vs 233 from SFT, 0 from current GRPO)
- Recall stays within 5pp of SFT recall (0.721)

## Connection to KD Pipeline

E1-E4 run on the 2B (Qwen3-VL) for speed, but the conclusions feed directly into the 9B teacher pipeline for KD.

### Why this matters for the teacher

The 9B training pipeline is:
```
ocr_lora_qwen35-9b  (OCR SFT)
  -> fmt_lora_qwen35-9b  (formatting SFT, the Phase 1 teacher)
  -> fmt_grpo_qwen35-9b  (formatting GRPO, the Phase 2 teacher)
```

The current `fmt_grpo_qwen35-9b` shows the same regression pattern as the 2B: contaminated bench results aside, the span count data still shows strikethrough at 104 predicted vs 279 from SFT. GRPO hurt the 9B teacher the same way it hurt the 2B student.

### How E1-E4 determine the teacher choice

| E1-E4 outcome | 9B teacher decision |
|---------------|---------------------|
| GRPO recovers with fixed reward | Re-run 9B GRPO with E1-E4 winning config; use `fmt_grpo_qwen35-9b` (re-run) as Phase 2 teacher |
| GRPO still regresses | Skip 9B GRPO re-run; use `fmt_lora_qwen35-9b` as the teacher (SFT-only is better) |

### Applying the winning config to 9B

If E1-E4 find a working reward config, the 9B GRPO re-run needs one config change: replace `reward: [edit_similarity, inline_formatting]` with the E1-E4 winning spec. Everything else (LR=2e-6, beta=0.1, temp=1.0, num_generations=4) stays.

The 9B used beta=0.1 (unlike 2B's 0.0), which may already partially prevent collapse. E4 (KL beta sweep) on 2B will tell us whether beta alone is sufficient, or whether the reward fix is the primary lever.

---

## Reference Baseline

From `formatting_9b.md` (2026-03-15):

| Model | P | R | F1 | Bold F1 | Italic F1 | Strike F1 |
|-------|---|---|----|---------|-----------|-----------|
| fmt_sft (2B) | 0.837 | 0.721 | 0.766 | 0.698 | 0.445 | 0.218 |
| fmt_grpo (2B) | 0.726 | 0.575 | 0.622 | 0.563 | 0.283 | 0.000 |
| fmt_lora_qwen35-9b | ~~0.346~~ | ~~0.563~~ | ~~0.405~~ | ~~0.364~~ | ~~0.248~~ | ~~0.222~~ |
| fmt_grpo_qwen35-9b | ~~0.321~~ | ~~0.496~~ | ~~0.365~~ | ~~0.327~~ | ~~0.209~~ | ~~0.100~~ |

The Qwen3-VL-2B rows (fmt_sft, fmt_grpo) are valid. The Qwen3.5-9B rows are contaminated by `</think>` leakage (172-187/250 generated files affected) and must be re-run. Use only the 2B numbers as baselines for E1-E4.
