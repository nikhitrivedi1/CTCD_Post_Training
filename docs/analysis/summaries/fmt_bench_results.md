# Formatting Bench Results & GRPO Regression Analysis (2026-03-11)

Inline formatting preservation benchmark on 250 synthetic pages. Measures bold, italic, and strikethrough reproduction in generated markdown. See [formatting.md](../bench/formatting.md) for methodology and raw results.

## Training Pipeline

```
Qwen3-VL-2B-Instruct
    |
    v  SFT on 5K olmocr-mix (CE-only, no formatting in GT)
ocr_sft (ablations_00)
    |
    v  SFT on 250 formatting-rich synthetic pages
fmt_sft
    |
    v  GRPO with inline_formatting + edit_similarity rewards
fmt_grpo (00_grpo)
    |
    v  GRPO v2 with per-type formatting reward + KL regularization
fmt_grpo_v2 (01_grpo) [planned]
```

## Results Summary

| Model | Bold F1 | Italic F1 | Strike F1 | Overall F1 |
|-------|:-------:|:---------:|:---------:|:----------:|
| ocr_sft | 0.000 | 0.000 | 0.000 | 0.000 |
| fmt_sft | 0.698 | 0.445 | 0.218 | 0.766 |
| fmt_grpo | 0.563 | 0.283 | 0.000 | 0.623 |
| **Delta (grpo - sft)** | **-0.135** | **-0.162** | **-0.218** | **-0.143** |

Span counts (GT: 2,070 bold, 951 italic, 756 strikethrough):

| Model | Bold | Italic | Strike |
|-------|:----:|:------:|:------:|
| fmt_sft | 2,048 | 826 | 233 |
| fmt_grpo | 2,107 | 577 | 0 |

fmt_grpo over-predicted bold (+59 vs GT), under-predicted italic (-374 vs GT), and produced zero strikethrough.

## GRPO Regression: Root Cause Analysis

fmt_grpo regressed -0.143 overall F1 from its own SFT base. Three problems compounded.

### 1. Pooled reward is blind to per-type balance

`_score_inline_formatting` in rewards.py pools all span types into one set before computing F1:

```python
all_gt = gt["bold"] | gt["italic"] | gt["strikethrough"]
all_pred = pred["bold"] | pred["italic"] | pred["strikethrough"]
return _fuzzy_set_match(all_pred, all_gt)["f1"]
```

Bold is 55% of GT spans (2,070 / 3,777). The model maximizes reward by focusing on bold alone. Italic and strikethrough contribute less to the pooled score, so the model has weak incentive to produce them. Over training, italic atrophied (826 -> 577 spans) and strikethrough collapsed entirely (233 -> 0).

### 2. beta=0.0 caused entropy collapse

The 00_grpo config set `beta: 0.0`, removing KL penalty entirely. Without any regularization pulling the policy back toward the reference distribution, entropy collapsed:

- Start of training: entropy ~1.0
- End of training: entropy ~0.059

The policy became near-deterministic. fmt_sft had learned diverse formatting patterns for different span types, but GRPO without KL regularization shed that diversity. The model converged on a narrow strategy: bold everything, ignore the rest.

### 3. edit_similarity competed with formatting

Two rewards with weights `[1.0, 3.0]`:

| Reward | Start | End | Delta |
|--------|:-----:|:---:|:-----:|
| edit_similarity (ocr_reward) | 0.875 | 0.961 | +0.086 |
| inline_formatting | 0.584 | 0.617 | +0.033 |

The model found it easier to optimize edit similarity (text accuracy) than formatting. Adding italic or strikethrough markers risks edit_similarity losses when the spans don't exactly match GT text, so the gradient signal pushed the model away from speculative formatting. The 3x weight on formatting wasn't enough to overcome this.

### How the three causes interacted

The causes are not independent. They formed a feedback loop:

1. Pooled reward makes bold the highest-return strategy
2. beta=0.0 allows the policy to collapse onto that strategy without resistance
3. edit_similarity penalizes formatting attempts that don't match, reinforcing the conservative (bold-only) equilibrium

Any one fix alone might not be sufficient. Bold dominance is harmless if the policy maintains diversity (beta > 0). Competing objectives are manageable if the reward balances all types. Entropy collapse is avoidable even without per-type rewards if beta provides enough regularization.

But with all three present, the model had every incentive to narrow and no mechanism to prevent it.

## v2 Design: 01_grpo.yaml

### Changes from 00_grpo

| Parameter | 00_grpo | 01_grpo | Rationale |
|-----------|---------|---------|-----------|
| Formatting reward | `inline_formatting` (pooled) | `inline_formatting_pertype` (per-type avg) | Fixes cause #1 |
| Formatting weight | 3.0 | 1.0 | Per-type reward is already balanced, less weight needed |
| edit_similarity weight | 1.0 | 0.25 | Fixes cause #3, reduced to light guardrail (1:4 ratio) |
| beta | 0.0 | 0.1 | Fixes cause #2, strong KL regularization |
| temperature | 1.3 | 1.0 | High temp was compensating for beta=0, no longer needed |

### Per-type formatting reward

New `_score_inline_formatting_pertype` in rewards.py scores each type independently:

```python
type_scores = []
for fmt_type in ["bold", "italic", "strikethrough"]:
    if gt[fmt_type]:  # skip types absent from GT
        f1 = fuzzy_set_match(pred[fmt_type], gt[fmt_type])["f1"]
        type_scores.append(f1)
return sum(type_scores) / len(type_scores)
```

Key properties:
- Bold, italic, strikethrough each contribute equally to the reward regardless of span count
- Types absent from GT are excluded (no penalty for missing strikethrough on a page with only bold)
- Reward scale is lower on average than pooled (italic ~0.45, strike ~0.22 pull down the mean), meaning smaller advantages and potentially slower but more balanced learning

### Why keep edit_similarity at 0.25

The formatting reward only checks formatting spans, not text quality. Without any text quality signal, the model could spray formatting markers while producing garbled text. beta=0.1 provides some implicit quality regularization (pulling toward the SFT reference), but it operates at the token distribution level, not output quality.

Keeping edit_similarity at 0.25 provides a light quality guardrail. At a 1:4 ratio with formatting, it's weak enough that formatting clearly drives learning, but present enough to catch text quality degradation. Note that reward weights are absolute multipliers, not just ratios - the low absolute scale (total reward ~0.0-1.25) keeps beta=0.1 meaningfully strong relative to the reward signal.

If v2 shows stable text quality, a future v3 could drop edit_similarity to 0.0 entirely.

### Why beta=0.1

Common range in the literature: 0.01-0.1. 0.1 is at the upper end, chosen because:

- v1 entropy collapsed from 1.0 to 0.059 with beta=0.0
- The formatting task requires maintaining diverse patterns across types
- 0.1 provides strong pull back toward the SFT reference distribution
- If learning is too slow, beta can be reduced. Starting high is safer than starting low.

### Why temperature 1.0

The 1.3 temperature in 00_grpo was compensating for beta=0.0. Without KL regularization, the only source of generation diversity was high temperature. With beta=0.1, the KL penalty naturally maintains diversity by penalizing policy divergence. Standard temperature (1.0) should provide sufficient variety across 4 generations per prompt.

## What to watch during v2 training

**Entropy trajectory.** Should stay above 0.3-0.5 throughout training (vs the 1.0 -> 0.059 collapse in v1). If it drops below 0.2, beta may need to increase.

**Per-reward trajectories.** Both rewards should increase or at least not regress. If edit_similarity stalls while formatting climbs, the balance is right. If edit_similarity drops, the model may be sacrificing text quality for formatting.

**Completion spot-checks.** Sample completions mid-training to verify:
- All three formatting types appear (not just bold)
- Text quality hasn't degraded
- Formatting spans align with visual content (not hallucinated)

**Span count distribution.** Track predicted span counts per type across training. In v1, italic dropped from 826 to 577 and strikethrough from 233 to 0. In v2, all three should remain stable or increase.

## Future directions

**v3 config options** (contingent on v2 results):

| Lever | Direction | When |
|-------|-----------|------|
| edit_similarity | 0.25 -> 0.0 | If text quality stays stable in v2 |
| beta | 0.1 -> 0.15 | If entropy still decays too fast |
| beta | 0.1 -> 0.05 | If learning is too slow (reward barely moves) |
| num_generations | 4 -> 8 | If advantage estimates are too noisy |
| epochs | 1 -> 2 | If reward is still climbing at end of epoch |

**Per-type reward logging.** Currently the reward function returns a single scalar (average of per-type F1s). Adding per-type logging to W&B (bold_f1, italic_f1, strike_f1 as separate metrics) would make it easier to diagnose which types are improving and which are stalling. This would require changes to the reward callback, not just the reward function.

**Weighted per-type averaging.** Equal weights (1/3 each) may not be optimal. Strikethrough is the hardest type (F1=0.218 in SFT) and rarest in training data. Upweighting it in the reward would push the model harder on the weakest type, but could also destabilize learning if strikethrough detection is genuinely harder for the model. Start equal, adjust based on v2 results.
