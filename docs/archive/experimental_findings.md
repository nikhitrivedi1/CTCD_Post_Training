# Experimental Findings

Observations and insights discovered during knowledge distillation experiments.

---

## Base Model Accuracies

**Date:** 2026-01-30

Zero-shot evaluation on MNLI validation_matched (n=9815):

| Model | Role | MNLI Accuracy | Notes |
|-------|------|---------------|-------|
| distilbert-base-uncased | Student | **32.11%** | Random classifier init |
| bert-base-uncased | Teacher (base) | **36.07%** | Random classifier init |
| textattack/bert-base-uncased-MNLI | Teacher (SFT) | **84.57%** | Fine-tuned on MNLI |

Zero-shot evaluation on QQP validation (n=40430):

| Model | Role | QQP Accuracy | Notes |
|-------|------|--------------|-------|
| distilbert-base-uncased | Student | **36.82%** | Random classifier init |
| bert-base-uncased | Teacher (base) | **37.68%** | Random classifier init |
| textattack/bert-base-uncased-QQP | Teacher (SFT) | **90.91%** | Fine-tuned on QQP |

**Note:** QQP baseline (~37%) is below 50% due to class imbalance - more "not duplicate" (0) than "duplicate" (1) examples. Models with random classifiers tend to predict the majority class.

---

## Final Ablation Results

**Date:** 2026-01-30

| Rank | Config | Val Accuracy | Δ from Student Baseline | Description |
|------|--------|--------------|------------------------|-------------|
| 1 | 04 | **82.62%** | +50.51% | Co-train SFT teacher |
| 2 | 03 | 82.06% | +49.95% | Co-train base teacher |
| 3 | 01 | 81.25% | +49.14% | Frozen base teacher |
| 4 | 00 | 80.94% | +48.83% | No teacher (CE only) |
| 5 | 02 | 79.26% | +47.15% | Frozen SFT teacher |
| 6 | 05 | 31.61% | -0.50% | KL only (no CE) |

**The full training journey:**
```
Student baseline:     32.11% (random)
    ↓ +48.83% with CE-only training
No teacher (00):      80.94%
    ↓ +1.68% with co-train SFT
Best config (04):     82.62%
```

---

## Co-training Robustness to Label Misalignment

**Date:** 2026-01-30

### Background

The TextAttack BERT-MNLI teacher uses a different label ordering than HuggingFace's MNLI dataset:

| Position | TextAttack | HuggingFace |
|----------|------------|-------------|
| 0 | contradiction | entailment |
| 1 | entailment | neutral |
| 2 | neutral | contradiction |

Without remapping (`teacher_label_map: [1, 2, 0]`), the teacher's soft labels are completely misaligned with the ground truth.

### Observation

In runs **without** the label mapping fix:
- **Config 04 (co-training SFT teacher)** achieved the highest accuracy
- Frozen teacher configs performed poorly

This was unexpected since the teacher's soft labels were "wrong."

### Explanation

Co-training (`freeze_teacher: false`) makes the teacher **self-correcting**:

1. Teacher starts with misaligned logit ordering
2. Both teacher and student train on the **same CE loss** with correct ground-truth labels
3. Teacher's classification head adapts to match dataset label ordering
4. Student eventually receives useful soft labels as teacher corrects itself

Frozen teachers (`freeze_teacher: true`) cannot adapt - they produce misaligned soft labels throughout training.

### Implication

**Co-training is more robust to initialization issues** than frozen distillation. This could be a useful property when:
- Teacher/student use different tokenizers or label schemes
- Teacher wasn't explicitly trained for the target task format
- There are subtle distribution shifts between teacher training and distillation data

### Verification

After adding `teacher_label_map: [1, 2, 0]`:

```
Raw accuracy (no mapping):      0/10 = 0%
Mapped accuracy (with [1,2,0]): 10/10 = 100%
```

New ablation runs (with correct mapping) will provide cleaner comparison between frozen vs co-training approaches.

---

## The Overconfident-Wrong Teacher Paradox

**Date:** 2026-01-30

### Background

Ablation results with label-corrected configs showed an unexpected ranking:

```
04 (co-train SFT) > 03 (co-train base) > 01 (frozen base) > 00 (no teacher) > 02 (frozen SFT) >>> 05 (KL only)
```

Config 02 uses a frozen SFT teacher (TextAttack BERT-MNLI) with **84.5% validation accuracy** - yet it underperforms:
- Config 00 (no teacher at all)
- Config 01 (untrained teacher with ~33% random accuracy)

### The Paradox

**Why does an 84.5% accurate teacher produce worse results than no teacher?**

### Explanation: Accuracy ≠ Distillation Quality

The critical insight is that accuracy only tells you *how often* the teacher is correct, not *how it behaves when wrong*.

#### The Confidence Problem

A fine-tuned teacher (SFT) is typically:
- **Highly confident when correct** (84.5% of samples) → helpful
- **Highly confident when wrong** (15.5% of samples) → harmful

With α=0.5 (50% CE + 50% KL):

| Teacher State | CE Loss Direction | KL Loss Direction | Net Gradient |
|---------------|-------------------|-------------------|--------------|
| Correct + confident | → correct label | → correct label | ✓ Aligned |
| Wrong + confident | → correct label | → **wrong label (strongly)** | ✗ Conflicting |

#### Untrained vs Trained-but-Wrong

```
Untrained Teacher (Config 01):
  When wrong (67% of cases):
    CE:  "Move toward label A" (confident)
    KL:  "Move... somewhere" (low confidence, high entropy)
    Net: Mostly correct direction, slight noise

Trained Teacher (Config 02):
  When wrong (15.5% of cases):
    CE:  "Move toward label A" (confident)
    KL:  "Move toward label B!" (high confidence)
    Net: Gradients fight → confused learning
```

**Key insight:** Random noise is less harmful than confident misinformation.

### The Distribution Mismatch Hypothesis

Even when the SFT teacher is correct, its soft labels may be problematic:

1. **Training regime mismatch**: TextAttack trained BERT-MNLI differently
   - Different preprocessing
   - Different hyperparameters
   - Potentially different data splits

2. **Overconfidence**: Soft labels like `[0.95, 0.03, 0.02]` don't transfer uncertainty
   - Student learns to be overconfident too
   - Loses the "dark knowledge" benefits of distillation

3. **Pattern misalignment**: Teacher learned patterns that don't transfer well
   - Different tokenization handling
   - Different attention patterns
   - Artifacts from TextAttack's training setup

### Why Other Configs Succeed

| Config | Why it works |
|--------|--------------|
| **04 (co-train SFT)** | Teacher adapts; wrong predictions get corrected over training |
| **03 (co-train base)** | Teacher learns task-appropriate soft labels from scratch |
| **01 (frozen base)** | Uniform-ish noise doesn't fight CE loss |
| **00 (no teacher)** | Pure CE, no interference |
| **02 (frozen SFT)** | Confident wrong predictions fight CE → interference |
| **05 (KL only)** | No CE to anchor learning; pure noise from untrained teacher |

### Implications for Knowledge Distillation

1. **Teacher quality matters differently for frozen vs co-training**
   - Frozen: Need well-calibrated teacher with matching distribution
   - Co-training: Can recover from poor initial teacher

2. **Overconfidence is worse than uncertainty**
   - High-entropy (uncertain) soft labels add noise
   - Low-entropy (confident) wrong soft labels actively mislead

3. **Pre-trained task-specific teachers may not transfer**
   - Training setup differences create distribution mismatch
   - "84.5% accurate" doesn't mean "good for distillation"

### Potential Mitigations

To make frozen SFT teacher work better:
- **Higher temperature** (T=4, 8): Soften overconfident predictions
- **Lower α** (α=0.3): Reduce KL influence, trust CE more
- **Temperature annealing**: Start high (explore), end low (sharpen)
- **Teacher calibration**: Post-hoc calibration before distillation

---

## Why α=1.0 (KL-Only) Catastrophically Fails

**Date:** 2026-01-30

### Background

Config 05 uses `alpha: 1.0`, meaning:
- 100% KL divergence loss (learn from teacher only)
- 0% CE loss (ignore ground truth labels)

### Results

Config 05 performed **dramatically worse** than all other configs.

### Explanation

With an untrained teacher (bert-base-uncased, not fine-tuned on MNLI):

```
Teacher's soft labels ≈ random distribution over 3 classes
Ground truth labels → not used (α=1.0)

Student learns to mimic random distributions
→ ~33% accuracy (random guessing)
```

### The Loss Function

```python
loss = α × CE_loss + (1 - α) × KL_loss

With α = 1.0:
loss = 1.0 × CE_loss + 0.0 × KL_loss
     = CE_loss only  # Wait, this seems wrong?
```

**Note:** The paper's code uses **inverted alpha convention**:
```python
# From paper's code (glue_notebook.py):
loss = self.args.alpha * loss_ce + (1. - self.args.alpha) * loss_kd
```

So α=1.0 means **CE only**, α=0.0 means **KL only**.

But our implementation uses the standard convention:
```python
# Our implementation:
loss = α × KL_loss + (1 - α) × CE_loss
```

So α=1.0 in our configs means **KL only** (no CE).

### Why This Matters

With KL-only and an untrained teacher:
- No ground truth signal to anchor learning
- Student tries to match teacher's random outputs
- Results in random-level performance

This validates that:
1. Ground truth (CE loss) is essential when teacher is untrained
2. Even with a trained teacher, pure KL may lose important hard-label signal
3. Balanced α (0.5) provides best of both worlds

---

## Summary: Ablation Ranking Explained

**Date:** 2026-01-30

### Final Ranking (with label mapping fix)

```
04 (co-train SFT)     >
03 (co-train base)    >
01 (frozen base)      >
00 (no teacher)       >
02 (frozen SFT)       >>>
05 (KL only)
```

### Explanation for Each Position

| Rank | Config | Why This Position |
|------|--------|-------------------|
| 1st | 04 | Best of both: task knowledge + adaptation |
| 2nd | 03 | Teacher learns alongside student; no prior baggage |
| 3rd | 01 | Noise doesn't hurt; slight regularization benefit? |
| 4th | 00 | Clean baseline; no interference |
| 5th | 02 | Confident wrong predictions interfere with learning |
| 6th | 05 | No ground truth + random teacher = disaster |

### Key Takeaways

1. **Co-training > Frozen** for robustness
2. **Untrained > Trained-misaligned** for frozen distillation
3. **Balanced α > Extreme α** for stable learning
4. **Ground truth is essential** when teacher quality is uncertain

---

## Meta: Why Ablations Matter

**Date:** 2026-01-30

> "Strong beliefs, loosely held." — Jeff Bezos

This quote applies at two levels:

### Level 1: The Models Themselves

The quote perfectly describes **why co-training beats frozen distillation**:

| Config | Strong Beliefs? | Loosely Held? | Outcome |
|--------|-----------------|---------------|---------|
| **02 (frozen SFT)** | ✓ High confidence | ✗ Frozen, cannot change | Dogmatic; wrong predictions stay wrong |
| **04 (co-train SFT)** | ✓ High confidence | ✓ Updates via gradients | Adaptive; corrects mistakes over training |

The frozen teacher is *dogmatic* - it cannot revise its beliefs even when the ground truth contradicts it. Every batch, it confidently repeats the same mistakes.

The co-training teacher is *adaptive* - when the CE loss says "you're wrong," the teacher updates. Strong initial beliefs, but willing to change when evidence accumulates.

**This is why 04 > 02:** Both start with the same 84.5% accurate teacher. But 04's teacher can learn from its mistakes; 02's teacher repeats them indefinitely.

### Level 2: The Assumption We Held

**Strong belief:** *"An 84.5% accurate teacher must produce better distillation than a 33% random one."*

This seems self-evident. A teacher that knows the task should transfer more useful knowledge than one producing random noise.

### What The Data Showed

```
01 (untrained, ~33% accuracy) > 02 (SFT, 84.5% accuracy)
```

The "worse" teacher produced better students.

### The Updated Belief

**Revised understanding:** *"For frozen distillation, confidence calibration matters more than accuracy. For co-training, adaptability matters most."*

- Accuracy measures *how often* the teacher is right
- Distillation quality depends on *how the teacher behaves when wrong*
- Frozen teachers: overconfident wrong predictions actively harm learning
- Co-training teachers: wrong predictions get corrected over time
- The ability to update beliefs (loosely held) trumps initial accuracy

### Why This Matters

Without systematic ablations, we would have:
1. Assumed "better teacher = better distillation"
2. Never discovered the overconfident-wrong teacher problem
3. Missed a key insight about frozen vs co-training robustness

The original paper likely held the same assumption - and never tested the baseline (α=0), missing this finding entirely.

### Lessons for Research

1. **Always include baselines** - especially the "obvious" ones like no-teacher
2. **Test assumptions explicitly** - intuition often fails in complex systems
3. **Be willing to update beliefs** - when evidence conflicts, trust the data
4. **Ablate systematically** - each component could behave unexpectedly

The ranking `04 > 03 > 01 > 00 > 02 >>> 05` told a story that no amount of theorizing would have predicted. The experiments revealed the truth.

---

## QQP Ablation Results

**Date:** 2026-01-31

| Rank | Config | Val Accuracy | Δ from Student Baseline | Description |
|------|--------|--------------|------------------------|-------------|
| 1 | 03 | **90.37%** | +53.55% | Co-train base teacher |
| 2 | 04 | 90.28% | +53.46% | Co-train SFT teacher |
| 3 | 01 | 90.19% | +53.37% | Frozen base teacher |
| 4 | 00 | 89.71% | +52.89% | No teacher (CE only) |
| 5 | 02 | 65.50% | +28.68% | Frozen SFT teacher ❌ |

**The full training journey:**
```
Student baseline:     36.82% (random)
    ↓ +52.89% with CE-only training
No teacher (00):      89.71%
    ↓ +0.66% with co-train base
Best config (03):     90.37%
```

**Key observation:** QQP ranking differs slightly from MNLI:
- MNLI: `04 > 03 > 01 > 00 > 02`
- QQP:  `03 > 04 > 01 > 00 > 02`

Co-training base teacher (03) slightly outperforms co-training SFT teacher (04) on QQP, while the reverse was true for MNLI. Both are within noise margin (~0.1%).

---

## TextAttack Label Mappings

**Date:** 2026-01-31

### Verification Method

Tested known examples against model predictions to determine actual label meanings:

### QQP Labels: ALIGNED ✓

| Index | TextAttack | HuggingFace |
|-------|------------|-------------|
| 0 | not_duplicate | not_duplicate |
| 1 | duplicate | duplicate |

**No `teacher_label_map` needed for QQP.**

### MNLI Labels: MISALIGNED ✗

| Index | TextAttack | HuggingFace |
|-------|------------|-------------|
| 0 | contradiction | entailment |
| 1 | entailment | neutral |
| 2 | neutral | contradiction |

**Required mapping:** `teacher_label_map: [1, 2, 0]`

Empirical verification:
```
Raw accuracy (no mapping):     5.5%
With [1,2,0] mapping:         83.5% ✓
With [2,0,1] mapping:         11.0%
```

---

## Frozen SFT Teacher Fails Even With Aligned Labels

**Date:** 2026-01-31

### Background

QQP config 02 (frozen SFT teacher) achieved only **65.50%** despite:
1. Labels being correctly aligned (no swap needed)
2. Teacher having **90.91%** validation accuracy

This confirms the "overconfident wrong teacher" effect is **not** due to label misalignment.

### Comparison

| Task | Config 02 Accuracy | Teacher Accuracy | Label Status |
|------|-------------------|------------------|--------------|
| MNLI | 79.26% | 84.57% | Misaligned (fixed with mapping) |
| QQP | 65.50% | 90.91% | Aligned |

QQP's larger gap (65.5% vs 90.9%) compared to MNLI (79.3% vs 84.6%) suggests the frozen SFT teacher problem may be worse when:
- Teacher is more confident (higher accuracy)
- Task is binary (confident wrong = 50% wrong direction vs 33% for 3-class)

### Conclusion

The frozen SFT teacher failure is **fundamental to frozen distillation**, not a label alignment bug. Co-training (03/04) consistently outperforms frozen approaches because it allows self-correction.

---

## Cross-Task Consistency

**Date:** 2026-01-31

Both MNLI and QQP show the same pattern:

| Observation | MNLI | QQP |
|-------------|------|-----|
| Best config | Co-train (03/04) | Co-train (03/04) |
| Worst config | Frozen SFT (02) | Frozen SFT (02) |
| CE-only baseline | Competitive | Competitive |
| Frozen base > Frozen SFT | ✓ (01 > 02) | ✓ (01 > 02) |

**Universal findings:**
1. Co-training > Frozen for distillation
2. Untrained teacher > Trained teacher when frozen
3. CE-only baseline is hard to beat with frozen distillation
4. Overconfident wrong predictions hurt more than uncertain noise

---

## RFA Factorial Study

**Date:** 2026-02-02

### Overview

Factorial study of three distillation signals:
- **R (Response)**: KL divergence on output logits - classic Hinton distillation
- **F (Feature)**: MSE on hidden states - FitNets-style intermediate matching
- **A (Attention)**: KL on attention distributions - attention transfer

8 configurations testing all combinations: 000, 001, 010, 011, 100, 101, 110, 111.

### Setup

| Parameter | Value |
|-----------|-------|
| Task | MNLI |
| Teacher | bert-base-uncased (frozen, untrained) |
| Student | distilbert-base-uncased |
| Alpha | 0.5 (50% CE + 50% distillation) |
| Temperature | 2.0 |
| Signal weights | Equal (1:1:1 when combined) |
| Training steps | 4602 |
| Runs per config | 2 |

### Results

| Config | R | F | A | Accuracy (Run 1) | Accuracy (Run 2) | Mean | Rank |
|--------|---|---|---|------------------|------------------|------|------|
| rfa-01 | 1 | 0 | 0 | 81.13% | 81.13% | **81.13%** | **1** |
| rfa-05 | 1 | 0 | 1 | 80.59% | 80.59% | 80.59% | 2 |
| rfa-04 | 1 | 1 | 0 | 80.49% | 80.49% | 80.49% | 3 |
| rfa-07 | 1 | 1 | 1 | 79.68% | 80.77% | 80.23% | 4 |
| rfa-02 | 0 | 1 | 0 | 79.42% | 79.42% | 79.42% | 5 |
| rfa-06 | 0 | 1 | 1 | 79.43% | 79.43% | 79.43% | 6 |
| rfa-03 | 0 | 0 | 1 | 78.69% | 78.69% | 78.69% | 7 |
| rfa-00 | 0 | 0 | 0 | 78.55% | 78.55% | 78.55% | 8 |

### Key Finding: R-Only Wins

**Classic KL distillation (R-only) outperforms all multi-signal combinations.**

```
rfa-01 (R only):  81.13%  ← Best
rfa-07 (RFA all): 80.23%  ← -0.9% worse
rfa-00 (CE only): 78.55%  ← Baseline
```

Adding Feature (F) or Attention (A) signals to Response (R) **hurts** performance:
- R → RF: 81.13% → 80.49% (-0.64%)
- R → RA: 81.13% → 80.59% (-0.54%)
- R → RFA: 81.13% → 80.23% (-0.90%)

### Signal Contribution Analysis

**Individual signal value (vs baseline rfa-00):**

| Signal | Config | Δ Accuracy |
|--------|--------|------------|
| R | rfa-01 | +2.58% |
| F | rfa-02 | +0.87% |
| A | rfa-03 | +0.14% |

**Ranking: R >> F > A > baseline**

### Why Combinations Underperform: Gradient Conflict

Observation from training metrics:
- **rfa-02 (F only)**: `feature_raw` decreases ✓
- **rfa-03 (A only)**: `attention_raw` decreases ✓
- **rfa-07 (RFA)**: `feature_raw` **increases**, `attention_raw` **increases** ✗

When combined, F and A signals *diverge* instead of converging. The optimizer cannot satisfy all three objectives simultaneously - gradients pull in conflicting directions.

**The conflict:**
```
R gradient: "adjust weights to match output distribution"
F gradient: "adjust weights to match hidden states"
A gradient: "adjust weights to match attention patterns"

These often point in different directions. Equal weighting (1:1:1) creates
a tug-of-war where no objective is satisfied well.
```

### Reproducibility

Results are highly reproducible:
- 7/8 configs: Identical accuracy across both runs (to 5 decimal places)
- rfa-07: Slight variance (79.68% vs 80.77%) - possibly due to combined signal instability

### Implications

1. **Simpler is better**: For BERT→DistilBERT on MNLI, classic KL distillation beats complex multi-signal approaches

2. **Signal interference**: Combining distillation signals requires careful tuning - equal weighting causes gradient conflict

3. **Diminishing returns**: Each additional signal adds complexity but reduces performance

4. **F and A work alone**: Both Feature and Attention signals improve over baseline when used individually, but interfere when combined

### Potential Improvements (Not Tested)

1. **Staged training**: Train F→A→R sequentially instead of simultaneously
2. **Adaptive weighting**: Reduce F/A weights as training progresses
3. **Gradient projection**: Project F/A gradients to not conflict with R
4. **Lower F/A weights**: Use R=0.8, F=0.1, A=0.1 instead of equal

### Conclusion

For knowledge distillation with a frozen untrained teacher:
- **Use R-only (KL divergence)** - it's simpler and performs best
- Feature and Attention signals add complexity without benefit in this setup
- Multi-signal distillation may require more sophisticated weighting or scheduling

---

## Future Findings

(Add new observations here as experiments progress)

---

## Related Documents

| Document | Purpose |
|----------|---------|
| [research-framing.md](research-framing.md) | Research questions, hypotheses, paper potential |
| [framework-direction.md](framework-direction.md) | Vision, mastery path, learning roadmap |
| [diffusion-distillation.md](diffusion-distillation.md) | Diffusion model distillation guide |
| [run-options.md](run-options.md) | CLI reference and run management |
