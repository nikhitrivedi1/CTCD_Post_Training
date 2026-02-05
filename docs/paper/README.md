# Knowledge Distillation: A Research Overview

A systematic investigation into when, why, and how knowledge distillation works.

---

## Summary: When KD Works

| Task | Type | Size | Balance | Classes | Difficulty | KD Effect |
|------|------|------|---------|---------|------------|-----------|
| MNLI | Classification | 393k ✓ | Balanced ✓ | 3 ✓ | Medium ✓ | ✓ **Helps** |
| QQP | Classification | 364k ✓ | 63/37 ⚠️ | 2 ⚠️ | Medium ✓ | ⚠️ **Risky** (catastrophic if frozen) |
| SST-2 | Classification | 67k ~ | Balanced ✓ | 2 ⚠️ | Easy ✗ | ~ **Neutral** |
| CoLA | Classification | 8.5k ✗ | 70/30 ⚠️ | 2 ⚠️ | Hard ✗ | ✗ **Hurts** |
| Europarl | Translation | 45k ~ | N/A | Seq2Seq | Medium ✓ | 🔬 **Testing** |

### Quick Decision

```
✗ Dataset < 10k?           → Skip KD
✗ Task saturated (>90%)?   → Skip KD
⚠️ Imbalanced + Binary?     → Co-train ONLY
✓ Otherwise                → KD likely helps
```

---

## What is Knowledge Distillation?

**Knowledge Distillation (KD)** transfers knowledge from a large "teacher" model to a smaller "student" model.

### Classification (BERT → DistilBERT)

```
Teacher: BERT-base (110M)  →  Student: DistilBERT (66M)
         84% accuracy              82% accuracy
```

### Translation (Llama 3B → Llama 1B)

```
Teacher: Llama-3.2-3B (3B)  →  Student: Llama-3.2-1B (1.2B)
         + LoRA fine-tuning
```

### The Core Idea

Train student on teacher's **soft labels** instead of hard labels:

| Input | Hard Label | Soft Label (Teacher) |
|-------|------------|----------------------|
| "Great movie!" | [0, 1] | [0.05, 0.95] |
| "It was okay" | [0, 1] | [0.35, 0.65] |

Soft labels encode **dark knowledge**:
- ✓ Relative similarities between classes
- ✓ Teacher's uncertainty on ambiguous examples
- ✓ Richer signal than binary labels

### 💡 Intuition: Why Soft Labels Help

Imagine learning to identify animals:

| Hard Label | Soft Label |
|------------|------------|
| "This is a dog" | "90% dog, 8% wolf, 2% fox" |

The soft label tells you:
- Dogs look similar to wolves (not to fish)
- The teacher is uncertain about edge cases
- There's a similarity structure between classes

A student trained on soft labels learns **relationships**, not just categories.

### The Loss Function

```
L = (1-α) × CE(student, labels) + α × KL(student, teacher)
    \_____________________/       \____________________/
         Ground truth                 Teacher guidance
```

| α | Mode | Use Case |
|---|------|----------|
| 0.0 | CE-only | Baseline (no teacher) |
| 0.5 | Balanced | Standard distillation |
| 1.0 | KL-only | Pure distillation |

---

## Why Use Knowledge Distillation?

### 1. Model Compression

KD is one of several techniques for reducing model size and memory footprint:

| Technique | What It Does | Trade-off |
|-----------|--------------|-----------|
| **Knowledge Distillation** | Train small model to mimic large one | Requires training, best quality |
| Pruning | Remove unimportant weights | Can hurt quality, needs fine-tuning |
| Quantization | Reduce precision (FP32→INT8) | Fast, some quality loss |
| Layer Reduction | Remove entire layers | Simple, may lose capacity |
| Head Grouping | Share attention heads | Architecture-specific |
| Dropout @ Inference | Use sub-networks | Probabilistic |

**Why KD stands out:**
- ✓ Preserves model architecture (easier deployment)
- ✓ Transfers "dark knowledge" (soft labels, relationships)
- ✓ Can combine with other techniques (distill → quantize)

| Benefit | Example |
|---------|---------|
| Smaller model | BERT 110M → DistilBERT 66M (40% smaller) |
| Faster inference | 2x speedup on CPU |
| Lower memory | Deploy on mobile/edge |

### 2. Knowledge Transfer
| From | To | Example |
|------|----|---------|
| Large model | Small model | Llama 3B → Llama 1B |
| Fine-tuned | Base | SFT teacher → pretrained student |
| One task | Another | Classification → Generation |

### 3. Regularization
- ✓ Soft labels prevent overconfident predictions
- ✓ Improves calibration
- ✓ Can help generalization

---

## When Does KD Work?

### Prerequisites for Success

| Factor | ✓ Helps KD | ✗ Hurts KD |
|--------|------------|------------|
| **Data Size** | >100k examples | <10k examples |
| **Class Balance** | Balanced | Highly imbalanced |
| **Num Classes** | 3+ classes | Binary |
| **Task Difficulty** | Medium (70-85%) | Too easy or too hard |
| **Teacher Quality** | Accurate + calibrated | Overconfident-wrong |

### Classification Tasks

| Task | Verdict | Why |
|------|---------|-----|
| MNLI | ✓ Use KD | Large, balanced, 3-class, medium difficulty |
| QQP | ⚠️ Co-train only | Imbalanced binary = catastrophic if frozen |
| SST-2 | ~ Skip KD | Task saturated, no room for improvement |
| CoLA | ✗ Skip KD | Too small, too hard, KD adds noise |

### Translation Tasks

| Factor | Europarl EN→FR |
|--------|----------------|
| Size | 45k ~ (medium) |
| Sequence length | Variable |
| Output space | 128k vocab |
| Teacher-student gap | 3B → 1B (2.5x compression) |
| Verdict | 🔬 Testing with LoRA |

**Translation-specific considerations:**
- ⚠️ Larger output space (vocab size) than classification
- ⚠️ Autoregressive generation (errors compound)
- ✓ KL on token distributions still applies
- ✓ Can use response-only distillation (R)

---

## How Can KD Fail?

### The Overconfident-Wrong Teacher Problem

When a frozen fine-tuned teacher is confidently wrong:

```
Ground Truth: "This IS a duplicate"
Teacher says: "NOT a duplicate" (95% confidence)

CE gradient  → Push toward correct answer  ✓
KL gradient  → Push toward teacher         ✗
             → CONFLICT!
```

### 💡 Intuition: The Stubborn Expert

Imagine learning from two sources:
- **Textbook** (ground truth): "The answer is A"
- **Expert** (frozen teacher): "I'm 95% sure it's B"

With α=0.5, you weight them equally. The expert is **confident but wrong**.

Now you're stuck:
- Half your learning signal says "go to A"
- Half says "go to B with high confidence"
- You end up somewhere in between — wrong

**The fix?** Let the expert learn too (co-training). When they see they're wrong, they become less confident, and you can learn the right answer.

### Catastrophic Example: QQP

| Config | F1 Score | Effect |
|--------|----------|--------|
| CE-only (no teacher) | 90.4% | Baseline |
| Co-train SFT | 90.5% | ✓ +0.1% |
| Frozen SFT | **55.9%** | ✗ **-34.5%** |

### 📊 Empirical Evidence: The 82% Problem

We analyzed the QQP teacher's predictions on 500 validation examples:

| Metric | Value |
|--------|-------|
| Overall accuracy | 91.6% ✓ |
| Wrong predictions | 42 (8.4%) |
| **Avg confidence when WRONG** | **82.1%** ⚠️ |
| >90% confident but wrong | 20 cases |
| >80% confident but wrong | 26 cases |

The teacher is **highly accurate** (91.6%) but **dangerously overconfident** when wrong. Those 42 wrong predictions with 82% average confidence create persistent gradient conflict that the frozen student can never escape.

### 💡 Intuition: The 8% That Ruins Everything

Think of it like a GPS that's right 92% of the time:
- 92% of directions are correct → you learn the route
- 8% are confidently wrong → "Turn left into the lake!"

If you **trust the GPS completely** (frozen teacher), those 8% wrong turns are catastrophic. But if the GPS **updates based on feedback** (co-training), it learns "oh, that's not a road" and corrects itself.

The problem isn't that the teacher is wrong 8% of the time—it's that it's **82% confident** about those wrong answers.

### Why Binary Tasks Are Riskier

| Classes | Wrong Prediction | Risk |
|---------|------------------|------|
| 3+ (MNLI) | Partial credit possible | Lower |
| 2 (QQP) | Maximally wrong (0↔100%) | ⚠️ Higher |

### 💡 Intuition: Binary vs Multi-class

**3-class (MNLI):** Teacher says "70% entailment, 20% neutral, 10% contradiction"
- Even if wrong about #1, still gives useful signal about #2 and #3
- Partial credit: "entailment is more like neutral than contradiction"

**Binary (QQP):** Teacher says "95% not duplicate, 5% duplicate"
- If wrong, it's **maximally wrong** (should be 5%/95%)
- No middle ground, no partial credit
- Confident wrong = complete misguidance

### Why Small Datasets Fail (CoLA)

| Problem | Effect |
|---------|--------|
| Teacher overfits | Systematic errors |
| Student inherits bias | Can't escape local minimum |
| Not enough examples | Can't average out noise |

### 💡 Intuition: Signal vs Noise

Imagine learning from a tutor who studied the same small textbook you have:

**Large dataset (393k examples):**
- Tutor has seen many variations
- Their mistakes are random, cancel out
- Their insights are genuine patterns

**Small dataset (8.5k examples):**
- Tutor memorized the specific examples
- Their "knowledge" is just memorization
- You inherit their overfitting, not wisdom

On CoLA, the teacher isn't teaching grammar—it's teaching "what answers worked on these 8k examples." That's not useful knowledge to transfer.

---

## How to Do KD Right

### ✓ 1. Always Include CE-Only Baseline

Without it, you can't measure KD's actual contribution.

```yaml
# Config 00: Baseline
distillation:
  alpha: 0.0  # CE-only, no teacher
```

### ✓ 2. Prefer Co-Training Over Frozen

| Approach | Safety | Why |
|----------|--------|-----|
| Frozen teacher | ⚠️ Risky | Can't correct errors |
| Co-training | ✓ Safe | Self-corrects over time |

```yaml
# Frozen (risky)
distillation:
  freeze_teacher: true

# Co-train (safe)
distillation:
  freeze_teacher: false
```

### 💡 Intuition: The Learning Expert

**Frozen teacher** = Expert who never admits mistakes
- "I'm 95% sure it's B" (even when wrong)
- Student keeps getting conflicting signals
- No way to resolve the disagreement

**Co-training** = Expert who learns alongside you
- Initially: "I'm 95% sure it's B"
- After seeing ground truth: "Hmm, maybe 60% B..."
- Eventually: "Actually, it's probably A"

The co-trained teacher **updates their confidence** when shown evidence. This means:
1. Their wrong predictions become less confident
2. The KL loss weakens on mistakes
3. CE loss can guide the student correctly

Key insight: *A humble teacher beats a confident fool.*

### ✓ 3. Match Strategy to Task Profile

| Profile | Recommendation |
|---------|----------------|
| Large + Balanced + Multi-class | ✓ KD with co-train |
| Large + Imbalanced + Binary | ⚠️ Co-train ONLY |
| Small (<10k) | ✗ Skip KD |
| Saturated (>90%) | ✗ Skip KD |

### ✓ 4. Monitor Training Dynamics

Watch for:
- ⚠️ Loss spikes (gradient conflict)
- ⚠️ Stuck training loss (teacher dominance)
- ⚠️ Eval metric diverging from loss

---

## What Signals to Distill?

### The FAR Framework

| Signal | Abbrev | What Transfers | Loss |
|--------|--------|----------------|------|
| **F**eature | F | Hidden representations | MSE on layers |
| **A**ttention | A | Attention patterns | MSE on attention |
| **R**esponse | R | Output distribution | KL divergence |

### Finding: R-Only Wins ✓

| Config | Signals | Result |
|--------|---------|--------|
| R-only | R | ✓ **Best** |
| F-only | F | ✗ Worse |
| A-only | A | ✗ Worse |
| FAR | F+A+R | ✗ Worse than R-only |

**Insight:** More signals ≠ better. Gradient conflict between signals hurts.

---

## Paper Series

| Paper | Question | Answer |
|-------|----------|--------|
| [Paper 0](paper-0-framework/draft.md) | How to do KD research? | Unified config-driven framework |
| [Paper 1](paper-1-ablation/draft.md) | When do teachers fail? | Frozen SFT on imbalanced binary |
| [Paper 2](paper-2-far/draft.md) | Which signals matter? | R-only beats multi-signal |
| [Paper 3](paper-3-translation/draft.md) | Does it generalize? | Testing on Llama translation |

### Narrative Arc

```
Paper 0: "Here's how to do KD research right"
Paper 1: "What we thought about KD is wrong"
Paper 2: "The fix is simpler than we thought"
Paper 3: "Does this generalize to generation?"
```

---

## Key Takeaways

| # | Takeaway |
|---|----------|
| 1 | ✗ KD is NOT always beneficial (CoLA, SST-2) |
| 2 | ✓ Teacher adaptability > teacher accuracy |
| 3 | ✓ Simplicity wins (R-only > FAR, co-train > complex fixes) |
| 4 | ✓ Baselines are essential (CE-only) |
| 5 | ⚠️ Binary + imbalanced = high risk |

---

## Open Questions & Next Investigations

### 🔬 Calibration & Confidence

| Question | Why It Matters |
|----------|----------------|
| Can we predict KD failure from teacher calibration? | Pre-training diagnostic |
| Does temperature help miscalibrated teachers? | Cheap fix for overconfidence |
| What's the confidence threshold for "safe" KD? | Decision rule |

**Investigation:** Measure Expected Calibration Error (ECE) before distillation. Hypothesis: ECE > 0.1 → skip KD or use higher temperature.

### 📊 Pre/Post Metrics

| Metric Set | What It Reveals |
|------------|-----------------|
| Teacher (Pre-run) | Baseline quality, calibration |
| Student (Pre-run) | Starting point |
| Teacher (Post-run) | Did co-training help/hurt teacher? |
| Student (Post-run) | Final performance |

**Investigation:** Log all four metric sets to understand knowledge transfer efficiency: `(student_post - student_pre) / (teacher_pre - student_pre)`

### 🎯 Class-Specific Analysis

| Question | Why It Matters |
|----------|----------------|
| Which class does the teacher fail on? | Minority class bias |
| Does co-training fix class imbalance? | Mechanism understanding |
| Can we weight KL loss by confidence? | Potential fix |

**Investigation:** Break down accuracy by class. Hypothesis: Teacher fails more on minority class with higher confidence.

### 🔄 Training Dynamics

| Signal | What To Watch |
|--------|---------------|
| eval/loss stuck from start | Teacher dominance (QQP pattern) |
| eval/loss increases mid-training | Overfitting (CoLA pattern) |
| train/teacher_ce_loss decreasing | Co-training working |

**Investigation:** Add early stopping based on eval/loss trend, not just final value.

---

## Resources

### Analysis
- [KD Prerequisites](../analysis/summaries/kd_prerequisites.md) - When KD works
- [Training Dynamics](../analysis/summaries/training_dynamics.md) - Loss curves & failure modes
- [Classification Ablations](../analysis/summaries/classification_ablations.md) - Full results
- [Ablation Explainer](../analysis/summaries/ablation_explainer.md) - Plain English
- [Raw Data](../analysis/data/) - W&B exports

### Papers
- [Paper 0: Framework](paper-0-framework/draft.md)
- [Paper 1: When Teachers Fail](paper-1-ablation/draft.md)
- [Paper 2: Simplicity Wins](paper-2-far/draft.md)
- [Paper 3: Translation Transfer](paper-3-translation/draft.md)
