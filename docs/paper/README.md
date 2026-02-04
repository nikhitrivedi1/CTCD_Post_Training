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

### Catastrophic Example: QQP

| Config | F1 Score | Effect |
|--------|----------|--------|
| CE-only (no teacher) | 90.4% | Baseline |
| Co-train SFT | 90.5% | ✓ +0.1% |
| Frozen SFT | **55.9%** | ✗ **-34.5%** |

### Why Binary Tasks Are Riskier

| Classes | Wrong Prediction | Risk |
|---------|------------------|------|
| 3+ (MNLI) | Partial credit possible | Lower |
| 2 (QQP) | Maximally wrong (0↔100%) | ⚠️ Higher |

### Why Small Datasets Fail (CoLA)

| Problem | Effect |
|---------|--------|
| Teacher overfits | Systematic errors |
| Student inherits bias | Can't escape local minimum |
| Not enough examples | Can't average out noise |

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

## Resources

### Analysis
- [KD Prerequisites](../analysis/summaries/kd_prerequisites.md) - When KD works
- [Classification Ablations](../analysis/summaries/classification_ablations.md) - Full results
- [Ablation Explainer](../analysis/summaries/ablation_explainer.md) - Plain English
- [Raw Data](../analysis/data/) - W&B exports

### Papers
- [Paper 0: Framework](paper-0-framework/draft.md)
- [Paper 1: When Teachers Fail](paper-1-ablation/draft.md)
- [Paper 2: Simplicity Wins](paper-2-far/draft.md)
- [Paper 3: Translation Transfer](paper-3-translation/draft.md)
