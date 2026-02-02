# Paper Critique: "Less is More"

Critical analysis of methodological issues in the original paper.

**Paper:** "Less is More: Achieving Smaller, Faster Models for Edge Computing with Knowledge Distillation"
**Authors:** Richard Yue & Vijay Venkatesan (Northeastern University)

---

## Issue 1: Untrained Teacher Model

### The Problem
Section 3.1 states:
> *"the 'bert-base-uncased' checkpoint was loaded for the teacher model"*

This is the **base pretrained BERT** - NOT fine-tuned on MNLI. The teacher only knows language modeling (MLM/NSP), not the MNLI classification task.

### Impact
- Teacher's soft labels are essentially **random for the task**
- KL loss component provides no useful task-specific signal
- Student learns primarily from CE loss with ground truth labels

### Evidence
Our config 00 (no teacher, CE-only) achieves ~81-82% accuracy - nearly identical to the paper's 81.77% result with "distillation."

---

## Issue 2: Missing Baseline (No α=0 Experiment)

### The Problem
The paper never trains the student model **without** a teacher to establish:
1. What accuracy the student achieves alone
2. Whether distillation actually contributes anything

### What Should Have Been Shown
```
Student alone (α=0, CE only):    ~81%
Student + Teacher (α=0.5):       81.77%
─────────────────────────────────────────
Distillation contribution:       ~0%
```

### Impact
Without this baseline, the paper cannot claim that knowledge distillation improved results. The observed accuracy may come entirely from fine-tuning DistilBERT with cross-entropy loss.

---

## Issue 3: Baseline Comparison Mismatch

### The Problem
Table 2 compares results against:
> *"BERT-base (baseline): 0.846/0.834"*

This baseline is from Devlin et al. (2019) where BERT was **fine-tuned on MNLI**.

But the paper's teacher is `bert-base-uncased` which is **NOT fine-tuned**.

### The Confusion
| What | Model State |
|------|-------------|
| Baseline they cite | BERT fine-tuned on MNLI (84.6%) |
| Teacher they use | BERT base pretrained (NOT fine-tuned) |

They're comparing apples to oranges.

---

## Issue 4: QQP Result Anomaly

### The Problem
Table 3 shows:
| Model | QQP Accuracy |
|-------|--------------|
| BERT-base (baseline) | 0.712 |
| DistilBERT @ 2e-05 | **0.9001** |

The student (90%) **far exceeds** the teacher baseline (71.2%) - a 19% improvement.

### Questions
1. Is the BERT baseline accurate? (Other sources report higher)
2. How can a distilled student dramatically exceed its teacher?
3. Is there a data leakage or evaluation issue?

This result is suspicious and warrants investigation.

---

## Issue 5: Loss Function Ambiguity

### The Problem
The paper describes alpha as controlling "how much the student model learns from the distillation loss compared to the cross-entropy loss" but doesn't provide the explicit formula.

### From Their Code (glue_notebook.py)
```python
loss = self.args.alpha * loss_ce + (1. - self.args.alpha) * loss_kd
```

With α=0.5: 50% CE + 50% KL

### Note
This convention is **inverted** from some implementations where α weights the KL term directly. The paper should have been explicit.

---

## Issue 6: Incomplete Ablation Tables

### The Problem
Section 5.2 mentions testing:
- Learning rates: 2e-05, 2e-07, 2e-09
- Batch sizes: 16, 32, 64
- Temperatures: 2, 11, 20
- Alpha values: 0.25, 0.5, 1.0

But only **learning rate results** are shown in tables. Other hyperparameter results are mentioned only in prose:
> *"The batch size of 16 yielded the strongest evaluation accuracy"*
> *"using a smaller temperature value led to better results"*

### Impact
- Results cannot be verified or reproduced precisely
- No clear picture of hyperparameter sensitivity
- Incomplete scientific documentation

---

## Issue 7: Student Initialization Discrepancy

### The Problem
Section 3.1 states:
> *"a freshly initialized 'distilbert-base-uncased' model configuration was loaded for the student model"*

But "freshly initialized" is ambiguous:
1. Random weights (true fresh initialization)
2. Pretrained DistilBERT weights (from HuggingFace)

### From Code Analysis
The actual code appears to use pretrained weights, not fresh initialization.

### Impact
This significantly affects reproducibility. A randomly initialized student would perform much worse than one starting from pretrained weights.

---

## Issue 8: Translation Alpha=1.0

### The Problem
Section 5.1 states translation uses:
> *"an alpha of 1"*

With their loss formula `α × CE + (1-α) × KL`, this means:
- 100% CE loss
- 0% KL loss

**This is NOT distillation** - it's just supervised fine-tuning with ground truth labels.

### Contradiction
The paper claims to do knowledge distillation for translation, but with α=1.0, no knowledge is being distilled from the teacher.

---

## Summary Table

| Issue | Severity | Impact on Claims |
|-------|----------|------------------|
| Untrained teacher | **Critical** | Invalidates KD benefit claims for GLUE |
| No α=0 baseline | **Critical** | Cannot prove distillation helped |
| Baseline mismatch | **High** | Misleading comparisons |
| QQP anomaly | **High** | Suspicious result needs verification |
| Loss ambiguity | Medium | Reproducibility issue |
| Incomplete tables | Medium | Cannot verify hyperparameter claims |
| Student init unclear | Medium | Reproducibility issue |
| Translation α=1.0 | **High** | Not actually doing distillation |

---

## How Our Ablations Address These Issues

| Paper Gap | Our Config | Purpose |
|-----------|------------|---------|
| No baseline | **00_no-teacher_ce-only** | Establish true baseline |
| Untrained teacher only | **01_frozen_base-teacher** | Match paper's actual setup |
| No trained teacher | **02_frozen_sft-teacher** | Proper KD with expert teacher |
| No co-training | **03, 04, 05** | Explore alternative approaches |
| Label mapping bug | **teacher_label_map: [1,2,0]** | Fix TextAttack label ordering |

Our experiments will provide the missing evidence to properly evaluate knowledge distillation's contribution.
