# Paper Evaluation: "Less is More"

A comprehensive analysis and critique of the foundational paper.

**Paper:** "Less is More: Achieving Smaller, Faster Models for Edge Computing with Knowledge Distillation"
**Authors:** Richard Yue & Vijay Venkatesan (Northeastern University)
**PDF:** [wip_docs/Yue_Venkatesan_Less_is_More.pdf](../../wip_docs/Yue_Venkatesan_Less_is_More.pdf)

---

## Table of Contents

1. [Target Results](#target-results)
2. [Model Configurations](#model-configurations)
3. [Hyperparameters](#hyperparameters)
4. [Reported Results](#reported-results)
5. [Key Observations](#key-observations)
6. [Critical Analysis](#critical-analysis)
7. [Ablation Study Assessment](#ablation-study-assessment)
8. [How Our Work Addresses the Gaps](#how-our-work-addresses-the-gaps)
9. [Reproduction Plan](#reproduction-plan)

---

## Target Results

| Task                 | Metric   | Target | Notes                |
|----------------------|----------|--------|----------------------|
| MNLI                 | Accuracy | 0.817  | 3-class classification |
| QQP                  | Accuracy | 0.900  | Binary classification |
| Translation (EN→FR)  | BLEU     | ~27    | Europarl subset      |
| Translation (EN→FR)  | COMET    | ~0.82  |                      |

---

## Model Configurations

### BERT → DistilBERT (GLUE Tasks)

| Model                 | Layers | Hidden Size | Attention Heads | Parameters |
|-----------------------|--------|-------------|-----------------|------------|
| BERT-base (teacher)   | 12     | 768         | 12              | 110M       |
| DistilBERT (student)  | 6      | 768         | 12              | 66M        |

**Key:** Student is freshly initialized (not pretrained DistilBERT checkpoint)

> ⚠️ **Paper vs Code Discrepancy:**
> - **Paper text (Section 3.1):** *"a freshly initialized 'distilbert-base-uncased' model configuration was loaded for the student model"*
> - **Actual code:** Uses `AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")` - pretrained weights
>
> Our configs use `student_init: pretrained` which matches the actual code behavior, not the paper text.

### Llama 3.2 (Translation)

| Model   | Parameters | Notes                               |
|---------|------------|-------------------------------------|
| Teacher | 3B         | `meta-llama/Llama-3.2-3B-Instruct`  |
| Student | 1B         | Fresh config (not pretrained weights) |

---

## Hyperparameters

### GLUE Tasks (Section 5.2)

| Hyperparameter | Values Tested       | Best       |
|----------------|---------------------|------------|
| Learning rate  | 2e-05, 2e-07, 2e-09 | **2e-05**  |
| Batch size     | 16, 32, 64          | **16**     |
| Temperature    | 2, 11, 20           | **2**      |
| Alpha          | 0.25, 0.5, 1.0      | **0.5**    |

**Other settings:**
- Epochs: 3
- GPU: P100
- Training time: 8 hours max

### Translation (Section 5.1)

| Hyperparameter | Values Tested         | Best       |
|----------------|-----------------------|------------|
| Learning rate  | 1e-03, 1e-04, 1e-05   | **1e-04**  |

**Fixed settings:**
- Alpha: 1.0 (pure distillation, no CE loss)
- Temperature: 2.0
- Batch size: 4 (with 24 gradient accumulation = effective 96)
- LoRA: r=16, alpha=16, dropout=0.05
- Target modules: down_proj, up_proj, gate_proj
- Warmup: 200 steps
- Weight decay: 0.01
- Max grad norm: 2.0
- Epochs: 3
- GPU: V100-SXM2

---

## Reported Results

### MNLI (Table 2)

| Model/Learning Rate       | Accuracy   |
|---------------------------|------------|
| BERT-base (baseline)      | 0.846 / 0.834 |
| DistilBERT @ 2e-05        | **0.8177** |
| DistilBERT @ 2e-07        | 0.6245     |
| DistilBERT @ 2e-09        | 0.3199     |

### QQP (Table 3)

| Model/Learning Rate       | Accuracy   |
|---------------------------|------------|
| BERT-base (baseline)      | 0.712      |
| DistilBERT @ 2e-05        | **0.9001** |
| DistilBERT @ 2e-07        | 0.617      |
| DistilBERT @ 2e-09        | 0.6318     |

**Note:** QQP result (0.9001) far exceeds BERT baseline (0.712) - remarkable and suspicious!

### Translation (Table 1)

| Learning Rate            | SacreBLEU | COMET    |
|--------------------------|-----------|----------|
| Baseline (1B pretrained) | 18.89     | 0.73     |
| 1e-03                    | 25.76     | 0.80     |
| **1e-04**                | **26.57** | **0.82** |
| 1e-05                    | 23.73     | 0.79     |

---

## Key Observations

### MNLI
- Learning rate 2e-05 achieved 81.77%, close to BERT's 84.6%
- Some overfitting observed between epochs 2-3
- Smaller batch size (16) surprisingly better than larger (32, 64)
- Lower temperature (2) better than higher (11, 20)
- Balanced alpha (0.5) worked well

### QQP
- Learning rate 2e-05 achieved 90.01%, **exceeding** BERT baseline
- Evaluation loss continued decreasing across all epochs
- Temperature 11 showed strongest results (different from MNLI)
- Alpha 0.5 again performed well

### Translation
- Best learning rate for metrics (1e-04) ≠ best for loss
- Created `<|end_of_translation|>` token to prevent hallucinations
- Quantization caused "abysmal" results - avoided
- Loss curves suggest more epochs could improve results

---

## Critical Analysis

### Issue 1: Untrained Teacher Model

**The Problem:** Section 3.1 states:
> *"the 'bert-base-uncased' checkpoint was loaded for the teacher model"*

This is the **base pretrained BERT** - NOT fine-tuned on MNLI. The teacher only knows language modeling (MLM/NSP), not the MNLI classification task.

**Impact:**
- Teacher's soft labels are essentially **random for the task**
- KL loss component provides no useful task-specific signal
- Student learns primarily from CE loss with ground truth labels

**Evidence:** Our config 00 (no teacher, CE-only) achieves ~81-82% accuracy - nearly identical to the paper's 81.77% result with "distillation."

---

### Issue 2: Missing Baseline (No α=0 Experiment)

**The Problem:** The paper never trains the student model **without** a teacher to establish:
1. What accuracy the student achieves alone
2. Whether distillation actually contributes anything

**What Should Have Been Shown:**
```
Student alone (α=0, CE only):    ~81%
Student + Teacher (α=0.5):       81.77%
─────────────────────────────────────────
Distillation contribution:       ~0%
```

**Impact:** Without this baseline, the paper cannot claim that knowledge distillation improved results.

---

### Issue 3: Baseline Comparison Mismatch

**The Problem:** Table 2 compares results against:
> *"BERT-base (baseline): 0.846/0.834"*

This baseline is from Devlin et al. (2019) where BERT was **fine-tuned on MNLI**. But the paper's teacher is `bert-base-uncased` which is **NOT fine-tuned**.

| What               | Model State                        |
|--------------------|-----------------------------------|
| Baseline they cite | BERT fine-tuned on MNLI (84.6%)   |
| Teacher they use   | BERT base pretrained (NOT fine-tuned) |

They're comparing apples to oranges.

---

### Issue 4: QQP Result Anomaly

**The Problem:** Table 3 shows:

| Model                    | QQP Accuracy |
|--------------------------|--------------|
| BERT-base (baseline)     | 0.712        |
| DistilBERT @ 2e-05       | **0.9001**   |

The student (90%) **far exceeds** the teacher baseline (71.2%) - a 19% improvement.

**Questions:**
1. Is the BERT baseline accurate? (Other sources report higher)
2. How can a distilled student dramatically exceed its teacher?
3. Is there a data leakage or evaluation issue?

This result is suspicious and warrants investigation.

---

### Issue 5: Loss Function Ambiguity

**The Problem:** The paper describes alpha as controlling "how much the student model learns from the distillation loss compared to the cross-entropy loss" but doesn't provide the explicit formula.

**From Their Code (glue_notebook.py):**
```python
loss = self.args.alpha * loss_ce + (1. - self.args.alpha) * loss_kd
```

With α=0.5: 50% CE + 50% KL

**Note:** This convention is **inverted** from some implementations where α weights the KL term directly.

---

### Issue 6: Incomplete Ablation Tables

**The Problem:** Section 5.2 mentions testing learning rates, batch sizes, temperatures, and alpha values, but only **learning rate results** are shown in tables. Other hyperparameter results are mentioned only in prose.

**Impact:**
- Results cannot be verified or reproduced precisely
- No clear picture of hyperparameter sensitivity
- Incomplete scientific documentation

---

### Issue 7: Student Initialization Discrepancy

**The Problem:** Section 3.1 states:
> *"a freshly initialized 'distilbert-base-uncased' model configuration was loaded for the student model"*

But "freshly initialized" is ambiguous:
1. Random weights (true fresh initialization)
2. Pretrained DistilBERT weights (from HuggingFace)

The actual code appears to use pretrained weights, not fresh initialization. This significantly affects reproducibility.

---

### Issue 8: Translation Alpha=1.0

**The Problem:** Section 5.1 states translation uses α=1. With their loss formula `α × CE + (1-α) × KL`, this means:
- 100% CE loss
- 0% KL loss

**This is NOT distillation** - it's just supervised fine-tuning with ground truth labels.

---

### Issue Severity Summary

| Issue                  | Severity     | Impact on Claims                       |
|------------------------|--------------|----------------------------------------|
| Untrained teacher      | **Critical** | Invalidates KD benefit claims for GLUE |
| No α=0 baseline        | **Critical** | Cannot prove distillation helped       |
| Baseline mismatch      | **High**     | Misleading comparisons                 |
| QQP anomaly            | **High**     | Suspicious result needs verification   |
| Loss ambiguity         | Medium       | Reproducibility issue                  |
| Incomplete tables      | Medium       | Cannot verify hyperparameter claims    |
| Student init unclear   | Medium       | Reproducibility issue                  |
| Translation α=1.0      | **High**     | Not actually doing distillation        |

---

## Ablation Study Assessment

### What They Did ✓

1. **Learning rate sensitivity** - Tested 3 values, showed dramatic impact
2. **Batch size variation** - Tested 16, 32, 64 (discussed in prose)
3. **Temperature variation** - Tested 2, 11, 20 (discussed in prose)
4. **Alpha variation** - Tested 0.25, 0.5, 1.0 (discussed in prose)

### What's Missing ✗

1. **No α=0 baseline** - Never trained student without teacher to prove distillation's contribution
2. **Results tables incomplete** - Only learning rate shown in tables; temperature, batch size, alpha results only in prose
3. **No component ablation** - e.g., "without temperature scaling" or "KL-only vs CE-only"
4. **Translation uses fixed hyperparameters** - Only learning rate varied

### Gap Analysis

The paper does **hyperparameter tuning** but NOT **formal component ablation**.

To strengthen claims, need:
```
Student alone (α=0):        ~75% (hypothetical)
With distillation (α=0.5):   81.77%
─────────────────────────────────────
Delta from distillation:     +6.77%
```

This would prove: "Distillation adds X% accuracy over training without a teacher."

---

## How Our Work Addresses the Gaps

| Paper Gap             | Our Config                    | Purpose                           |
|-----------------------|-------------------------------|-----------------------------------|
| No baseline           | **00_no-teacher_ce-only**     | Establish true baseline           |
| Untrained teacher only| **01_frozen_base-teacher**    | Match paper's actual setup        |
| No trained teacher    | **02_frozen_sft-teacher**     | Proper KD with expert teacher     |
| No co-training        | **03, 04, 05**                | Explore alternative approaches    |
| Label mapping bug     | **teacher_label_map: [1,2,0]**| Fix TextAttack label ordering     |

Our experiments will provide the missing evidence to properly evaluate knowledge distillation's contribution.

---

## Reproduction Plan

### Phase 1: Exact Reproduction

Reproduce the paper's reported results with identical settings:

**MNLI Config:**
```yaml
model:
  teacher: bert-base-uncased
  student: distilbert-base-uncased
  student_init: fresh

training:
  num_epochs: 3
  batch_size: 16
  learning_rate: 2e-05
  warmup_steps: 200
  weight_decay: 0.01

distillation:
  temperature: 2
  alpha: 0.5
```

**Expected:** MNLI accuracy ~0.817, QQP accuracy ~0.900

### Phase 2: Add Missing Baselines

| Experiment             | Alpha | Description                         |
|------------------------|-------|-------------------------------------|
| `baseline_no_distill`  | 0.0   | Student trained with CE only        |
| `distill_kl_only`      | 1.0   | Pure distillation (no CE loss)      |
| `distill_balanced`     | 0.5   | Paper's best setting                |

This proves: **Distillation contributes +X% over baseline**

### Phase 3: Enhancements Beyond Paper

**Alternative Loss Functions:**

| Loss          | Formula                        | Expected Benefit        |
|---------------|--------------------------------|-------------------------|
| KL Divergence | `KL(student || teacher)`       | Paper's default         |
| Reverse KL    | `KL(teacher || student)`       | Mode-seeking, sharper   |
| MSE           | `MSE(student, teacher)`        | Simpler gradient        |
| Cosine        | `1 - cos_sim(student, teacher)`| Direction alignment     |
| JS Divergence | `0.5*KL(s||m) + 0.5*KL(t||m)`  | Symmetric, stable       |

**Temperature Annealing:**
```python
# Linear annealing: T=4 → T=1 over training
T = T_start - (T_start - T_end) * (step / total_steps)

# Cosine annealing
T = T_end + 0.5 * (T_start - T_end) * (1 + cos(π * step / total_steps))
```

---

## Loss Function Details

The paper uses standard knowledge distillation loss:

```
L_total = (1 - α) × L_CE + α × L_KL

Where:
- L_CE  = Cross-entropy with ground truth labels
- L_KL  = KL divergence between softened distributions
- α     = 0.5 (GLUE) or 1.0 (Translation)
- T     = 2 (temperature for softening)
```

KL divergence formula:
```python
kl_loss = F.kl_div(
    F.log_softmax(student_logits / T, dim=-1),
    F.softmax(teacher_logits / T, dim=-1),
    reduction='batchmean'
) * (T ** 2)
```

The `T²` scaling preserves gradient magnitudes when temperature > 1.

---

## References

- Hinton et al. (2015): "Distilling the Knowledge in a Neural Network"
- Sanh et al. (2019): "DistilBERT, a distilled version of BERT"
- Jiao et al. (2019): "TinyBERT: Distilling BERT for NLU"
- Original code:
  - https://github.com/rtotheich/knowledge_distillation
  - https://github.com/vijayvenkatesan005/Knowledge-Distillation-GLUE-Tasks
