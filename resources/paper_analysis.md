# "Less is More" Paper Analysis

**Paper:** "Less is More: Achieving Smaller, Faster Models for Edge Computing with Knowledge Distillation"
**Authors:** Richard Yue & Vijay Venkatesan (Northeastern University)
**PDF:** [wip_docs/Yue_Venkatesan_Less_is_More.pdf](wip_docs/Yue_Venkatesan_Less_is_More.pdf)

---

## Target Results to Reproduce

| Task | Metric | Target | Notes |
|------|--------|--------|-------|
| MNLI | Accuracy | 0.817 | 3-class classification |
| QQP | Accuracy | 0.900 | Binary classification |
| Translation (EN→FR) | BLEU | ~27 | Europarl subset |
| Translation (EN→FR) | COMET | ~0.82 | |

---

## Model Configurations

### BERT → DistilBERT (GLUE Tasks)

| Model | Layers | Hidden Size | Attention Heads | Parameters |
|-------|--------|-------------|-----------------|------------|
| BERT-base (teacher) | 12 | 768 | 12 | 110M |
| DistilBERT (student) | 6 | 768 | 12 | 66M |

**Key:** Student is freshly initialized (not pretrained DistilBERT checkpoint)

### Llama 3.2 (Translation)

| Model | Parameters | Notes |
|-------|------------|-------|
| Teacher | 3B | `meta-llama/Llama-3.2-3B-Instruct` |
| Student | 1B | Fresh config (not pretrained weights) |

---

## Hyperparameters Tested

### GLUE Tasks (Section 5.2)

| Hyperparameter | Values Tested | Best |
|----------------|---------------|------|
| Learning rate | 2e-05, 2e-07, 2e-09 | **2e-05** |
| Batch size | 16, 32, 64 | **16** |
| Temperature | 2, 11, 20 | **2** |
| Alpha | 0.25, 0.5, 1.0 | **0.5** |

**Other settings:**
- Epochs: 3
- GPU: P100
- Training time: 8 hours max

### Translation (Section 5.1)

| Hyperparameter | Values Tested | Best |
|----------------|---------------|------|
| Learning rate | 1e-03, 1e-04, 1e-05 | **1e-04** |

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

## Results from Paper

### MNLI (Table 2)

| Model/Learning Rate | Accuracy |
|---------------------|----------|
| BERT-base (baseline) | 0.846 / 0.834 |
| DistilBERT @ 2e-05 | **0.8177** |
| DistilBERT @ 2e-07 | 0.6245 |
| DistilBERT @ 2e-09 | 0.3199 |

### QQP (Table 3)

| Model/Learning Rate | Accuracy |
|---------------------|----------|
| BERT-base (baseline) | 0.712 |
| DistilBERT @ 2e-05 | **0.9001** |
| DistilBERT @ 2e-07 | 0.617 |
| DistilBERT @ 2e-09 | 0.6318 |

**Note:** QQP result (0.9001) far exceeds BERT baseline (0.712) - remarkable!

### Translation (Table 1)

| Learning Rate | SacreBLEU | COMET |
|---------------|-----------|-------|
| Baseline (1B pretrained) | 18.89 | 0.73 |
| 1e-03 | 25.76 | 0.80 |
| **1e-04** | **26.57** | **0.82** |
| 1e-05 | 23.73 | 0.79 |

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

## Key Observations from Paper

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

## Reproduction Checklist

### Must Have
- [ ] Match hyperparameters exactly (lr=2e-05, batch=16, temp=2, α=0.5)
- [ ] Fresh student initialization (not pretrained weights)
- [ ] 3 epochs training
- [ ] Same loss function (KL divergence + CE)

### Should Add (for stronger paper)
- [ ] Baseline configs (α=0) to show distillation delta
- [ ] Reproducibility seeds
- [ ] Full hyperparameter tables (not just learning rate)
- [ ] Temperature/alpha ablation tables

### Nice to Have
- [ ] Different loss functions (reverse KL, MSE, cosine)
- [ ] Temperature annealing experiments
- [ ] More epochs for translation

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

---

## Paper Reproduction Plan

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

Create configs to establish proper ablation baselines:

| Experiment | Alpha | Description |
|------------|-------|-------------|
| `baseline_no_distill` | 0.0 | Student trained with CE only (no teacher) |
| `distill_kl_only` | 1.0 | Pure distillation (no CE loss) |
| `distill_balanced` | 0.5 | Paper's best setting |

This proves: **Distillation contributes +X% over baseline**

### Phase 3: Enhancements Beyond Paper

#### 3a. Alternative Loss Functions

| Loss | Formula | Expected Benefit |
|------|---------|------------------|
| KL Divergence | `KL(student || teacher)` | Paper's default |
| Reverse KL | `KL(teacher || student)` | Mode-seeking, sharper |
| MSE | `MSE(student_logits, teacher_logits)` | Simpler gradient |
| Cosine | `1 - cos_sim(student, teacher)` | Direction alignment |
| JS Divergence | `0.5*KL(s||m) + 0.5*KL(t||m)` | Symmetric, stable |

#### 3b. Temperature Annealing

Instead of fixed temperature, decay over training:

```python
# Linear annealing: T=4 → T=1 over training
T = T_start - (T_start - T_end) * (step / total_steps)

# Cosine annealing
T = T_end + 0.5 * (T_start - T_end) * (1 + cos(π * step / total_steps))
```

**Hypothesis:** High T early (explore soft labels) → Low T late (sharpen)

#### 3c. Learning Rate Schedules

- Linear warmup + decay (paper's approach)
- Cosine annealing with warm restarts
- One-cycle policy

---

## Grid Search Strategy

### Approach Comparison

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Grid Search** | Exhaustive, reproducible | Expensive O(n^k) | Small search spaces |
| **Random Search** | More efficient | Less systematic | Medium spaces |
| **Optuna (Bayesian)** | Smart sampling, pruning | Less reproducible | Large spaces |
| **W&B Sweeps** | Visual, distributed | Requires W&B | Team collaboration |

### Recommended: Optuna + W&B Tracking

Use Optuna for optimization, W&B for tracking:

```python
import optuna
import wandb

def objective(trial):
    # Suggest hyperparameters
    config = {
        "learning_rate": trial.suggest_float("lr", 1e-6, 1e-4, log=True),
        "temperature": trial.suggest_float("temp", 1.0, 10.0),
        "alpha": trial.suggest_float("alpha", 0.0, 1.0),
        "batch_size": trial.suggest_categorical("batch", [16, 32, 64]),
        "loss_fn": trial.suggest_categorical("loss", ["kl", "mse", "cosine"]),
    }

    # Log to W&B
    wandb.init(project="distillation-sweep", config=config)

    # Train and return metric
    accuracy = train(config)
    wandb.log({"accuracy": accuracy})

    return accuracy

# Run optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
```

### Search Space Definition

**Tier 1 (Most Important):**
```yaml
learning_rate: [1e-6, 1e-4]  # log scale
alpha: [0.0, 1.0]            # continuous
temperature: [1, 10]         # continuous
```

**Tier 2 (Secondary):**
```yaml
batch_size: [16, 32, 64, 128, 256]
loss_function: [kl, reverse_kl, mse, cosine, js]
warmup_steps: [100, 200, 500]
```

**Tier 3 (Advanced):**
```yaml
temp_annealing: [none, linear, cosine]
temp_start: [2, 4, 8]
temp_end: [1, 2]
lr_schedule: [linear, cosine, one_cycle]
```

### Recommended Search Order

1. **Quick validation** (5 trials): Verify training works
2. **Coarse search** (20 trials): Find promising regions
3. **Fine search** (50 trials): Optimize in promising regions
4. **Final training** (3 seeds): Best config with multiple seeds

---

## Experiment Tracking Schema

### W&B Run Naming Convention

```
{task}_{teacher}_{student}_{experiment_type}_{timestamp}
```

Examples:
- `mnli_bert_distilbert_baseline_20240115`
- `mnli_bert_distilbert_sweep_alpha_20240115`
- `qqp_bert_distilbert_loss_mse_20240115`

### Metrics to Log

```python
wandb.log({
    # Loss components
    "train/loss_total": loss.item(),
    "train/loss_ce": ce_loss.item(),
    "train/loss_kl": kl_loss.item(),

    # Evaluation
    "eval/accuracy": accuracy,
    "eval/f1": f1,
    "eval/loss": eval_loss,

    # Training dynamics
    "train/learning_rate": lr,
    "train/temperature": current_temp,  # if annealing
    "train/epoch": epoch,
    "train/step": step,

    # System
    "system/gpu_memory_gb": gpu_mem,
})
```

### Experiment Tags

```python
wandb.init(
    tags=[
        "reproduction",      # or "enhancement", "ablation"
        "mnli",             # task
        "bert-distilbert",  # model pair
        "alpha-sweep",      # experiment type
    ]
)
```

---

## Implementation Priority

### Week 1: Reproduction
1. ✅ Core training pipeline
2. ⬜ Add reproducibility seeds
3. ⬜ Verify MNLI result (~0.817)
4. ⬜ Verify QQP result (~0.900)

### Week 2: Baselines & Ablation
1. ⬜ Create baseline config (α=0)
2. ⬜ Run baseline experiments
3. ⬜ Create ablation table (temperature, alpha)
4. ⬜ Document distillation delta

### Week 3: Enhancements
1. ⬜ Implement loss functions module
2. ⬜ Add temperature annealing
3. ⬜ Set up Optuna integration
4. ⬜ Run hyperparameter sweep

### Week 4: Analysis & Report
1. ⬜ Analyze sweep results
2. ⬜ Create visualizations
3. ⬜ Write comparison tables
4. ⬜ Document findings

---

## References

- Hinton et al. (2015): "Distilling the Knowledge in a Neural Network"
- Sanh et al. (2019): "DistilBERT, a distilled version of BERT"
- Jiao et al. (2019): "TinyBERT: Distilling BERT for NLU"
- Original code:
  - https://github.com/rtotheich/knowledge_distillation
  - https://github.com/vijayvenkatesan005/Knowledge-Distillation-GLUE-Tasks
