# Metrics Reference

This document describes all metrics logged during training, including task metrics, distillation metrics, and knowledge transfer metrics.

---

## Task Metrics

These are the standard metrics computed during evaluation, depending on the task type.

### Classification Tasks

| Metric | Description | Tasks | Range |
|--------|-------------|-------|-------|
| `accuracy` | Fraction of correct predictions | MNLI, SST-2 | 0-1 |
| `f1` | Weighted F1 score | QQP | 0-1 |
| `mcc` | Matthews Correlation Coefficient | CoLA | -1 to 1 |

**Primary metrics by task (per GLUE benchmark):**
- MNLI: `accuracy`
- QQP: `f1`
- SST-2: `accuracy`
- CoLA: `mcc`

### Translation Tasks

| Metric | Description | Tasks | Range |
|--------|-------------|-------|-------|
| `bleu` | SacreBLEU score | Europarl | 0-100 |
| `comet` | COMET score (if available) | Europarl | ~0-1 |

---

## Loss Metrics

### Training Loss Components

| Metric | Description | When Logged |
|--------|-------------|-------------|
| `train/loss` | Total training loss (CE + α×KL) | Every logging step |
| `train/loss_total` | Same as train/loss | Every logging step |
| `train/loss_ce` | Cross-entropy loss component | When using distillation |
| `train/loss_kl` | KL divergence loss component | When using distillation |
| `train/teacher_ce_loss` | Teacher's CE loss | Co-training only |

### Evaluation Loss

| Metric | Description | Notes |
|--------|-------------|-------|
| `eval/loss` | Evaluation loss | For distillation: includes KL term |

**Important:** `eval/loss` is misleading for comparing distillation vs CE-only configs because:
- CE-only: `loss = CE`
- Distillation: `loss = CE + α×KL`

Always compare task metrics (accuracy, F1, MCC), not loss.

---

## Pre/Post Training Metrics

When `evaluation.log_pre_post_metrics: true` (default), metrics are logged for both teacher and student at the start and end of training.

### Metric Sets

| Prefix | Model | Phase | Purpose |
|--------|-------|-------|---------|
| `student/pre/*` | Student | Before training | Baseline starting point |
| `student/post/*` | Student | After training | Final performance |
| `teacher/pre/*` | Teacher | Before training | Target to match |
| `teacher/post/*` | Teacher | After training | Co-training effect |

### Example W&B Keys

```
student/pre/accuracy
student/pre/f1
student/pre/mcc
student/pre/loss
student/post/accuracy
...
teacher/pre/accuracy
teacher/post/accuracy
```

---

## Knowledge Transfer Metrics

These metrics quantify how effectively knowledge was transferred from teacher to student.

### Teacher Advantage

```
teacher_advantage = teacher_pre[metric] - student_pre[metric]
```

| W&B Key | Description |
|---------|-------------|
| `transfer/teacher_advantage` | Room for improvement (gap between teacher and student at start) |

**Interpretation:**
- Large positive value → teacher is much better → more potential for transfer
- Near zero → teacher and student similar → limited transfer opportunity
- Negative → student already better than teacher (shouldn't happen)

**Example:**
```
Teacher accuracy (pre): 85%
Student accuracy (pre): 70%
Teacher advantage: +15%  (15 percentage points of room to improve)
```

### Student Gain

```
student_gain = student_post[metric] - student_pre[metric]
```

| W&B Key | Description |
|---------|-------------|
| `transfer/student_gain` | How much the student improved during training |

**Interpretation:**
- Positive → student improved
- Negative → student got worse (training failure)
- Compare to CE-only baseline to see if KD helped

**Example:**
```
Student accuracy (pre):  70%
Student accuracy (post): 82%
Student gain: +12%
```

### Knowledge Transfer Efficiency

```
efficiency = student_gain / teacher_advantage
          = (student_post - student_pre) / (teacher_pre - student_pre)
```

| W&B Key | Description |
|---------|-------------|
| `transfer/efficiency` | Fraction of teacher's advantage captured by student |

**Interpretation:**
- `1.0` (100%) → Student matched teacher exactly
- `>1.0` (>100%) → Student exceeded teacher
- `0.5` (50%) → Student captured half the advantage
- `0` → No improvement (student stayed at baseline)
- `<0` → Negative transfer (KD hurt performance)

**Example:**
```
Teacher advantage: 15%
Student gain: 12%
Transfer efficiency: 12/15 = 80%
```

The student captured 80% of the teacher's advantage.

---

## Comparison to RL Concepts

The transfer metrics are conceptually similar to advantage functions in Reinforcement Learning, particularly GRPO (Group Relative Policy Optimization):

| Concept | RL (GRPO) | KD Transfer |
|---------|-----------|-------------|
| Reference | Group of k responses | Teacher performance |
| Baseline | Group mean reward | Student pre-training |
| Signal | "Better than average?" | "Closed gap to teacher?" |
| Normalization | By group std | By teacher_advantage |

**Key difference:** RL advantages are training signals (shape gradients), while KD transfer metrics are evaluation metrics (measure success after training).

---

## Peak Metrics

The `PeakMetricsCallback` tracks the best value seen for each metric during training.

| W&B Key | Description |
|---------|-------------|
| `peak/eval_accuracy` | Best accuracy seen during training |
| `peak/eval_f1` | Best F1 seen during training |
| `peak/eval_mcc` | Best MCC seen during training |

Useful when `load_best_model_at_end=True` may select a different checkpoint than the peak.

---

## Configuration

### Enable/Disable Pre/Post Metrics

```yaml
evaluation:
  log_pre_post_metrics: true  # default: true
```

### Select Primary Metric

The primary metric is automatically determined by task (per GLUE benchmark), but you can verify it in the config:

```yaml
# Automatically set based on task:
# - mnli, sst2: accuracy
# - qqp: f1
# - cola: mcc
```

---

## Analyzing Transfer Efficiency

### Healthy Transfer (efficiency > 0.5)

```
teacher_pre: 85%
student_pre: 70%
student_post: 82%
efficiency: 80%
```

Student learned effectively from teacher.

### Poor Transfer (efficiency < 0.3)

```
teacher_pre: 85%
student_pre: 70%
student_post: 73%
efficiency: 20%
```

KD barely helped. Consider:
- Is teacher overconfident when wrong?
- Is dataset too small?
- Try co-training instead of frozen teacher

### Negative Transfer (efficiency < 0)

```
teacher_pre: 85%
student_pre: 70%
student_post: 68%
efficiency: -13%
```

KD hurt performance. This indicates catastrophic failure, often caused by:
- Frozen teacher confidently wrong (QQP pattern)
- Too small dataset (CoLA pattern)

**Fix:** Use co-training or skip KD entirely.

---

## W&B Dashboard Tips

### Useful Filters

1. **Compare transfer efficiency across runs:**
   - Y-axis: `transfer/efficiency`
   - Group by: config

2. **Check if KD helped:**
   - Compare `student/post/accuracy` vs CE-only baseline

3. **Monitor training health:**
   - Watch `eval/loss` trend (should decrease or stabilize)
   - Watch `train/teacher_ce_loss` in co-training (should decrease)

### Warning Signs

| Signal | What It Means |
|--------|---------------|
| `eval/loss` stuck from start | Teacher dominance (QQP frozen pattern) |
| `eval/loss` increases mid-training | Overfitting (CoLA pattern) |
| `transfer/efficiency < 0` | Negative transfer (KD hurting) |
| `transfer/teacher_advantage ≈ 0` | No room for transfer |
