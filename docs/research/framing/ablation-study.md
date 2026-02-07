# Ablation Study: Knowledge Distillation Configurations

This document describes the ablation study configurations for systematically comparing different knowledge distillation approaches.

---

## Overview

The ablation study isolates the effect of two key factors:
1. **Teacher training state**: LM-pretrained vs SFT (task-trained)
2. **Teacher training mode**: Frozen vs Co-training (learning alongside student)

All experiments use:
- **Student**: DistilBERT (66M params) with pretrained LM weights
- **Task**: MNLI (393K training examples, 3-way classification)
- **Training**: 3 epochs, batch size 16, lr 2e-5

---

## Ablation Configurations

### 00_no-teacher_ce-only (Baseline)

| Setting | Value |
|---------|-------|
| Alpha | 0.0 (CE only, no KL) |
| Teacher | Unused |
| Purpose | Establishes baseline |

---

### 01_frozen_base-teacher_kl

| Setting | Value |
|---------|-------|
| Alpha | 0.5 (CE + KL) |
| Teacher | `bert-base-uncased` (LM-pretrained) |
| Freeze Teacher | Yes |
| Purpose | Tests whether an untrained teacher provides useful soft labels |

---

### 02_frozen_sft-teacher_kl (Traditional KD)

| Setting | Value |
|---------|-------|
| Alpha | 0.5 (CE + KL) |
| Teacher | `textattack/bert-base-uncased-MNLI` (MNLI-trained) |
| Freeze Teacher | Yes |
| Purpose | Standard knowledge distillation setup |

---

### 03_cotrain_base-teacher_kl

| Setting | Value |
|---------|-------|
| Alpha | 0.5 (CE + KL) |
| Teacher | `bert-base-uncased` (learns during training) |
| Freeze Teacher | No |
| Purpose | Teacher and student learn together |

---

### 04_cotrain_sft-teacher_kl

| Setting | Value |
|---------|-------|
| Alpha | 0.5 (CE + KL) |
| Teacher | `textattack/bert-base-uncased-MNLI` (MNLI-trained) |
| Freeze Teacher | No |
| Purpose | Tests whether continued teacher training helps |

---

## Results Matrix

| Config | Teacher State | Training Mode | MNLI Acc | QQP Acc |
|--------|---------------|---------------|----------|---------|
| 00_no-teacher_ce-only | N/A | N/A | 80.94% | 89.71% |
| 01_frozen_base-teacher_kl | Untrained | Frozen | 81.25% | 90.19% |
| 02_frozen_sft-teacher_kl | Task-trained | Frozen | 79.26% | 65.50% |
| 03_cotrain_base-teacher_kl | Untrained→Trained | Co-training | 82.06% | 90.37% |
| 04_cotrain_sft-teacher_kl | Task-trained | Co-training | 82.62% | 90.28% |

---

## Key Insights

1. **Co-training works**: Co-training consistently outperforms frozen approaches

2. **Frozen SFT can hurt**: The frozen expert teacher paradoxically underperforms the CE-only baseline

3. **Untrained > Trained when frozen**: Due to overconfident wrong predictions in SFT teachers

4. **Distillation gains are marginal**: Best distillation adds only ~1-2% over CE-only baseline

---

## Running the Ablations

```bash
# Run individual ablation
python distill.py --config configs/mnli/bert_distilbert/ablations/00_no-teacher_ce-only.yaml

# Run with custom name
python distill.py --config configs/mnli/bert_distilbert/ablations/03_cotrain_base-teacher_kl.yaml \
    --run_name ablation_cotrain
```
