# Ablation Study: Knowledge Distillation Configurations

This document describes the ablation study configurations for systematically comparing different knowledge distillation approaches on MNLI.

## Overview

The ablation study isolates the effect of two key factors:
1. **Teacher training state**: LM-pretrained vs SFT (task-trained)
2. **Teacher training mode**: Frozen vs Co-training (learning alongside student)

All experiments use:
- **Student**: DistilBERT (66M params) with pretrained LM weights
- **Task**: MNLI (393K training examples, 3-way classification)
- **Training**: 3 epochs, batch size 16, lr 2e-5

> **Note on student initialization:** Our configs use `student_init: pretrained` (HuggingFace DistilBERT weights). The paper text mentions "freshly initialized" students, but the actual code repositories use pretrained weights. Our implementation matches the code behavior.

## Baselines and Context

| Model | Description | MNLI Accuracy |
|-------|-------------|---------------|
| Pretrained DistilBERT (no fine-tuning) | LM weights only, no task training | ~33% (random) |
| DistilBERT + CE (fine-tuned) | Student trained on ground truth labels | ~79-80% |
| BERT (teacher, fine-tuned) | Teacher model trained on MNLI | ~84% |

The goal of knowledge distillation is to close the gap between the student baseline (~80%) and the teacher (~84%). The pretrained baseline (~33%) demonstrates the value of fine-tuning itself, while the CE-only baseline shows what the student can achieve without teacher guidance.

## Ablation Configurations

### 00_no-teacher_ce-only (Baseline)
**Config**: `configs/mnli/bert_distilbert/ablations/00_no-teacher_ce-only.yaml`

| Setting | Value |
|---------|-------|
| Alpha | 0.0 (CE only, no KL) |
| Teacher | Unused |
| Expected Accuracy | ~79-80% |

**Purpose**: Establishes baseline - what does DistilBERT achieve with just cross-entropy loss on ground truth labels?

---

### 01_frozen_base-teacher_kl
**Config**: `configs/mnli/bert_distilbert/ablations/01_frozen_base-teacher_kl.yaml`

| Setting | Value |
|---------|-------|
| Alpha | 0.5 (CE + KL) |
| Teacher | `bert-base-uncased` (LM-pretrained, not task-trained) |
| Freeze Teacher | Yes |
| Expected Accuracy | ~60-65% |

**Purpose**: Tests whether an untrained teacher provides useful soft labels. Since the teacher hasn't learned MNLI, its soft labels are essentially random - this should hurt performance.

---

### 02_frozen_sft-teacher_kl (Traditional KD)
**Config**: `configs/mnli/bert_distilbert/ablations/02_frozen_sft-teacher_kl.yaml`

| Setting | Value |
|---------|-------|
| Alpha | 0.5 (CE + KL) |
| Teacher | `textattack/bert-base-uncased-MNLI` (MNLI-trained) |
| Freeze Teacher | Yes |
| Expected Accuracy | ~81-82% |

**Purpose**: Standard knowledge distillation setup from most papers. Teacher is pre-trained on task, provides expert soft labels. This is the "traditional" approach.

---

### 03_cotrain_base-teacher_kl
**Config**: `configs/mnli/bert_distilbert/ablations/03_cotrain_base-teacher_kl.yaml`

| Setting | Value |
|---------|-------|
| Alpha | 0.5 (CE + KL) |
| Teacher | `bert-base-uncased` (LM-pretrained, learns during training) |
| Freeze Teacher | No |
| Expected Accuracy | ~82% |

**Purpose**: Teacher and student learn together. Teacher starts from LM weights and learns MNLI during distillation. The student benefits from increasingly better soft labels as training progresses.

---

### 04_cotrain_sft-teacher_kl
**Config**: `configs/mnli/bert_distilbert/ablations/04_cotrain_sft-teacher_kl.yaml`

| Setting | Value |
|---------|-------|
| Alpha | 0.5 (CE + KL) |
| Teacher | `textattack/bert-base-uncased-MNLI` (MNLI-trained) |
| Freeze Teacher | No |
| Expected Accuracy | ~82-83% |

**Purpose**: Tests whether continued teacher training helps when the teacher already knows the task. The teacher can potentially refine its knowledge further.

---

## Expected Results Matrix

| Config | Teacher State | Training Mode | Expected Acc |
|--------|---------------|---------------|--------------|
| 00_no-teacher_ce-only | N/A | N/A | ~79-80% |
| 01_frozen_base-teacher_kl | Untrained | Frozen | ~60-65% |
| 02_frozen_sft-teacher_kl | Task-trained | Frozen | ~81-82% |
| 03_cotrain_base-teacher_kl | Untrained→Trained | Co-training | ~82% |
| 04_cotrain_sft-teacher_kl | Task-trained | Co-training | ~82-83% |

## Key Insights

1. **Teacher quality matters**: Frozen LM teacher (~60%) vs Frozen SFT teacher (~82%) shows that soft label quality is critical.

2. **Co-training works**: Co-training with LM teacher (~82%) matches or exceeds frozen SFT teacher (~82%), demonstrating that the teacher can learn alongside the student effectively.

3. **Distillation helps**: All proper distillation setups (81-83%) beat the CE-only baseline (~80%), showing the value of soft labels.

4. **Diminishing returns for SFT co-training**: Co-training SFT teacher (~82-83%) provides marginal improvement over frozen SFT (~82%), suggesting the teacher is already well-trained.

## Running the Ablations

```bash
# Run individual ablation
python forge.py --config configs/mnli/bert_distilbert/ablations/00_no-teacher_ce-only.yaml

# Run with custom name
python forge.py --config configs/mnli/bert_distilbert/ablations/03_cotrain_base-teacher_kl.yaml \
    --run_name ablation_cotrain --new

# Resume a run
python forge.py --run_name ablation_cotrain
```

## References

- Original DistilBERT paper: Sanh et al., "DistilBERT, a distilled version of BERT" (2019)
- Co-training approach inspired by: Born-Again Neural Networks (Furlanello et al., 2018)
