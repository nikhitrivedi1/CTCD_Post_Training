# MNLI Model Label Mapping Issue

## Problem Summary

Different pre-trained MNLI models have **inconsistent internal logit orderings** that don't match the GLUE standard expectation of `[entailment, neutral, contradiction]` (0, 1, 2).

## GLUE Standard Labels

GLUE MNLI expects:
- **0 = entailment**
- **1 = neutral**
- **2 = contradiction**

## Model Internal Orderings

### TextAttack BERT MNLI (`textattack/bert-base-uncased-MNLI`)
- **Internal ordering**: `[contradiction, entailment, neutral]` (CEN)
- **Config labels**: Generic `LABEL_0, LABEL_1, LABEL_2` (no semantic meaning)
- **Distillation logit mapping**: `[1, 2, 0]` (reorders teacher logits)
- **Evaluation prediction mapping**: `[2, 0, 1]` (maps predictions)
- **Accuracy**: ~79% distillation accuracy, 84% teacher evaluation accuracy

### TextAttack DistilBERT MNLI (`textattack/distilbert-base-uncased-MNLI`)
- **Internal ordering**: `[contradiction, entailment, neutral]` (CEN)
- **Config labels**: Generic `LABEL_0, LABEL_1, LABEL_2`
- **Distillation logit mapping**: `[1, 2, 0]` (reorders teacher logits)
- **Evaluation prediction mapping**: `[2, 0, 1]` (maps predictions)

### Facebook BART MNLI (`facebook/bart-large-mnli`)
- **Internal ordering**: `[contradiction, neutral, entailment]` (CNE)
- **Config labels**: Proper `0=contradiction, 1=neutral, 2=entailment`
- **Required mapping**: `[2, 1, 0]` (different from TextAttack!)

## How to Apply the Mapping

**Important**: Apply different mappings for distillation vs evaluation!

### For Distillation (Logits)
Reorder teacher logit columns to align soft targets:
```python
# teacher_label_map: [1, 2, 0] for TextAttack models
teacher_logits = teacher_logits[:, [1, 2, 0]]  # Reorder columns
```

### For Evaluation (Predictions)  
Map predicted class indices:
```python
# label_map: [2, 0, 1] for TextAttack models  
pred_corrected = label_map[pred_raw]
```

## Evidence from Testing

### On GLUE Validation Examples

Testing on first 5 GLUE validation examples:

| Example | True Label | Raw Pred | Mapped Pred | Correct? |
|---------|------------|----------|-------------|----------|
| 1 | neutral (1) | 2 | 1 | ✓ |
| 2 | contradiction (2) | 0 | 2 | ✓ |
| 3 | entailment (0) | 1 | 0 | ✓ |
| 4 | contradiction (2) | 0 | 2 | ✓ |
| 5 | contradiction (2) | 0 | 2 | ✓ |

**Raw accuracy**: 0/5 = 0.0%
**Mapped accuracy**: 5/5 = 100.0%

### Overall Performance

- **TextAttack MNLI teacher evaluation**: ~84% accuracy with prediction mapping `[2, 0, 1]`
- **Distillation accuracy**: ~79% with logit mapping `[1, 2, 0]`
- **Without mapping**: ~8.97% accuracy (random performance)

## Root Cause

- **Model cards are incomplete**: TextAttack models use generic `LABEL_0/1/2` instead of semantic labels
- **Inconsistent training**: Different research groups fine-tune MNLI models with different internal label orderings
- **Documentation gap**: Neither TextAttack docs nor model cards explain the internal ordering

## Implications for Knowledge Distillation

- **Teacher logit reordering**: Use `teacher_label_map` in distillation configs to reorder teacher soft targets
- **Teacher prediction mapping**: Apply inverse mapping when evaluating teacher models directly
- **Student evaluation**: Distilled students should predict in GLUE order (no mapping needed)
- **GLUE compliance**: All models should follow GLUE's `[entailment, neutral, contradiction]` standard

## Recommendations

1. **For distillation**: Use `teacher_label_map: [1, 2, 0]` to reorder teacher logits
2. **For evaluation**: Use `label_map: [2, 0, 1]` to map predictions  
3. **For new MNLI teachers**: Verify both logit and prediction mappings experimentally
4. **For model cards**: Include semantic label mappings, not just generic LABEL_0/1/2

## Files Affected

- `configs/mnli/bert_distilbert/ablations/02_frozen_sft-teacher_kl.yaml`
- `configs/mnli/bert_distilbert/ablations/04_cotrain_sft-teacher_kl.yaml`
- `scripts/eval_baselines.py` (includes mapping logic)
- `src/training/trainer.py` (applies mapping during distillation)

## Date Documented

February 2, 2026</content>
<parameter name="filePath">/Users/karthik/Projects/Coursework/CS-7150-DL/Distillation/docs/mnli-model-label-mapping.md