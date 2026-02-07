# Workbench Post-Hoc Analysis Summary

Analysis of Ablation 04 (co-train + SFT teacher + KL) on MNLI.

**Models:**
- Teacher: `textattack/bert-base-uncased-MNLI` (12L, 109.5M)
- Student: `outputs/mnli_ablations_04_cotrain_sft-teacher_kl_baseline` (6L, 66.4M)

---

## 1. Linear Probing

Measures linearly-extractable task information per layer.

| Layer | Teacher | Student |
|-------|---------|---------|
| embeddings | 34.49% | 34.49% |
| layer_1 | 35.76% | 37.39% |
| layer_2 | 37.60% | 44.98% |
| layer_3 | 42.44% | 52.01% |
| layer_4 | 45.34% | 64.29% |
| layer_5 | 42.38% | 71.22% |
| layer_6 | 51.35% | **80.54%** |
| ... | ... | - |
| layer_12 | **61.90%** | - |

**Finding:** Student (80.5%) > Teacher (61.9%). Compression forces more linearly-separable representations.

---

## 2. CKA Layer Alignment

Representational similarity between teacher and student layers.

| Student Layer | Best Teacher Match | CKA |
|---------------|-------------------|-----|
| layer_1 | layer_1 | 0.841 |
| layer_2 | layer_3 | 0.845 |
| layer_3 | layer_5 | 0.783 |
| layer_4 | layer_5 | 0.537 |
| layer_5 | layer_11 | 0.339 |
| layer_6 | layer_12 | 0.224 |

**Finding:** Early layers align well (CKA ~0.84), final layers diverge (CKA ~0.22). Student develops own representations.

---

## 3. Attention Entropy

Average entropy across attention heads (lower = more focused).

| Model | Mean Entropy |
|-------|--------------|
| Teacher | 2.137 |
| Student | 1.901 |
| **Ratio** | **0.889** |

**Finding:** Student has 11% lower entropy—more focused attention patterns.

---

## 4. Prediction Disagreement

| Metric | Teacher | Student |
|--------|---------|---------|
| Accuracy | 84.57% | 82.67% |
| Agreement | - | 89.3% |

| Category | % |
|----------|---|
| Both correct | 78.7% |
| Both wrong | 10.6% |
| Teacher only correct | 5.9% |
| Student only correct | 4.0% |

| Confidence | Teacher | Student |
|------------|---------|---------|
| Mean | 0.910 | 0.931 |
| When wrong | 0.781 | 0.825 |

**Finding:** 89% agreement. Student is overconfident when wrong (calibration issue).

---

## Summary Table

| Analysis | Key Finding |
|----------|-------------|
| Probing | Student 80.5% > Teacher 62% |
| CKA | Early layers align, final diverge |
| Attention | Student 11% lower entropy |
| Disagreement | 89% agree, student overconfident |

---

## Interpretation

Distillation produces a student with:
1. **More task-focused representations** — higher probing accuracy despite fewer layers
2. **Sharper attention** — lower entropy, more selective token focus
3. **Different final-layer strategy** — low CKA with teacher but effective
4. **Calibration issues** — overconfident on wrong predictions

The 2% accuracy gap (84.6% → 82.7%) is explained by the 5.9% "teacher-only-correct" cases—knowledge that didn't transfer due to student capacity constraints.

---

*Source: `workbench/results/` (2026-02-06)*
*Commands: `probe`, `cka`, `attention-entropy`, `disagreement`*
