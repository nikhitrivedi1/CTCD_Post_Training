# GLUE Ablation Coefficients

Resolved coefficient values for all KD ablation configs across GLUE classification tasks.
Each task lives under `configs/classification/{task}/distilbert/bert-base/ablations/`.

**Student loss:** `L_student = alpha_h * CE(student) + alpha_s * KL(student || teacher_detached)`
**Teacher loss:** `L_teacher = beta_h * CE(teacher) + beta_s * KL(teacher || student_detached)`

Beta values for frozen teachers are irrelevant (no teacher loss is computed), shown as `—`.

---

## Ablation Definitions

| # | Config | freeze | alpha_h | alpha_s | beta_h | beta_s | Description |
|---|--------|--------|---------|---------|--------|--------|-------------|
| 00 | `00_sft` | `true` | 1.0 | 0.0 | — | — | CE-only baseline; no teacher loaded |
| 01 | `01_kd_frozen-base` | `true` | 1.0 | 1.0 | — | — | Frozen LM-pretrained teacher; untrained soft labels |
| 02 | `02_kd_frozen-sft` | `true` | 1.0 | 1.0 | — | — | Frozen task-trained teacher; standard KD |
| 03 | `03_kd_cotrain-base` | `false` | 1.0 | 1.0 | 1.0 | 0.0 | Co-training, LM-pretrained teacher; CE-only teacher loss |
| 04 | `04_kd_cotrain-sft` | `false` | 1.0 | 1.0 | 1.0 | 0.0 | Co-training, task-trained teacher; CE-only teacher loss |
| 06 | `06_ctcd_sample` *(MNLI only)* | `false` | 1.0 | 1.0 | 1.0 | 1.0 | CTCD; both student and teacher receive CE + KL |

> **Note on beta defaults:** Configs 03 and 04 do not set `beta_h`/`beta_s` in YAML.
> `resolve_distillation_coefficients` applies the co-training default of `(beta_h=1.0, beta_s=0.0)`.

---

## Per-Task Status

| Task | 00 | 01 | 02 | 03 | 04 | Notes |
|------|----|----|----|----|----|----|
| **cola** | ✓ | ✓ | ✓ | ✓ | ✓ | |
| **mnli** | ✓ | ✓ | ✓ | ✓ | ✓ | Also has `06_ctcd_sample` |
| **mrpc** | ✓ | ✓ | ✓ | ✓ | ✓ | |
| **qnli** | ✓ | ✓ | ✓ | ✓ | ✓ | |
| **qqp** | ✓ | ✓ | ✓ | ✓ | ✓ | |
| **rte** | ✓ | ✓ | ✓ | ✓ | ✓ | |
| **sst2** | ✓ | ✓ | ✓ | ✓ | ✓ | |
| **wnli** | ✓ | ✓ | ✓ | ✓ | ✓ | |

All configs verified as of 2026-04-01. Ablations 01–04 unified at `alpha_h: 1.0, alpha_s: 1.0`.
Ablation 00 uses `alpha_h: 1.0, alpha_s: 0.0` by design (CE-only baseline, no teacher loaded).

---

## Design Notes

**01 vs 03** — Same LM-pretrained teacher, frozen vs co-training. Isolates the value of letting the teacher learn alongside the student.

**02 vs 04** — Same task-trained teacher, frozen vs co-training. Tests whether continued teacher training helps when the teacher already knows the task.

**03/04 vs 06 (MNLI)** — Pure co-training vs CTCD. The only difference is `beta_s: 1.0` in 06, which routes student logits back into the teacher loss.

**Coefficient scale** — All configs are unified at `alpha_h=1.0, alpha_s=1.0`, enabling direct comparison of loss curves across all tasks and ablations.

---

## Entropy Metrics

All KD runs using `DistillationTrainer` also log two always-on confidence metrics:

- `teacher_entropy` — Batch-mean Shannon entropy (in nats) of the teacher’s output distribution over task labels, computed from the teacher logits after label remapping and any vocab alignment.
- `student_entropy` — Batch-mean Shannon entropy (in nats) of the student’s output distribution over the same label space.

Formally, for logits \(\ell\) and induced probabilities \(p = \mathrm{softmax}(\ell)\), entropy is
\(H(p) = -\mathbb{E}_{c \sim p}[\log p(c)]\), averaged over the batch.
These metrics are logged at training time for every KD configuration, independent of \(\alpha_h\), \(\alpha_s\), \(\beta_h\), or \(\beta_s\), and surface in W&B alongside the loss components.
