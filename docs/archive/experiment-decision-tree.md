# Experiment Decision Tree

This document maps the experimental hierarchy for knowledge distillation research.

---

## Overview

```
L0: Teacher Setup (Ablation)
 └── L1: Distillation Signal (FAR Factorial)
      └── L2: Hyperparameters (Tuning)
```

Each level builds on the best configuration from the previous level.

---

## L0: Teacher Setup Ablation

**Question:** How should the teacher be prepared and used?

```
Teacher Setup
├── freeze_teacher: true
│   ├── base-teacher    → 01_frozen_base-teacher_kl   (≈ baseline)
│   └── sft-teacher     → 02_frozen_sft-teacher_kl    ❌ WORST
│
└── freeze_teacher: false (co-train)
    ├── base-teacher    → 03_cotrain_base-teacher_kl  ✓ 2nd BEST
    └── sft-teacher     → 04_cotrain_sft-teacher_kl   ✓ BEST
```

| ID | Config | Teacher | Mode | MNLI | QQP |
|----|--------|---------|------|------|-----|
| 00 | `00_no-teacher_ce-only` | None | CE baseline | 80.7% | 89.8% |
| 01 | `01_frozen_base-teacher_kl` | bert-base | Frozen | 81.0% | 89.8% |
| 02 | `02_frozen_sft-teacher_kl` | textattack/bert-MNLI | Frozen | 79.2% | 65% ❌ |
| 03 | `03_cotrain_base-teacher_kl` | bert-base | Co-train | 81.6% | 90.3% |
| 04 | `04_cotrain_sft-teacher_kl` | textattack/bert-MNLI | Co-train | **82.3%** | **90.3%** |

**Ranking:** 04 > 03 > 01 ≈ 00 >>> 02

**Winner:** `cotrain_sft-teacher` — Best accuracy. Close 2nd: `cotrain_base-teacher`.

**Surprise:** Frozen SFT (02) underperforms badly, especially on QQP (~65%). Co-training appears to overcome potential label mapping issues with the TextAttack model.

---

## L1: FAR Factorial Study

**Question:** Which distillation signals (Feature/Attention/Response) help most?

Built on top of L0 winner: `cotrain_sft-teacher`

```
Distillation Signals (2³ = 8 combinations)
│
├── R=0 (no response/KL)
│   ├── F=0, A=0  → 00_ce-only      (baseline)
│   ├── F=1, A=0  → 02_f-only       (feature only)
│   ├── F=0, A=1  → 03_a-only       (attention only)
│   └── F=1, A=1  → 06_fa           (feature + attention)
│
└── R=1 (with response/KL)
    ├── F=0, A=0  → 01_r-only       (classic KD)
    ├── F=1, A=0  → 04_rf           (response + feature)
    ├── F=0, A=1  → 05_ra           (response + attention)
    └── F=1, A=1  → 07_far          (full combination)
```

### FAR Matrix

| ID | F | A | R | Config | Description |
|----|---|---|---|--------|-------------|
| 00 | 0 | 0 | 0 | `00_ce-only.yaml` | CE baseline (no distillation) |
| 01 | 0 | 0 | 1 | `01_r-only.yaml` | Response only (classic KL) |
| 02 | 1 | 0 | 0 | `02_f-only.yaml` | Feature only (FitNets-style) |
| 03 | 0 | 1 | 0 | `03_a-only.yaml` | Attention only |
| 04 | 1 | 0 | 1 | `04_rf.yaml` | Response + Feature |
| 05 | 0 | 1 | 1 | `05_ra.yaml` | Response + Attention |
| 06 | 1 | 1 | 0 | `06_fa.yaml` | Feature + Attention |
| 07 | 1 | 1 | 1 | `07_far.yaml` | Full FAR |

### Loss Functions

| Signal | Loss Type | Implementation |
|--------|-----------|----------------|
| F (Feature) | MSE | `feature_mse` on hidden states |
| A (Attention) | KL Divergence | `attention_kl` on attention weights |
| R (Response) | KL Divergence | `kl_divergence` on softened logits |

---

## L2: Hyperparameter Tuning

**Question:** What hyperparameters optimize the best L1 configuration?

Built on top of L1 winner (TBD after L1 experiments).

```
Hyperparameters
├── Temperature (T)
│   ├── T=1 (sharp)
│   ├── T=2 (default)
│   ├── T=4
│   └── T=8 (soft)
│
├── Alpha (α) — distillation weight
│   ├── α=0.3
│   ├── α=0.5 (default)
│   ├── α=0.7
│   └── α=0.9
│
├── Layer Mapping (for F/A)
│   ├── uniform
│   ├── last_n
│   └── first_last
│
└── Learning Rate / Batch Size
    └── (scaled together per linear scaling rule)
```

---

## Naming Convention

```
{task}_{l0-setup}_{l1-far}_{timestamp}
```

**Examples:**
- `mnli_cotrain-sft_far-00` — MNLI, co-trained SFT teacher, CE baseline
- `mnli_cotrain-sft_far-07` — MNLI, co-trained SFT teacher, full FAR
- `mnli_cotrain-base_far-01` — MNLI, co-trained base teacher, R only

**Config paths:**
```
configs/
└── mnli/
    └── bert_distilbert/
        ├── ablations/          # L0 experiments
        │   ├── 00_no-teacher_ce-only.yaml
        │   ├── 01_frozen_base-teacher_kl.yaml
        │   ├── 02_frozen_sft-teacher_kl.yaml
        │   ├── 03_cotrain_base-teacher_kl.yaml
        │   └── 04_cotrain_sft-teacher_kl.yaml
        │
        └── far/                # L1 experiments (cotrain-sft base)
            ├── 00_ce-only.yaml
            ├── 01_r-only.yaml
            ├── 02_f-only.yaml
            ├── 03_a-only.yaml
            ├── 04_rf.yaml
            ├── 05_ra.yaml
            ├── 06_fa.yaml
            └── 07_far.yaml
```

---

## Execution Order

1. **L0 Ablation** — Find best teacher setup
   ```bash
   sbatch scripts/run_slurm.sh mnli_00 configs/mnli/.../ablations/00_no-teacher_ce-only.yaml
   sbatch scripts/run_slurm.sh mnli_01 configs/mnli/.../ablations/01_frozen_base-teacher_kl.yaml
   # ... etc
   ```

2. **L1 FAR Factorial** — Test signal combinations on L0 winner
   ```bash
   DEBUG=1 sbatch scripts/run_slurm.sh mnli_cotrain-sft_far-00 configs/mnli/.../far/00_ce-only.yaml
   DEBUG=1 sbatch scripts/run_slurm.sh mnli_cotrain-sft_far-01 configs/mnli/.../far/01_r-only.yaml
   # ... etc
   ```

3. **L2 Hyperparameter Tuning** — Optimize L1 winner
   ```bash
   # TBD after L1 results
   ```

---

## Full Decision Tree Visualization

```
Knowledge Distillation Experiments
│
├─ L0: Teacher Setup (which teacher? frozen or co-trained?)
│  │
│  ├─ 00: No teacher (CE baseline)
│  │
│  ├─ Frozen teacher
│  │  ├─ 01: Base (LM-pretrained)     ~ OK (≈ CE baseline)
│  │  └─ 02: SFT (task-finetuned)     ❌ Poor (label issues?)
│  │
│  └─ Co-trained teacher
│     ├─ 03: Base (LM-pretrained)     ✓ 2nd BEST
│     └─ 04: SFT (task-finetuned)     ✓ BEST
│
│  Winner: cotrain_sft-teacher
│  │
│  └─────────────────────────────────────────────────────────┐
│                                                            │
├─ L1: FAR Factorial (which signals? F/A/R combinations)     │
│  │                                                         │
│  │  Base: frozen_sft-teacher ◄─────────────────────────────┘
│  │
│  ├─ 00: CE only (R=0, F=0, A=0)     baseline
│  ├─ 01: R only  (R=1, F=0, A=0)     classic KD
│  ├─ 02: F only  (R=0, F=1, A=0)     FitNets
│  ├─ 03: A only  (R=0, F=0, A=1)     attention transfer
│  ├─ 04: R+F     (R=1, F=1, A=0)
│  ├─ 05: R+A     (R=1, F=0, A=1)
│  ├─ 06: F+A     (R=0, F=1, A=1)
│  └─ 07: R+F+A   (R=1, F=1, A=1)     full combination
│
│  Winner: TBD
│  │
│  └─────────────────────────────────────────────────────────┐
│                                                            │
└─ L2: Hyperparameter Tuning                                 │
   │                                                         │
   │  Base: L1 winner ◄──────────────────────────────────────┘
   │
   ├─ Temperature: 1, 2, 4, 8
   ├─ Alpha: 0.3, 0.5, 0.7, 0.9
   ├─ Layer mapping: uniform, last_n, first_last
   └─ LR/batch scaling
```

---

## Research Questions by Level

| Level | Question | Method |
|-------|----------|--------|
| L0 | Does teacher quality matter? | Ablation |
| L0 | Is co-training worth the cost? | Ablation |
| L1 | Which signals transfer knowledge best? | Factorial |
| L1 | Do signals combine additively? | Factorial |
| L1 | Can F+A replace R (logit-free distillation)? | Factorial |
| L2 | What temperature is optimal? | Grid search |
| L2 | How to balance CE vs distillation? | Grid search |
