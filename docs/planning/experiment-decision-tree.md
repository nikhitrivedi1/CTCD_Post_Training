# Experiment Decision Tree

Mapping the experimental hierarchy for knowledge distillation research.

---

## Overview

```
L0: Teacher Setup (Ablation)
 └── L1: Distillation Signal (RFA Factorial)
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
| 02 | `02_frozen_sft-teacher_kl` | textattack/bert-MNLI | Frozen | 79.2% | 65% |
| 03 | `03_cotrain_base-teacher_kl` | bert-base | Co-train | 81.6% | 90.3% |
| 04 | `04_cotrain_sft-teacher_kl` | textattack/bert-MNLI | Co-train | **82.3%** | **90.3%** |

**Ranking:** 04 > 03 > 01 ≈ 00 >>> 02

**Winner:** `cotrain_sft-teacher`

---

## L1: RFA Factorial Study

**Question:** Which distillation signals (Response/Feature/Attention) help most?

Built on L0 winner: `cotrain_sft-teacher`

| ID | R | F | A | Config | Description |
|----|---|---|---|--------|-------------|
| 00 | 0 | 0 | 0 | `00_ce-only.yaml` | CE baseline |
| 01 | 1 | 0 | 0 | `01_r-only.yaml` | Response only (classic KL) |
| 02 | 0 | 1 | 0 | `02_f-only.yaml` | Feature only |
| 03 | 0 | 0 | 1 | `03_a-only.yaml` | Attention only |
| 04 | 1 | 1 | 0 | `04_rf.yaml` | Response + Feature |
| 05 | 1 | 0 | 1 | `05_ra.yaml` | Response + Attention |
| 06 | 0 | 1 | 1 | `06_fa.yaml` | Feature + Attention |
| 07 | 1 | 1 | 1 | `07_rfa.yaml` | Full RFA |

### Loss Functions

| Signal | Loss Type | Implementation |
|--------|-----------|----------------|
| R (Response) | KL Divergence | `kl_divergence` on softened logits |
| F (Feature) | MSE | `feature_mse` on hidden states |
| A (Attention) | KL Divergence | `attention_kl` on attention weights |

---

## L2: Hyperparameter Tuning

**Question:** What hyperparameters optimize the best L1 configuration?

Built on L1 winner (TBD).

```
Hyperparameters
├── Temperature (T): 1, 2, 4, 8
├── Alpha (α): 0.3, 0.5, 0.7, 0.9
├── Layer Mapping: uniform, last_n, first_last
└── Learning Rate / Batch Size (scaled together)
```

---

## Naming Convention

```
{task}_{l0-setup}_{l1-rfa}_{timestamp}
```

**Examples:**
- `mnli_cotrain-sft_rfa-00` - MNLI, co-trained SFT teacher, CE baseline
- `mnli_cotrain-sft_rfa-07` - MNLI, co-trained SFT teacher, full RFA

**Config paths:**
```
configs/
└── mnli/
    └── bert_distilbert/
        ├── ablations/          # L0 experiments
        │   ├── 00_no-teacher_ce-only.yaml
        │   ├── 01_frozen_base-teacher_kl.yaml
        │   └── ...
        │
        └── rfa/                # L1 experiments
            ├── 00_ce-only.yaml
            ├── 01_r-only.yaml
            └── ...
```

---

## Full Decision Tree

```
Knowledge Distillation Experiments
│
├─ L0: Teacher Setup (which teacher? frozen or co-trained?)
│  │
│  ├─ 00: No teacher (CE baseline)
│  │
│  ├─ Frozen teacher
│  │  ├─ 01: Base (LM-pretrained)     ~ OK
│  │  └─ 02: SFT (task-finetuned)     ❌ Poor
│  │
│  └─ Co-trained teacher
│     ├─ 03: Base (LM-pretrained)     ✓ 2nd
│     └─ 04: SFT (task-finetuned)     ✓ BEST
│
│  Winner: cotrain_sft-teacher ─────────────────────────┐
│                                                       │
├─ L1: RFA Factorial (R/F/A combinations)               │
│  │                                                    │
│  │  Base: cotrain_sft-teacher ◄───────────────────────┘
│  │
│  ├─ 00: CE only      baseline
│  ├─ 01: R only       classic KD
│  ├─ 02: F only       FitNets
│  ├─ 03: A only       attention transfer
│  ├─ 04: R+F
│  ├─ 05: R+A
│  ├─ 06: F+A
│  └─ 07: R+F+A        full combination
│
│  Winner: TBD ─────────────────────────────────────────┐
│                                                       │
└─ L2: Hyperparameter Tuning                            │
   │                                                    │
   │  Base: L1 winner ◄─────────────────────────────────┘
   │
   ├─ Temperature: 1, 2, 4, 8
   ├─ Alpha: 0.3, 0.5, 0.7, 0.9
   └─ Layer mapping variations
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
