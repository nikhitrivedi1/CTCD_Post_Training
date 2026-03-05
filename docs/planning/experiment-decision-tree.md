# Experiment Decision Tree

Mapping the experimental hierarchy for knowledge distillation research.

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
├── teacher.freeze: true
│   ├── base-teacher    → 01_kd_frozen-base   (≈ baseline)
│   └── sft-teacher     → 02_kd_frozen-sft    ❌ WORST
│
└── teacher.freeze: false (co-train)
    ├── base-teacher    → 03_kd_cotrain-base  ✓ 2nd BEST
    └── sft-teacher     → 04_kd_cotrain-sft   ✓ BEST
```

| ID | Config | Teacher | Mode | MNLI | QQP |
|----|--------|---------|------|------|-----|
| 00 | `00_sft` | None | CE baseline | 80.7% | 89.8% |
| 01 | `01_kd_frozen-base` | bert-base | Frozen | 81.0% | 89.8% |
| 02 | `02_kd_frozen-sft` | textattack/bert-MNLI | Frozen | 79.2% | 65% |
| 03 | `03_kd_cotrain-base` | bert-base | Co-train | 81.6% | 90.3% |
| 04 | `04_kd_cotrain-sft` | textattack/bert-MNLI | Co-train | **82.3%** | **90.3%** |

**Ranking:** 04 > 03 > 01 ≈ 00 >>> 02

**Winner:** `cotrain_sft-teacher`

---

## L1: FAR Factorial Study

**Question:** Which distillation signals (Feature/Attention/Response) help most?

Built on L0 winner: `cotrain_sft-teacher`

| ID | F | A | R | Config | Description |
|----|---|---|---|--------|-------------|
| 00 | 0 | 0 | 0 | `00_ce-only.yaml` | CE baseline |
| 01 | 1 | 0 | 0 | `01_f-only.yaml` | Feature only |
| 02 | 0 | 1 | 0 | `02_a-only.yaml` | Attention only |
| 03 | 0 | 0 | 1 | `03_r-only.yaml` | Response only (classic KL) |
| 04 | 1 | 1 | 0 | `04_fa.yaml` | Feature + Attention |
| 05 | 1 | 0 | 1 | `05_fr.yaml` | Feature + Response |
| 06 | 0 | 1 | 1 | `06_ar.yaml` | Attention + Response |
| 07 | 1 | 1 | 1 | `07_far.yaml` | Full FAR |

### Loss Functions

| Signal | Loss Type | Implementation |
|--------|-----------|----------------|
| F (Feature) | Cosine Similarity | `feature_cosine` on hidden states |
| A (Attention) | KL Divergence | `attention_kl` on attention weights |
| R (Response) | KL Divergence | `response_kl` on softened logits |

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
{task}_{l0-setup}_{l1-far}_{timestamp}
```

**Examples:**
- `mnli_cotrain-sft_far-00` - MNLI, co-trained SFT teacher, CE baseline
- `mnli_cotrain-sft_far-07` - MNLI, co-trained SFT teacher, full FAR

**Config paths:**
```
configs/
└── classification/
    └── mnli/
        └── distilbert/
            └── bert-base/
                ├── 00_sft.yaml         # L0: SFT baseline (at model root)
                ├── ablations/          # L0 experiments
                │   ├── 01_kd_frozen-base.yaml
                │   ├── 02_kd_frozen-sft.yaml
                │   └── ...
                │
                └── far/                # L1 experiments
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
├─ L1: FAR Factorial (F/A/R combinations)               │
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

## Diffusion Experiments

Separate track from text KD. Two compression dimensions:

```
Diffusion Distillation
├── Step Reduction (same model, fewer steps)
│   ├── Progressive Distillation: 1024→512→...→4 steps
│   ├── Consistency Distillation: any→1 step
│   └── Baseline: DDIM naive reduction
│
└── Parameter Reduction (smaller model, same steps)
    ├── Standard KD: large UNet → small UNet (F/A/R signals)
    └── Combined: smaller model + fewer steps
```

| # | Config | Approach | Compression |
|---|--------|----------|-------------|
| D0 | DDPM teacher (1000 steps) | Baseline | None |
| D1 | DDIM naive (100, 10, 4 steps) | No distillation | Steps only |
| D2 | Progressive distillation (4 steps) | Step reduction | Steps only |
| D3 | Small UNet + KD (1000 steps) | Parameter reduction | Params only |
| D4 | Small UNet + progressive (4 steps) | Both | Params + steps |

**Teacher:** `google/ddpm-cifar10-32` (~35M params, 32x32)
**Dataset:** CIFAR-10
**Metric:** FID (50K samples)

See [docs/planning/diffusion-distillation.md](diffusion-distillation.md) for full plan.

---

## Voice Experiments (StyleTTS2)

Capstone modality combining all KD approaches:

```
StyleTTS2 Distillation
├── Style Diffusion Step Reduction (diffusion KD)
│   ├── V0: Teacher (100 steps)
│   ├── V1: Naive step reduction (10, 4 steps)
│   └── V2: Progressive distillation (4 steps)
│
├── PL-BERT Encoder Compression (text KD)
│   ├── V4: Full PL-BERT (12 layers)
│   └── V5-V6: DistilPL-BERT (6 layers, R/FAR)
│
├── HiFi-GAN Decoder Compression (feature KD)
│   ├── V7: Full decoder
│   └── V8: 50% channels (Feature KD)
│
└── Combined Pipeline
    └── V9: All components compressed
```

**Teacher:** `yl4579/StyleTTS2-LJSpeech`
**Dataset:** LJSpeech (24h, single-speaker)
**Metrics:** MOS (UTMOS proxy), PESQ, STOI, RTF

See [docs/planning/voice-distillation.md](voice-distillation.md) for full plan.

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
| Diffusion | Does step reduction + param reduction compose? | Compare D2, D3, D4 |
| Diffusion | Do F/A/R signals help for UNet distillation? | Factorial on D3 |
| Voice | Does module-wise KD compose across components? | Compare V2+V6+V8 vs V9 |
| Voice | Which component benefits most from KD? | Compare V2, V5, V8 individually |
