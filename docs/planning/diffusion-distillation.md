# Diffusion Distillation Implementation Plan

Two compression dimensions for diffusion models: fewer denoising steps AND smaller architecture.

---

## Two Compression Axes

| Axis | What compresses | Method | Goal |
|------|----------------|--------|------|
| **Step reduction** | Sampling iterations | Progressive Distillation | 1000 → 4 steps |
| **Parameter reduction** | Model size | Standard KD (R/F/A) | Large UNet → Small UNet |
| **Combined** | Both | Progressive + KD | Smaller model, fewer steps |

## Overview

| Aspect | Text KD (existing) | Diffusion Step KD | Diffusion Param KD |
|--------|--------------------|--------------------|---------------------|
| **What compresses** | Parameters | Steps | Parameters |
| **Signal** | Output logits (KL) | Denoised samples (MSE) | R/F/A on UNet outputs |
| **Teacher** | Fine-tuned classifier | Pre-trained DDPM | Large UNet |
| **Student** | Smaller architecture | Same architecture | Smaller UNet |
| **Training loop** | HF Trainer subclass | Custom loop | Custom loop |

**Key insight:** Parameter reduction for UNets reuses our existing R/F/A signals. The UNet has attention layers and feature maps — Feature (cosine on intermediate activations) and Attention (KL on self-attention) directly apply. This connects diffusion KD back to our text KD framework.

---

## Approaches

### 1. Progressive Distillation (Salimans & Ho, 2022)

Train student to match teacher's 2-step output in 1 step. Repeat to halve steps each round.

**Algorithm:**
```
For each round (1024→512→...→4):
  1. Initialize student = copy of teacher
  2. For each batch:
     a. Sample x_0, timestep t, noise epsilon
     b. Create x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon
     c. Teacher: 2 DDIM steps (x_t → x_{t-1} → x_{t-2}), no grad
     d. Student: 1 step (x_t → x_{t-2})
     e. Loss = MSE(student_output, teacher_output)
  3. Student becomes next round's teacher
```

**Halving cascade:**

| Round | Teacher Steps | Student Steps | Iterations |
|-------|---------------|---------------|------------|
| 1 | 1024 | 512 | 50K |
| 2 | 512 | 256 | 50K |
| 3 | 256 | 128 | 50K |
| 4 | 128 | 64 | 50K |
| 5 | 64 | 32 | 50K |
| 6 | 32 | 16 | 50K |
| 7 | 16 | 8 | 50K |
| 8 | 8 | 4 | 100K |

**Critical details:**
- Must use **v-prediction** parameterization (not epsilon-prediction). v = alpha_t * epsilon - sigma_t * x. Epsilon-prediction is unstable for few-step models.
- Both teacher steps use the **same noise** (deterministic target).
- **No dropout** during distillation.
- DDIM sampling (deterministic ODE, not stochastic SDE).

**Results:** CIFAR-10 FID 3.0 with 4 steps (from 8192-step teacher).

### 2. Consistency Models (Song et al., 2023)

Map any noisy point on the ODE trajectory directly to x_0.

**Self-consistency property:** f(x_t, t) = f(x_s, s) for all points on the same trajectory.

**Two modes:**
- **Consistency Distillation (CD):** Uses pre-trained teacher to generate adjacent trajectory points. Enforce f(x_t, t) = f(x_{t-delta}, t-delta).
- **Consistency Training (CT):** No teacher. Self-supervised consistency from data alone.

**Advantages over progressive:**
- Single training phase (no iterative rounds)
- 1-step generation by design
- Zero-shot capabilities (inpainting, super-resolution)

**Results:** CIFAR-10 FID 3.55 (1 step).

### 3. Simple & Fast Distillation (Zhou & Chen, 2024)

Achieves FID 4.53 (2-step) on CIFAR-10 in **0.64h on a single A100**. Most practical approach if time is limited.

---

## Implementation Plan

### Teacher Model

| Model | Resolution | Params | HuggingFace ID |
|-------|-----------|--------|----------------|
| DDPM CIFAR-10 | 32x32 | ~35M | `google/ddpm-cifar10-32` |

Use pre-trained teacher. No need to train from scratch.

### Dataset

**CIFAR-10**: 50K train, 10K test, 32x32. Standard benchmark in all diffusion distillation papers. Available via `torchvision.datasets.CIFAR10`.

### Evaluation

| Metric | What it measures | Library | Baseline |
|--------|-----------------|---------|----------|
| **FID** | Distribution similarity (lower=better) | `torch-fidelity` | DDPM: 3.17, ProgDistill 4-step: 3.0 |
| **IS** | Quality + diversity (higher=better) | `torch-fidelity` | Secondary |
| **NFE** | Number of function evaluations | Manual | Steps × 1 |
| **Wall-clock** | Sampling time per batch | Manual | - |

Generate 50K samples for stable FID estimates.

### GPU Requirements

| Task | Memory | Time (A100) |
|------|--------|-------------|
| Progressive distillation (1 round) | ~8-16GB | 2-4h |
| Full cascade (8 rounds) | ~8-16GB | 16-32h |
| FID evaluation (50K samples) | ~4GB | 30min |

### New Components

```
src/diffusion/
    __init__.py
    trainer.py              # Progressive distillation training loop
    noise_schedule.py       # Alpha/sigma schedules, DDIM step logic
    evaluation.py           # FID/IS computation wrapper
    v_prediction.py         # v-parameterization conversion utilities

configs/diffusion/
    cifar10/
        base.yaml           # Shared config (model, dataset, eval)
        progressive.yaml    # Progressive distillation settings
        consistency.yaml    # Consistency distillation settings (stretch)
```

### Training Loop (Progressive Distillation)

```python
for batch in dataloader:
    x_0 = batch["images"]
    t = sample_timestep(N)
    noise = torch.randn_like(x_0)
    x_t = add_noise(x_0, noise, t)

    # Teacher: 2 DDIM steps (no grad)
    with torch.no_grad():
        x_mid = teacher_ddim_step(teacher, x_t, t, t-1)
        x_target = teacher_ddim_step(teacher, x_mid, t-1, t-2)

    # Student: 1 step to match teacher's 2-step output
    v_pred = student(x_t, t)
    x_student = reconstruct_from_v(x_t, v_pred, t, t-2)

    loss = F.mse_loss(x_student, x_target)
    loss.backward()
```

---

## Reuse from Existing Framework

| Component | Reusable? | Notes |
|-----------|-----------|-------|
| YAML config inheritance | Yes | Same pattern |
| W&B logging | Yes | Via callbacks |
| Loss strategy pattern | Yes (conceptually) | New strategies needed |
| `--force` / `--debug` flags | Yes | Same train.py CLI |
| HF Trainer | No | Custom loop required |
| Existing loss strategies | **Yes (for param reduction)** | R/F/A on UNet outputs/features |
| Data pipeline | No | Images, not text |
| Layer mapping / projections | **Yes (for param reduction)** | Map large UNet layers → small UNet |

---

## Experiment Design

### Baselines

| # | Method | Steps | Description |
|---|--------|-------|-------------|
| B0 | DDPM (teacher) | 1000 | Full sampling (reference quality) |
| B1 | DDIM (no distill) | 100 | Naive step reduction |
| B2 | DDIM (no distill) | 10 | Aggressive naive reduction |
| B3 | DDIM (no distill) | 4 | Lower bound without distillation |

### Progressive Distillation Runs

| # | From→To Steps | Round | Expected FID |
|---|---------------|-------|-------------|
| P1 | 1024→512 | 1 | ~3.2 |
| P2 | 512→256 | 2 | ~3.2 |
| P3 | 256→128 | 3 | ~3.3 |
| P4 | 128→64 | 4 | ~3.5 |
| P5 | 64→32 | 5 | ~3.8 |
| P6 | 32→16 | 6 | ~4.0 |
| P7 | 16→8 | 7 | ~4.5 |
| P8 | 8→4 | 8 | ~3.0 (published) |

### Parameter Reduction (Standard KD on UNets)

Compress the UNet itself. The teacher UNet (~35M params) has:
- Residual blocks with feature maps → **Feature (F) signal** applies
- Self-attention layers → **Attention (A) signal** applies
- Noise/v predictions → **Response (R) signal** applies

| # | Teacher UNet | Student UNet | Steps | Signals |
|---|-------------|-------------|-------|---------|
| K0 | ~35M (full) | ~10M (smaller channels) | 1000 | CE only (baseline) |
| K1 | ~35M | ~10M | 1000 | R (KL on v-predictions) |
| K2 | ~35M | ~10M | 1000 | R + F (v-pred + feature cosine) |
| K3 | ~35M | ~10M | 1000 | R + F + A (full FAR) |

Student UNet: reduce channel multipliers (e.g., [128,256,256,256] → [64,128,128,128]).

### Combined: Step + Parameter Reduction

The ultimate compression: smaller model AND fewer steps.

| # | Method | Params | Steps | Description |
|---|--------|--------|-------|-------------|
| C0 | Teacher (reference) | 35M | 1000 | Baseline |
| C1 | Step reduction only | 35M | 4 | Progressive distillation |
| C2 | Param reduction only | ~10M | 1000 | Standard KD (R/F/A) |
| C3 | Both (sequential) | ~10M | 4 | First param KD, then progressive distill |
| C4 | Both (joint) | ~10M | 4 | Simultaneous param + step reduction |

**Key question:** Does step + param reduction compose? Or does quality degrade multiplicatively?

### Key Outputs

**1. FID vs. Steps curve:**
- DDIM (no distillation) — degrades rapidly
- Progressive Distillation — maintains quality to 4 steps
- (Stretch) Consistency Model — 1-step generation

**2. FID vs. Parameters curve:**
- Standard KD R/F/A on UNets
- Same FAR factorial from text KD, applied to diffusion

**3. FID vs. (Steps × Params) Pareto frontier:**
- Do step and param reduction compose?
- What's the optimal trade-off?

---

## Paper Narrative

"We study knowledge distillation across three modalities and two compression axes:
1. **Classification** (BERT→DistilBERT): parameter compression, output distribution transfer
2. **Translation** (Llama 3B→1B): parameter compression, token-level distribution transfer
3. **Generation** (DDPM): step compression (1000→4) AND parameter compression (35M→10M)

Text KD transfers output distributions. Diffusion step KD transfers sampling trajectories. Diffusion parameter KD bridges both paradigms — the same R/F/A signals that work for text also apply to UNet feature maps and attention layers. The core principle is universal: a teacher model provides richer training signal than ground truth alone."

---

## Key Papers

1. **Ho et al., 2020** — "Denoising Diffusion Probabilistic Models" (DDPM background)
2. **Salimans & Ho, 2022** — "Progressive Distillation for Fast Sampling" (core method) — [arXiv:2202.00512](https://arxiv.org/abs/2202.00512)
3. **Song et al., 2023** — "Consistency Models" (alternative) — [arXiv:2303.01469](https://arxiv.org/abs/2303.01469)
4. **Zhou & Chen et al., 2024** — "Simple and Fast Distillation of Diffusion Models" (practical, NeurIPS 2024)

## Reference Implementations

| Repository | Approach | Notes |
|-----------|----------|-------|
| [Hramchenko/diffusion_distiller](https://github.com/Hramchenko/diffusion_distiller) | Progressive Distillation | Clean PyTorch, v-prediction |
| [openai/consistency_models](https://github.com/openai/consistency_models) | Consistency Models | Official, full implementation |
| [cloneofsimo/consistency_models](https://github.com/cloneofsimo/consistency_models) | Consistency Training | Minimal, single-file |
