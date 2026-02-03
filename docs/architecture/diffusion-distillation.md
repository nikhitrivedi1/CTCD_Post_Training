# Diffusion Model Distillation

Reducing inference steps while preserving generation quality.

---

## Overview

Diffusion model distillation differs fundamentally from classification distillation. The bottleneck isn't model size—it's the number of iterative denoising steps.

### Key Differences from Classification KD

| Aspect | Classification KD | Diffusion KD |
|--------|------------------|--------------|
| **Goal** | Match output distributions | Reduce sampling steps |
| **Bottleneck** | Model size | Inference iterations (50-1000 steps) |
| **Loss** | KL divergence on logits | Score matching / Denoising |
| **Output** | Discrete labels | Continuous samples |
| **Success metric** | Accuracy preservation | Quality + step reduction |

---

## Distillation Strategies

| Strategy | Step Reduction | Quality | Best For |
|----------|---------------|---------|----------|
| **Progressive** | Gradual (halving) | High | Production |
| **Consistency** | Aggressive (1-4 steps) | Medium-High | Real-time |
| **Adversarial** | Aggressive | High | Quality-critical |
| **Score** | Moderate | High | Research |

---

## 1. Progressive Distillation

Train student to match teacher's 2-step output in 1 step, then repeat.

```
Training Stages:
  Stage 1: 1000 steps → 500 steps
  ┌─────────────────────────────────────────────────────────────┐
  │  Teacher: x_t ──[step]──> x_{t-1} ──[step]──> x_{t-2}       │
  │                              ↓                              │
  │  Student: x_t ─────────[1 step]─────────────> x_{t-2}       │
  │                                                             │
  │  Loss: MSE(student_output, teacher_2step_output)            │
  └─────────────────────────────────────────────────────────────┘

  Stage 2: 500 → 250 (Stage 1 student becomes teacher)
  ...
  Stage 8: 8 → 4
```

| Advantages | Limitations |
|------------|-------------|
| Stable training | Many stages required |
| Predictable degradation | Cannot skip to few steps directly |
| Well-understood | Accumulates errors |

---

## 2. Consistency Distillation

Train model to predict same clean output regardless of starting timestep.

```
Consistency Property:
  x_T ─────────────────────────────────────> x_0
                  ↑                          ↑
               f(x_T, T)                  f(x_t, t)
                  ↑                          ↑
  x_t ─────────────────────────────────────> x_0

  f(x_T, T) = f(x_t, t) for all t on same trajectory
```

---

## 3. Adversarial Distillation

Use discriminator to ensure student outputs match teacher quality.

```
  Noise ──> Student (4 steps) ──> Generated Sample
                                        │
                                        ↓
                               ┌─────────────────┐
                               │  Discriminator  │
                               └─────────────────┘
                                        ↑
  Noise ──> Teacher (50 steps) ──> Reference Sample

  Loss = Reconstruction + λ * Adversarial
```

---

## Application: StyleTTS2

StyleTTS2 text-to-speech has multiple distillable components.

### Component Strategy

| Component | Strategy | Teacher | Student | Speedup |
|-----------|----------|---------|---------|---------|
| **Text Encoder** | Layer pruning + KD | 6 layers | 3 layers | 2x |
| **Duration Predictor** | MSE regression | Full | Smaller | 1.5x |
| **Style Diffusion** | Progressive | 100 steps | 4-8 steps | 12-25x |
| **HiFi-GAN Vocoder** | Pruning + KD | Full | Tiny | 2-4x |

### Quality vs Speed Trade-off

| Steps | RTF* | MOS (est.) | Notes |
|-------|------|------------|-------|
| 100 | 0.8x | 4.2 | Original quality |
| 50 | 1.5x | 4.1 | Minimal degradation |
| 25 | 3.0x | 4.0 | Still high quality |
| 12 | 6.0x | 3.9 | Acceptable for most |
| 8 | 9.0x | 3.7 | Noticeable but usable |
| 4 | 18.0x | 3.4 | Real-time, quality drop |

*RTF = Real-Time Factor (higher = faster)

---

## Other Diffusion Applications

### Image Generation

| Model | Original | Distilled | Method |
|-------|----------|-----------|--------|
| SD 1.5 | 50 | 4-8 | LCM |
| SDXL | 50 | 4 | SDXL-Turbo |
| SD 3 | 28 | 4-8 | Progressive |

### Audio

| Model | Original | Distilled | Method |
|-------|----------|-----------|--------|
| AudioLDM | 200 | 4 | Consistency |
| MusicGen | 50 | 8 | Progressive |

---

## Research Questions

| Question | Why It Matters |
|----------|----------------|
| Minimum viable steps? | Quality degradation threshold |
| Progressive vs one-shot? | Is iterative training necessary? |
| Style diversity preservation? | Does distillation cause mode collapse? |
| Perceptual vs pixel loss? | Which preserves quality better? |

---

## Key Papers

| Paper | Year | Contribution |
|-------|------|--------------|
| Progressive Distillation | 2022 | Salimans and Ho - iterative step reduction |
| Consistency Models | 2023 | Song et al. - one-step generation |
| SDXL-Turbo | 2023 | Adversarial diffusion distillation |
| Latent Consistency Models | 2023 | Few-step latent diffusion |

---

## Framework Extension Required

```
NEW COMPONENTS:
├── losses/
│   ├── diffusion_mse.py      # Step-matching
│   ├── consistency.py        # Self-consistency
│   └── adversarial.py        # GAN-style
│
├── trainers/
│   ├── progressive_trainer.py
│   └── consistency_trainer.py
│
└── configs/tts/styletts2/
    └── progressive_distillation.yaml
```

---

## Diffusion Basics

### Forward Process (Adding Noise)
```
x_0 (clean) → x_1 → x_2 → ... → x_T (pure noise)
x_t = √(α_t) * x_0 + √(1 - α_t) * ε
```

### Reverse Process (Denoising)
```
x_T (noise) → x_{T-1} → ... → x_0 (clean)
x_{t-1} = μ_θ(x_t, t) + σ_t * z
```

### Why So Many Steps?
- Each step makes a small correction
- Larger steps = larger errors = worse quality
- Distillation: Learn to make bigger, accurate steps
