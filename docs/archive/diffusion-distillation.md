# Diffusion Model Distillation

> Reducing Inference Steps While Preserving Generation Quality

---

## Overview

Diffusion model distillation is fundamentally different from classification distillation. The primary bottleneck isn't model size—it's the number of iterative denoising steps required for generation.

### Key Differences from Classification KD

| Aspect | Classification KD | Diffusion KD |
|--------|------------------|--------------|
| **Goal** | Match output distributions | Reduce sampling steps |
| **Bottleneck** | Model size / parameters | Inference iterations (50-1000 steps) |
| **Loss** | KL divergence on logits | Score matching / Denoising objectives |
| **Output** | Discrete labels | Continuous samples (audio, images, video) |
| **Teacher output** | Soft probability distribution | Denoised sample at step t |
| **Success metric** | Accuracy preservation | Quality + step reduction |

---

## Distillation Strategies for Diffusion Models

### Strategy Comparison

| Strategy | Step Reduction | Quality | Training Complexity | Best For |
|----------|---------------|---------|---------------------|----------|
| **Progressive Distillation** | Gradual (iterative halving) | High | Medium | Production systems |
| **Consistency Distillation** | Aggressive (1-4 steps) | Medium-High | High | Real-time applications |
| **Adversarial Distillation** | Aggressive | High | Very High | Quality-critical |
| **Score Distillation** | Moderate | High | Medium | Research |
| **Guided Distillation** | Moderate | High | Low | Conditional generation |

---

## 1. Progressive Distillation

The most common and stable approach. Train student to match teacher's 2-step output in 1 step, then repeat.

```
Training Stages:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  Stage 1: 1000 steps → 500 steps                                    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Teacher: x_t ──[step]──> x_{t-1} ──[step]──> x_{t-2}       │    │
│  │                              ↓                               │    │
│  │  Student: x_t ─────────[1 step]─────────────> x_{t-2}       │    │
│  │                                                              │    │
│  │  Loss: MSE(student_output, teacher_2step_output)            │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Stage 2: 500 steps → 250 steps (Stage 1 student becomes teacher)   │
│  Stage 3: 250 steps → 125 steps                                     │
│  Stage 4: 125 steps → 64 steps                                      │
│  Stage 5: 64 steps → 32 steps                                       │
│  Stage 6: 32 steps → 16 steps                                       │
│  Stage 7: 16 steps → 8 steps                                        │
│  Stage 8: 8 steps → 4 steps                                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
# Progressive distillation pseudocode
def progressive_distillation_stage(teacher, student, dataloader, num_steps):
    """Train student to do 2 teacher steps in 1 student step."""

    for batch in dataloader:
        # Sample random timestep
        t = torch.randint(0, num_steps, (batch_size,))

        # Add noise to clean data
        x_t = add_noise(batch, t)

        # Teacher takes 2 steps
        with torch.no_grad():
            x_mid = teacher.denoise_step(x_t, t)
            x_target = teacher.denoise_step(x_mid, t - 1)

        # Student tries to match in 1 step
        x_pred = student.denoise_step(x_t, t, step_size=2)

        # Loss on denoised output
        loss = F.mse_loss(x_pred, x_target)
        loss.backward()
        optimizer.step()

    return student  # This becomes next stage's teacher
```

### Advantages & Limitations

| Advantages | Limitations |
|------------|-------------|
| Stable training | Many training stages required |
| Predictable quality degradation | Cannot skip to very few steps directly |
| Well-understood theoretically | Accumulates errors across stages |

---

## 2. Consistency Distillation

Train model to predict the same clean output regardless of starting timestep. Enables single-step generation.

```
Consistency Property:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  The same noise should map to the same clean sample:                │
│                                                                      │
│       x_T ─────────────────────────────────────> x_0                │
│                        ↑                          ↑                  │
│                     f(x_T, T)                  f(x_t, t)             │
│                        ↑                          ↑                  │
│       x_t ─────────────────────────────────────> x_0                │
│                                                                      │
│  Consistency: f(x_T, T) = f(x_t, t) for all t on same trajectory   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Loss Function

```python
def consistency_loss(model, x_0, noise_schedule):
    # Sample two adjacent timesteps
    t = torch.randint(1, T, (batch_size,))
    t_next = t - 1

    # Add noise
    x_t = add_noise(x_0, t)
    x_t_next = add_noise(x_0, t_next)

    # Both should predict same x_0
    pred_t = model(x_t, t)
    pred_t_next = model(x_t_next, t_next)

    # Consistency loss
    loss = F.mse_loss(pred_t, pred_t_next.detach())
    return loss
```

---

## 3. Adversarial Distillation

Use discriminator to ensure student outputs are indistinguishable from teacher outputs.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ADVERSARIAL DISTILLATION                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Noise ──> Student (4 steps) ──> Generated Sample                   │
│                                        │                             │
│                                        ↓                             │
│                               ┌─────────────────┐                   │
│                               │  Discriminator  │                   │
│                               │  "Real or Fake?"│                   │
│                               └─────────────────┘                   │
│                                        ↑                             │
│                                        │                             │
│  Noise ──> Teacher (50 steps) ──> Reference Sample                  │
│                                                                      │
│  Student Loss = Reconstruction + λ * Adversarial                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Combined Loss

```python
def adversarial_distillation_loss(student, teacher, discriminator, x_0):
    # Teacher generates high-quality reference
    with torch.no_grad():
        teacher_sample = teacher.sample(x_0.shape, steps=50)

    # Student generates with fewer steps
    student_sample = student.sample(x_0.shape, steps=4)

    # Discriminator scores
    real_score = discriminator(teacher_sample)
    fake_score = discriminator(student_sample)

    # Adversarial loss (student wants to fool discriminator)
    adv_loss = F.binary_cross_entropy_with_logits(
        fake_score, torch.ones_like(fake_score)
    )

    # Reconstruction loss
    recon_loss = F.mse_loss(student_sample, teacher_sample)

    return recon_loss + 0.01 * adv_loss
```

---

## 4. Score/Velocity Distillation

Match the learned score function (gradient of log probability) directly.

```python
def score_distillation_loss(student, teacher, x_t, t):
    # Teacher's score prediction
    with torch.no_grad():
        teacher_score = teacher.predict_score(x_t, t)

    # Student's score prediction
    student_score = student.predict_score(x_t, t)

    # Match scores
    loss = F.mse_loss(student_score, teacher_score)
    return loss
```

---

## Application: StyleTTS2 Distillation

StyleTTS2 is a text-to-speech system with multiple distillable components.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        StyleTTS2 Architecture                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  TEXT INPUT                                                          │
│      │                                                               │
│      ↓                                                               │
│  ┌─────────────────┐                                                │
│  │ Text Encoder    │  ← Standard KD (smaller encoder)               │
│  │ (Transformer)   │     Complexity: Low                            │
│  └────────┬────────┘                                                │
│           │                                                          │
│           ↓                                                          │
│  ┌─────────────────┐                                                │
│  │ Duration        │  ← Regression KD                               │
│  │ Predictor       │     Loss: MSE on durations                     │
│  └────────┬────────┘     Complexity: Low                            │
│           │                                                          │
│           ↓                                                          │
│  ┌─────────────────┐     ┌─────────────────┐                        │
│  │ Style Diffusion │ ←───│ Reference Audio │                        │
│  │ (50-100 steps)  │     │ (optional)      │                        │
│  └────────┬────────┘     └─────────────────┘                        │
│           │                                                          │
│           │  ← PRIMARY DISTILLATION TARGET                          │
│           │     Strategy: Progressive or Consistency                │
│           │     Goal: 50-100 steps → 4-8 steps                      │
│           │     Complexity: High                                    │
│           ↓                                                          │
│  ┌─────────────────┐                                                │
│  │ Decoder/Vocoder │  ← Pruning + Standard KD                       │
│  │ (HiFi-GAN)      │     Target: HiFi-GAN Tiny                      │
│  └────────┬────────┘     Complexity: Medium                         │
│           │                                                          │
│           ↓                                                          │
│      AUDIO OUTPUT                                                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Component-wise Distillation Strategy

| Component | Strategy | Teacher | Student | Expected Speedup |
|-----------|----------|---------|---------|------------------|
| **Text Encoder** | Layer pruning + KD | 6 layers | 3 layers | 2x |
| **Duration Predictor** | MSE regression KD | Full model | Smaller MLP | 1.5x |
| **Style Diffusion** | Progressive distillation | 100 steps | 4-8 steps | 12-25x |
| **HiFi-GAN Vocoder** | Pruning + logit KD | Full HiFi-GAN | HiFi-GAN Tiny | 2-4x |

### Style Diffusion: The Primary Target

The style diffusion model is the main bottleneck. Distillation approach:

```python
# StyleTTS2 style diffusion distillation

class StyleDiffusionDistiller:
    def __init__(self, teacher_steps=100, target_steps=4):
        self.teacher_steps = teacher_steps
        self.target_steps = target_steps
        self.stages = int(np.log2(teacher_steps / target_steps))

    def train_stage(self, teacher, student, dataloader, current_steps):
        """Progressive distillation: halve steps each stage."""

        for text, ref_audio in dataloader:
            # Get text encoding
            text_enc = text_encoder(text)

            # Teacher: 2 diffusion steps for style
            with torch.no_grad():
                # Sample initial noise
                z_T = torch.randn(batch_size, style_dim)

                # Teacher takes 2 steps
                z_mid = teacher.diffusion_step(z_T, text_enc, t)
                z_target = teacher.diffusion_step(z_mid, text_enc, t-1)

            # Student: 1 step to match
            z_pred = student.diffusion_step(z_T, text_enc, t, step_size=2)

            # Style matching loss
            loss = F.mse_loss(z_pred, z_target)

            # Optional: Add perceptual loss on generated audio
            if use_perceptual:
                audio_pred = vocoder(decoder(z_pred, text_enc))
                audio_target = vocoder(decoder(z_target, text_enc))
                loss += 0.1 * perceptual_loss(audio_pred, audio_target)

            loss.backward()
            optimizer.step()
```

### Quality vs Speed Trade-off

```
┌─────────────────────────────────────────────────────────────────────┐
│              STYLETTS2 DISTILLATION: QUALITY VS SPEED               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Steps    RTF*     MOS (est.)    Notes                              │
│  ─────────────────────────────────────────────────────────────────  │
│  100      0.8x     4.2           Original quality                   │
│   50      1.5x     4.1           Minimal degradation                │
│   25      3.0x     4.0           Still high quality                 │
│   12      6.0x     3.9           Acceptable for most uses           │
│    8      9.0x     3.7           Noticeable but usable              │
│    4     18.0x     3.4           Real-time capable, quality drop    │
│    1     70.0x     2.8           One-shot, significant artifacts    │
│                                                                      │
│  * RTF = Real-Time Factor (higher = faster than real-time)          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Configuration Schema

### Proposed Config for Diffusion KD

```yaml
# configs/tts/styletts2/progressive_distillation.yaml

model:
  teacher: styletts2-base
  student: styletts2-fast

  # Component-specific settings
  components:
    text_encoder:
      teacher_layers: 6
      student_layers: 3
      distill: true

    style_diffusion:
      teacher_steps: 100
      student_steps: 4
      distill: true  # Primary target

    vocoder:
      teacher: hifigan-v1
      student: hifigan-tiny
      distill: true

distillation:
  # Diffusion-specific
  strategy: progressive
  stages: 5                    # 100 → 50 → 25 → 12 → 6 → 4
  stage_epochs: 10

  # Loss weighting
  diffusion_weight: 1.0
  perceptual_weight: 0.1

  # Optional adversarial refinement (final stage)
  adversarial:
    enabled: true
    discriminator: multi_scale
    weight: 0.01
    start_stage: 4             # Only in final stages

training:
  batch_size: 32
  learning_rate: 1e-4
  warmup_steps: 1000

data:
  dataset: ljspeech
  sample_rate: 22050
  max_duration: 10.0

evaluation:
  metrics:
    - mos_predictor           # Predicted MOS
    - rtf                     # Real-time factor
    - mel_cepstral_distortion # MCD
  eval_steps: 1000
```

---

## Other Diffusion Applications

The same principles apply to other diffusion-based models:

### Image Generation (Stable Diffusion)

| Model | Original Steps | Distilled Steps | Method |
|-------|----------------|-----------------|--------|
| SD 1.5 | 50 | 4-8 | LCM (Latent Consistency) |
| SDXL | 50 | 4 | SDXL-Turbo (Adversarial) |
| SD 3 | 28 | 4-8 | Progressive |

### Video Generation (SVD)

| Model | Original Steps | Distilled Steps | Method |
|-------|----------------|-----------------|--------|
| SVD | 25 | 4 | AnimateLCM |

### Audio (Music, Speech)

| Model | Original Steps | Distilled Steps | Method |
|-------|----------------|-----------------|--------|
| AudioLDM | 200 | 4 | Consistency |
| MusicGen | 50 | 8 | Progressive |

---

## Research Questions

| Question | Why It Matters |
|----------|----------------|
| **Minimum viable steps?** | How few steps before quality degrades unacceptably? |
| **Progressive vs one-shot?** | Is iterative training necessary or can we skip? |
| **Style diversity preservation?** | Does distillation cause mode collapse? |
| **Perceptual vs pixel loss?** | Which loss better preserves audio quality? |
| **Joint vs sequential distillation?** | Distill all components together or one at a time? |
| **Adversarial necessity?** | Is GAN refinement worth the complexity? |

---

## Key Papers

| Paper | Year | Contribution |
|-------|------|--------------|
| **Progressive Distillation for Fast Sampling** | 2022 | Salimans & Ho - iterative step reduction |
| **Consistency Models** | 2023 | Song et al. - one-step generation |
| **SDXL-Turbo** | 2023 | Adversarial diffusion distillation |
| **Latent Consistency Models (LCM)** | 2023 | Few-step latent diffusion |
| **AnimateLCM** | 2024 | Video diffusion distillation |
| **DMD (Distribution Matching Distillation)** | 2024 | One-step distillation |

---

## Framework Extension Required

To support diffusion distillation in our framework:

```
NEW COMPONENTS NEEDED:
├── losses/
│   ├── diffusion_mse.py          # Step-matching loss
│   ├── consistency.py            # Self-consistency loss
│   ├── adversarial.py            # GAN-style discriminator losses
│   └── perceptual.py             # Audio/image perceptual losses
│
├── trainers/
│   ├── progressive_trainer.py    # Multi-stage progressive training
│   └── consistency_trainer.py    # Consistency model training
│
├── models/
│   ├── diffusion_wrapper.py      # Hook intermediate diffusion steps
│   └── discriminator.py          # For adversarial refinement
│
└── configs/
    └── tts/
        └── styletts2/
            ├── progressive_distillation.yaml
            └── consistency_distillation.yaml
```

---

## Comparison: Classification KD vs Diffusion KD

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DISTILLATION PARADIGM COMPARISON                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  CLASSIFICATION KD (Current Focus)                                  │
│  ─────────────────────────────────                                  │
│  Input ──> [Teacher] ──> Soft Labels                               │
│              ↓              ↓                                        │
│  Input ──> [Student] ──> Match soft labels                         │
│                                                                      │
│  Key insight: Soft labels contain "dark knowledge"                  │
│  Our finding: Overconfident wrong teachers hurt                     │
│                                                                      │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                      │
│  DIFFUSION KD (Future Extension)                                    │
│  ───────────────────────────────                                    │
│  Noise ──> [Teacher: N steps] ──> Clean Sample                     │
│              ↓                        ↓                              │
│  Noise ──> [Student: N/2 steps] ──> Match clean sample             │
│                                                                      │
│  Key insight: Intermediate steps can be compressed                  │
│  Open question: Does overconfidence affect diffusion KD?            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Next Steps

1. **Literature review** - Survey diffusion distillation papers in detail
2. **Baseline implementation** - Add progressive distillation to framework
3. **StyleTTS2 experiments** - Test on actual TTS model
4. **Cross-paradigm analysis** - Do our classification findings transfer?

---

## Appendix: Diffusion Basics

For those unfamiliar with diffusion models:

### Forward Process (Adding Noise)
```
x_0 (clean) → x_1 → x_2 → ... → x_T (pure noise)

x_t = √(α_t) * x_0 + √(1 - α_t) * ε,  where ε ~ N(0, I)
```

### Reverse Process (Denoising)
```
x_T (noise) → x_{T-1} → x_{T-2} → ... → x_0 (clean)

x_{t-1} = μ_θ(x_t, t) + σ_t * z,  where z ~ N(0, I)
```

### Why So Many Steps?
- Each step makes a small correction
- Larger steps = larger errors = worse quality
- Distillation: Learn to make bigger, accurate steps

---

## Related Documents

| Document | Purpose |
|----------|---------|
| [research-framing.md](research-framing.md) | Classification KD research, findings, paper potential |
| [framework-direction.md](framework-direction.md) | Vision, mastery path, learning roadmap |
| [experimental_findings.md](experimental_findings.md) | Detailed experiment results (MNLI, QQP) |
| [run-options.md](run-options.md) | CLI reference and run management |
