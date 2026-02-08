# Voice Distillation: StyleTTS2 Compression Plan

Compressing StyleTTS2 as a fourth modality — combining text encoder KD, diffusion step reduction, and decoder compression.

---

## StyleTTS2 Architecture

| Component | Role | Approx Size | KD Approach |
|-----------|------|-------------|-------------|
| **PL-BERT** | Text encoder (12 layers) | ~110M | BERT-style KD (R/F/A) |
| **Style Diffusion** | Latent style vector generation | ~20-30M | Progressive / Consistency distillation |
| **Duration Predictor** | Phoneme duration modeling | ~5M | Response KD (MSE) |
| **Prosody Predictor** | Pitch/energy modeling | ~5M | Response KD (MSE) |
| **Pitch Extractor (JDC)** | F0 extraction (pre-trained) | ~10M | Frozen (no distillation needed) |
| **HiFi-GAN Decoder** | Mel-spectrogram to waveform | ~14M | Feature KD on upsampling blocks |
| **WavLM Discriminator** | Adversarial training signal | ~95M | Training-only (not deployed) |

**Key insight:** Each component maps to a KD approach we've already studied:
- PL-BERT → BERT distillation (our classification/translation work)
- Style Diffusion → Diffusion distillation (our CIFAR-10 work)
- HiFi-GAN → Feature/Response KD on convolutional decoder

---

## Datasets

| Dataset | Speakers | Hours | Resolution | Use |
|---------|----------|-------|------------|-----|
| **LJSpeech** | 1 (female) | 24h | 22.05kHz | Single-speaker baseline |
| **LibriTTS** | 2,456 | ~245h | 24kHz | Multi-speaker, zero-shot |
| **VCTK** | 44 | ~44h | 48kHz | Multi-speaker evaluation |

**Recommendation:** Start with **LJSpeech** (simpler, single-speaker, fast iteration). Extend to LibriTTS for multi-speaker experiments.

---

## Evaluation Metrics

| Metric | What it measures | Type | Tool |
|--------|-----------------|------|------|
| **MOS** | Naturalness (1-5 scale) | Subjective | Human eval or UTMOS proxy |
| **PESQ** | Perceptual speech quality | Objective | `pesq` library |
| **STOI** | Intelligibility (0-1) | Objective | `pystoi` library |
| **Speaker Similarity** | Cosine sim of speaker embeddings | Objective | ECAPA-TDNN / WavLM |
| **RTF** | Real-time factor (speed) | Efficiency | Wall-clock / audio duration |

**Primary metric:** MOS (or UTMOS proxy for automated evaluation).
**Secondary:** PESQ + STOI for objective quality, RTF for efficiency.

---

## Pre-trained Models (Teachers)

| Model | Dataset | HuggingFace ID | Notes |
|-------|---------|----------------|-------|
| StyleTTS2-LJSpeech | LJSpeech | `yl4579/StyleTTS2-LJSpeech` | Single-speaker, best starting point |
| StyleTTS2-LibriTTS | LibriTTS | `yl4579/StyleTTS2-LibriTTS` | Multi-speaker, zero-shot capable |

**GitHub:** [yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2)

---

## Compression Strategies

### Strategy 1: Component-wise Distillation

Distill each component independently, then assemble.

```
Teacher StyleTTS2 (full)
├── PL-BERT (12 layers) ──────→ DistilPL-BERT (6 layers)     [BERT KD: R/F/A]
├── Style Diffusion ───────────→ Fewer diffusion steps         [Progressive Distill]
├── Duration/Prosody Predictors → Smaller predictors           [Response KD: MSE]
├── Pitch Extractor ───────────→ Keep frozen (small already)
└── HiFi-GAN Decoder ─────────→ Smaller HiFi-GAN              [Feature KD]
```

**Prior art:** Nix-TTS achieved 89% parameter reduction via module-wise distillation on a similar TTS architecture (~40M → 5.2M params).

### Strategy 2: Diffusion Step Reduction Only

Keep all components same size, reduce style diffusion steps.

```
Style Diffusion: 100 steps → 4 steps (Progressive Distillation)
                 100 steps → 1 step  (Consistency Distillation)
```

**Advantage:** Simplest to implement. Direct application of our CIFAR-10 diffusion work.
**Limitation:** Only speeds up inference, doesn't reduce model size.

### Strategy 3: End-to-End Distillation

Train a smaller student model end-to-end on teacher-generated speech.

```
Teacher: StyleTTS2-LJSpeech (full) generates mel-spectrograms
Student: Smaller architecture trained to match teacher output
Loss: L1/MSE on mel + adversarial (WavLM discriminator)
```

**Advantage:** Holistic optimization.
**Disadvantage:** Harder to train, less interpretable.

---

## Experiment Design

### Phase 1: Style Diffusion Step Reduction (connects to CIFAR-10 work)

| # | Method | Steps | Description |
|---|--------|-------|-------------|
| V0 | Teacher (reference) | 100 | Full StyleTTS2 |
| V1 | Fewer DDIM steps (no distill) | 10, 4 | Naive reduction baseline |
| V2 | Progressive distillation | 4 | Step reduction |
| V3 | Consistency distillation | 1 | Single-step generation |

### Phase 2: PL-BERT Compression (connects to text KD work)

| # | Method | Layers | Description |
|---|--------|--------|-------------|
| V4 | Full PL-BERT | 12 | Baseline |
| V5 | DistilPL-BERT (R only) | 6 | Response KD |
| V6 | DistilPL-BERT (R+F+A) | 6 | Full FAR |

### Phase 3: Decoder Compression

| # | Method | Channels | Description |
|---|--------|----------|-------------|
| V7 | Full HiFi-GAN | Original | Baseline |
| V8 | Smaller HiFi-GAN | 50% channels | Feature KD on upsampling |

### Phase 4: Combined

| # | Method | Compression | Description |
|---|--------|-------------|-------------|
| V9 | All components compressed | Steps + params | Full pipeline distillation |

---

## GPU Requirements

| Task | Memory | Time (4x A100) |
|------|--------|-----------------|
| Fine-tune teacher (LJSpeech) | ~40GB | ~4h |
| Style diffusion step reduction | ~16GB | ~4-8h |
| PL-BERT distillation | ~16GB | ~2-4h |
| Full pipeline compression | ~32GB | ~8-16h |
| Evaluation (MOS proxy) | ~8GB | ~1h |

---

## New Components

```
src/voice/
    __init__.py
    styletts2_wrapper.py     # Load/wrap StyleTTS2 components
    diffusion_distill.py     # Style diffusion step reduction
    encoder_distill.py       # PL-BERT compression (reuse text KD)
    decoder_distill.py       # HiFi-GAN compression
    evaluation.py            # MOS/PESQ/STOI computation

configs/voice/
    ljspeech/
        base.yaml            # Shared config
        step_reduction.yaml  # Style diffusion steps
        encoder_kd.yaml      # PL-BERT distillation
        decoder_kd.yaml      # HiFi-GAN compression
        combined.yaml        # Full pipeline
```

---

## Paper Narrative

"We study knowledge distillation across four modalities:
1. **Classification** (BERT→DistilBERT): parameter compression, output distribution transfer
2. **Translation** (Llama 3B→1B): parameter compression, token-level distribution transfer
3. **Image Generation** (DDPM): step compression AND parameter compression
4. **Speech Synthesis** (StyleTTS2): **all three** — text encoder KD (R/F/A), diffusion step reduction, and decoder compression

StyleTTS2 is the capstone: it contains a BERT-like encoder (PL-BERT), a diffusion model (style generation), and a convolutional decoder (HiFi-GAN). Every KD technique we've developed applies to a different component. This demonstrates the universality of our distillation framework across architectures and modalities."

---

## Key Papers

1. **Li et al., 2023** — "StyleTTS 2: Towards Human-Level Text-to-Speech" (NeurIPS 2023) — [arXiv:2306.07691](https://arxiv.org/abs/2306.07691)
2. **Chevi et al., 2022** — "Nix-TTS: Lightweight TTS via Module-wise Distillation" — [arXiv:2203.15643](https://arxiv.org/abs/2203.15643)
3. **Salimans & Ho, 2022** — "Progressive Distillation for Fast Sampling" — [arXiv:2202.00512](https://arxiv.org/abs/2202.00512)
4. **DMDSpeech, 2024** — "Distilled Diffusion Model for TTS" — [arXiv:2410.11097](https://arxiv.org/abs/2410.11097)

## Reference Implementations

| Repository | Approach | Notes |
|-----------|----------|-------|
| [yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2) | Full model | Official, PyTorch |
| [rendchevi/nix-tts](https://github.com/rendchevi/nix-tts) | Module-wise KD | 89% param reduction |
| [dangtr0408/StyleTTS2-lite](https://huggingface.co/dangtr0408/StyleTTS2-lite) | Lightweight variant | Community, initialized from LibriTTS |
