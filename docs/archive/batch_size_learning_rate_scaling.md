# Batch Size and Learning Rate Scaling: The Complete Guide

This document provides a deep dive into the relationship between batch size and learning rate, with specific calculations for our BERT→DistilBERT distillation task.

---

## Table of Contents

1. [Why Batch Size Affects Learning Rate](#why-batch-size-affects-learning-rate)
2. [The Mathematics](#the-mathematics)
3. [Linear Scaling Rule](#linear-scaling-rule)
4. [Square Root Scaling (Alternative)](#square-root-scaling-alternative)
5. [Warmup Requirements](#warmup-requirements)
6. [Practical Limits](#practical-limits)
7. [Our Project: Specific Calculations](#our-project-specific-calculations)
8. [Recipes for Different GPUs](#recipes-for-different-gpus)

---

## Why Batch Size Affects Learning Rate

### The Intuition

Imagine you're trying to find the lowest point in a foggy valley:

```
Small batch (16 samples):
  - You ask 16 people which way is downhill
  - They point in slightly different directions (noisy)
  - You take a small step (can't trust the average fully)

Large batch (1024 samples):
  - You ask 1024 people which way is downhill
  - Their average direction is very accurate
  - You can confidently take a larger step
```

### The Key Insight

**Gradient quality improves with batch size.**

```
True gradient:        →→→→→→→→→→ (unknown, what we want)

Batch of 1:          ↗ (very noisy, could be any direction)
Batch of 16:         →↗ (somewhat noisy)
Batch of 256:        →→ (fairly accurate)
Batch of 4096:       →→→→ (very accurate)
```

---

## The Mathematics

### Gradient Variance

For a mini-batch of size B, the gradient estimate is:

```
ĝ = (1/B) Σᵢ ∇L(xᵢ)

Where each ∇L(xᵢ) has variance σ²
```

The variance of the batch gradient estimate:

```
Var(ĝ) = σ² / B

Key insight: Variance decreases linearly with batch size!
```

### Numerical Example

```
Assume per-sample gradient variance σ² = 1.0

Batch 16:   Var(ĝ) = 1.0 / 16 = 0.0625
Batch 64:   Var(ĝ) = 1.0 / 64 = 0.0156   (4× smaller)
Batch 256:  Var(ĝ) = 1.0 / 256 = 0.0039  (16× smaller)
Batch 1024: Var(ĝ) = 1.0 / 1024 = 0.00098 (64× smaller)
```

### Standard Deviation (More Intuitive)

```
Batch 16:   std(ĝ) = 0.25
Batch 64:   std(ĝ) = 0.125  (2× smaller, not 4×)
Batch 256:  std(ĝ) = 0.0625 (4× smaller)
Batch 1024: std(ĝ) = 0.031  (8× smaller)

Standard deviation scales as 1/√B
```

---

## Linear Scaling Rule

### The Rule (Goyal et al., 2017)

> When batch size increases by factor k, multiply learning rate by k

```
lr_new = lr_base × (batch_new / batch_base)
```

### Derivation

**Goal:** Keep the expected weight update magnitude constant across batch sizes.

With SGD, the weight update is:
```
Δw = -lr × ĝ
```

The expected squared magnitude:
```
E[||Δw||²] = lr² × E[||ĝ||²]
           = lr² × (||g_true||² + Var(ĝ))
           ≈ lr² × ||g_true||²  (for large batches, variance small)
```

To keep E[||Δw||²] constant when variance decreases:
```
If Var(ĝ) decreases by k (from k× larger batch)
Then lr can increase by √k to maintain same update variance

BUT: We also do k× fewer updates per epoch!

Combined: lr should increase by k (linear scaling)
```

### Why Linear, Not Square Root?

The confusion arises from two competing effects:

```
Effect 1: Lower gradient variance → can take larger steps
          Suggests: lr ∝ √B (square root scaling)

Effect 2: Fewer updates per epoch → need larger steps per update
          Suggests: lr ∝ B (linear with batch)

Combined: lr ∝ B (linear scaling wins)
```

**The key is "equivalent training"**: same number of samples seen should produce same result.

### Calculation Table

| Batch Size | Scaling Factor | Learning Rate |
|------------|----------------|---------------|
| 16 (base)  | 1×             | 2e-5          |
| 32         | 2×             | 4e-5          |
| 64         | 4×             | 8e-5          |
| 128        | 8×             | 1.6e-4        |
| 256        | 16×            | 3.2e-4        |
| 512        | 32×            | 6.4e-4        |
| 1024       | 64×            | 1.28e-3       |
| 2048       | 128×           | 2.56e-3       |

---

## Square Root Scaling (Alternative)

### When to Use √B Scaling

Some practitioners use square root scaling:

```
lr_new = lr_base × √(batch_new / batch_base)
```

| Batch Size | Linear LR | Square Root LR |
|------------|-----------|----------------|
| 16         | 2e-5      | 2e-5           |
| 64         | 8e-5      | 4e-5           |
| 256        | 3.2e-4    | 8e-5           |
| 1024       | 1.28e-3   | 1.6e-4         |

### Arguments for Square Root

1. **Noise scale:** Gradient noise std scales as 1/√B, so lr can scale as √B
2. **Adam already adapts:** Adam's per-parameter scaling handles some variance
3. **More conservative:** Less likely to diverge

### When Each Works Best

```
Linear scaling:    SGD, large-scale distributed training
Square root:       Adam/AdamW, fine-tuning pretrained models
Empirical tuning:  Always the best approach!
```

### Our Recommendation for This Project

**Use linear scaling with AdamW**, but be conservative:

```python
# Start with linear scaling
lr_scaled = lr_base * (batch_new / batch_base)

# But cap at reasonable maximum
lr_final = min(lr_scaled, 1e-3)  # Don't go above 1e-3 for fine-tuning
```

---

## Warmup Requirements

### Why Warmup Matters More for Large Batches

At training start:
- Adam's moment estimates (m, v) are zero
- Bias correction is active but imperfect
- First few updates can be very large/unstable

With large batches:
- Already using large learning rate
- Instability is amplified
- More warmup needed to stabilize

### Warmup Scaling Rule

```
warmup_steps ∝ batch_size (approximately)

Or more conservatively:
warmup_steps = base_warmup × √(batch_new / batch_base)
```

### Calculation Table

| Batch Size | Linear Warmup | Conservative Warmup |
|------------|---------------|---------------------|
| 16         | 200           | 200                 |
| 64         | 800           | 400                 |
| 256        | 3200          | 800                 |
| 512        | 6400          | 1131                |
| 1024       | 12800         | 1600                |

**Practical recommendation:** Use 5-10% of total training steps as warmup.

### Our Settings

```yaml
# Batch 16 (paper)
batch_size: 16
warmup_steps: 200  # ~0.3% of 73,750 steps

# Batch 256 (scaled)
batch_size: 256
warmup_steps: 500  # ~10% of 4,609 steps

# Batch 1024 (aggressive)
batch_size: 1024
warmup_steps: 1000  # ~87% of 1,152 steps (yes, lots of warmup!)
```

---

## Practical Limits

### The Generalization Gap

Beyond a certain batch size, test accuracy degrades:

```
Batch Size:    16    64   256   1K    4K    8K   16K   32K
Test Acc:     81.7  81.8  81.5  81.0  79.5  77.0  74.0  70.0
                                            ↑
                                    Generalization gap appears
```

### Why Large Batches Hurt Generalization

**Sharp vs Flat Minima:**

```
Loss Landscape:

Sharp minimum (large batch finds this):
        ╱╲
       ╱  ╲      ← Narrow valley, sensitive to perturbation
      ╱    ╲
   __╱      ╲__

Flat minimum (small batch finds this):
     ____
    ╱    ╲     ← Wide valley, robust to perturbation
   ╱      ╲
__╱        ╲__
```

Small batch gradient noise helps "bounce out" of sharp minima into flat ones.

### Critical Batch Size

For most tasks, there's a "critical batch size" beyond which:
- Training becomes less efficient (need more total samples)
- Generalization degrades

```
BERT fine-tuning critical batch: ~256-512
ImageNet training critical batch: ~8K-16K
GPT pretraining critical batch: ~1M+ (with proper techniques)
```

### Techniques for Very Large Batches

If you must use batch > 1024:

1. **LARS (Layer-wise Adaptive Rate Scaling)**
   ```python
   # Per-layer learning rate based on ||weights|| / ||gradients||
   layer_lr = base_lr * ||w|| / ||∇w||
   ```

2. **LAMB (Layer-wise Adaptive Moments)**
   ```python
   # LARS + Adam = LAMB
   # Used to train BERT with batch size 65K
   ```

3. **Gradient noise injection**
   ```python
   # Add artificial noise to simulate smaller batch
   gradient = gradient + noise * (1/sqrt(batch_size))
   ```

---

## Our Project: Specific Calculations

### MNLI Dataset Statistics

```
Training samples:     392,702
Validation samples:   9,815 (matched)
Epochs:              3
Total samples seen:  1,178,106
```

### Steps per Epoch by Batch Size

```python
steps_per_epoch = ceil(392,702 / batch_size)

Batch 16:   24,544 steps/epoch × 3 = 73,632 total
Batch 64:   6,136 steps/epoch × 3 = 18,408 total
Batch 256:  1,534 steps/epoch × 3 = 4,602 total
Batch 512:  767 steps/epoch × 3 = 2,301 total
Batch 1024: 384 steps/epoch × 3 = 1,152 total
```

### Complete Scaling Table for MNLI

| Batch | LR | Warmup | Steps | Time (P100) | Time (H200) |
|-------|-----|--------|-------|-------------|-------------|
| 16 | 2e-5 | 200 | 73,632 | ~7 hr | ~2 hr |
| 64 | 8e-5 | 200 | 18,408 | ~2 hr | ~30 min |
| 256 | 3.2e-4 | 500 | 4,602 | ~45 min | ~10 min |
| 512 | 6.4e-4 | 500 | 2,301 | ~25 min | ~6 min |
| 1024 | 1.28e-3 | 1000 | 1,152 | ~15 min | ~4 min |

### Gradient Accumulation Alternative

Can't fit large batch in memory? Simulate it:

```python
# Physical batch 64, effective batch 256
batch_size = 64
gradient_accumulation_steps = 4
effective_batch = 64 × 4 = 256

# Use lr for effective batch size
learning_rate = 3.2e-4  # As if batch=256
```

**Tradeoff:** Same gradient quality, but 4× more forward passes (slower).

---

## Recipes for Different GPUs

### Recipe: Paper Reproduction (Any GPU)

```yaml
# Exact paper settings - guaranteed to work
training:
  batch_size: 16
  learning_rate: 0.00002
  warmup_steps: 200
  num_epochs: 3

# Time: ~7 hours on P100, ~2 hours on H200
# Memory: ~2-3 GB
# Expected: 81.77% accuracy
```

### Recipe: T4 16GB (Colab Free/Pro)

```yaml
# Maximize T4 utilization
training:
  batch_size: 128
  learning_rate: 0.00016  # 8× base
  warmup_steps: 400
  num_epochs: 3

# Time: ~1 hour
# Memory: ~10 GB
# Expected: ~81.5% accuracy
```

### Recipe: A100 40GB

```yaml
# Good balance of speed and accuracy
training:
  batch_size: 256
  learning_rate: 0.00032  # 16× base
  warmup_steps: 500
  num_epochs: 3

# Time: ~20 minutes
# Memory: ~15 GB
# Expected: ~81.5% accuracy
```

### Recipe: A100 80GB

```yaml
# Push for speed
training:
  batch_size: 512
  learning_rate: 0.00064  # 32× base
  warmup_steps: 500
  num_epochs: 3

# Time: ~12 minutes
# Memory: ~28 GB
# Expected: ~81.0% accuracy
```

### Recipe: H200 144GB (Maximum Speed)

```yaml
# Aggressive but may lose accuracy
training:
  batch_size: 1024
  learning_rate: 0.001    # Capped, not full 64× scale
  warmup_steps: 1000
  num_epochs: 3

# Time: ~5 minutes
# Memory: ~55 GB
# Expected: ~80.0-80.5% accuracy
```

### Recipe: H200 144GB (Balanced)

```yaml
# Best accuracy/speed tradeoff on H200
training:
  batch_size: 512
  learning_rate: 0.00064
  warmup_steps: 500
  num_epochs: 3

# Time: ~8 minutes
# Memory: ~28 GB
# Expected: ~81.0% accuracy
```

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────┐
│          BATCH SIZE / LEARNING RATE SCALING             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Linear Scaling Rule:                                   │
│  ────────────────────                                   │
│  lr_new = lr_base × (batch_new / batch_base)            │
│                                                         │
│  Base (paper): batch=16, lr=2e-5                        │
│                                                         │
│  ┌────────┬─────────┬─────────┬─────────┐              │
│  │ Batch  │   LR    │ Warmup  │ Quality │              │
│  ├────────┼─────────┼─────────┼─────────┤              │
│  │   16   │  2e-5   │   200   │  Best   │              │
│  │   64   │  8e-5   │   200   │  ~Same  │              │
│  │  256   │ 3.2e-4  │   500   │  Good   │              │
│  │  512   │ 6.4e-4  │   500   │  OK     │              │
│  │ 1024   │  1e-3*  │  1000   │  Lower  │              │
│  └────────┴─────────┴─────────┴─────────┘              │
│  * Capped at 1e-3 for stability                        │
│                                                         │
│  Rules of Thumb:                                        │
│  ───────────────                                        │
│  • Warmup = 5-10% of total steps                       │
│  • Max effective lr ≈ 1e-3 for fine-tuning             │
│  • Critical batch ~256-512 for BERT                    │
│  • Beyond critical: speed gain, accuracy loss          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## References

- [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677) - Linear scaling rule
- [Don't Decay the Learning Rate, Increase the Batch Size](https://arxiv.org/abs/1711.00489) - Equivalence proof
- [Large Batch Training of Convolutional Networks (LARS)](https://arxiv.org/abs/1708.03888)
- [Large Batch Optimization for Deep Learning: Training BERT in 76 minutes (LAMB)](https://arxiv.org/abs/1904.00962)
- [On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima](https://arxiv.org/abs/1609.04836)
