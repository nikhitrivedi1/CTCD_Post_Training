# How Training Differs with Batch Size

This document explains the fundamental differences in training dynamics when using small vs large batch sizes, and why these differences matter for model quality.

---

## Table of Contents

1. [The Training Loop Visualized](#the-training-loop-visualized)
2. [Small Batch Training](#small-batch-training)
3. [Large Batch Training](#large-batch-training)
4. [The Loss Landscape Perspective](#the-loss-landscape-perspective)
5. [Gradient Noise as Regularization](#gradient-noise-as-regularization)
6. [Generalization Gap](#generalization-gap)
7. [Practical Implications](#practical-implications)

---

## The Training Loop Visualized

### The Key Difference

```
Small batch (B=16):
  - Average of 16 gradients
  - High variance estimate
  - Update direction varies significantly step-to-step

Large batch (B=1024):
  - Average of 1024 gradients
  - Low variance estimate
  - Update direction is consistent step-to-step
```

---

## Small Batch Training

### Characteristics

```
Batch Size: 16-64
Learning Rate: Small (e.g., 2e-5)
Steps per Epoch: Many (e.g., 24,000 for MNLI)
Gradient Quality: Noisy
```

### Training Trajectory

```
Start ●
       ↘
        ↗    ← Noisy, zigzag path
         ↘
          ↗
           ↘
            → ● End (flat minimum)

Overall: Explores the loss landscape widely
```

---

## Large Batch Training

### Characteristics

```
Batch Size: 512-2048
Learning Rate: Large (e.g., 6.4e-4), scaled up
Steps per Epoch: Few (e.g., 767 for MNLI)
Gradient Quality: Clean
```

### Training Trajectory

```
Start ●
       \
        \    ← Smooth, direct path
         \
          \
           \ ● End (sharp minimum!)

Overall: Efficient path to nearest minimum
```

---

## The Loss Landscape Perspective

### Sharp vs Flat Minima

```
Sharp minimum (large batch finds this):
        ╱╲
       ╱  ╲      ← Narrow valley, sensitive to perturbation
      ╱    ╲

Flat minimum (small batch finds this):
     ____
    ╱    ╲     ← Wide valley, robust to perturbation
   ╱      ╲
```

Small batch gradient noise helps "bounce out" of sharp minima into flat ones.

### Why Flat Minima Generalize Better

```
Training data:  Minimum at θ*
Test data:      Minimum at θ* + δ  (slightly different)

Sharp minimum: Small δ causes big loss increase
Flat minimum:  Small δ causes small loss increase

Flat minima are more robust to distribution shift!
```

---

## Gradient Noise as Regularization

### The Noise Provides Implicit Regularization

```
Small batch training is like SGD + noise injection:

θ_new = θ - lr × (true_gradient + noise)

Where noise ~ N(0, σ²/B)

This noise:
1. Prevents overfitting to specific samples
2. Helps escape sharp minima
3. Acts like a form of regularization
```

---

## Generalization Gap

### How Batch Size Affects It

```
Batch Size:     16      64     256    512    1024   2048
─────────────────────────────────────────────────────────
Train Acc:     82.5%   82.8%  83.0%  83.5%  84.0%  85.0%
Test Acc:      81.7%   81.8%  81.5%  81.0%  80.0%  78.0%
─────────────────────────────────────────────────────────
Gap:            0.8%    1.0%   1.5%   2.5%   4.0%   7.0%
                 ↑                              ↑
              Good!                      Overfitting!
```

### Why Large Batches Overfit

1. Less gradient noise → less implicit regularization
2. Finds sharper minima → more sensitive to data shift
3. Fewer gradient updates → less "averaging" over data

---

## Practical Implications

### When to Use Small Batches

- Final training for best accuracy
- When generalization is critical
- Research requiring exact reproduction

### When to Use Large Batches

- Hyperparameter sweeps (need many runs)
- Debugging and iteration
- When speed matters more than last 1%

### Hybrid Approach (Recommended)

```
Phase 1: Exploration (Large Batch)
├── batch_size: 512
├── learning_rate: 6.4e-4
├── goal: Find good hyperparameters
└── time: ~10 min per experiment

Phase 2: Validation (Medium Batch)
├── batch_size: 128
├── learning_rate: 1.6e-4
├── goal: Verify promising configs
└── time: ~30 min per experiment

Phase 3: Final Training (Small Batch)
├── batch_size: 16-32
├── learning_rate: 2e-5 to 4e-5
├── goal: Best possible model
└── time: ~7 hours, run once
```

---

## Summary Table

```
┌────────────────┬─────────────────┬─────────────────┐
│    Aspect      │   Small Batch   │   Large Batch   │
├────────────────┼─────────────────┼─────────────────┤
│ Batch size     │ 16-64           │ 512-2048        │
│ Learning rate  │ Small (2e-5)    │ Large (6e-4)    │
│ Steps/epoch    │ Many (24,000)   │ Few (767)       │
├────────────────┼─────────────────┼─────────────────┤
│ Gradient noise │ High            │ Low             │
│ Path to min    │ Wandering       │ Direct          │
│ Minimum type   │ Flat            │ Sharp           │
├────────────────┼─────────────────┼─────────────────┤
│ Test accuracy  │ Best            │ Worse           │
│ Generalization │ Strong          │ Weak            │
├────────────────┼─────────────────┼─────────────────┤
│ Best for       │ Final training  │ Exploration     │
│                │ Benchmarks      │ Debugging       │
└────────────────┴─────────────────┴─────────────────┘
```

---

## Key Takeaways

1. **Same samples, different experience**: Both see the same data, but with different dynamics.

2. **Noise is a feature, not a bug**: Small batch gradient noise provides implicit regularization.

3. **Sharp vs flat minima**: Large batches find sharp minima that overfit.

4. **Speed vs quality tradeoff**: Large batches are faster but produce slightly worse models.

5. **Hybrid is often best**: Use large batches for exploration, small batches for final training.
