# How Training Differs with Batch Size

This document explains the fundamental differences in training dynamics when using small vs large batch sizes, and why these differences matter for model quality.

---

## Table of Contents

1. [The Training Loop Visualized](#the-training-loop-visualized)
2. [Small Batch Training](#small-batch-training)
3. [Large Batch Training](#large-batch-training)
4. [Side-by-Side Comparison](#side-by-side-comparison)
5. [The Loss Landscape Perspective](#the-loss-landscape-perspective)
6. [Gradient Noise as Regularization](#gradient-noise-as-regularization)
7. [Convergence Behavior](#convergence-behavior)
8. [Generalization Gap](#generalization-gap)
9. [What Happens During Training](#what-happens-during-training)
10. [Practical Implications](#practical-implications)

---

## The Training Loop Visualized

### What Happens Each Step

```
┌─────────────────────────────────────────────────────────────┐
│                    ONE TRAINING STEP                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Sample batch of B examples from training data           │
│     ┌───┬───┬───┬───┬───┐                                  │
│     │ x₁│ x₂│ x₃│...│ xB│  ← B samples                     │
│     └───┴───┴───┴───┴───┘                                  │
│                                                             │
│  2. Forward pass: compute predictions                       │
│     ŷᵢ = model(xᵢ) for each sample                         │
│                                                             │
│  3. Compute loss for each sample                            │
│     Lᵢ = loss(ŷᵢ, yᵢ)                                      │
│                                                             │
│  4. Average losses to get batch loss                        │
│     L_batch = (1/B) Σᵢ Lᵢ                                  │
│                                                             │
│  5. Backward pass: compute gradient                         │
│     ∇L = (1/B) Σᵢ ∇Lᵢ  ← Average of per-sample gradients  │
│                                                             │
│  6. Update parameters                                       │
│     θ = θ - lr × ∇L                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

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

### What It Looks Like

```
Training trajectory (2D projection of parameter space):

Start ●
       ↘
        ↗    ← Noisy, zigzag path
         ↘
          ↗
           ↘
            →
             ↘
              ↗
               ↘
                ● End (flat minimum)

Each step: Small, somewhat random direction
Overall: Explores the loss landscape widely
```

### Loss Curve

```
Loss
│
│●
│ ●
│  ●●
│    ●●●
│       ●●●●
│           ●●●●●●●●
│                    ●●●●●●●●●●●●●●
│                                   ●●●●●●●●●●●●
└───────────────────────────────────────────────→ Steps
 0                                            73,000

Characteristics:
- Noisy, lots of small fluctuations
- Gradual descent
- Many small improvements
```

### Per-Step Analysis

```
Step 1:    gradient = [0.12, -0.08, 0.15, ...]  ← sample 1-16
Step 2:    gradient = [0.09, -0.11, 0.13, ...]  ← sample 17-32
Step 3:    gradient = [0.14, -0.06, 0.11, ...]  ← sample 33-48
...

Gradient direction changes significantly each step
because different 16 samples give different signals.
```

### Memory and Compute

```
Memory per step:      ~4 GB (activations for 16 samples)
Forward passes/epoch: 24,544 (MNLI)
Time per step:        ~50ms
Total time:           ~7 hours
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

### What It Looks Like

```
Training trajectory (2D projection of parameter space):

Start ●
       \
        \    ← Smooth, direct path
         \
          \
           \
            \
             \
              \
               ● End (sharp minimum!)

Each step: Large, consistent direction
Overall: Efficient path to nearest minimum
```

### Loss Curve

```
Loss
│
│●
│ \
│  \
│   \
│    \
│     \____
│          \____
│               \________
└───────────────────────────────────────────────→ Steps
 0                                             2,300

Characteristics:
- Smooth, clear descent
- Rapid initial drop
- Quick convergence
```

### Per-Step Analysis

```
Step 1:    gradient = [0.11, -0.09, 0.12, ...]  ← sample 1-512
Step 2:    gradient = [0.11, -0.09, 0.12, ...]  ← sample 513-1024
Step 3:    gradient = [0.11, -0.08, 0.12, ...]  ← sample 1025-1536
...

Gradient direction is nearly identical each step
because 512 samples give a stable average.
```

### Memory and Compute

```
Memory per step:      ~24 GB (activations for 512 samples)
Forward passes/epoch: 767 (MNLI)
Time per step:        ~200ms
Total time:           ~10 minutes
```

---

## Side-by-Side Comparison

### Same Epoch, Different Experience

```
                    Small Batch (16)         Large Batch (512)
                    ────────────────         ─────────────────
Steps per epoch:    24,544                   767
Gradient updates:   24,544                   767
Samples seen:       392,702                  392,702 (same!)

Per step:
  Samples:          16                       512
  Gradient noise:   High                     Low
  Step size:        Small                    Large
  Direction:        Variable                 Consistent

Overall:
  Path to minimum:  Wandering, exploratory   Direct, efficient
  Time:             Long                     Short
  Final minimum:    Often flat               Often sharp
```

### Visualizing the Difference

```
Small Batch (many small noisy steps):

    Loss
    │ ●
    │  ●
    │   ●
    │  ●     ← Sometimes goes up!
    │   ●
    │    ●●
    │      ●●●
    │         ●●●●
    │             ●●●●●
    └──────────────────→ 73,000 steps


Large Batch (few large clean steps):

    Loss
    │●
    │ \
    │  \
    │   \
    │    \
    │     \____
    │          \_____
    │                \______
    │                       \______
    └──────────────────────────────→ 2,300 steps
```

---

## The Loss Landscape Perspective

### What is the Loss Landscape?

```
Imagine the loss as a terrain over parameter space:

        Loss (height)
             │
             │     ╱╲
             │    ╱  ╲    ╱╲
             │   ╱    ╲__╱  ╲
             │  ╱            ╲___
             │ ╱                 ╲____
             │╱    ↑               ↑  ╲___
             └─────┴───────────────┴──────→ Parameters
                   │               │
              Sharp minimum    Flat minimum

Sharp minimum: High curvature, narrow valley
Flat minimum:  Low curvature, wide basin
```

### Small Batch: Finds Flat Minima

```
Small batch gradient noise acts like a "temperature":

    │     ╱╲
    │    ╱  ╲    ╱╲
    │   ╱    ╲__╱  ╲
    │  ╱    ●→●→●   ╲___
    │ ╱        ↓        ╲____
    │╱         ●→●→●→●→●→●   ╲___
    └────────────────────────────→

The noise "bounces" the optimization out of sharp minima
and into wide, flat basins.
```

### Large Batch: Finds Sharp Minima

```
Large batch has low noise, goes straight to nearest minimum:

    │     ╱╲
    │    ╱  ╲    ╱╲
    │   ╱    ╲__╱  ╲
    │  ╱      ●     ╲___
    │ ╱       ↓          ╲____
    │╱        ●               ╲___
    └─────────↓───────────────────→
              ● (stuck in sharp minimum)

Without noise to escape, it settles in the first minimum found.
```

### Why Flat Minima Generalize Better

```
Training data:  Minimum at θ*
Test data:      Minimum at θ* + δ  (slightly different)

Sharp minimum:
    │    ╱╲
    │   ╱  ╲      ← Small δ causes big loss increase
    │  ╱ ●  ╲
    │ ╱  ↑   ╲
    └────┴────→
         θ*

Flat minimum:
    │  ______
    │ ╱  ●   ╲    ← Small δ causes small loss increase
    │╱   ↑    ╲
    └────┴─────→
         θ*

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

### Quantifying the Noise

```
Gradient covariance for batch size B:

Cov(ĝ) = (1/B) × Σ  (where Σ is per-sample gradient covariance)

Small B → Large noise → More regularization
Large B → Small noise → Less regularization
```

### Effective Temperature

```
The "temperature" of optimization:

T_eff ∝ lr / B

Small batch, small lr:  T_eff = 2e-5 / 16 = 1.25e-6
Large batch, large lr:  T_eff = 6.4e-4 / 512 = 1.25e-6

Linear scaling keeps temperature constant!
But: large batch still has less inherent noise.
```

---

## Convergence Behavior

### Small Batch Convergence

```
Epoch:  1         2         3
        │         │         │
Loss:   ●●●●●     │         │
            ●●●●●●●         │
                  ●●●●●●●●●●●
                            ●●●●●●●●●●●

Characteristics:
- Slow, steady progress
- Lots of micro-improvements
- Loss decreases gradually throughout
- Rarely overshoots
```

### Large Batch Convergence

```
Epoch:  1         2         3
        │         │         │
Loss:   ●         │         │
        │\        │         │
        │ \       │         │
        │  \______│_________│__________
        │         │         │

Characteristics:
- Rapid initial drop
- Quick plateau
- Most progress in first epoch
- Can overshoot if lr too high
```

### Learning Curves Comparison

```
        │
   Loss │●                            ← Both start same
        │ ●\
        │  ● \
        │   ●  \____________________  ← Large batch plateaus
        │    ●  \
        │     ●   \
        │      ●    \
        │       ●●●●●●●●●●●●●●●●●●●  ← Small batch keeps improving
        │
        └────────────────────────────→ Samples seen

Large batch reaches its (worse) minimum faster.
Small batch takes longer but finds better minimum.
```

---

## Generalization Gap

### Definition

```
Generalization Gap = Test Loss - Training Loss

Or equivalently:
Generalization Gap = Training Accuracy - Test Accuracy
```

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

### Visualizing Overfitting

```
        │
   Acc  │                    ●───────● Train (large batch)
        │               ●────
        │          ●────     ●───────● Train (small batch)
        │     ●────
        │●────
        │●────
        │     ●────          ●───────● Test (small batch)
        │          ●────
        │               ●────
        │                    ●───────● Test (large batch)
        │
        └────────────────────────────→ Epochs

Large batch: Train acc keeps rising, test acc drops
Small batch: Train and test acc rise together
```

### Why Large Batches Overfit

```
1. Less gradient noise → less implicit regularization
2. Finds sharper minima → more sensitive to data shift
3. Fewer gradient updates → less "averaging" over data

Result: Model memorizes training patterns
        that don't generalize to test data.
```

---

## What Happens During Training

### Epoch 1: Early Training

```
Small Batch:
┌─────────────────────────────────────────┐
│ Step 1-1000:   Learning basic patterns  │
│ Step 1000-5000: Refining representations│
│ Step 5000-24000: Gradual improvement    │
│                                         │
│ Many updates, each small adjustment     │
│ Model sees each sample ~1 time          │
└─────────────────────────────────────────┘

Large Batch:
┌─────────────────────────────────────────┐
│ Step 1-100:   Learning basic patterns   │
│ Step 100-400: Rapid convergence         │
│ Step 400-767: Near plateau              │
│                                         │
│ Few updates, each large adjustment      │
│ Model sees each sample ~1 time          │
└─────────────────────────────────────────┘
```

### Epoch 2-3: Late Training

```
Small Batch:
┌─────────────────────────────────────────┐
│ Continue making small improvements      │
│ Noise helps escape local minima         │
│ Explores different configurations       │
│ Finds robust, generalizable features    │
└─────────────────────────────────────────┘

Large Batch:
┌─────────────────────────────────────────┐
│ Already converged, minimal change       │
│ Small oscillations around minimum       │
│ May start overfitting                   │
│ Features are optimal for training data  │
└─────────────────────────────────────────┘
```

### What Each Update "Feels Like"

```
Small Batch Update:
  "I just saw 16 examples. Some said go left, some said
   go right. On average, I'll take a small step left,
   but I'm not very confident."

Large Batch Update:
  "I just saw 512 examples. They overwhelmingly agree:
   go left. I'll take a big confident step left."
```

---

## Practical Implications

### When to Use Small Batches

```
✓ Final training for best accuracy
✓ When generalization is critical
✓ Limited memory (only option)
✓ When you have plenty of time
✓ Research requiring exact reproduction
```

### When to Use Large Batches

```
✓ Hyperparameter sweeps (need many runs)
✓ Debugging and iteration
✓ When speed matters more than last 1%
✓ Distributed training (large clusters)
✓ When you'll do multiple runs and pick best
```

### Hybrid Approach

```
1. Use large batch for initial exploration
   - Quick feedback on what works
   - Test hyperparameter ranges
   - ~80% of final accuracy in 10% of time

2. Use small batch for final training
   - Best model quality
   - Submit to benchmark with this
   - Worth the extra time for final result
```

### Our Recommended Strategy

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
│ Time/epoch     │ Long (~2 hr)    │ Short (~3 min)  │
├────────────────┼─────────────────┼─────────────────┤
│ Gradient noise │ High            │ Low             │
│ Update size    │ Small           │ Large           │
│ Path to min    │ Wandering       │ Direct          │
│ Minimum type   │ Flat            │ Sharp           │
├────────────────┼─────────────────┼─────────────────┤
│ Train accuracy │ Good            │ Better          │
│ Test accuracy  │ Best            │ Worse           │
│ Generalization │ Strong          │ Weak            │
│ Overfitting    │ Rare            │ Common          │
├────────────────┼─────────────────┼─────────────────┤
│ Best for       │ Final training  │ Exploration     │
│                │ Benchmarks      │ Debugging       │
│                │ Production      │ Sweeps          │
└────────────────┴─────────────────┴─────────────────┘
```

---

## Key Takeaways

1. **Same samples, different experience**: Both see the same data, but in different-sized chunks with different dynamics.

2. **Noise is a feature, not a bug**: Small batch gradient noise provides implicit regularization that improves generalization.

3. **Sharp vs flat minima**: Large batches find sharp minima that overfit; small batches find flat minima that generalize.

4. **Speed vs quality tradeoff**: Large batches are faster but produce slightly worse models. Choose based on your goal.

5. **Linear scaling helps but doesn't fully compensate**: Scaling learning rate with batch size keeps some properties constant, but not all.

6. **Hybrid is often best**: Use large batches for exploration, small batches for final training.
