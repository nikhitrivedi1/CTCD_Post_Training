# Optimizers and Learning Rates: A Deep Dive

This document explains how optimizers work, why learning rate scaling matters with batch size, and how all the pieces fit together in deep learning training.

---

## Table of Contents

1. [Gradient Descent Basics](#gradient-descent-basics)
2. [SGD and Its Limitations](#sgd-and-its-limitations)
3. [Momentum](#momentum)
4. [Adam Optimizer](#adam-optimizer)
5. [AdamW (Weight Decay Fix)](#adamw-weight-decay-fix)
6. [Learning Rate Schedules](#learning-rate-schedules)
7. [Batch Size and Learning Rate Scaling](#batch-size-and-learning-rate-scaling)
8. [Putting It All Together](#putting-it-all-together)

---

## Gradient Descent Basics

### The Core Idea

Training a neural network = finding parameters θ that minimize a loss function L(θ).

```
θ_new = θ_old - lr × ∇L(θ)

Where:
- θ = model parameters (weights)
- lr = learning rate (step size)
- ∇L(θ) = gradient of loss with respect to parameters
```

### The Gradient

The gradient tells us:
- **Direction**: Which way is "downhill" (decreases loss)
- **Magnitude**: How steep the slope is

```
Loss Landscape (2D visualization):

        ↑ Loss
        │    ╱╲
        │   ╱  ╲    ← Local minimum
        │  ╱    ╲__╱╲
        │ ╱          ╲___  ← Global minimum
        │╱               ╲
        └─────────────────→ Parameter θ

Gradient points toward steeper ascent → we go opposite direction
```

### Learning Rate: The Step Size

```
Too small (lr = 0.00001):
  ○ → ○ → ○ → ○ → ○ → ○ → ... (forever)
  Takes forever to converge

Too large (lr = 10):
  ○ ────────→ ○ ←──────── ○ → ...
  Overshoots, oscillates, diverges

Just right (lr = 0.001):
  ○ ──→ ○ ─→ ○ → ○ ✓
  Converges efficiently
```

---

## SGD and Its Limitations

### Vanilla SGD (Stochastic Gradient Descent)

```python
# Simplest optimizer
for batch in data:
    gradient = compute_gradient(batch)
    params = params - lr * gradient
```

### Problems with SGD

**1. Same learning rate for all parameters**
```
Parameter A: Large gradients, needs small steps
Parameter B: Tiny gradients, needs large steps

SGD: Uses same lr for both → suboptimal
```

**2. Noisy gradients**
```
Each mini-batch gives slightly different gradient estimate.

True gradient:     →→→→→→→→→→ (consistent direction)
Mini-batch grads:  →↗→↘→→↗→↘→ (noisy, zigzag path)
```

**3. Saddle points and plateaus**
```
        Loss
          │    ___________
          │   /           \
          │  /   plateau   \
          │ /               \
          └─────────────────→

Gradient ≈ 0 on plateau → training stalls
```

---

## Momentum

### The Idea: Remember Past Gradients

Like a ball rolling downhill, accumulate velocity:

```python
velocity = 0
for batch in data:
    gradient = compute_gradient(batch)
    velocity = momentum * velocity + gradient  # Accumulate
    params = params - lr * velocity
```

### Why It Helps

```
Without momentum (SGD):
  →↗→↘→→↗→↘→  (zigzag, slow)

With momentum (β=0.9):
  →→→→→→→→→→  (smoothed, faster)

The oscillations cancel out, consistent direction amplifies.
```

### Momentum Hyperparameter

```
β = 0.0: No momentum (vanilla SGD)
β = 0.9: Standard choice (remembers ~10 steps)
β = 0.99: Heavy momentum (remembers ~100 steps)
```

---

## Adam Optimizer

### The Key Innovation: Adaptive Learning Rates

Adam = **Ada**ptive **M**oment Estimation

It maintains TWO running averages per parameter:
- **m (first moment)**: Average of gradients (like momentum)
- **v (second moment)**: Average of squared gradients (variance)

```python
m = 0  # First moment (mean)
v = 0  # Second moment (variance)
t = 0  # Timestep

for batch in data:
    t += 1
    gradient = compute_gradient(batch)

    # Update moments
    m = β1 * m + (1 - β1) * gradient          # Momentum
    v = β2 * v + (1 - β2) * gradient²         # Variance

    # Bias correction (important early in training)
    m_hat = m / (1 - β1^t)
    v_hat = v / (1 - β2^t)

    # Update parameters
    params = params - lr * m_hat / (√v_hat + ε)
```

### What Each Part Does

**First moment (m) - Momentum:**
```
Smooths gradient direction, reduces noise
Same benefit as momentum in SGD
```

**Second moment (v) - Adaptive scaling:**
```
Parameter with large gradients → large v → smaller effective lr
Parameter with small gradients → small v → larger effective lr

This is the "adaptive" part!
```

**Bias correction:**
```
At t=1: m and v are initialized to 0, biased toward 0
Correction factor (1 - β^t) compensates for this
After ~10 steps, correction ≈ 1 (no effect)
```

### Default Hyperparameters

```python
Adam(
    lr=0.001,       # Base learning rate
    betas=(0.9, 0.999),  # β1 for momentum, β2 for variance
    eps=1e-8,       # Prevents division by zero
)
```

### Per-Parameter Effective Learning Rate

```
effective_lr[param_i] = lr * m_hat[i] / (√v_hat[i] + ε)

Example:
- Parameter A: gradient always ~0.1 → v ≈ 0.01 → effective_lr = lr/0.1 = 10×lr
- Parameter B: gradient always ~10 → v ≈ 100 → effective_lr = lr/10 = 0.1×lr

Adam automatically gives larger steps to "starved" parameters!
```

---

## AdamW (Weight Decay Fix)

### The Problem with L2 Regularization in Adam

Original Adam with L2 regularization:
```python
gradient = compute_gradient(batch) + weight_decay * params  # L2 added to gradient
```

**Issue:** The adaptive scaling in Adam scales down the weight decay too!

```
Large gradient params: v is large → weight decay effect is small
Small gradient params: v is small → weight decay effect is large

This is backwards! We want equal regularization on all params.
```

### AdamW: Decoupled Weight Decay

```python
# AdamW: Apply weight decay AFTER Adam update
params = params - lr * (adam_update + weight_decay * params)
```

This applies weight decay equally to all parameters, regardless of their gradient history.

### Why HuggingFace Uses AdamW

```python
# transformers default
from transformers import AdamW

optimizer = AdamW(
    model.parameters(),
    lr=2e-5,
    weight_decay=0.01,  # Decoupled weight decay
)
```

---

## Learning Rate Schedules

### Why Schedule the Learning Rate?

```
Early training: Far from optimum → large steps OK
Late training: Near optimum → need small steps for fine-tuning

Fixed lr = compromise that's suboptimal at both stages
```

### Common Schedules

**1. Constant (no schedule)**
```
lr: ═══════════════════════════
    |                          |
    0                        steps
```

**2. Linear Decay**
```
lr: ╲
     ╲
      ╲
       ╲________________
    |                   |
    0                 steps
```

**3. Linear Warmup + Linear Decay** (used in BERT/transformers)
```
lr:    ╱╲
      ╱  ╲
     ╱    ╲
    ╱      ╲___________
    |  |              |
    0 warmup        steps
```

**4. Cosine Annealing**
```
lr: ╭──╮
    │   ╲
    │    ╲
    │     ╰──────────
    |                |
    0              steps

Smooth decay, popular in vision models
```

**5. Cosine with Warm Restarts**
```
lr: ╭╮  ╭╮  ╭╮
    │╲  │╲  │╲
    │ ╲ │ ╲ │ ╲
    │  ╰│  ╰│  ╰
    |            |
    0          steps

"Shake" out of local minima periodically
```

### Our Setup (HuggingFace Default)

```python
# Linear warmup + linear decay
TrainingArguments(
    warmup_steps=200,      # Linear increase from 0 to lr
    # Then linear decay to 0 over remaining steps
)
```

```
lr:     ╱╲
       ╱  ╲
      ╱    ╲
     ╱      ╲
    ╱        ╲
   |  |       ╲________
   0  200            steps
      ↑
   warmup ends
```

---

## Batch Size and Learning Rate Scaling

### The Key Insight: Gradient Variance

```
Mini-batch gradient = estimate of true gradient

Small batch (16):  High variance, noisy estimate
Large batch (1024): Low variance, accurate estimate
```

**Mathematically:**
```
Var(gradient_estimate) ∝ 1/batch_size

Batch 16:   variance = σ²/16
Batch 1024: variance = σ²/1024 (64× smaller!)
```

### Why This Matters for Learning Rate

With **noisy gradients** (small batch):
- Taking large steps is risky (might go wrong direction)
- Need smaller learning rate

With **accurate gradients** (large batch):
- Taking large steps is safer
- Can use larger learning rate

### The Linear Scaling Rule

> When you multiply batch size by k, multiply learning rate by k

```
Batch 16,  lr = 2e-5    (baseline)
Batch 32,  lr = 4e-5    (2× batch → 2× lr)
Batch 64,  lr = 8e-5    (4× batch → 4× lr)
Batch 256, lr = 3.2e-4  (16× batch → 16× lr)
Batch 1024, lr = 1.28e-3 (64× batch → 64× lr)
```

### Intuition: Keeping "Progress per Sample" Constant

```
Batch 16, lr = 2e-5:
  - See 16 samples
  - Update params by 2e-5 × gradient
  - Progress per sample: 2e-5 × gradient / 16

Batch 1024, lr = 1.28e-3:
  - See 1024 samples
  - Update params by 1.28e-3 × gradient
  - Progress per sample: 1.28e-3 × gradient / 1024 = 2e-5 × gradient / 16

Same progress per sample → similar training dynamics!
```

### But There Are Limits

**Linear scaling breaks down at very large batches:**

```
Batch size:     16    64    256   1K    4K    8K    16K
Accuracy:      81.7  81.8  81.5  81.0  79.5  77.0  74.0
               ↑                              ↑
            optimal                    generalization gap
```

**Why?** Large batches converge to "sharp" minima that don't generalize well.

### Additional Tricks for Large Batch Training

**1. Warmup is crucial**
```python
# Small batch: warmup_steps = 200
# Large batch: warmup_steps = 1000-2000

# Start with tiny lr, ramp up to full lr
# Prevents divergence at start
```

**2. LARS/LAMB optimizers**
```
Layer-wise Adaptive Rate Scaling
Adjusts lr per layer based on weight/gradient norms
Enables training with batch sizes up to 32K
```

**3. Gradient accumulation (alternative approach)**
```python
# Instead of large batch, simulate it:
batch_size = 16
gradient_accumulation_steps = 64
effective_batch = 16 × 64 = 1024

# Same gradient quality, but slower (more forward passes)
```

---

## Putting It All Together

### Our Training Setup

```python
# Config
batch_size = 16
learning_rate = 2e-5
warmup_steps = 200
weight_decay = 0.01

# What happens
optimizer = AdamW(
    params,
    lr=2e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01,
)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    warmup_steps=200,
    num_training_steps=total_steps,
)
```

### Timeline of One Training Run

```
Step 0:
  - lr = 0 (warmup starting)
  - Adam moments m, v = 0

Step 100:
  - lr = 1e-5 (halfway through warmup)
  - Adam moments building up

Step 200:
  - lr = 2e-5 (peak, warmup complete)
  - Adam moments stabilized

Step 1000:
  - lr = 1.8e-5 (decaying)
  - Adam adapting per-parameter

Step 10000:
  - lr = 0.5e-5 (near end)
  - Fine-tuning with small steps

End:
  - lr ≈ 0
  - Model converged
```

### What Adam Does vs What We Control

| Component | Controlled By | Effect |
|-----------|---------------|--------|
| Base step size | `learning_rate` | Overall speed |
| Per-param adaptation | Adam (automatic) | Balances parameters |
| Training schedule | `warmup_steps`, scheduler | When to step big/small |
| Batch size effect | Manual LR scaling | Gradient quality |
| Regularization | `weight_decay` | Prevents overfitting |

### Scaling Recipe for This Project

| Batch Size | Learning Rate | Warmup Steps | Expected Accuracy |
|------------|---------------|--------------|-------------------|
| 16 (paper) | 2e-5 | 200 | ~81.7% |
| 64 | 8e-5 | 200 | ~81.7% |
| 256 | 3.2e-4 | 500 | ~81.5% |
| 512 | 6.4e-4 | 500 | ~81.0% |
| 1024 | 1.28e-3 | 1000 | ~80.0% |

---

## Common Misconceptions

### "Adam adjusts learning rate automatically, so I don't need to tune lr"

**Wrong.** Adam adjusts *per-parameter* scaling. The base `lr` still determines overall step magnitude and needs tuning.

### "Larger batch = faster training = better"

**Partially wrong.** Larger batch = fewer gradient updates. Need to scale lr. Very large batches can hurt generalization.

### "More warmup is always better"

**Wrong.** Too much warmup wastes steps at suboptimal lr. Rule of thumb: 5-10% of total steps.

### "AdamW and Adam are basically the same"

**Wrong for fine-tuning.** AdamW's decoupled weight decay is crucial for transformer fine-tuning. Always use AdamW for BERT/LLMs.

---

## Further Reading

- [On the Variance of Adaptive Learning Rate and Beyond (RAdam)](https://arxiv.org/abs/1908.03265)
- [Accurate, Large Minibatch SGD (Linear Scaling)](https://arxiv.org/abs/1706.02677)
- [Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101)
- [LAMB: Large Batch Optimization for Deep Learning](https://arxiv.org/abs/1904.00962)
