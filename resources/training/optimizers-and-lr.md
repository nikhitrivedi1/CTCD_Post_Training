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
7. [Putting It All Together](#putting-it-all-together)

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

### Problems with SGD

**1. Same learning rate for all parameters**
**2. Noisy gradients**
**3. Saddle points and plateaus**

---

## Momentum

### The Idea: Remember Past Gradients

Like a ball rolling downhill, accumulate velocity:

```python
velocity = momentum * velocity + gradient  # Accumulate
params = params - lr * velocity
```

### Why It Helps

The oscillations cancel out, consistent direction amplifies.

---

## Adam Optimizer

### The Key Innovation: Adaptive Learning Rates

Adam = **Ada**ptive **M**oment Estimation

It maintains TWO running averages per parameter:
- **m (first moment)**: Average of gradients (like momentum)
- **v (second moment)**: Average of squared gradients (variance)

```python
# Update parameters
params = params - lr * m_hat / (√v_hat + ε)
```

### What Each Part Does

**First moment (m):** Smooths gradient direction, reduces noise
**Second moment (v):** Adaptive scaling - large gradients get smaller lr

### Default Hyperparameters

```python
Adam(
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
)
```

---

## AdamW (Weight Decay Fix)

### The Problem with L2 Regularization in Adam

The adaptive scaling in Adam scales down the weight decay too!

### AdamW: Decoupled Weight Decay

```python
# AdamW: Apply weight decay AFTER Adam update
params = params - lr * (adam_update + weight_decay * params)
```

This applies weight decay equally to all parameters.

---

## Learning Rate Schedules

### Common Schedules

**Linear Warmup + Linear Decay** (used in BERT/transformers):
```
lr:    ╱╲
      ╱  ╲
     ╱    ╲
    ╱      ╲___________
    |  |              |
    0 warmup        steps
```

### Our Setup (HuggingFace Default)

```python
TrainingArguments(
    warmup_steps=200,      # Linear increase from 0 to lr
    # Then linear decay to 0 over remaining steps
)
```

---

## Putting It All Together

### Our Training Setup

```python
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

**Wrong.** Adam adjusts *per-parameter* scaling. The base `lr` still needs tuning.

### "Larger batch = faster training = better"

**Partially wrong.** Need to scale lr. Very large batches can hurt generalization.

### "AdamW and Adam are basically the same"

**Wrong for fine-tuning.** AdamW's decoupled weight decay is crucial for transformers.
