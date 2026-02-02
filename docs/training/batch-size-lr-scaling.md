# Batch Size and Learning Rate Scaling: The Complete Guide

This document provides a deep dive into the relationship between batch size and learning rate, with specific calculations for our BERT→DistilBERT distillation task.

---

## Table of Contents

1. [Why Batch Size Affects Learning Rate](#why-batch-size-affects-learning-rate)
2. [The Linear Scaling Rule](#the-linear-scaling-rule)
3. [Warmup Requirements](#warmup-requirements)
4. [Practical Limits](#practical-limits)
5. [Our Project: Specific Calculations](#our-project-specific-calculations)
6. [Recipes for Different GPUs](#recipes-for-different-gpus)

---

## Why Batch Size Affects Learning Rate

### The Intuition

**Gradient quality improves with batch size.**

```
True gradient:        →→→→→→→→→→ (unknown, what we want)

Batch of 16:         →↗ (somewhat noisy)
Batch of 256:        →→ (fairly accurate)
Batch of 4096:       →→→→ (very accurate)
```

### The Mathematics

```
Var(gradient_estimate) ∝ 1/batch_size

Batch 16:   Var(ĝ) = σ²/16
Batch 1024: Var(ĝ) = σ²/1024 (64× smaller!)
```

---

## The Linear Scaling Rule

### The Rule (Goyal et al., 2017)

> When batch size increases by factor k, multiply learning rate by k

```
lr_new = lr_base × (batch_new / batch_base)
```

### Calculation Table

| Batch Size | Scaling Factor | Learning Rate |
|------------|----------------|---------------|
| 16 (base)  | 1×             | 2e-5          |
| 64         | 4×             | 8e-5          |
| 256        | 16×            | 3.2e-4        |
| 512        | 32×            | 6.4e-4        |
| 1024       | 64×            | 1.28e-3       |

---

## Warmup Requirements

### Why Warmup Matters More for Large Batches

With large batches:
- Already using large learning rate
- Instability is amplified
- More warmup needed to stabilize

### Our Settings

```yaml
# Batch 16 (paper)
warmup_steps: 200  # ~0.3% of steps

# Batch 256 (scaled)
warmup_steps: 500  # ~10% of steps

# Batch 1024 (aggressive)
warmup_steps: 1000
```

---

## Practical Limits

### The Generalization Gap

Beyond a certain batch size, test accuracy degrades:

```
Batch Size:    16    64   256   1K    4K    8K
Test Acc:     81.7  81.8  81.5  81.0  79.5  77.0
```

### Critical Batch Size

```
BERT fine-tuning critical batch: ~256-512
```

---

## Our Project: Specific Calculations

### MNLI Dataset Statistics

```
Training samples:     392,702
Epochs:              3
Total samples seen:  1,178,106
```

### Steps per Epoch by Batch Size

```
Batch 16:   24,544 steps/epoch × 3 = 73,632 total
Batch 256:  1,534 steps/epoch × 3 = 4,602 total
Batch 1024: 384 steps/epoch × 3 = 1,152 total
```

### Complete Scaling Table for MNLI

| Batch | LR | Warmup | Steps | Time (H200) |
|-------|-----|--------|-------|-------------|
| 16 | 2e-5 | 200 | 73,632 | ~2 hr |
| 256 | 3.2e-4 | 500 | 4,602 | ~10 min |
| 512 | 6.4e-4 | 500 | 2,301 | ~6 min |
| 1024 | 1.28e-3 | 1000 | 1,152 | ~4 min |

---

## Recipes for Different GPUs

### Recipe: Paper Reproduction (Any GPU)

```yaml
batch_size: 16
learning_rate: 0.00002
warmup_steps: 200
# Expected: 81.77% accuracy
```

### Recipe: T4 16GB (Colab)

```yaml
batch_size: 128
learning_rate: 0.00016  # 8× base
warmup_steps: 400
# Expected: ~81.5% accuracy
```

### Recipe: A100 40GB

```yaml
batch_size: 256
learning_rate: 0.00032  # 16× base
warmup_steps: 500
# Expected: ~81.5% accuracy
```

### Recipe: H200 (Maximum Speed)

```yaml
batch_size: 1024
learning_rate: 0.001    # Capped
warmup_steps: 1000
# Expected: ~80.0-80.5% accuracy
```

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────┐
│          BATCH SIZE / LEARNING RATE SCALING             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Linear Scaling Rule:                                   │
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
└─────────────────────────────────────────────────────────┘
```
