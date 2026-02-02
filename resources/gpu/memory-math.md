# GPU Memory Math for Deep Learning Training

A comprehensive guide to understanding, calculating, and optimizing GPU memory usage during model training.

---

## Table of Contents

1. [Overview](#overview)
2. [Memory Components](#memory-components)
3. [Model Weights](#1-model-weights)
4. [Gradients](#2-gradients)
5. [Optimizer States](#3-optimizer-states)
6. [Activations](#4-activations)
7. [Mixed Precision Training](#mixed-precision-training)
8. [Memory Optimization Techniques](#memory-optimization-techniques)
9. [Practical Examples](#practical-examples)
10. [Quick Reference Formulas](#quick-reference-formulas)

---

## Overview

### Why Training Uses More Memory Than Inference

| Mode | What's Stored |
|------|---------------|
| **Inference** | Model weights only |
| **Training** | Weights + Gradients + Optimizer States + Activations |

**Rule of thumb:** Training requires **4-20x** more memory than inference.

---

## Memory Components

```
Total Training Memory =
    Model Parameters (weights)
  + Gradients
  + Optimizer States
  + Activations (forward pass cache)
  + Temporary Buffers
  + CUDA/Framework Overhead
```

---

## 1. Model Weights

### Bytes Per Parameter

| Precision | Bytes | Use Case |
|-----------|-------|----------|
| fp32 (float32) | 4 | Default, high precision |
| fp16 (float16) | 2 | Mixed precision training |
| bf16 (bfloat16) | 2 | Better range than fp16 |
| int8 | 1 | Quantized inference |
| int4 | 0.5 | Aggressive quantization |

### Common Model Sizes

| Model | Parameters | fp32 | fp16 |
|-------|------------|------|------|
| BERT-base | 110M | 440 MB | 220 MB |
| DistilBERT | 66M | 264 MB | 132 MB |
| Llama 7B | 7B | 28 GB | 14 GB |
| Llama 70B | 70B | 280 GB | 140 GB |

---

## 2. Gradients

### Memory Calculation

```
Gradient Memory = num_trainable_parameters × bytes_per_gradient
```

**Important:** Gradients are typically stored in **fp32** even during mixed precision training.

### Frozen Parameters

Frozen parameters (like teacher models in distillation) have **no gradients**:

```python
for param in teacher.parameters():
    param.requires_grad = False  # No gradient memory needed
```

---

## 3. Optimizer States

### Adam / AdamW

**States per parameter:** 2
- **m** (first moment): Running average of gradients
- **v** (second moment): Running average of squared gradients

```
Adam Memory = 2 × num_trainable_params × 4 bytes
```

### Comparison Table

| Optimizer | States/Param | Example (100M params) |
|-----------|--------------|----------------------|
| SGD | 0 | 0 MB |
| SGD + Momentum | 1 | 400 MB |
| Adam/AdamW | 2 | 800 MB |

---

## 4. Activations

During the forward pass, intermediate outputs must be saved for backpropagation.

### Activation Memory Scaling

```
Activations ∝ batch_size × seq_length × num_layers
```

| Batch Size | Relative Memory |
|------------|-----------------|
| 1 | 1× |
| 8 | 8× |
| 32 | 32× |

**This is why large batch sizes quickly exhaust GPU memory!**

---

## Mixed Precision Training

### Memory Savings

| Component | Full fp32 | Mixed Precision |
|-----------|-----------|-----------------|
| Master weights | 4 bytes | 4 bytes |
| fp16 weights copy | - | 2 bytes |
| Activations | 4 bytes | 2 bytes |
| Gradients | 4 bytes | 2-4 bytes |

**Net effect:** ~30-50% memory reduction, primarily from activations.

---

## Memory Optimization Techniques

### 1. Gradient Checkpointing
Trade compute for memory: recompute activations during backward pass.
**Savings:** Up to 5-10× reduction in activation memory
**Cost:** ~30% slower training

### 2. Gradient Accumulation
Simulate large batches with small memory:
```python
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 3. LoRA (Low-Rank Adaptation)
Only train small adapter matrices.
**Savings:** ~99%+ reduction in gradient/optimizer memory

---

## Practical Examples

### BERT→DistilBERT Distillation

| Component | Memory |
|-----------|--------|
| Teacher weights (fp16) | 220 MB |
| Student weights (fp16) | 132 MB |
| Student master weights (fp32) | 264 MB |
| Gradients (fp32) | 264 MB |
| AdamW states | 528 MB |
| Activations | ~675 MB |
| **Total** | **~2.3 GB** |

---

## Quick Reference Formulas

### Total Memory Estimate (Full Fine-tuning, AdamW, fp16 mixed)

```
Memory (GB) ≈ params_billions × 18 + activation_memory
```

### Activation Memory Estimate

```
Activation (GB) ≈ 12 × batch × seq_len × hidden × layers × 2 / 1e9
```

---

## GPU Memory Reference

| GPU | VRAM | Good For |
|-----|------|----------|
| RTX 3090/4090 | 24 GB | 7B models with LoRA |
| V100 | 16-32 GB | Mid-size model training |
| A100 | 40-80 GB | Large model training |
| H100/H200 | 80-141 GB | Very large models |

---

## Summary: Memory Reduction Techniques

| Technique | Effort | Memory Savings | Speed Impact |
|-----------|--------|----------------|--------------|
| Mixed precision (fp16) | Low | 30-50% | +10-50% faster |
| Gradient accumulation | Low | Proportional to steps | Neutral |
| LoRA/PEFT | Medium | 90-99%+ | Similar |
| Gradient checkpointing | Medium | 50-90% | -20-30% |
| DeepSpeed ZeRO | High | 4-N× | Some overhead |
