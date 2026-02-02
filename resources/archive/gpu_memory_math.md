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

**Rule of thumb:** Training requires **4-20x** more memory than inference, depending on:
- Optimizer choice (SGD vs Adam)
- Precision (fp32 vs fp16)
- Batch size
- Sequence length
- Whether using gradient checkpointing

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

### Calculation

```
Weight Memory = num_parameters × bytes_per_parameter
```

### Common Model Sizes

| Model | Parameters | fp32 | fp16 |
|-------|------------|------|------|
| BERT-base | 110M | 440 MB | 220 MB |
| BERT-large | 340M | 1.36 GB | 680 MB |
| DistilBERT | 66M | 264 MB | 132 MB |
| GPT-2 Small | 124M | 496 MB | 248 MB |
| GPT-2 XL | 1.5B | 6 GB | 3 GB |
| Llama 7B | 7B | 28 GB | 14 GB |
| Llama 13B | 13B | 52 GB | 26 GB |
| Llama 70B | 70B | 280 GB | 140 GB |

### Multi-Model Scenarios (e.g., Knowledge Distillation)

When loading multiple models:
```
Total Weight Memory = Teacher Weights + Student Weights
```

For BERT→DistilBERT distillation:
```
= 110M × 2 bytes + 66M × 2 bytes  (fp16)
= 220 MB + 132 MB
= 352 MB
```

---

## 2. Gradients

### What Are Gradients?

During backpropagation, we compute the partial derivative of the loss with respect to each trainable parameter:

```
∂Loss/∂weight for every trainable weight
```

### Memory Calculation

```
Gradient Memory = num_trainable_parameters × bytes_per_gradient
```

**Important:** Gradients are typically stored in **fp32** even during mixed precision training for numerical stability.

### Examples

| Model | Trainable Params | Gradient Memory (fp32) |
|-------|------------------|------------------------|
| DistilBERT (full) | 66M | 264 MB |
| BERT-base (full) | 110M | 440 MB |
| Llama 7B (full) | 7B | 28 GB |
| Llama 7B (LoRA r=16) | ~4M | 16 MB |

### Frozen Parameters

Frozen parameters (like teacher models in distillation) have **no gradients**:

```python
for param in teacher.parameters():
    param.requires_grad = False  # No gradient memory needed
```

---

## 3. Optimizer States

Different optimizers maintain different amounts of state per parameter.

### SGD (Stochastic Gradient Descent)

**States per parameter:** 0 (or 1 with momentum)

```
SGD Memory = 0  (vanilla)
SGD + Momentum Memory = num_params × 4 bytes  (momentum buffer)
```

### Adam / AdamW

**States per parameter:** 2
- **m** (first moment / momentum): Running average of gradients
- **v** (second moment / velocity): Running average of squared gradients

```
Adam Memory = 2 × num_trainable_params × 4 bytes
```

**Why fp32?** Optimizer states must be high precision for stable training. Even with fp16 training, optimizer states are fp32.

### Adafactor

Memory-efficient alternative to Adam:
- Factorizes second moment into row and column vectors
- Reduces memory from O(n×m) to O(n+m)

```
Adafactor Memory ≈ 0.5 × Adam Memory
```

### Comparison Table

| Optimizer | States/Param | Memory Multiplier | Example (100M params) |
|-----------|--------------|-------------------|----------------------|
| SGD | 0 | 0× | 0 MB |
| SGD + Momentum | 1 | 1× | 400 MB |
| Adam/AdamW | 2 | 2× | 800 MB |
| Adafactor | ~1 | ~1× | ~400 MB |
| LAMB | 2 | 2× | 800 MB |

### Complete Example: AdamW with DistilBERT

```
Optimizer Memory = 2 × 66M params × 4 bytes = 528 MB
```

---

## 4. Activations

### What Are Activations?

During the forward pass, intermediate outputs must be **saved** for backpropagation:

```python
# Conceptual view
x = input_embeddings        # saved for backward
h1 = attention(x)           # saved for backward
h2 = feedforward(h1)        # saved for backward
...
output = classifier(hN)     # saved for backward
```

### Why Save Activations?

The chain rule requires intermediate values:
```
∂Loss/∂W1 = ∂Loss/∂h2 × ∂h2/∂h1 × ∂h1/∂W1
                              ↑
                    Needs saved h1 value
```

### Activation Memory Formula

```
Activation Memory ≈ batch_size × seq_length × hidden_dim × num_layers × bytes × constant
```

The constant varies by architecture (attention matrices, layer norms, etc.)

### Transformer Activation Breakdown

For each transformer layer, we save:

| Component | Shape | Memory |
|-----------|-------|--------|
| Input to attention | (batch, seq, hidden) | B × S × H |
| Q, K, V projections | 3 × (batch, seq, hidden) | 3 × B × S × H |
| Attention weights | (batch, heads, seq, seq) | B × A × S × S |
| Attention output | (batch, seq, hidden) | B × S × H |
| FFN intermediate | (batch, seq, 4×hidden) | 4 × B × S × H |
| Layer norm inputs | 2 × (batch, seq, hidden) | 2 × B × S × H |

**Total per layer:** ~12 × B × S × H (approximate)

### Example: BERT-base

```
batch_size = 16
seq_length = 128
hidden_dim = 768
num_layers = 12
bytes = 2 (fp16)

Per layer ≈ 12 × 16 × 128 × 768 × 2 ≈ 37 MB
Total ≈ 37 MB × 12 layers ≈ 450 MB
```

### Activation Memory Scaling

```
Activations ∝ batch_size × seq_length × num_layers
```

| Batch Size | Relative Memory |
|------------|-----------------|
| 1 | 1× |
| 8 | 8× |
| 32 | 32× |

| Seq Length | Relative Memory |
|------------|-----------------|
| 128 | 1× |
| 512 | 4× |
| 2048 | 16× |

**This is why large batch sizes and long sequences quickly exhaust GPU memory!**

---

## Mixed Precision Training

### How It Works

1. **Master weights:** Keep in fp32 (for precision)
2. **Forward/backward:** Compute in fp16 (for speed + memory)
3. **Gradients:** Computed in fp16, accumulated in fp32
4. **Optimizer:** Always fp32

### Memory Savings

| Component | Full fp32 | Mixed Precision |
|-----------|-----------|-----------------|
| Master weights | 4 bytes | 4 bytes |
| fp16 weights copy | - | 2 bytes |
| Activations | 4 bytes | 2 bytes |
| Gradients | 4 bytes | 2-4 bytes |

**Net effect:** ~30-50% memory reduction, primarily from activations.

### Loss Scaling

fp16 has limited range (max ~65,504). Small gradients can underflow to zero.

**Solution:** Scale loss up, compute gradients, scale back down:
```python
scaled_loss = loss * scale_factor  # e.g., 1024
scaled_loss.backward()
gradients = gradients / scale_factor
```

Modern frameworks (PyTorch AMP) handle this automatically.

---

## Memory Optimization Techniques

### 1. Gradient Checkpointing (Activation Recomputation)

**Trade compute for memory:** Don't save all activations; recompute during backward pass.

```python
# PyTorch
from torch.utils.checkpoint import checkpoint
output = checkpoint(layer, input)
```

**Memory savings:** Up to 5-10× reduction in activation memory
**Cost:** ~30% slower training (recomputation overhead)

### 2. Gradient Accumulation

**Simulate large batches with small memory:**

```python
# Instead of batch_size=64
# Use batch_size=8 with accumulation_steps=8

for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Memory:** Only need memory for small batch
**Effective batch size:** batch_size × accumulation_steps

### 3. LoRA (Low-Rank Adaptation)

**Only train small adapter matrices:**

```
Original: W (d × d matrix, d² parameters)
LoRA: W + BA where B (d × r), A (r × d), r << d

Trainable parameters: 2 × d × r instead of d²
```

**Example:** Llama 7B
- Full fine-tuning: 7B trainable params → 28 GB gradients + 56 GB optimizer
- LoRA (r=16): ~4M trainable params → 16 MB gradients + 32 MB optimizer

**Memory savings:** ~99%+ reduction in gradient/optimizer memory

### 4. Quantization (QLoRA)

Combine quantization with LoRA:
- Base model in 4-bit (1.75 GB for 7B model)
- LoRA adapters in fp16
- Optimizer states in fp32

### 5. CPU Offloading

Move optimizer states or model layers to CPU RAM:
```python
# DeepSpeed ZeRO-Offload
# Keeps optimizer states in CPU, moves to GPU for updates
```

### 6. Model Parallelism

Split model across multiple GPUs:
- **Tensor Parallelism:** Split layers across GPUs
- **Pipeline Parallelism:** Split model vertically (layers on different GPUs)

### 7. DeepSpeed ZeRO

Partition optimizer states, gradients, and parameters across GPUs:

| Stage | What's Partitioned | Memory Reduction |
|-------|-------------------|------------------|
| ZeRO-1 | Optimizer states | ~4× |
| ZeRO-2 | + Gradients | ~8× |
| ZeRO-3 | + Parameters | ~N× (N = num GPUs) |

---

## Practical Examples

### Example 1: BERT→DistilBERT Distillation

**Setup:**
- Teacher: BERT-base (110M params, frozen)
- Student: DistilBERT (66M params, trainable)
- Batch size: 16, Seq length: 128
- Optimizer: AdamW
- Precision: fp16 mixed

**Memory Breakdown:**

| Component | Calculation | Memory |
|-----------|-------------|--------|
| Teacher weights (fp16) | 110M × 2 | 220 MB |
| Student weights (fp16) | 66M × 2 | 132 MB |
| Student master weights (fp32) | 66M × 4 | 264 MB |
| Gradients (fp32) | 66M × 4 | 264 MB |
| AdamW states | 66M × 2 × 4 | 528 MB |
| Teacher activations | ~450 MB | 450 MB |
| Student activations | ~225 MB | 225 MB |
| Buffers/overhead | - | ~200 MB |
| **Total** | | **~2.3 GB** |

**Fits easily on:** T4 (16 GB), even CPU with patience

---

### Example 2: Llama 3B→1B with LoRA

**Setup:**
- Teacher: Llama 3B (frozen)
- Student: Llama 1B (LoRA adapters only)
- LoRA: r=16, targeting MLP layers
- Batch size: 4, Seq length: 256
- Optimizer: AdamW
- Precision: fp16

**LoRA Trainable Parameters:**
```
Target modules: down_proj, up_proj, gate_proj
Per module: 2 × hidden × r = 2 × 2048 × 16 = 65,536
Total modules: 3 × 16 layers = 48
Total LoRA params: 48 × 65,536 ≈ 3.1M
```

**Memory Breakdown:**

| Component | Calculation | Memory |
|-----------|-------------|--------|
| Teacher (3B, fp16) | 3B × 2 | 6 GB |
| Student (1B, fp16) | 1B × 2 | 2 GB |
| LoRA weights (fp16) | 3.1M × 2 | 6 MB |
| LoRA master weights | 3.1M × 4 | 12 MB |
| Gradients | 3.1M × 4 | 12 MB |
| AdamW states | 3.1M × 2 × 4 | 25 MB |
| Teacher activations | ~8 GB | 8 GB |
| Student activations | ~4 GB | 4 GB |
| **Total** | | **~20 GB** |

**Requires:** A100 (40 GB) or V100 (32 GB)

---

### Example 3: Full Fine-tuning Llama 7B (What NOT to do on consumer GPUs)

**Setup:**
- Model: Llama 7B (all trainable)
- Batch size: 1, Seq length: 512
- Optimizer: AdamW

**Memory Breakdown:**

| Component | Calculation | Memory |
|-----------|-------------|--------|
| Model weights (fp16) | 7B × 2 | 14 GB |
| Master weights (fp32) | 7B × 4 | 28 GB |
| Gradients | 7B × 4 | 28 GB |
| AdamW states | 7B × 8 | 56 GB |
| Activations | ~20 GB | 20 GB |
| **Total** | | **~146 GB** |

**Requires:** 4× A100 80GB with ZeRO-3, or 8× A100 40GB

---

## Quick Reference Formulas

### Total Memory Estimate (Full Fine-tuning, AdamW, fp16 mixed)

```
Memory (GB) ≈ params_billions × 18 + activation_memory
```

Where:
- 18 = 2 (fp16 weights) + 4 (master) + 4 (grads) + 8 (Adam)
- Activation memory depends heavily on batch size and sequence length

### Activation Memory Estimate

```
Activation (GB) ≈ 12 × batch × seq_len × hidden × layers × 2 / 1e9
```

### LoRA Memory Savings

```
LoRA params ≈ 2 × hidden × rank × num_modules
Savings ≈ (full_params - lora_params) × 16 bytes  (grads + optimizer)
```

### Batch Size Calculator

Given available GPU memory and model requirements:
```
max_batch ≈ (GPU_memory - model_memory - optimizer_memory) / activation_per_sample
```

---

## GPU Memory Reference

| GPU | VRAM | Good For |
|-----|------|----------|
| RTX 3060 | 12 GB | BERT fine-tuning, small models |
| RTX 3090/4090 | 24 GB | 7B models with LoRA |
| A10 | 24 GB | 7B models with LoRA |
| V100 | 16-32 GB | Mid-size model training |
| A100 | 40-80 GB | Large model training |
| H100 | 80 GB | Very large models |
| 8× A100 | 320-640 GB | 70B+ models |

---

## Tools for Memory Profiling

### PyTorch

```python
# Check current memory usage
print(torch.cuda.memory_allocated() / 1e9, "GB allocated")
print(torch.cuda.memory_reserved() / 1e9, "GB reserved")

# Memory snapshot
torch.cuda.memory._record_memory_history()
# ... run code ...
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
```

### nvidia-smi

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Detailed memory breakdown
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv
```

### HuggingFace Accelerate

```python
from accelerate import Accelerator
accelerator = Accelerator()
print(accelerator.state)  # Shows memory usage
```

---

## Summary: Memory Reduction Techniques

| Technique | Effort | Memory Savings | Speed Impact |
|-----------|--------|----------------|--------------|
| Mixed precision (fp16) | Low | 30-50% | +10-50% faster |
| Gradient accumulation | Low | Proportional to steps | Neutral |
| Smaller batch size | Low | Linear | May hurt convergence |
| LoRA/PEFT | Medium | 90-99%+ | Similar |
| Gradient checkpointing | Medium | 50-90% | -20-30% |
| DeepSpeed ZeRO | High | 4-N× | Some overhead |
| CPU offloading | High | Large | -50%+ |
| Model parallelism | High | N× (N = GPUs) | Communication overhead |

---

## Further Reading

- [Efficient Training on a Single GPU (HuggingFace)](https://huggingface.co/docs/transformers/perf_train_gpu_one)
- [DeepSpeed ZeRO Paper](https://arxiv.org/abs/1910.02054)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Mixed Precision Training (NVIDIA)](https://developer.nvidia.com/automatic-mixed-precision)
- [Gradient Checkpointing](https://github.com/cybertronai/gradient-checkpointing)
