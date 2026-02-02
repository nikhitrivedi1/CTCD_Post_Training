# GPU Utilization & Training Dynamics

Balancing optimal training dynamics with maximum GPU utilization for efficient deep learning.

> Merged from: `gpu_utilization_guide.md` + `balancing_training_dynamics_gpu_utilization.md`

---

## Table of Contents

1. [The Core Trade-Off](#the-core-trade-off)
2. [GPU Specifications](#gpu-specifications)
3. [Training Dynamics: Why Batch Size Matters](#training-dynamics-why-batch-size-matters)
4. [Understanding GPU Bottlenecks](#understanding-gpu-bottlenecks)
5. [Memory Budget Analysis](#memory-budget-analysis)
6. [GPU-Specific Configurations](#gpu-specific-configurations)
7. [Mixed Precision Training](#mixed-precision-training)
8. [Tensor Core Utilization](#tensor-core-utilization)
9. [DataLoader Optimization](#dataloader-optimization)
10. [Monitoring and Profiling](#monitoring-and-profiling)
11. [Troubleshooting](#troubleshooting)
12. [Quick Reference](#quick-reference)

---

## The Core Trade-Off

### Two Competing Goals

```
┌─────────────────────────────────────────────────────────────────┐
│                    THE FUNDAMENTAL TENSION                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  TRAINING DYNAMICS                    GPU UTILIZATION            │
│  (Model Quality)                      (Hardware Efficiency)      │
│  ─────────────────                    ─────────────────────      │
│                                                                  │
│  Small batch (16-64):                 Large batch (512-2048):    │
│  ✓ High gradient noise               ✓ High memory utilization  │
│  ✓ Better generalization             ✓ High bandwidth usage     │
│  ✓ Escapes sharp minima              ✓ Tensor cores saturated   │
│  ✗ Low GPU utilization               ✗ Less gradient noise      │
│  ✗ Slower wall-clock time            ✗ May hurt generalization  │
│                                                                  │
│                         BALANCE POINT                            │
│                              ↓                                   │
│                    batch_size = 256-512                          │
│                    (for BERT-scale models)                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### The Decision Framework

```
Question 1: What's your priority?

┌─────────────────────┬──────────────────────┬────────────────────┐
│ Priority            │ Batch Size           │ Trade-off          │
├─────────────────────┼──────────────────────┼────────────────────┤
│ Paper reproduction  │ Use paper settings   │ Accept slower time │
│ Maximum accuracy    │ Small (16-64)        │ Accept lower util  │
│ Fastest iteration   │ Large (512-2048)     │ Accept ±1% acc     │
│ Balanced default    │ Medium (256-512)     │ Best of both       │
└─────────────────────┴──────────────────────┴────────────────────┘
```

### When to Favor Each Side

```
FAVOR TRAINING DYNAMICS (smaller batch):
├── Final production model
├── Paper reproduction
├── Maximum accuracy critical
├── Plenty of time available
└── Overfitting concerns

FAVOR GPU UTILIZATION (larger batch):
├── Hyperparameter search
├── Quick prototyping
├── Comparing architectures
├── Time-constrained experiments
└── Development/debugging
```

---

## GPU Specifications

### NVIDIA A100

| Variant | VRAM | Memory Bandwidth | FP16 TFLOPS |
|---------|------|------------------|-------------|
| A100 40GB | 40 GB HBM2e | 1,555 GB/s | 312 |
| A100 80GB | 80 GB HBM2e | 2,039 GB/s | 312 |

**Available on:** Google Colab Pro/Pro+ (40GB), Northeastern Discovery (40GB/80GB), Cloud providers (AWS p4d, GCP a2)

### NVIDIA H200

| Spec | Value |
|------|-------|
| VRAM | 141 GB HBM3 |
| Memory Bandwidth | 4,800 GB/s |
| FP16 TFLOPS | 989 |
| FP8 TFLOPS | 1,979 |

**Available on:** Northeastern Discovery (gpu partition)

### Comparison

```
Performance (relative to V100):

           Memory    Bandwidth    Compute
V100:      32 GB     900 GB/s     125 TFLOPS    (1.0×)
A100:      80 GB     2,039 GB/s   312 TFLOPS    (2.5×)
H200:      141 GB    4,800 GB/s   989 TFLOPS    (8.0×)
```

---

## Training Dynamics: Why Batch Size Matters

### Gradient Noise

**Gradient Noise** is the variance in your gradient estimate because you're only looking at a subset of data.

```
True Gradient (if you used ALL training samples):
  → Exactly points toward the loss minimum
  → Deterministic, same every time

Mini-Batch Gradient (using 16 samples):
  → Estimate of the true gradient
  → Different 16 samples = different direction
  → This randomness IS the "noise"
```

**The Math:**
```
Gradient variance:  Var(ĝ) = σ² / B

Where:
  σ² = per-sample gradient variance (fixed by your data)
  B  = batch size

Batch 16:   Var = σ²/16   = 0.0625σ²  (high noise)
Batch 512:  Var = σ²/512  = 0.00195σ² (32× less noise)
```

### Why Gradient Noise Helps (Implicit Regularization)

```
Loss Landscape with Two Minima:

        Sharp (overfits)          Flat (generalizes)
              ╱╲                    ____
             ╱  ╲                  ╱    ╲
            ╱    ╲                ╱      ╲
        ___╱      ╲______________╱        ╲___


Small Batch (noisy):
  - Gets "bounced around" by the noise
  - Can't settle in narrow sharp minimum (bounces out!)
  - Eventually lands in wide flat minimum (stable)

Large Batch (clean):
  - Goes straight to nearest minimum
  - Gets stuck in first minimum found (often sharp)
  - Never explores alternatives
```

### Sharp vs Flat Minima

```
Sharp Minimum (large batch finds):
├── Narrow valley
├── High curvature
├── Sensitive to perturbation
└── Overfits: training ✓, test ✗

Flat Minimum (small batch finds):
├── Wide basin
├── Low curvature
├── Robust to perturbation
└── Generalizes: training ✓, test ✓
```

### The Generalization Gap

```
Batch Size:     16      64     256    512    1024   2048
─────────────────────────────────────────────────────────
Train Acc:     82.5%   82.8%  83.0%  83.5%  84.0%  85.0%
Test Acc:      81.7%   81.8%  81.5%  81.0%  80.0%  78.0%
─────────────────────────────────────────────────────────
Gap:            0.8%    1.0%   1.5%   2.5%   4.0%   7.0%
                ↑                               ↑
             Optimal                    Overfitting!
```

---

## Understanding GPU Bottlenecks

### The Three GPU Resources

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU RESOURCE TRIANGLE                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│                        COMPUTE                               │
│                       (TFLOPS)                               │
│                          △                                   │
│                         ╱ ╲                                  │
│                        ╱   ╲                                 │
│                       ╱     ╲                                │
│                      ╱       ╲                               │
│           MEMORY ◇───────────◇ BANDWIDTH                    │
│         (Capacity)           (GB/s)                          │
│                                                              │
│  Optimal performance = Balance all three                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

| Resource | What It Is | Bottleneck When... |
|----------|------------|-------------------|
| **Compute** | TFLOPS (operations/sec) | Math operations dominate |
| **Memory** | GB capacity | Can't fit model/activations |
| **Bandwidth** | GB/s data transfer | Moving data dominates |

### The Roofline Model

```
Performance
(TFLOPS)
    │
    │                    ┌─────────── Peak Compute (312 TFLOPS)
    │                   ╱
    │                  ╱
    │                 ╱    ← Compute-bound region
    │                ╱
    │               ╱
    │              ╱
    │             ╱
    │            ╱  ← Memory-bound region
    │           ╱
    │          ╱
    └───────┴──────────────────────────────────
            ↑                              Arithmetic
         Ridge point                       Intensity
       (ops per byte)                    (FLOPS/byte)
```

### Common Operations by Category

```
Memory-Bound (optimize bandwidth):
├── Activation functions (ReLU, GELU, softmax)
├── Layer normalization
├── Dropout
├── Embedding lookups
└── Attention score computation

Compute-Bound (optimize compute):
├── Matrix multiplications (Linear layers)
├── Convolutions
└── Large batch GEMM operations
```

---

## Memory Budget Analysis

### BERT→DistilBERT Memory Breakdown

```
Teacher (BERT-base):
  - Parameters: 110M
  - FP16: 110M × 2 bytes = 220 MB
  - Frozen (no gradients/optimizer states)

Student (DistilBERT):
  - Parameters: 66M
  - FP16 weights: 66M × 2 bytes = 132 MB
  - FP32 gradients: 66M × 4 bytes = 264 MB
  - Adam states (m, v): 66M × 2 × 4 bytes = 528 MB

Total model memory: ~1.15 GB
```

### Memory Formula

```
Total Memory ≈ Models + Activations × Batch + Overhead

For batch B and seq_len 128:
Memory (GB) ≈ 1.15 + 0.04 × B + 2.0  (overhead buffer)
            ≈ 3.15 + 0.04 × B
```

### Maximum Batch Size by GPU

```
A100 40GB:  Safe maximum: 768
A100 80GB:  Safe maximum: 1536
H200 141GB: Safe maximum: 2560
```

---

## GPU-Specific Configurations

### A100 40GB - Recommended

```yaml
training:
  batch_size: 256
  learning_rate: 0.00032  # 16× scaled from 2e-5
  warmup_steps: 500
  fp16: true

# Memory usage: ~13 GB (leaves 27 GB headroom)
# Time: ~20 minutes for MNLI
# Expected accuracy: ~81.5%
```

### A100 80GB - Recommended

```yaml
training:
  batch_size: 512
  learning_rate: 0.00064  # 32× scaled
  warmup_steps: 500
  fp16: true

# Memory usage: ~24 GB
# Time: ~10 minutes
# Expected accuracy: ~81.0%
```

### H200 - Balanced

```yaml
training:
  batch_size: 512
  learning_rate: 0.00064
  warmup_steps: 500
  bf16: true
  dataloader_num_workers: 8

# Memory usage: ~24 GB
# Time: ~8 minutes
# Expected accuracy: ~81.0%
```

### H200 - Maximum Speed

```yaml
training:
  batch_size: 1024
  learning_rate: 0.001    # Capped at 1e-3
  warmup_steps: 1000
  bf16: true
  dataloader_num_workers: 16

# Memory usage: ~45 GB
# Time: ~4-5 minutes
# Expected accuracy: ~80.0%
```

---

## Mixed Precision Training

### Benefits

```
┌──────────────────────────────────────────────────────────┐
│              MIXED PRECISION BENEFITS                     │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  Memory Savings:                                          │
│  • Activations: 2× smaller (FP16 vs FP32)                │
│  • Can fit 2× larger batch                               │
│                                                           │
│  Bandwidth Savings:                                       │
│  • 2× more values per memory transfer                    │
│  • Effectively doubles memory bandwidth                   │
│                                                           │
│  Compute Boost:                                           │
│  • Tensor Cores: 8× faster than FP32 CUDA cores          │
│  • Only available for FP16/BF16/FP8                      │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

### FP16 vs BF16 Selection

```
FP16 (float16):
├── Range: ±65,504
├── Risk: Overflow/underflow (needs loss scaling)
└── Best for: V100, A100, T4

BF16 (bfloat16):
├── Range: ±3.4×10³⁸ (same as FP32!)
├── Risk: Less precision, but no overflow
└── Best for: A100, H100, H200 (native support)
```

---

## Tensor Core Utilization

### Requirements

```
┌─────────────────────────────────────────────────────────────┐
│              TENSOR CORE REQUIREMENTS                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Data Type: FP16, BF16, TF32, or FP8                     │
│     └── FP32 operations use CUDA cores, not Tensor Cores    │
│                                                              │
│  2. Dimensions divisible by 8 (or 16 for best performance)  │
│     └── batch_size % 8 == 0                                 │
│     └── hidden_dim % 8 == 0                                 │
│     └── sequence_length % 8 == 0                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Optimal Dimensions

```
Batch sizes:    8, 16, 32, 64, 128, 256, 512, 1024, 2048
Hidden dims:    256, 512, 768, 1024, 2048, 4096
Sequence lens:  64, 128, 256, 512
```

---

## DataLoader Optimization

### The Key Insight: Batch Size ≠ Data Loading Parallelism

```
Batch Size (training dynamics):
  - Controls gradient quality (variance)
  - Affects generalization (sharp vs flat minima)
  - Requires learning rate scaling

Num Workers (data loading):
  - Controls CPU parallelism for data preparation
  - Affects GPU utilization (keeps GPU fed)
  - No impact on training dynamics
```

### Optimal num_workers Settings

```
┌─────────────┬──────────────┬─────────────────────┐
│ Batch Size  │ num_workers  │ Reasoning           │
├─────────────┼──────────────┼─────────────────────┤
│     16      │      4       │ Small batch = quick │
│     64      │      8       │ Moderate workload   │
│    256      │      8       │ Standard setting    │
│    512+     │    8-16      │ Large batch benefit │
└─────────────┴──────────────┴─────────────────────┘
```

### The Best of Both Worlds

You CAN have small batch (for quality) + many workers (for throughput):

```yaml
training:
  batch_size: 16              # Paper setting (training dynamics)
  learning_rate: 0.00002      # Paper setting
  dataloader_num_workers: 8   # Keep GPU fed (throughput)
  dataloader_pin_memory: true # Faster CPU→GPU transfer
```

---

## Monitoring and Profiling

### nvidia-smi Metrics

```bash
# Watch GPU utilization in real-time
watch -n 1 nvidia-smi

# Key metrics to monitor:
# - GPU-Util: Should be 90%+ (compute utilization)
# - Memory-Usage: Should be 70-90% of total
# - Power: Near TDP indicates high utilization
```

### Memory Profiling

```python
import torch

def print_memory_stats():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9

        print(f"Allocated: {allocated:.2f} GB")
        print(f"Reserved:  {reserved:.2f} GB")
        print(f"Peak:      {max_allocated:.2f} GB")
```

---

## Troubleshooting

### Out of Memory (OOM)

**Solutions:**
1. Reduce batch size (scale LR proportionally)
2. Enable gradient checkpointing (`model.gradient_checkpointing_enable()`)
3. Use gradient accumulation
4. Reduce sequence length

### Training Too Slow

**Solutions:**
1. Increase batch size (scale LR)
2. Increase dataloader workers
3. Enable mixed precision (`fp16: true`)
4. Disable deterministic mode

### Accuracy Lower Than Expected

**Solutions:**
1. Increase warmup steps
2. Cap learning rate at ~1e-3
3. Use smaller batch size
4. Add label smoothing

---

## Quick Reference

```
┌────────────────────────────────────────────────────────────┐
│                GPU CONFIGURATION CHEAT SHEET               │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  A100 40GB (Colab Pro, Discovery)                         │
│  ─────────────────────────────────                        │
│  Reproduction:  batch=256, lr=3.2e-4, warmup=500          │
│  Fast:          batch=512, lr=6.4e-4, warmup=500          │
│                                                            │
│  A100 80GB (Discovery)                                    │
│  ─────────────────────                                    │
│  Reproduction:  batch=512, lr=6.4e-4, warmup=500          │
│  Fast:          batch=1024, lr=1e-3, warmup=1000          │
│                                                            │
│  H200 141GB (Discovery)                                   │
│  ──────────────────────                                   │
│  Reproduction:  batch=512, lr=6.4e-4, warmup=500          │
│  Fast:          batch=1024, lr=1e-3, warmup=1000          │
│  Ultra:         batch=2048, lr=1e-3, warmup=1500          │
│                                                            │
│  Time Estimates (MNLI, 3 epochs):                         │
│  ────────────────────────────────                         │
│  A100 40GB: batch=256 → 20 min | batch=512 → 12 min      │
│  A100 80GB: batch=512 → 10 min | batch=1024 → 6 min      │
│  H200:      batch=512 → 8 min  | batch=1024 → 5 min      │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## References

- [NVIDIA Mixed Precision Training](https://developer.nvidia.com/automatic-mixed-precision)
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Roofline Model Paper](https://dl.acm.org/doi/10.1145/1498765.1498785)

---

*See also: [memory-math.md](memory-math.md), [comparison.md](comparison.md)*
