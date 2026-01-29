# Balancing Training Dynamics and GPU Utilization

This guide explains how to balance the trade-off between **optimal training dynamics** (gradient quality, generalization) and **maximum GPU utilization** (memory, bandwidth, compute) to get the best results efficiently.

---

## Table of Contents

1. [The Core Trade-Off](#the-core-trade-off)
2. [Training Dynamics: Why Batch Size Matters](#training-dynamics-why-batch-size-matters)
   - [What is Gradient Noise?](#what-is-gradient-noise)
   - [Why Gradient Noise Helps](#why-gradient-noise-helps-implicit-regularization)
   - [Why Flat Minima Generalize Better](#why-flat-minima-generalize-better)
3. [Understanding GPU Bottlenecks](#understanding-the-bottlenecks)
4. [The Roofline Model](#the-roofline-model)
5. [Memory Bandwidth Optimization](#memory-bandwidth-optimization)
6. [Memory Capacity Optimization](#memory-capacity-optimization)
7. [Mixed Precision Training](#mixed-precision-training)
8. [Tensor Core Utilization](#tensor-core-utilization)
9. [Practical Optimization Strategies](#practical-optimization-strategies)
10. [GPU-Specific Configurations](#gpu-specific-configurations)
11. [Monitoring and Profiling](#monitoring-and-profiling)

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

Question 2: How do you compensate for batch size choice?

Small Batch + High Throughput:
├── Use more DataLoader workers (8-16)
├── Enable pin_memory
├── Increase prefetch_factor
└── GPU stays busy between small batches

Large Batch + Good Training:
├── Scale learning rate (linear rule)
├── Increase warmup steps
├── Cap LR at ~1e-3 for fine-tuning
└── Accept slight generalization gap
```

### Visual: The Sweet Spot

```
                 Accuracy
                    │
                    │    ●──────●
                    │   ╱        ╲
                    │  ●          ╲
                    │ ╱            ●
                    │╱              ╲
                    ●                ●
                    │
                    └───────────────────→ Batch Size
                    16   64   256  512  1024  2048

                 GPU Utilization
                    │
                    │                    ●────●
                    │              ●────╱
                    │        ●────╱
                    │   ●───╱
                    │  ╱
                    ●─╱
                    │
                    └───────────────────→ Batch Size
                    16   64   256  512  1024  2048

         ┌──────────────────────────────────────┐
         │  SWEET SPOT: Batch 256-512           │
         │  • GPU utilization: 60-80%           │
         │  • Accuracy: Within 0.5% of optimal  │
         │  • Training time: 4-8× faster        │
         └──────────────────────────────────────┘
```

### Summary: When to Favor Each Side

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

## Training Dynamics: Why Batch Size Matters

### The Fundamental Difference

```
Small Batch (16 samples):
─────────────────────────
Each gradient update averages only 16 sample gradients.
→ High variance estimate
→ Direction changes step-to-step
→ "Noisy" but exploratory optimization

Large Batch (512 samples):
─────────────────────────
Each gradient update averages 512 sample gradients.
→ Low variance estimate
→ Direction is consistent
→ "Clean" but direct optimization
```

### Visualization: Training Trajectories

```
Small Batch (many noisy steps):

Start ●
       ↘
        ↗    ← Bounces around
         ↘
          ↗
           ↘
            →
             ↘
              ↗
               ↘
                ● End (flat minimum - generalizes well!)


Large Batch (few direct steps):

Start ●
       \
        \    ← Straight path
         \
          \
           \
            \
             \
              \
               ● End (sharp minimum - may overfit!)
```

### Sharp vs Flat Minima

```
Loss Landscape:

        ╱╲
       ╱  ╲    ╱╲
      ╱    ╲__╱  ╲
     ╱    ↑       ╲___
    ╱     │            ╲____
   ╱      │        ↑        ╲___
           │        │
      Sharp Min  Flat Min


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

### What is Gradient Noise?

**Gradient Noise** is the variance in your gradient estimate because you're only looking at a subset of data.

```
True Gradient (if you used ALL 390,000 training samples):
  → Exactly points toward the loss minimum
  → Deterministic, same every time

Mini-Batch Gradient (using 16 samples):
  → Estimate of the true gradient
  → Different 16 samples = different direction
  → This randomness IS the "noise"
```

**Numerical Example:**

```
Suppose the true gradient says: "Move parameters by [0.10, -0.08, 0.12]"

Batch A (samples 1-16):   gradient = [0.15, -0.05, 0.09]  ← noisy estimate
Batch B (samples 17-32):  gradient = [0.08, -0.12, 0.14]  ← different noise
Batch C (samples 33-48):  gradient = [0.11, -0.07, 0.10]  ← closer to true

Large batch (512 samples): gradient = [0.10, -0.08, 0.12] ← almost exact
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

The noise acts as a form of regularization that helps find better minima:

```
Loss Landscape with Two Minima:

        Sharp (overfits)          Flat (generalizes)
              ╱╲                    ____
             ╱  ╲                  ╱    ╲
            ╱    ╲                ╱      ╲
        ___╱      ╲______________╱        ╲___


Small Batch (noisy):
  - Gradient points slightly wrong direction each step
  - Gets "bounced around" by the noise
  - Can't settle in narrow sharp minimum (bounces out!)
  - Eventually lands in wide flat minimum (stable)

Large Batch (clean):
  - Gradient points precisely toward nearest minimum
  - Goes straight there, no bouncing
  - Gets stuck in first minimum found (often sharp)
  - Never explores alternatives
```

**The Bouncing Ball Analogy:**

```
Sharp minimum = narrow wine glass
Flat minimum = wide bowl

Small batch = ball with vibration
  → Bounces out of wine glass
  → Settles in bowl

Large batch = ball with no vibration
  → Falls into first container (wine glass)
  → Stays stuck there
```

### Why Flat Minima Generalize Better

```
Training Data:  Optimum at θ = 5.0
Test Data:      Optimum at θ = 5.1  (slightly different distribution)

Sharp minimum at θ = 5.0:
    Loss │    ╱╲
         │   ╱  ╲
         │  ╱ ●  ╲      Train: 0.01  ✓
         │ ╱  ↑   ╲     Test:  0.50  ✗ (big jump!)
         └────┴────→
              5.0

Flat minimum at θ = 5.0:
    Loss │  ______
         │ ╱  ●   ╲
         │╱   ↑    ╲    Train: 0.02  ✓
         └────┴─────→   Test:  0.03  ✓ (small change)
              5.0

Flat minima are robust to distribution shift between train and test!
```

### Gradient Noise Summary

| Aspect | Small Batch | Large Batch |
|--------|-------------|-------------|
| Gradient noise | High | Low |
| Finds | Flat minima | Sharp minima |
| Training loss | Good | Better |
| **Test loss** | **Best** | Worse |
| Regularization | Implicit (from noise) | Need explicit |

**Key insight:** Small batch isn't "better" at finding the minimum faster—it's better at finding a minimum that **generalizes to unseen data**.

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

Rule: Larger batch → better training, worse testing
```

### Convergence Behavior

```
Small Batch:
Loss │●●●●●
     │    ●●●●●●●
     │           ●●●●●●●●●●
     │                     ●●●●●●●●●●●●●●
     └─────────────────────────────────────→
      73,000 steps (slow but thorough)

Large Batch:
Loss │●
     │ \
     │  \_______
     │          \_________
     └─────────────────────────────────────→
      2,300 steps (fast but plateaus)
```

### Practical Implications

| Scenario | Recommended Batch | Reason |
|----------|-------------------|--------|
| Paper reproduction | 16 | Match exact dynamics |
| Final production model | 16-64 | Best generalization |
| Hyperparameter sweep | 256-512 | Fast iteration |
| Debugging/prototyping | 512-1024 | Quick feedback |
| Architecture comparison | 256 | Balanced speed/quality |

See [training_dynamics_batch_size.md](training_dynamics_batch_size.md) for comprehensive details.

---

## Understanding the Bottlenecks

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

### GPU Specifications Comparison

```
                    Memory      Bandwidth     Compute (FP16)
                    ────────    ──────────    ──────────────
V100 32GB:          32 GB       900 GB/s      125 TFLOPS
A100 40GB:          40 GB       1,555 GB/s    312 TFLOPS
A100 80GB:          80 GB       2,039 GB/s    312 TFLOPS
H100 80GB:          80 GB       3,350 GB/s    989 TFLOPS
H200 141GB:         141 GB      4,800 GB/s    989 TFLOPS
```

### Memory × Bandwidth Product

The "Memory × Bandwidth" product indicates total data throughput potential:

```
V100:   32 GB  ×  900 GB/s  = 28,800 GB²/s
A100:   80 GB  × 2,039 GB/s = 163,120 GB²/s  (5.7× V100)
H200:  141 GB  × 4,800 GB/s = 676,800 GB²/s  (23× V100)
```

**To maximize this product, you need to:**
1. Use as much memory as possible (large batches)
2. Keep bandwidth saturated (efficient data access patterns)

---

## The Roofline Model

### What Is It?

The roofline model helps identify whether your workload is:
- **Compute-bound**: Limited by FLOPS (good!)
- **Memory-bound**: Limited by bandwidth (room to optimize)

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
    │         ╱
    │        ╱
    └───────┴──────────────────────────────────
            ↑                              Arithmetic
         Ridge point                       Intensity
       (ops per byte)                    (FLOPS/byte)
```

### Arithmetic Intensity

```
Arithmetic Intensity = FLOPS performed / Bytes accessed

Low AI (<10):   Memory-bound (element-wise ops, activations)
Medium AI:      Balanced
High AI (>100): Compute-bound (matrix multiply, convolutions)
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

## Memory Bandwidth Optimization

### 1. Use Smaller Data Types

Smaller types = more data per memory transfer:

```
Data Type    Bytes    Relative Bandwidth
─────────    ─────    ──────────────────
FP32         4        1.0× (baseline)
FP16/BF16    2        2.0× effective bandwidth
FP8          1        4.0× effective bandwidth
INT8         1        4.0× effective bandwidth
```

**PyTorch Implementation:**
```python
# Automatic Mixed Precision (AMP)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    with autocast(dtype=torch.float16):  # or torch.bfloat16
        outputs = model(batch)
        loss = criterion(outputs, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 2. Optimize Memory Access Patterns

**Coalesced vs Non-Coalesced Access:**

```
Coalesced (GOOD):
Thread 0 reads memory[0]
Thread 1 reads memory[1]    → Single memory transaction
Thread 2 reads memory[2]
...

Non-Coalesced (BAD):
Thread 0 reads memory[0]
Thread 1 reads memory[128]  → Multiple memory transactions
Thread 2 reads memory[256]
...
```

**Practical tips:**
```python
# BAD: Strided access
for i in range(0, len(data), stride):
    process(data[i])

# GOOD: Contiguous access
data_contiguous = data[::stride].contiguous()
process(data_contiguous)

# Ensure tensors are contiguous
x = x.contiguous()  # Before operations that assume contiguous memory
```

### 3. Reduce Memory Transfers

```python
# BAD: Multiple kernel launches, multiple transfers
x = x + 1
x = x * 2
x = torch.relu(x)

# GOOD: Fused operations (single kernel, single transfer)
# Use torch.compile or custom CUDA kernels
@torch.compile
def fused_ops(x):
    return torch.relu((x + 1) * 2)
```

### 4. Pin Memory for Faster CPU→GPU Transfer

```python
# In DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=64,
    pin_memory=True,  # Faster CPU→GPU transfer
    num_workers=8,
)

# Or manually
tensor = tensor.pin_memory()
tensor_gpu = tensor.to('cuda', non_blocking=True)  # Async transfer
```

---

## Memory Capacity Optimization

### 1. Fill Available Memory

The goal: Use as much memory as possible without OOM.

```
Memory Budget:
┌────────────────────────────────────────┐
│  Model Weights (fixed)          ~1 GB  │
│  Optimizer States (fixed)       ~0.5 GB│
│  Gradients (fixed)              ~0.3 GB│
│  ─────────────────────────────────────│
│  Activations (scales with batch) ???   │
│  ─────────────────────────────────────│
│  Headroom (for safety)          ~2 GB  │
└────────────────────────────────────────┘

Available for activations = Total VRAM - Fixed - Headroom
```

### 2. Find Optimal Batch Size

```python
def find_max_batch_size(model, sample_input_fn, device='cuda'):
    """Binary search for maximum batch size."""
    low, high = 1, 4096
    max_batch = 1

    while low <= high:
        mid = (low + high) // 2
        torch.cuda.empty_cache()

        try:
            inputs = sample_input_fn(batch_size=mid)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                _ = model(**inputs)

            max_batch = mid
            low = mid + 1

        except RuntimeError as e:
            if "out of memory" in str(e):
                high = mid - 1
            else:
                raise

        torch.cuda.empty_cache()

    # Use 80% of max for safety
    return int(max_batch * 0.8)
```

### 3. Gradient Checkpointing (Trade Compute for Memory)

```python
# Without checkpointing: Store all activations
# Memory: O(layers × batch × hidden)

# With checkpointing: Recompute activations during backward
# Memory: O(sqrt(layers) × batch × hidden)
# Compute: ~1.3× more (recomputation cost)

model.gradient_checkpointing_enable()

# Or selectively:
from torch.utils.checkpoint import checkpoint

class CheckpointedBlock(nn.Module):
    def forward(self, x):
        return checkpoint(self._forward_impl, x, use_reentrant=False)

    def _forward_impl(self, x):
        # Your computation here
        return self.layers(x)
```

### 4. Gradient Accumulation (Simulate Larger Batches)

```python
# Physical batch: 64 (fits in memory)
# Effective batch: 256 (4 accumulation steps)

accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    outputs = model(batch)
    loss = criterion(outputs) / accumulation_steps  # Scale loss
    loss.backward()  # Accumulate gradients

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## Mixed Precision Training

### Why Mixed Precision Maximizes Memory × Bandwidth

```
FP32 Training:
├── Weights: 4 bytes × N params
├── Gradients: 4 bytes × N params
├── Optimizer: 8 bytes × N params (Adam m, v)
├── Activations: 4 bytes × batch × ...
└── Total: ~16 bytes per parameter + activations

FP16/BF16 Mixed Precision:
├── Weights (FP16): 2 bytes × N params
├── Master weights (FP32): 4 bytes × N params
├── Gradients (FP16): 2 bytes × N params
├── Optimizer (FP32): 8 bytes × N params
├── Activations (FP16): 2 bytes × batch × ...
└── Total: ~16 bytes per param, but 2× activation savings!
```

### Benefits Breakdown

```
┌──────────────────────────────────────────────────────────┐
│              MIXED PRECISION BENEFITS                     │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  Memory Savings:                                          │
│  ───────────────                                          │
│  • Activations: 2× smaller (FP16 vs FP32)                │
│  • Can fit 2× larger batch                               │
│  • Model weights: 2× smaller in forward pass             │
│                                                           │
│  Bandwidth Savings:                                       │
│  ─────────────────                                        │
│  • 2× more values per memory transfer                    │
│  • Effectively doubles memory bandwidth                   │
│                                                           │
│  Compute Boost:                                           │
│  ──────────────                                           │
│  • Tensor Cores: 8× faster than FP32 CUDA cores          │
│  • Only available for FP16/BF16/FP8                      │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

### HuggingFace Configuration

```python
# TrainingArguments
training_args = TrainingArguments(
    fp16=True,                    # Use FP16 mixed precision
    # OR
    bf16=True,                    # Use BF16 (better for H100/H200)

    fp16_full_eval=False,         # Keep eval in FP32 for accuracy

    # For maximum memory efficiency
    gradient_checkpointing=True,
)
```

### FP16 vs BF16 Selection

```
FP16 (float16):
├── Range: ±65,504
├── Precision: ~3-4 decimal digits
├── Risk: Overflow/underflow (needs loss scaling)
└── Best for: V100, A100, T4

BF16 (bfloat16):
├── Range: ±3.4×10³⁸ (same as FP32!)
├── Precision: ~2-3 decimal digits
├── Risk: Less precision, but no overflow
└── Best for: A100, H100, H200 (native support)

Recommendation:
├── A100: bf16=True (if available) or fp16=True
├── H100/H200: bf16=True (best native support)
└── V100/T4: fp16=True (no BF16 support)
```

---

## Tensor Core Utilization

### What Are Tensor Cores?

Tensor Cores are specialized units for matrix multiply-accumulate:

```
Regular CUDA Cores:
  One multiply-add per cycle per core

Tensor Cores:
  4×4×4 matrix multiply-accumulate per cycle
  = 64 multiply-adds per cycle per tensor core
```

### Requirements for Tensor Core Activation

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
│  3. Memory alignment (handled automatically by PyTorch)     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Practical Tensor Core Optimization

```python
# Ensure batch size is multiple of 8
batch_size = (desired_batch // 8) * 8

# Pad sequence length to multiple of 8
def pad_to_multiple(length, multiple=8):
    return ((length + multiple - 1) // multiple) * multiple

max_length = pad_to_multiple(128, 8)  # 128 is already good

# Check tensor dimensions
def check_tensor_core_compatible(tensor):
    for dim in tensor.shape:
        if dim % 8 != 0:
            print(f"Warning: dimension {dim} not divisible by 8")
```

### Dimension Guidelines

```
Optimal dimensions for Tensor Cores:
─────────────────────────────────────

Batch sizes:    8, 16, 32, 64, 128, 256, 512, 1024, 2048
Hidden dims:    256, 512, 768, 1024, 2048, 4096
Sequence lens:  64, 128, 256, 512

Non-optimal (slower):
Batch sizes:    17, 33, 100, 1000
Hidden dims:    300, 500, 700
Sequence lens:  100, 150, 200
```

---

## Practical Optimization Strategies

### Strategy 1: Maximum Throughput (Speed Priority)

```yaml
# Goal: Fastest training time, accuracy within ±1%

training:
  # Large batch to saturate compute
  batch_size: 512           # Or maximum that fits
  learning_rate: 0.00064    # Scaled appropriately

  # Mixed precision for memory and bandwidth
  fp16: true                # Or bf16 on H100/H200

  # Keep GPU fed
  dataloader_num_workers: 8
  dataloader_pin_memory: true

  # Tensor core friendly
  # (batch 512 is divisible by 8 ✓)
```

### Strategy 2: Maximum Memory Efficiency (Large Model Priority)

```yaml
# Goal: Fit largest possible model/batch

training:
  batch_size: 64
  gradient_accumulation_steps: 8  # Effective batch 512

  # Memory savers
  fp16: true
  gradient_checkpointing: true    # Trade 20% speed for 40% memory

  # Aggressive memory management
  dataloader_num_workers: 4       # Workers use CPU memory
  dataloader_pin_memory: true
```

### Strategy 3: Balanced (Recommended Default)

```yaml
# Goal: Good speed, good accuracy, reliable training

training:
  batch_size: 256
  learning_rate: 0.00032
  warmup_steps: 500

  fp16: true
  gradient_checkpointing: false  # Not needed for BERT-scale

  dataloader_num_workers: 8
  dataloader_pin_memory: true
```

### The Optimization Checklist

```
□ Enable mixed precision (fp16/bf16)
□ Batch size divisible by 8 (Tensor Cores)
□ Batch size maximizes memory usage (80-90% target)
□ Learning rate scaled with batch size
□ num_workers sufficient (4-16)
□ pin_memory enabled
□ Sequence length padded to multiple of 8
□ Gradient checkpointing if memory-constrained
□ torch.compile for kernel fusion (PyTorch 2.0+)
```

---

## GPU-Specific Configurations

### A100 40GB - Balanced

```yaml
training:
  batch_size: 256         # ~13 GB, leaves headroom
  learning_rate: 0.00032
  fp16: true              # Or bf16
  dataloader_num_workers: 8
  dataloader_pin_memory: true

# Memory utilization: ~65%
# Bandwidth utilization: ~80%
# Tensor Core utilization: High (batch % 8 == 0)
```

### A100 80GB - Aggressive

```yaml
training:
  batch_size: 768         # Push memory higher
  learning_rate: 0.00096  # 48× scaled
  warmup_steps: 800
  bf16: true              # Native BF16 support
  dataloader_num_workers: 16
  dataloader_pin_memory: true

# Memory utilization: ~80%
# Bandwidth utilization: ~85%
```

### H200 141GB - Maximum Performance

```yaml
training:
  batch_size: 1024        # Large batch for high throughput
  learning_rate: 0.001    # Capped for stability
  warmup_steps: 1000
  bf16: true              # H200 excels at BF16
  dataloader_num_workers: 16
  dataloader_pin_memory: true
  dataloader_prefetch_factor: 4

# Memory utilization: ~45%
# Bandwidth utilization: ~90%
# Could push batch higher if accuracy allows
```

### H200 141GB - Memory Maximized

```yaml
training:
  batch_size: 2048        # Use more of 141GB
  learning_rate: 0.001    # Still capped
  warmup_steps: 1500
  bf16: true
  gradient_checkpointing: false  # Not needed
  dataloader_num_workers: 16

# Memory utilization: ~75%
# Warning: Generalization may degrade at this batch size
```

### Configuration Comparison Table

```
┌──────────┬───────┬────────┬─────────┬──────────┬──────────┐
│   GPU    │ Batch │   LR   │ Memory  │ Bandwidth│ Time Est │
│          │       │        │  Used   │   Util   │ (MNLI)   │
├──────────┼───────┼────────┼─────────┼──────────┼──────────┤
│ A100 40G │  256  │ 3.2e-4 │   65%   │   80%    │  20 min  │
│ A100 40G │  512  │ 6.4e-4 │   85%   │   85%    │  12 min  │
│ A100 80G │  768  │ 9.6e-4 │   60%   │   85%    │   8 min  │
│ A100 80G │ 1024  │ 1.0e-3 │   75%   │   90%    │   6 min  │
│ H200     │  512  │ 6.4e-4 │   25%   │   70%    │   8 min  │
│ H200     │ 1024  │ 1.0e-3 │   45%   │   85%    │   5 min  │
│ H200     │ 2048  │ 1.0e-3 │   75%   │   90%    │   3 min  │
└──────────┴───────┴────────┴─────────┴──────────┴──────────┘
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

### PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=tensorboard_trace_handler('./logs/profiler'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(dataloader):
        if step >= 5:
            break

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        prof.step()

# View with: tensorboard --logdir=./logs/profiler
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

# Call during training
print_memory_stats()

# Reset peak stats
torch.cuda.reset_peak_memory_stats()
```

### Bandwidth Estimation

```python
def estimate_bandwidth_utilization(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_layers: int,
    step_time_ms: float,
    gpu_bandwidth_gbs: float,
) -> float:
    """Estimate memory bandwidth utilization."""

    # Approximate bytes moved per step (forward + backward)
    # This is a rough estimate for transformer-like models
    bytes_per_sample = seq_len * hidden_dim * num_layers * 2 * 3  # FP16, fwd+bwd
    total_bytes = bytes_per_sample * batch_size

    # Bandwidth used
    bandwidth_used = total_bytes / (step_time_ms / 1000) / 1e9  # GB/s

    utilization = bandwidth_used / gpu_bandwidth_gbs * 100

    return utilization

# Example for BERT on H200
util = estimate_bandwidth_utilization(
    batch_size=512,
    seq_len=128,
    hidden_dim=768,
    num_layers=12,
    step_time_ms=50,  # Measure this
    gpu_bandwidth_gbs=4800,  # H200
)
print(f"Estimated bandwidth utilization: {util:.1f}%")
```

---

## Quick Reference

```
┌──────────────────────────────────────────────────────────────┐
│           MEMORY × BANDWIDTH OPTIMIZATION CHEAT SHEET         │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  MAXIMIZE MEMORY USAGE:                                       │
│  ─────────────────────                                        │
│  • Increase batch size until 80-90% memory used              │
│  • Use gradient checkpointing if needed                       │
│  • Gradient accumulation for effective larger batch          │
│                                                               │
│  MAXIMIZE BANDWIDTH:                                          │
│  ──────────────────                                           │
│  • Use FP16/BF16 (2× more data per transfer)                 │
│  • Enable pin_memory for DataLoader                          │
│  • Use torch.compile for kernel fusion                        │
│  • Ensure contiguous tensors                                  │
│                                                               │
│  ENABLE TENSOR CORES:                                         │
│  ────────────────────                                         │
│  • Dimensions divisible by 8 (batch, hidden, seq_len)        │
│  • Use FP16/BF16/TF32 data types                             │
│                                                               │
│  MONITORING:                                                  │
│  ──────────────                                               │
│  • nvidia-smi: GPU-Util should be 90%+                       │
│  • Memory: Target 80-90% utilization                         │
│  • Profile to find bottlenecks                               │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## References

- [NVIDIA Mixed Precision Training](https://developer.nvidia.com/automatic-mixed-precision)
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Roofline Model Paper](https://dl.acm.org/doi/10.1145/1498765.1498785)
- [NVIDIA Tensor Core Programming](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)
- [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
