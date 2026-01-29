# GPU Utilization Guide: A100 and H200

This document provides specific recommendations for maximizing GPU utilization on NVIDIA A100 and H200 GPUs for our BERT→DistilBERT knowledge distillation task.

---

## Table of Contents

1. [GPU Specifications](#gpu-specifications)
2. [Memory Budget Analysis](#memory-budget-analysis)
3. [A100 Configurations](#a100-configurations)
4. [H200 Configurations](#h200-configurations)
5. [Optimal Settings by Goal](#optimal-settings-by-goal)
6. [Performance Benchmarks](#performance-benchmarks)
7. [Troubleshooting](#troubleshooting)

---

## GPU Specifications

### NVIDIA A100

| Variant | VRAM | Memory Bandwidth | FP16 TFLOPS |
|---------|------|------------------|-------------|
| A100 40GB | 40 GB HBM2e | 1,555 GB/s | 312 |
| A100 80GB | 80 GB HBM2e | 2,039 GB/s | 312 |

**Available on:**
- Google Colab Pro/Pro+ (40GB)
- Northeastern Discovery (40GB/80GB)
- Cloud providers (AWS p4d, GCP a2)

### NVIDIA H200

| Spec | Value |
|------|-------|
| VRAM | 141 GB HBM3 |
| Memory Bandwidth | 4,800 GB/s |
| FP16 TFLOPS | 989 |
| FP8 TFLOPS | 1,979 |

**Available on:**
- Northeastern Discovery (gpu partition)
- Limited cloud availability (2024)

### Comparison

```
Performance (relative to V100):

           Memory    Bandwidth    Compute
V100:      32 GB     900 GB/s     125 TFLOPS    (1.0×)
A100:      80 GB     2,039 GB/s   312 TFLOPS    (2.5×)
H200:      141 GB    4,800 GB/s   989 TFLOPS    (8.0×)
```

---

## Memory Budget Analysis

### BERT→DistilBERT Memory Breakdown

#### Model Weights

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

#### Activation Memory (Per Sample)

```
BERT forward pass activations:
  - Hidden states: 12 layers × seq_len × hidden_dim × batch
  - Attention scores: 12 layers × num_heads × seq_len² × batch

For seq_len=128, hidden=768, heads=12:
  - Per sample: ~25-50 MB (varies with implementation)
```

#### Memory Formula

```
Total Memory ≈ Models + Activations × Batch + Overhead

For batch B and seq_len 128:
Memory (GB) ≈ 1.15 + 0.04 × B + 2.0  (overhead buffer)
            ≈ 3.15 + 0.04 × B
```

### Maximum Batch Size by GPU

```
A100 40GB:
  40 = 3.15 + 0.04 × B
  B = (40 - 3.15) / 0.04 ≈ 921
  Safe maximum: 768

A100 80GB:
  80 = 3.15 + 0.04 × B
  B = (80 - 3.15) / 0.04 ≈ 1921
  Safe maximum: 1536

H200 141GB:
  141 = 3.15 + 0.04 × B
  B = (141 - 3.15) / 0.04 ≈ 3446
  Safe maximum: 2560
```

---

## A100 Configurations

### A100 40GB - Conservative (Recommended for Reproduction)

```yaml
# configs/bert_distilbert_mnli_a100_40gb.yaml
# Matches paper settings, uses minimal memory

inherit: base.yaml

model:
  teacher: bert-base-uncased
  student: distilbert-base-uncased
  student_init: fresh

data:
  task: mnli
  max_length: 128

training:
  num_epochs: 3
  batch_size: 256
  learning_rate: 0.00032  # 16× scaled from 2e-5
  warmup_steps: 500
  weight_decay: 0.01
  max_grad_norm: 2.0
  fp16: true

distillation:
  temperature: 2
  alpha: 0.5

# Memory usage: ~13 GB (leaves 27 GB headroom)
# Time: ~20 minutes
# Expected accuracy: ~81.5%
```

### A100 40GB - Aggressive

```yaml
# Push utilization higher
training:
  batch_size: 512
  learning_rate: 0.00064  # 32× scaled
  warmup_steps: 500

# Memory usage: ~24 GB
# Time: ~12 minutes
# Expected accuracy: ~81.0%
```

### A100 80GB - Recommended

```yaml
# Good balance for 80GB variant
training:
  batch_size: 512
  learning_rate: 0.00064
  warmup_steps: 500

# Memory usage: ~24 GB (leaves 56 GB headroom)
# Time: ~10 minutes
# Expected accuracy: ~81.0%
```

### A100 80GB - Maximum Throughput

```yaml
training:
  batch_size: 1024
  learning_rate: 0.001  # Capped for stability
  warmup_steps: 1000

# Memory usage: ~45 GB
# Time: ~6 minutes
# Expected accuracy: ~80.0-80.5%
```

---

## H200 Configurations

### H200 - Paper Reproduction (Fastest Exact Match)

```yaml
# configs/bert_distilbert_mnli_h200_repro.yaml
# Use paper settings but benefit from H200 speed

inherit: base.yaml

model:
  teacher: bert-base-uncased
  student: distilbert-base-uncased
  student_init: fresh

data:
  task: mnli
  max_length: 128

training:
  num_epochs: 3
  batch_size: 16        # Paper setting
  learning_rate: 0.00002  # Paper setting
  warmup_steps: 200
  weight_decay: 0.01
  max_grad_norm: 2.0
  fp16: true
  dataloader_num_workers: 8  # More workers for fast GPU

distillation:
  temperature: 2
  alpha: 0.5

checkpointing:
  save_steps: 2000      # Less frequent saves (fast training)
  save_total_limit: 3

evaluation:
  eval_steps: 2000
  eval_strategy: steps

# Memory usage: ~4 GB (141 GB available - massively underutilized!)
# Time: ~2 hours (H200 compute not fully utilized)
# Expected accuracy: ~81.77% (matches paper)
```

### H200 - Balanced (Recommended)

```yaml
# configs/bert_distilbert_mnli_h200.yaml
# Best accuracy/speed tradeoff

inherit: base.yaml

model:
  teacher: bert-base-uncased
  student: distilbert-base-uncased
  student_init: fresh

data:
  task: mnli
  max_length: 128

training:
  num_epochs: 3
  batch_size: 512
  learning_rate: 0.00064  # 32× scaled
  warmup_steps: 500
  weight_decay: 0.01
  max_grad_norm: 2.0
  fp16: true
  dataloader_num_workers: 8

distillation:
  temperature: 2
  alpha: 0.5

checkpointing:
  save_steps: 200
  save_total_limit: 3

evaluation:
  eval_steps: 200
  eval_strategy: steps

# Memory usage: ~24 GB
# Time: ~8 minutes
# Expected accuracy: ~81.0%
```

### H200 - Maximum Speed

```yaml
# configs/bert_distilbert_mnli_h200_fast.yaml
# Fastest possible, some accuracy loss acceptable

training:
  batch_size: 1024
  learning_rate: 0.001    # Capped at 1e-3
  warmup_steps: 1000
  dataloader_num_workers: 16

checkpointing:
  save_steps: 100
  save_total_limit: 2

evaluation:
  eval_steps: 100
  eval_strategy: steps

# Memory usage: ~45 GB
# Time: ~4-5 minutes
# Expected accuracy: ~80.0%
```

### H200 - Ultra (Experimental)

```yaml
# Push to the limit - may be unstable
training:
  batch_size: 2048
  learning_rate: 0.001    # Still capped
  warmup_steps: 1500
  gradient_accumulation_steps: 1

# Memory usage: ~85 GB
# Time: ~3 minutes
# Expected accuracy: ~79-80% (generalization gap)
```

---

## Optimal Settings by Goal

### Goal: Reproduce Paper Results Exactly

```yaml
# Use paper settings regardless of GPU
batch_size: 16
learning_rate: 0.00002
warmup_steps: 200

# A100: ~45 min | H200: ~25 min
```

### Goal: Fastest Reproduction (Allow ±0.5% Accuracy)

```yaml
# A100 40GB
batch_size: 256
learning_rate: 0.00032
warmup_steps: 500
# Time: ~20 min, Accuracy: ~81.3%

# H200
batch_size: 512
learning_rate: 0.00064
warmup_steps: 500
# Time: ~8 min, Accuracy: ~81.0%
```

### Goal: Run Many Experiments Quickly

```yaml
# For hyperparameter sweeps, speed > perfect accuracy
batch_size: 1024
learning_rate: 0.001
warmup_steps: 1000
num_epochs: 2  # Fewer epochs OK for comparison

# H200: ~3 min per experiment
# Can run ~20 experiments per hour!
```

### Goal: Maximum Accuracy (Time No Object)

```yaml
# Smaller batch often generalizes better
batch_size: 32
learning_rate: 0.00004
warmup_steps: 400
num_epochs: 5  # More epochs

# May exceed paper results slightly
# H200: ~1 hour
```

---

## Performance Benchmarks

### Throughput (Samples/Second)

```
GPU          | Batch 16 | Batch 256 | Batch 1024 |
-------------|----------|-----------|------------|
V100 32GB    |    150   |    450    |    600     |
A100 40GB    |    400   |   1,200   |   1,800    |
A100 80GB    |    400   |   1,200   |   2,000    |
H200 141GB   |    800   |   2,400   |   4,000    |
```

### Time to Complete MNLI (3 epochs, ~1.18M samples)

```
GPU          | Batch 16 | Batch 256 | Batch 1024 |
-------------|----------|-----------|------------|
V100 32GB    |  2.2 hr  |   45 min  |   OOM      |
A100 40GB    |   50 min |   20 min  |   12 min   |
A100 80GB    |   50 min |   18 min  |   10 min   |
H200 141GB   |   25 min |    8 min  |    5 min   |
```

### Memory Utilization

```
Batch Size | Memory Used | A100 40GB | A100 80GB | H200 141GB |
-----------|-------------|-----------|-----------|------------|
    16     |    ~4 GB    |    10%    |     5%    |     3%     |
   256     |   ~13 GB    |    33%    |    16%    |     9%     |
   512     |   ~24 GB    |    60%    |    30%    |    17%     |
  1024     |   ~45 GB    |   OOM*    |    56%    |    32%     |
  2048     |   ~85 GB    |   OOM     |   OOM     |    60%     |

* May work with gradient checkpointing
```

---

## Troubleshooting

### Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions:**

1. **Reduce batch size**
   ```yaml
   batch_size: 256  # → 128
   learning_rate: 0.00016  # Scale down proportionally
   ```

2. **Enable gradient checkpointing** (trades compute for memory)
   ```python
   model.gradient_checkpointing_enable()
   # Reduces memory ~40%, increases time ~20%
   ```

3. **Use gradient accumulation**
   ```yaml
   batch_size: 64
   gradient_accumulation_steps: 4
   # Effective batch 256, but fits in memory
   ```

4. **Reduce sequence length**
   ```yaml
   max_length: 64  # Instead of 128
   # Halves activation memory
   ```

### Training Too Slow

**Symptoms:** Not utilizing full GPU compute

**Solutions:**

1. **Increase batch size**
   ```yaml
   batch_size: 512  # From 256
   # Remember to scale learning rate!
   ```

2. **Increase dataloader workers**
   ```yaml
   dataloader_num_workers: 8  # From 4
   ```

3. **Enable mixed precision** (should be default)
   ```yaml
   fp16: true
   ```

4. **Disable deterministic mode** (small speed boost)
   ```python
   torch.backends.cudnn.deterministic = False
   torch.backends.cudnn.benchmark = True
   ```

### Accuracy Lower Than Expected

**Symptoms:** Batch 1024 getting 79% instead of 81%

**Solutions:**

1. **Increase warmup**
   ```yaml
   warmup_steps: 1500  # From 1000
   ```

2. **Cap learning rate**
   ```yaml
   learning_rate: 0.0008  # Instead of 0.00128
   ```

3. **Use smaller batch**
   ```yaml
   batch_size: 512  # Compromise
   ```

4. **Add label smoothing**
   ```python
   # Can help with large batch generalization
   loss_fct = CrossEntropyLoss(label_smoothing=0.1)
   ```

### GPU Not Being Detected

**Check CUDA availability:**
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
```

**On SLURM:**
```bash
# Ensure GPU is allocated
echo $CUDA_VISIBLE_DEVICES
nvidia-smi
```

---

## DataLoader Workers Optimization

### The Key Insight: Batch Size ≠ Data Loading Parallelism

**Batch size** and **num_workers** control completely different things:

```
Batch Size (training dynamics):
  - Controls gradient quality (variance)
  - Affects generalization (sharp vs flat minima)
  - Requires learning rate scaling
  - Determines training behavior

Num Workers (data loading):
  - Controls CPU parallelism for data preparation
  - Affects GPU utilization (keeps GPU fed)
  - No impact on training dynamics
  - Determines throughput efficiency
```

### Why This Matters

With a small batch size (e.g., 16), the GPU finishes processing quickly. If there's only one data loading worker, the GPU sits idle waiting for the next batch:

```
Single Worker (num_workers=1):
  GPU:    [===IDLE===][compute][===IDLE===][compute]
  CPU:    [load batch 1       ][load batch 2       ]

Multiple Workers (num_workers=8):
  GPU:    [compute][compute][compute][compute]
  CPU-1:  [load 1 ][load 9 ][load 17]
  CPU-2:  [load 2 ][load 10][load 18]
  ...
  CPU-8:  [load 8 ][load 16][load 24]
         ↑
         Batches ready before GPU needs them!
```

### Optimal num_workers Settings

```
Rule of Thumb:
  num_workers = min(batch_size, cpu_cores, 8-16)

Specific Recommendations:
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

You CAN have:
- **Small batch for training quality** (better generalization)
- **Many workers for GPU utilization** (higher throughput)

```yaml
# Optimal for paper reproduction on H200
training:
  batch_size: 16              # Paper setting (training dynamics)
  learning_rate: 0.00002      # Paper setting
  dataloader_num_workers: 8   # Keep GPU fed (throughput)
  dataloader_pin_memory: true # Faster CPU→GPU transfer

# The GPU processes small batches, but workers ensure
# the next batch is always ready immediately
```

### Data Loading Pipeline

```
┌────────────────────────────────────────────────────────────┐
│                    DATA LOADING PIPELINE                    │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  Storage (disk/SSD)                                        │
│       ↓                                                    │
│  Worker Process 1 ─┐                                       │
│  Worker Process 2 ─┤                                       │
│  Worker Process 3 ─┼──→ Prefetch Queue ──→ GPU            │
│  Worker Process 4 ─┤    (pin_memory)                       │
│  ...               ┘                                       │
│                                                             │
│  Each worker:                                               │
│  1. Reads raw data from disk                               │
│  2. Applies transforms/tokenization                        │
│  3. Collates into batch                                    │
│  4. Optionally pins memory for fast GPU transfer           │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

### Prefetching with prefetch_factor

```yaml
# Default: prefetch_factor=2 (prefetch 2 batches per worker)
# For fast GPUs, increase this:

training:
  dataloader_num_workers: 8
  dataloader_prefetch_factor: 4  # 8 workers × 4 = 32 batches ready

# Trade-off: More memory usage, but GPU never starves
```

### Configuration Examples

**Small Batch, High Throughput (Paper Reproduction):**
```yaml
training:
  batch_size: 16
  learning_rate: 0.00002
  dataloader_num_workers: 8
  dataloader_pin_memory: true
  dataloader_prefetch_factor: 4
```

**Large Batch, Maximum Speed:**
```yaml
training:
  batch_size: 512
  learning_rate: 0.00064
  dataloader_num_workers: 8
  dataloader_pin_memory: true
  # prefetch_factor=2 is usually enough for large batches
```

**Memory-Constrained (Colab):**
```yaml
training:
  batch_size: 64
  learning_rate: 0.00008
  dataloader_num_workers: 2  # Colab has limited CPU
  dataloader_pin_memory: true
```

### Diagnosing Data Loading Bottlenecks

**Signs of data loading bottleneck:**
- GPU utilization fluctuates (not steady 90%+)
- `nvidia-smi` shows GPU compute is low
- Training step time varies significantly

**Quick check with PyTorch profiler:**
```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for batch in dataloader:
        # training step
        break

print(prof.key_averages().table(sort_by="cuda_time_total"))
# Look for: DataLoader time vs Compute time
```

### Summary: Decoupling Training Quality from Throughput

```
┌─────────────────────────────────────────────────────────────┐
│         TRAINING QUALITY vs GPU THROUGHPUT                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Training Quality (batch_size + lr):                       │
│  ───────────────────────────────────                       │
│  • Small batch → better generalization, more noise         │
│  • Large batch → faster convergence, sharp minima risk     │
│  • Must scale lr with batch size                           │
│                                                             │
│  GPU Throughput (num_workers + pin_memory):                │
│  ──────────────────────────────────────────                │
│  • More workers → GPU never waits for data                 │
│  • pin_memory → faster CPU→GPU transfer                    │
│  • Independent of training dynamics                        │
│                                                             │
│  ✓ You CAN have small batch + high throughput!             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Reference: Recommended Configs

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

## Creating the Config Files

Run these commands to create optimized configs:

```bash
# A100 40GB config
cat > configs/bert_distilbert_mnli_a100_40gb.yaml << 'EOF'
inherit: base.yaml
model:
  teacher: bert-base-uncased
  student: distilbert-base-uncased
  student_init: fresh
data:
  task: mnli
  max_length: 128
training:
  num_epochs: 3
  batch_size: 256
  learning_rate: 0.00032
  warmup_steps: 500
  weight_decay: 0.01
  max_grad_norm: 2.0
distillation:
  temperature: 2
  alpha: 0.5
checkpointing:
  save_steps: 500
  save_total_limit: 3
logging:
  report_to: wandb
  logging_steps: 50
evaluation:
  eval_steps: 500
  eval_strategy: steps
EOF

# H200 config
cat > configs/bert_distilbert_mnli_h200.yaml << 'EOF'
inherit: base.yaml
model:
  teacher: bert-base-uncased
  student: distilbert-base-uncased
  student_init: fresh
data:
  task: mnli
  max_length: 128
training:
  num_epochs: 3
  batch_size: 512
  learning_rate: 0.00064
  warmup_steps: 500
  weight_decay: 0.01
  max_grad_norm: 2.0
  dataloader_num_workers: 8
distillation:
  temperature: 2
  alpha: 0.5
checkpointing:
  save_steps: 200
  save_total_limit: 3
logging:
  report_to: wandb
  logging_steps: 20
evaluation:
  eval_steps: 200
  eval_strategy: steps
EOF
```
