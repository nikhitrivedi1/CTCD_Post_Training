# GPU Comparison Guide for Deep Learning

A comprehensive reference for GPUs commonly used in deep learning training, with specifications, capabilities, availability, and recommendations.

---

## Table of Contents

1. [GPU Overview](#gpu-overview)
2. [NVIDIA V100](#nvidia-v100)
3. [NVIDIA T4](#nvidia-t4)
4. [NVIDIA A100](#nvidia-a100)
5. [NVIDIA H100](#nvidia-h100)
6. [NVIDIA H200](#nvidia-h200)
7. [Apple Silicon (M1/M2/M3)](#apple-silicon-m1m2m3)
8. [Comparison Tables](#comparison-tables)
9. [Availability Guide](#availability-guide)
10. [Recommendations by Use Case](#recommendations-by-use-case)

---

## GPU Overview

### The Key Specifications

```
┌────────────────────────────────────────────────────────────────┐
│                    GPU SPECIFICATIONS EXPLAINED                 │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  VRAM (Video RAM)                                               │
│  ─────────────────                                              │
│  Memory capacity for model weights, activations, gradients.    │
│  More VRAM = larger models, larger batch sizes.                │
│                                                                 │
│  Memory Bandwidth                                               │
│  ────────────────                                               │
│  Speed of data transfer between VRAM and compute units.        │
│  Higher = faster for memory-bound operations.                  │
│  Measured in GB/s.                                              │
│                                                                 │
│  TFLOPS (Teraflops)                                            │
│  ─────────────────                                              │
│  Trillion floating-point operations per second.                │
│  Higher = faster for compute-bound operations.                 │
│  FP16 TFLOPS typically 2× FP32 TFLOPS.                        │
│                                                                 │
│  Tensor Cores                                                   │
│  ────────────                                                   │
│  Specialized units for matrix multiply-accumulate.             │
│  8-16× faster than CUDA cores for supported operations.        │
│  Require FP16/BF16/TF32 and dimensions divisible by 8.        │
│                                                                 │
│  TDP (Thermal Design Power)                                    │
│  ─────────────────────────                                      │
│  Power consumption in watts.                                    │
│  Higher TDP = more heat, higher electricity cost.              │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Generation Timeline

```
2017: V100 (Volta architecture)
2018: T4 (Turing architecture)
2020: A100 (Ampere architecture)
2022: H100 (Hopper architecture)
2024: H200 (Hopper architecture, more memory)
```

---

## NVIDIA V100

### Specifications

| Spec | Value |
|------|-------|
| Architecture | Volta (2017) |
| VRAM | 16 GB or 32 GB HBM2 |
| Memory Bandwidth | 900 GB/s |
| FP32 TFLOPS | 15.7 |
| FP16 TFLOPS | 125 (with Tensor Cores) |
| Tensor Cores | Yes (1st generation) |
| TDP | 300W |

### Characteristics

```
Strengths:
├── Mature, well-supported
├── Good price/performance on cloud
├── Reliable for production workloads
└── Wide driver compatibility

Weaknesses:
├── No BF16 support (FP16 only)
├── Lower memory bandwidth than newer GPUs
├── Tensor Cores less efficient than A100/H100
└── Limited availability (being phased out)
```

### Our Workload Performance

```
BERT→DistilBERT distillation on MNLI:

Batch 16:   ~2.2 hours
Batch 256:  ~45 minutes
Batch 512:  OOM (16GB) or ~30 min (32GB)

Memory limit: batch ~400 on 16GB, ~900 on 32GB
```

### Availability

```
✓ AWS: p3.2xlarge (1× V100 16GB) - ~$3/hour
✓ GCP: n1-standard + V100 - ~$2.50/hour
✓ Paperspace: V100 instances
✗ Colab: Not available
✗ Discovery HPC: Limited/deprecated
```

---

## NVIDIA T4

### Specifications

| Spec | Value |
|------|-------|
| Architecture | Turing (2018) |
| VRAM | 16 GB GDDR6 |
| Memory Bandwidth | 320 GB/s |
| FP32 TFLOPS | 8.1 |
| FP16 TFLOPS | 65 (with Tensor Cores) |
| Tensor Cores | Yes (2nd generation) |
| TDP | 70W |

### Characteristics

```
Strengths:
├── Low power consumption (70W)
├── Widely available (Colab Free!)
├── Good for inference
├── Cost-effective for small models
└── Fits in standard PCIe slots

Weaknesses:
├── Lower memory bandwidth (GDDR6 vs HBM2)
├── Less compute than V100/A100
├── 16GB can be limiting for large batches
└── Slower than datacenter GPUs
```

### Our Workload Performance

```
BERT→DistilBERT distillation on MNLI:

Batch 16:   ~3-4 hours
Batch 128:  ~1 hour
Batch 256:  May OOM (tight on 16GB)

Recommended config: batch=128, lr=1.6e-4
```

### Availability

```
✓ Google Colab: Free tier (usage limits)
✓ Google Colab Pro: Priority access
✓ AWS: g4dn instances - ~$0.50/hour
✓ GCP: n1-standard + T4 - ~$0.35/hour
✓ Paperspace: Gradient free tier
```

### Colab-Specific Notes

```
Colab Free:
├── Random GPU assignment (may get T4, P100, or K80)
├── 12-hour session limit
├── May disconnect if idle
└── Limited to 2 CPUs (use num_workers=2)

Colab Pro ($10/month):
├── Higher chance of T4 or better
├── Longer session limits
├── Background execution
└── More RAM (25GB vs 12GB)
```

---

## NVIDIA A100

### Specifications

| Variant | VRAM | Bandwidth | FP16 TFLOPS |
|---------|------|-----------|-------------|
| A100 40GB | 40 GB HBM2e | 1,555 GB/s | 312 |
| A100 80GB | 80 GB HBM2e | 2,039 GB/s | 312 |

| Spec | Value |
|------|-------|
| Architecture | Ampere (2020) |
| FP32 TFLOPS | 19.5 |
| TF32 TFLOPS | 156 |
| BF16 TFLOPS | 312 |
| Tensor Cores | Yes (3rd generation) |
| TDP | 400W |

### Characteristics

```
Strengths:
├── Native BF16 support (more stable than FP16)
├── TF32 mode (FP32 input, Tensor Core speed)
├── Multi-Instance GPU (MIG) partitioning
├── 2× bandwidth vs V100
├── 80GB variant handles very large models
└── Current datacenter standard

Weaknesses:
├── Expensive (~$10-15K per GPU)
├── High power consumption (400W)
├── Requires good cooling
└── 40GB can still be limiting for LLMs
```

### Our Workload Performance

```
BERT→DistilBERT distillation on MNLI:

A100 40GB:
├── Batch 256:  ~20 minutes
├── Batch 512:  ~12 minutes
└── Batch 768:  ~10 minutes (pushing limits)

A100 80GB:
├── Batch 512:  ~10 minutes
├── Batch 1024: ~6 minutes
└── Batch 1536: ~5 minutes

Recommended: batch=256-512, lr=3.2e-4 to 6.4e-4
```

### Availability

```
✓ Google Colab Pro+: A100 40GB available
✓ AWS: p4d.24xlarge (8× A100 40GB) - ~$32/hour
✓ GCP: a2-highgpu-1g (1× A100 40GB) - ~$4/hour
✓ Lambda Labs: ~$1.10/hour (A100 40GB)
✓ Discovery HPC: gpu partition (check availability)
✓ Paperspace: A100 instances
```

---

## NVIDIA H100

### Specifications

| Variant | VRAM | Bandwidth | FP16 TFLOPS |
|---------|------|-----------|-------------|
| H100 PCIe | 80 GB HBM3 | 2,000 GB/s | 756 |
| H100 SXM5 | 80 GB HBM3 | 3,350 GB/s | 989 |

| Spec | Value |
|------|-------|
| Architecture | Hopper (2022) |
| FP32 TFLOPS | 67 |
| TF32 TFLOPS | 495 |
| BF16 TFLOPS | 989 |
| FP8 TFLOPS | 1,979 |
| Tensor Cores | Yes (4th generation) |
| TDP | 350-700W |

### Characteristics

```
Strengths:
├── FP8 support (2× throughput vs BF16)
├── Transformer Engine (automatic precision)
├── 3× compute vs A100
├── NVLink 4.0 (900 GB/s GPU-to-GPU)
├── Ideal for LLM training
└── Best for large-scale distributed training

Weaknesses:
├── Very expensive (~$25-40K per GPU)
├── High power consumption (up to 700W)
├── Limited availability
├── Still 80GB (same as A100 80GB)
└── Overkill for small models
```

### Our Workload Performance

```
BERT→DistilBERT distillation on MNLI:

H100 SXM5:
├── Batch 256:  ~8 minutes
├── Batch 512:  ~5 minutes
├── Batch 1024: ~3 minutes
└── Batch 2048: ~2 minutes (may hurt accuracy)

For BERT-scale models, H100 is overkill.
Better suited for LLM training (7B+ parameters).
```

### Availability

```
✓ AWS: p5.48xlarge (8× H100) - ~$98/hour
✓ GCP: a3-highgpu-8g (8× H100) - limited
✓ Lambda Labs: ~$2/hour (H100 80GB)
✓ CoreWeave: ~$2.50/hour
△ Discovery HPC: Check gpu partition
✗ Colab: Not available
```

---

## NVIDIA H200

### Specifications

| Spec | Value |
|------|-------|
| Architecture | Hopper (2024) |
| VRAM | 141 GB HBM3e |
| Memory Bandwidth | 4,800 GB/s |
| FP16 TFLOPS | 989 |
| BF16 TFLOPS | 989 |
| FP8 TFLOPS | 1,979 |
| Tensor Cores | Yes (4th generation) |
| TDP | 700W |

### Characteristics

```
Strengths:
├── 141 GB VRAM (76% more than H100!)
├── 4,800 GB/s bandwidth (43% more than H100!)
├── Same compute as H100
├── Can fit 70B+ models without sharding
├── Ideal for LLM inference and fine-tuning
└── Best memory bandwidth available

Weaknesses:
├── Extremely expensive
├── Very limited availability (2024)
├── Same compute as H100 (memory-focused upgrade)
├── High power consumption (700W)
└── Massive overkill for BERT-scale
```

### Our Workload Performance

```
BERT→DistilBERT distillation on MNLI:

H200:
├── Batch 512:   ~8 minutes
├── Batch 1024:  ~5 minutes
├── Batch 2048:  ~3 minutes
└── Batch 3072:  ~2.5 minutes (possible but accuracy may suffer)

Memory utilization at batch=1024: only ~45%!
H200 is designed for much larger models than BERT.

For our use case: Similar speed to H100.
The extra memory doesn't help for BERT-scale models.
```

### Availability

```
✓ Discovery HPC: gpu partition (if available)
✓ Lambda Labs: Limited availability
✓ CoreWeave: Limited availability
✗ AWS/GCP: Not yet widely available
✗ Colab: Not available
```

---

## Apple Silicon (M1/M2/M3)

### Specifications

| Chip | GPU Cores | Unified Memory | Bandwidth |
|------|-----------|----------------|-----------|
| M1 | 7-8 | 8-16 GB | 68 GB/s |
| M1 Pro | 14-16 | 16-32 GB | 200 GB/s |
| M1 Max | 24-32 | 32-64 GB | 400 GB/s |
| M1 Ultra | 48-64 | 64-128 GB | 800 GB/s |
| M2 | 8-10 | 8-24 GB | 100 GB/s |
| M2 Pro | 16-19 | 16-32 GB | 200 GB/s |
| M2 Max | 30-38 | 32-96 GB | 400 GB/s |
| M2 Ultra | 60-76 | 64-192 GB | 800 GB/s |
| M3 | 8-10 | 8-24 GB | 100 GB/s |
| M3 Pro | 14-18 | 18-36 GB | 150 GB/s |
| M3 Max | 30-40 | 36-128 GB | 400 GB/s |

### Characteristics

```
Strengths:
├── Unified memory (CPU and GPU share RAM)
├── Very power efficient
├── Silent operation (no fans on base models)
├── Large memory possible (M2 Ultra: 192GB)
├── Good for development and testing
└── No separate GPU purchase needed

Weaknesses:
├── No FP16 training support (use FP32)
├── MPS backend less mature than CUDA
├── Lower raw compute than datacenter GPUs
├── PyTorch MPS still has bugs
├── No multi-GPU support
└── num_workers must be 0 for stability
```

### Our Workload Performance

```
BERT→DistilBERT distillation on MNLI (M2 base):

Batch 16:   ~15-20 hours (paper settings)
Batch 64:   ~5-7 hours
Batch 128:  ~3-4 hours (may OOM on 8GB)

Speed comparison (batch 64):
├── M2:     ~3.0 s/step
├── T4:     ~0.8 s/step
├── A100:   ~0.1 s/step
└── Ratio:  M2 is ~30× slower than A100

Good for: Development, debugging, small experiments
Bad for:  Production training, large-scale experiments
```

### MPS-Specific Configuration

```yaml
# Optimized for Apple Silicon
training:
  batch_size: 64
  learning_rate: 0.00008
  fp16: false              # MPS doesn't support FP16 well
  dataloader_num_workers: 0  # Must be 0 for MPS stability
  dataloader_pin_memory: false  # Not applicable to MPS
```

### Common MPS Issues

```
Issue: "MPS backend out of memory"
Fix: Reduce batch size, close other apps

Issue: "RuntimeError: Placeholder storage has not been allocated"
Fix: Use num_workers=0

Issue: NaN losses with FP16
Fix: Use fp16=false (FP32 training)

Issue: Slow training speed
Fix: Expected - MPS is slower than CUDA

Issue: Multiprocessing errors
Fix: Add to script start:
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
```

---

## Comparison Tables

### Raw Specifications

```
┌──────────┬────────┬───────────┬───────────┬──────────┬───────┐
│   GPU    │  VRAM  │ Bandwidth │ FP16 TF   │ BF16 TF  │  TDP  │
├──────────┼────────┼───────────┼───────────┼──────────┼───────┤
│ V100 16G │  16 GB │   900 GB/s│    125    │    -     │ 300W  │
│ V100 32G │  32 GB │   900 GB/s│    125    │    -     │ 300W  │
│ T4       │  16 GB │   320 GB/s│     65    │    -     │  70W  │
│ A100 40G │  40 GB │ 1,555 GB/s│    312    │   312    │ 400W  │
│ A100 80G │  80 GB │ 2,039 GB/s│    312    │   312    │ 400W  │
│ H100 SXM │  80 GB │ 3,350 GB/s│    989    │   989    │ 700W  │
│ H200     │ 141 GB │ 4,800 GB/s│    989    │   989    │ 700W  │
│ M2 (8C)  │ 8-24GB │   100 GB/s│    ~3*    │    -     │  22W  │
└──────────┴────────┴───────────┴───────────┴──────────┴───────┘
* M2 FP16 TFLOPS is approximate, MPS performance varies
```

### Performance for Our Workload

```
BERT→DistilBERT on MNLI (3 epochs):

┌──────────┬───────────┬────────────┬──────────────┐
│   GPU    │  Batch    │    Time    │  Est. Cost   │
├──────────┼───────────┼────────────┼──────────────┤
│ T4       │   128     │   ~1 hr    │  ~$0.50      │
│ V100     │   256     │  ~45 min   │  ~$2.25      │
│ A100 40G │   256     │  ~20 min   │  ~$1.30      │
│ A100 80G │   512     │  ~10 min   │  ~$0.80      │
│ H100     │  1024     │   ~5 min   │  ~$0.80      │
│ H200     │  1024     │   ~5 min   │  ~$1.00      │
│ M2       │    64     │  ~5-7 hr   │  electricity │
└──────────┴───────────┴────────────┴──────────────┘

Note: Costs are rough estimates based on typical cloud pricing.
```

### Memory Capacity Comparison

```
Maximum batch size for BERT→DistilBERT (seq_len=128):

┌──────────┬────────┬───────────┬─────────────────────┐
│   GPU    │  VRAM  │ Max Batch │ Memory Utilization  │
├──────────┼────────┼───────────┼─────────────────────┤
│ T4       │  16 GB │    ~200   │  85% at batch 128   │
│ V100 16G │  16 GB │    ~200   │  85% at batch 128   │
│ V100 32G │  32 GB │    ~600   │  70% at batch 512   │
│ A100 40G │  40 GB │    ~800   │  60% at batch 512   │
│ A100 80G │  80 GB │  ~1,600   │  50% at batch 1024  │
│ H100     │  80 GB │  ~1,600   │  50% at batch 1024  │
│ H200     │ 141 GB │  ~3,000   │  30% at batch 1024  │
│ M2 8GB   │   8 GB │    ~100   │  80% at batch 64    │
│ M2 24GB  │  24 GB │    ~400   │  65% at batch 256   │
└──────────┴────────┴───────────┴─────────────────────┘
```

### Feature Support

```
┌──────────┬───────┬───────┬───────┬────────┬─────────┐
│   GPU    │ FP16  │ BF16  │ TF32  │  FP8   │ MIG     │
├──────────┼───────┼───────┼───────┼────────┼─────────┤
│ V100     │  ✓    │  ✗    │  ✗    │   ✗    │   ✗     │
│ T4       │  ✓    │  ✗    │  ✗    │   ✗    │   ✗     │
│ A100     │  ✓    │  ✓    │  ✓    │   ✗    │   ✓     │
│ H100     │  ✓    │  ✓    │  ✓    │   ✓    │   ✓     │
│ H200     │  ✓    │  ✓    │  ✓    │   ✓    │   ✓     │
│ M1/M2/M3 │  ✗*   │  ✗    │  ✗    │   ✗    │   ✗     │
└──────────┴───────┴───────┴───────┴────────┴─────────┘
* MPS has FP16 support but it's buggy for training
```

---

## Availability Guide

### Cloud Providers

```
┌─────────────────────────────────────────────────────────────┐
│                   CLOUD GPU AVAILABILITY                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Google Colab:                                               │
│  ├── Free: T4, P100, or K80 (random assignment)             │
│  ├── Pro ($10/mo): T4, P100, V100                           │
│  └── Pro+ ($50/mo): A100 40GB                               │
│                                                              │
│  AWS:                                                        │
│  ├── g4dn.*: T4 (~$0.50/hr)                                 │
│  ├── p3.*: V100 (~$3/hr)                                    │
│  ├── p4d.*: A100 40GB (~$4/hr per GPU)                      │
│  └── p5.*: H100 (~$12/hr per GPU)                           │
│                                                              │
│  GCP:                                                        │
│  ├── T4: ~$0.35/hr                                          │
│  ├── V100: ~$2.50/hr                                        │
│  ├── A100 40GB: ~$4/hr                                      │
│  └── A100 80GB: ~$5/hr                                      │
│                                                              │
│  Lambda Labs:                                                │
│  ├── A100 40GB: ~$1.10/hr                                   │
│  ├── A100 80GB: ~$1.40/hr                                   │
│  └── H100: ~$2/hr                                           │
│                                                              │
│  Paperspace:                                                 │
│  ├── Free: M4000 (limited)                                  │
│  ├── Pro: V100, A100                                        │
│  └── Gradient: Various GPUs                                 │
│                                                              │
│  Vast.ai (community cloud):                                 │
│  ├── Variable pricing based on supply                       │
│  ├── Often cheaper than major clouds                        │
│  └── Less reliability guarantees                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Academic/HPC Resources

```
┌─────────────────────────────────────────────────────────────┐
│                   ACADEMIC GPU ACCESS                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Northeastern Discovery:                                     │
│  ├── gpu partition: V100, A100, H200                        │
│  ├── Request via SLURM: sbatch --partition=gpu              │
│  └── Check availability: sinfo -p gpu                       │
│                                                              │
│  Google Cloud Research Credits:                              │
│  ├── Apply for research credits                             │
│  └── Up to $5,000 for students                              │
│                                                              │
│  AWS Research Credits:                                       │
│  ├── AWS Cloud Credit for Research program                  │
│  └── Various amounts available                              │
│                                                              │
│  NSF ACCESS (formerly XSEDE):                               │
│  ├── Allocations for research                               │
│  └── Large-scale GPU clusters                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Recommendations by Use Case

### Development & Debugging

```
Recommended: Apple Silicon (M1/M2/M3) or Colab Free (T4)

Why:
├── No cost (already own laptop, Colab is free)
├── Fast iteration cycles for small tests
├── Sufficient for debugging code logic
└── Can verify training works before scaling up

Config:
├── batch_size: 16-64
├── max_steps: 100-500 (quick test)
└── Verify loss decreases, no errors
```

### Paper Reproduction (Exact Settings)

```
Recommended: T4 (Colab) or A100 (if available)

Why:
├── Paper uses batch=16, need exact settings
├── T4 handles batch=16 fine
├── A100 for faster runs with same settings
└── Cost-effective for single runs

Config:
├── batch_size: 16 (paper setting)
├── learning_rate: 2e-5 (paper setting)
├── Use paper's exact hyperparameters
└── Takes longer but matches paper
```

### Hyperparameter Search

```
Recommended: A100 or H100

Why:
├── Need many fast runs
├── Large batch + scaled LR for speed
├── A100: 20 experiments/hour at batch=256
├── H100: 30+ experiments/hour at batch=512
└── Worth the cost for thorough search

Config:
├── batch_size: 256-512
├── learning_rate: scaled appropriately
├── num_epochs: 2 (enough for comparison)
└── Run many configs in parallel if possible
```

### Final Training (Best Model)

```
Recommended: A100 40GB or V100 32GB

Why:
├── Use smaller batch for best generalization
├── Single longer run (not many experiments)
├── A100 faster, V100 cheaper
└── Time investment worth it for final model

Config:
├── batch_size: 16-64 (smaller for quality)
├── learning_rate: appropriately scaled
├── num_epochs: 3-5
└── Full training for best results
```

### LLM Training (7B+ Parameters)

```
Recommended: H100 or H200 (multi-GPU)

Why:
├── Large models need 80GB+ per GPU
├── H200's 141GB ideal for 70B models
├── Multi-GPU for distributed training
└── High bandwidth crucial for large batches

Note: Our BERT→DistilBERT task doesn't need this.
This is for future reference if scaling to LLMs.
```

---

## Quick Reference Card

```
┌────────────────────────────────────────────────────────────────┐
│                    GPU SELECTION CHEAT SHEET                    │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  BUDGET CONSCIOUS:                                              │
│  ├── Free: Colab (T4) or own laptop (M1/M2/M3)                 │
│  ├── Cheap: Lambda Labs A100 ($1.10/hr)                        │
│  └── Academic: Discovery HPC (if available)                    │
│                                                                 │
│  SPEED PRIORITY:                                                │
│  ├── Best: H200 or H100 (if available)                         │
│  ├── Great: A100 80GB                                          │
│  └── Good: A100 40GB                                           │
│                                                                 │
│  MEMORY PRIORITY (large models):                                │
│  ├── Maximum: H200 (141 GB)                                    │
│  ├── Large: A100 80GB or H100 (80 GB)                          │
│  └── Medium: A100 40GB or V100 32GB                            │
│                                                                 │
│  FOR OUR WORKLOAD (BERT→DistilBERT):                           │
│  ├── Sweet spot: A100 40GB (fast, not overkill)                │
│  ├── Budget: T4 on Colab (slower but free)                     │
│  └── Overkill: H100/H200 (designed for LLMs)                   │
│                                                                 │
│  RULE OF THUMB:                                                 │
│  ├── Model < 1B params: T4/A100 sufficient                     │
│  ├── Model 1-10B params: A100 80GB / H100                      │
│  └── Model 10B+ params: H100/H200 multi-GPU                    │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```
