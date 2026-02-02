# GPU Comparison Guide for Deep Learning

A comprehensive reference for GPUs commonly used in deep learning training.

---

## GPU Overview

### Generation Timeline

```
2017: V100 (Volta architecture)
2018: T4 (Turing architecture)
2020: A100 (Ampere architecture)
2022: H100 (Hopper architecture)
2024: H200 (Hopper architecture, more memory)
```

---

## Specifications Comparison

```
┌──────────┬────────┬───────────┬───────────┬──────────┬───────┐
│   GPU    │  VRAM  │ Bandwidth │ FP16 TF   │ BF16 TF  │  TDP  │
├──────────┼────────┼───────────┼───────────┼──────────┼───────┤
│ V100 16G │  16 GB │   900 GB/s│    125    │    -     │ 300W  │
│ T4       │  16 GB │   320 GB/s│     65    │    -     │  70W  │
│ A100 40G │  40 GB │ 1,555 GB/s│    312    │   312    │ 400W  │
│ A100 80G │  80 GB │ 2,039 GB/s│    312    │   312    │ 400W  │
│ H100 SXM │  80 GB │ 3,350 GB/s│    989    │   989    │ 700W  │
│ H200     │ 141 GB │ 4,800 GB/s│    989    │   989    │ 700W  │
└──────────┴────────┴───────────┴───────────┴──────────┴───────┘
```

---

## Performance for BERT→DistilBERT

```
┌──────────┬───────────┬────────────┬──────────────┐
│   GPU    │  Batch    │    Time    │  Est. Cost   │
├──────────┼───────────┼────────────┼──────────────┤
│ T4       │   128     │   ~1 hr    │  ~$0.50      │
│ V100     │   256     │  ~45 min   │  ~$2.25      │
│ A100 40G │   256     │  ~20 min   │  ~$1.30      │
│ A100 80G │   512     │  ~10 min   │  ~$0.80      │
│ H100     │  1024     │   ~5 min   │  ~$0.80      │
│ H200     │  1024     │   ~5 min   │  ~$1.00      │
└──────────┴───────────┴────────────┴──────────────┘
```

---

## Feature Support

```
┌──────────┬───────┬───────┬───────┬────────┬─────────┐
│   GPU    │ FP16  │ BF16  │ TF32  │  FP8   │ MIG     │
├──────────┼───────┼───────┼───────┼────────┼─────────┤
│ V100     │  ✓    │  ✗    │  ✗    │   ✗    │   ✗     │
│ T4       │  ✓    │  ✗    │  ✗    │   ✗    │   ✗     │
│ A100     │  ✓    │  ✓    │  ✓    │   ✗    │   ✓     │
│ H100     │  ✓    │  ✓    │  ✓    │   ✓    │   ✓     │
│ H200     │  ✓    │  ✓    │  ✓    │   ✓    │   ✓     │
└──────────┴───────┴───────┴───────┴────────┴─────────┘
```

---

## Availability Guide

### Cloud Providers

| Provider | T4 | V100 | A100 | H100 |
|----------|-----|------|------|------|
| Google Colab | Free/Pro | Pro | Pro+ | ✗ |
| AWS | g4dn ~$0.50/hr | p3 ~$3/hr | p4d ~$4/hr | p5 ~$12/hr |
| GCP | ~$0.35/hr | ~$2.50/hr | ~$4/hr | Limited |
| Lambda Labs | - | - | ~$1.10/hr | ~$2/hr |

### Academic Resources

- **Northeastern Discovery**: V100, A100, H200 (gpu partition)
- **Google Cloud Research Credits**: Up to $5,000 for students
- **NSF ACCESS**: Large-scale GPU clusters

---

## Recommendations by Use Case

### Development & Debugging
**Recommended:** Apple Silicon or Colab Free (T4)
- No cost, fast iteration for small tests

### Paper Reproduction
**Recommended:** T4 (Colab) or A100
- T4 handles batch=16 fine
- A100 for faster runs

### Hyperparameter Search
**Recommended:** A100 or H100
- Need many fast runs
- A100: 20 experiments/hour at batch=256

### Final Training
**Recommended:** A100 40GB or V100 32GB
- Use smaller batch for best generalization

---

## Quick Reference Card

```
┌────────────────────────────────────────────────────────────────┐
│                    GPU SELECTION CHEAT SHEET                    │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  BUDGET CONSCIOUS:                                              │
│  ├── Free: Colab (T4) or own laptop                            │
│  ├── Cheap: Lambda Labs A100 ($1.10/hr)                        │
│  └── Academic: Discovery HPC                                    │
│                                                                 │
│  SPEED PRIORITY:                                                │
│  ├── Best: H200 or H100                                        │
│  ├── Great: A100 80GB                                          │
│  └── Good: A100 40GB                                           │
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
