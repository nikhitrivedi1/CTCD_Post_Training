# Memory Landscape: Ceiling vs Optimization

How GPU memory requirements and memory-saving techniques interact across Forge's model and training configurations. Two orthogonal axes determine whether a workload fits on a given GPU setup.

---

## The Two Axes

```
Memory Ceiling (what the workload demands)
    ▲
    │
    │   ┌─────────────────────────────────────┐
    │   │                                     │  Doesn't fit
    │   │         MULTI-GPU REQUIRED          │  (even with all optimizations)
    │   │                                     │
    │   ├─────────────────────────────────────┤ ← Hardware capacity (e.g. 140GB H200)
    │   │                                     │
    │   │         FITS WITH OPTIMIZATION      │  Apply techniques to
    │   │                                     │  push ceiling below line
    │   ├─────────────────────────────────────┤
    │   │                                     │
    │   │         FITS COMFORTABLY            │  No optimization needed
    │   │                                     │
    │   └─────────────────────────────────────┘
    └──────────────────────────────────────────────→
                Memory Optimization (techniques applied)
```

**Memory Ceiling** = total GPU memory a workload *needs* before any optimization. Driven by model size, precision, training mode, batch/sequence dimensions, and VLM-specific factors.

**Memory Optimization** = techniques that reduce effective memory usage. Each has a cost (slower training, fewer trainable params, reduced quality). Stack them until the ceiling drops below hardware capacity.

The goal: find the cheapest combination of optimizations that fits your workload on your hardware.

---

## Axis 1: Memory Ceiling

### Components

Every training workload's memory footprint breaks down into:

```
Total Memory = Weights + Gradients + Optimizer States + Activations + Overhead
               ├── Fixed ──────────────────────────┤   ├── Variable ────────┤
               (depends on model + training mode)       (depends on batch,
                                                         sequence, images)
```

| Component | Full Fine-Tune (bf16 + AdamW) | LoRA (bf16 + AdamW) |
|-----------|-------------------------------|---------------------|
| Weights | 2 bytes/param | 2 bytes/param (frozen, no grads) |
| Gradients | 4 bytes/param | 4 bytes/trainable param only |
| Optimizer (AdamW) | 8 bytes/param (m + v, fp32) | 8 bytes/trainable param only |
| Master weights | 4 bytes/param (fp32 copy) | 4 bytes/trainable param only |
| **Per-param total** | **18 bytes/param** | **2 bytes/param + 16 bytes/trainable** |

### Model Ceilings (Fixed Costs)

Using the Qwen3-VL and Qwen3.5 model families:

| Model | Params | Weights (bf16) | Full FT (AdamW bf16) | LoRA (~0.1%) |
|-------|--------|---------------|----------------------|--------------|
| Qwen3-VL-2B | 2.13B | 4.3 GB | ~38 GB | ~4.3 GB + ~34 MB |
| Qwen3-VL-4B | 4.0B | 8.0 GB | ~72 GB | ~8.0 GB + ~64 MB |
| Qwen3-VL-8B | 8.2B | 16.4 GB | ~148 GB | ~16.4 GB + ~131 MB |
| Qwen3-VL-32B | 32B | 64 GB | ~576 GB | ~64 GB + ~512 MB |
| Qwen3.5-4B | 4B | 8 GB | ~72 GB | ~8 GB + ~64 MB |
| Qwen3.5-9B | 9B | 18 GB | ~162 GB | ~18 GB + ~144 MB |
| Qwen3.5-27B | 27B | 54 GB | ~486 GB | ~54 GB + ~432 MB |

**Key takeaway:** Full fine-tuning at 8B+ exceeds a single H200 (140GB). LoRA collapses optimizer/gradient costs by ~1000x, making the model weights themselves the dominant fixed cost. Even with LoRA, Qwen3-VL-32B's weights alone (64 GB) consume nearly half the H200.

### Variable Costs: Activations

Activation memory scales with batch size, sequence length, and model depth:

```
Activations ~ batch_size x seq_len x hidden_dim x num_layers x 2 bytes
```

For VLMs, sequence length includes visual tokens, which can be thousands per image. A single high-res document scan might produce 3000+ visual tokens.

| Scenario | Typical Activation Cost |
|----------|------------------------|
| SFT, batch=1, 2K tokens | ~2-4 GB |
| SFT, batch=4, 2K tokens | ~8-16 GB |
| GRPO generation, batch=1 x 4 completions | ~15-30 GB (KV cache dominated) |
| GRPO generation, high-res document image | ~30-50 GB |

### VLM-Specific Ceiling Multipliers

Vision-language models add unique memory pressures beyond standard LLM training:

| Factor | Impact | Why |
|--------|--------|-----|
| Visual tokens per image | +2-10 GB per image | High-res documents produce thousands of patch tokens |
| KV cache during generation | Dominates GRPO memory | Grows with `max_completion_length x num_generations` |
| Vision encoder | +0.5-2 GB | Frozen but still loaded in memory |
| Image preprocessing buffers | ~0.5 GB | Pixel tensors before encoding |

### Ceiling Summary by Workload

Estimated peak GPU memory (bf16, LoRA unless noted):

| Workload | Qwen3-VL-2B | Qwen3-VL-8B | Qwen3-VL-32B | Qwen3.5-27B |
|----------|:-----------:|:-----------:|:------------:|:-----------:|
| Inference only | ~6 GB | ~20 GB | ~68 GB | ~58 GB |
| SFT (LoRA, batch=1) | ~10 GB | ~24 GB | ~72 GB | ~62 GB |
| SFT (full FT, batch=1) | ~42 GB | ~152 GB | ~580 GB | ~492 GB |
| KD (LoRA, teacher frozen) | ~14 GB | ~28 GB | ~76 GB | ~66 GB |
| GRPO (LoRA, num_gen=4) | ~50-80 GB | ~80-120 GB | ~120-160 GB | ~120-170 GB |
| GRPO (LoRA, num_gen=4, high-res) | ~80-120 GB | ~120-160 GB | ~160-200 GB | ~170-220 GB |

*These are rough estimates. Actual usage varies with sequence length, image resolution, and framework overhead.*

---

## Axis 2: Memory Optimization

### Technique Inventory

Ordered from cheapest (least impact on quality/speed) to most expensive:

| # | Technique | Memory Reduction | Speed Cost | Quality Cost | Applies To |
|---|-----------|:----------------:|:----------:|:------------:|------------|
| 1 | **Mixed precision (bf16)** | ~40% vs fp32 | Faster | None | All |
| 2 | **LoRA** | ~90-99% (grad/optim) | Neutral | Minor | All training |
| 3 | **Gradient accumulation** | Proportional | Neutral | None | All training |
| 4 | **Visual token cap** (`max_visual_tokens`) | Variable, up to ~50% activation reduction | Neutral | Reduced OCR on fine text | VLM workloads |
| 5 | **Chunked generation** (`generation_batch_size`) | Decouples peak mem from `num_gen` | Slower generation | None | GRPO |
| 6 | **Gradient checkpointing** | ~50-90% activations | -20-30% throughput | None | All training |
| 7 | **Reduced `max_completion_length`** | Linear KV cache reduction | N/A | Truncated outputs | GRPO |
| 8 | **FSDP / ZeRO-2** | ~N x reduction (optimizer + grads) | Communication overhead | None | Multi-GPU |
| 9 | **ZeRO-3** | ~N x reduction (all params) | High communication | None | Multi-GPU |
| 10 | **CPU offloading** | Moves optimizer to RAM | PCIe bottleneck | None | Near-fit cases |

### Optimization Stacking

Techniques are largely independent and composable. Apply them in order until the workload fits:

```
Start: Raw ceiling (full precision, full fine-tune, no tricks)
  │
  ├─ [1] bf16                    ─── Always on. Free performance + memory.
  ├─ [2] LoRA                   ─── Default for Forge. Cuts optimizer/grad by 99%.
  ├─ [3] Grad accumulation      ─── Already used. Effective batch=32 with micro-batch=1.
  │
  │   ── Usually sufficient for SFT/KD ──
  │
  ├─ [4] Visual token cap       ─── Forge default (max_visual_tokens in config).
  ├─ [5] Chunked generation     ─── Forge default (generation_batch_size: 4).
  │
  │   ── Usually sufficient for GRPO at 2B ──
  │
  ├─ [6] Gradient checkpointing ─── Trade 20-30% speed for activation memory.
  ├─ [7] Shorter completions    ─── Last resort before multi-GPU.
  │
  │   ── Single-GPU optimizations exhausted ──
  │
  ├─ [8] FSDP / ZeRO-2 (2+ GPUs) ─── Shard optimizer + grads.
  ├─ [9] ZeRO-3 (2+ GPUs)        ─── Shard everything.
  └─ [10] CPU offload              ─── Bridge small memory gaps.
```

---

## The Intersection: Fitting Workloads to Hardware

### Single H200 (140 GB)

| Workload | Optimizations Applied | Est. Peak | Fits? |
|----------|----------------------|:---------:|:-----:|
| SFT (LoRA), Qwen3-VL-2B | bf16 + LoRA + grad accum | ~10 GB | Yes |
| SFT (LoRA), Qwen3-VL-8B | bf16 + LoRA + grad accum | ~24 GB | Yes |
| SFT (LoRA), Qwen3-VL-32B | bf16 + LoRA + grad accum | ~72 GB | Yes |
| SFT (LoRA), Qwen3.5-27B | bf16 + LoRA + grad accum | ~62 GB | Yes |
| SFT (full FT), Qwen3-VL-2B | bf16 + grad accum | ~42 GB | Yes |
| SFT (full FT), Qwen3-VL-8B | bf16 + grad accum | ~152 GB | **No** |
| SFT (full FT), Qwen3-VL-32B | bf16 + grad accum | ~580 GB | **No** |
| SFT (full FT), Qwen3.5-27B | bf16 + grad accum | ~492 GB | **No** |
| GRPO (LoRA), Qwen3-VL-2B | bf16 + LoRA + visual cap + chunked gen | ~60-80 GB | Yes (tight) |
| GRPO (LoRA), Qwen3-VL-8B | All single-GPU optimizations | ~100-130 GB | Marginal |
| GRPO (LoRA), Qwen3-VL-32B | All single-GPU optimizations | ~130-170 GB | **No** |
| GRPO (LoRA), Qwen3.5-27B | All single-GPU optimizations | ~130-170 GB | **No** |

### 2x H200 (FSDP, ~240 GB effective)

FSDP shards optimizer states and gradients. Per-GPU memory drops by roughly half for those components, but activations remain per-GPU.

| Workload | Fits on 1 GPU? | Fits on 2 GPUs (FSDP)? |
|----------|:--------------:|:----------------------:|
| SFT (full FT), Qwen3-VL-8B | No (~152 GB) | Yes (~84 GB/GPU) |
| GRPO (LoRA), Qwen3-VL-8B | Marginal | Yes (~70 GB/GPU) |
| SFT (LoRA), Qwen3-VL-32B | Yes (~72 GB) | Yes (~40 GB/GPU) |
| GRPO (LoRA), Qwen3.5-27B | No | Marginal (~100 GB/GPU) |

### 4x H200 (FSDP/ZeRO-3, ~480 GB effective)

| Workload | Fits on 2 GPUs? | Fits on 4 GPUs? |
|----------|:--------------:|:----------------------:|
| SFT (full FT), Qwen3-VL-8B | Yes (~84 GB/GPU) | Yes (~45 GB/GPU) |
| SFT (full FT), Qwen3.5-27B | No (~250 GB/GPU) | Yes (~130 GB/GPU) |
| SFT (full FT), Qwen3-VL-32B | No (~295 GB/GPU) | **No** (~150 GB/GPU) |
| GRPO (LoRA), Qwen3-VL-32B | Marginal | Yes (~55 GB/GPU) |
| GRPO (LoRA), Qwen3.5-27B | Marginal | Yes (~65 GB/GPU) |

### 8x H200 (ZeRO-3, ~960 GB effective)

| Workload | Fits on 4 GPUs? | Fits on 8 GPUs? |
|----------|:--------------:|:----------------------:|
| SFT (full FT), Qwen3-VL-32B | No (~150 GB/GPU) | Yes (~80 GB/GPU) |
| SFT (full FT), Qwen3.5-27B | Yes (~130 GB/GPU) | Yes (~70 GB/GPU) |

### MoE Models: Different Math

MoE models store all expert weights but only activate a subset per token:

| Model | Total Params | Active Params | Experts (active/total) | Weights (bf16) | Active Compute |
|-------|:------------:|:-------------:|:---------------------:|:--------------:|:--------------:|
| Qwen3-VL-30B-A3B | 30B | 3B | - | ~60 GB | Like a 3B model |
| Qwen3.5-35B-A3B | 35B | 3B | - | ~70 GB | Like a 3B model |
| Qwen3.5-122B-A10B | 122B | 10B | 9/256 | ~244 GB | Like a 10B model |
| Qwen3-VL-235B-A22B | 235B | 22B | - | ~470 GB | Like a 22B model |
| Qwen3.5-397B-A17B | 397B | 17B | - | ~794 GB | Like a 17B model |

Memory is driven by *total* params (all experts loaded), but compute and activation memory scale with *active* params. This makes MoE models memory-heavy but compute-light relative to their total size. LoRA on MoE models typically targets the shared attention layers, not the expert MLPs.

---

## Decision Framework

### Step 1: Calculate your ceiling

```
Fixed = model_params x bytes_per_param (see table above)
      + trainable_params x 16 (gradients + optimizer, AdamW bf16)

Variable = activation_estimate (batch, seq_len, image resolution)
         + KV_cache (GRPO: num_gen x max_completion_len x hidden x layers x 4 bytes)

Ceiling = Fixed + Variable
```

### Step 2: Apply optimizations until it fits

Walk down the stacking order. Stop when `Ceiling_optimized < GPU_capacity x 0.85` (leave 15% for framework overhead and fragmentation).

### Step 3: If single-GPU optimizations aren't enough

```
Ceiling after all single-GPU optimizations > 140 GB?
  │
  ├─ By how much?
  │   ├─ < 2x over  →  2 GPUs, FSDP
  │   ├─ 2-4x over  →  4 GPUs, FSDP or ZeRO-2
  │   └─ > 4x over  →  4-8 GPUs, ZeRO-3
  │
  └─ Is the bottleneck optimizer/grads or activations?
      ├─ Optimizer/grads  →  FSDP/ZeRO shards these well
      └─ Activations      →  Gradient checkpointing first, then FSDP
```

---

## Forge's Current Position

```
                    Memory Ceiling
                    ▲
                    │                          ● Full FT 32B (~580 GB)
        500 GB ─    │               ● Full FT 27B
                    │
                    │
        200 GB ─    │          ● GRPO 27B/32B (LoRA)
                    │       ● Full FT 8B
  H200 (140GB) ─ ──│── ── ── ── ── ── ── ── ── ── ── hardware line
                    │    ● GRPO 8B (LoRA)
        100 GB ─    │
                    │ ● GRPO 2B (LoRA, high-res)
                    │ ● SFT 32B (LoRA)
         50 GB ─    │ ● Full FT 2B
                    │
                    │ ● SFT 8B (LoRA)
         20 GB ─    │ ● SFT 2B (LoRA)
                    │
                    └──────────────────────────────────→
                      none    bf16   LoRA   visual  chunked  grad    FSDP
                                            cap     gen      ckpt    2-4 GPU
                                     Memory Optimization →
```

We're currently operating at the "GRPO 2B (LoRA, high-res)" point, just below the hardware line with all single-GPU optimizations applied. Scaling to 8B+ GRPO or 8B+ full fine-tuning pushes above the line. Even LoRA SFT on 32B sits at ~72 GB, comfortable on one H200 but leaving little room for GRPO's generation overhead. Multi-GPU is the next step.

---

*See also: [memory-math.md](memory-math.md) (per-component formulas), [multi-gpu.md](multi-gpu.md) (distributed strategies), [parallelization.md](parallelization.md) (CPU/GPU workload mapping), [utilization.md](utilization.md) (throughput optimization)*
