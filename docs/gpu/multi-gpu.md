# Multi-GPU Training

Strategies for scaling Forge training beyond a single GPU.

---

## When You Need Multi-GPU

Single-GPU training hits limits when:

1. **Model + activations don't fit** - Even with LoRA, VLM generation with large images can exhaust 140GB (H200)
2. **Training is too slow** - Throughput scales linearly with GPUs for data-parallel strategies
3. **Effective batch size needs more memory** - GRPO with many generations per prompt

For Forge's current workloads (Qwen3-VL-2B with LoRA), the main bottleneck is activation memory during generation, not model size.

---

## Strategy Overview

```
                       Multi-GPU Strategies
                              |
        ┌─────────────┬───────┴───────┬─────────────┐
        |             |               |             |
  Independent    Data Parallel   Model Parallel  Pipeline
  Sharding       (DDP/FSDP)     (Tensor/ZeRO)   Parallel
        |             |               |             |
  Separate jobs  Each GPU gets   Model layers    Model stages
  no comms       full model +    split across    on different
                 data subset     GPUs            GPUs
```

### Independent Sharding (Simplest)

Not multi-GPU in the distributed sense, but the most practical way to use multiple GPUs for inference workloads. Each GPU runs as a separate SLURM job with no communication.

```bash
# Split 1355 pages across 4 GPUs
for i in 1 2 3 4; do
  sbatch run_bench.sh omnidoc --checkpoint outputs/sft/ --shard $i/4
done
# Each job processes ~339 pages independently
# Score-only job aggregates after all complete
```

Closer to MapReduce than DDP: embarrassingly parallel map (4 inference jobs), single reduce (score job). No synchronization, no shared state, any shard can fail and restart independently.

**When to use:** Inference, benchmarking, any workload where samples are independent.
**Not suitable for:** Training (needs gradient synchronization).

### Decision Matrix

| Strategy | Best For | Forge Use Case | Complexity |
|----------|----------|----------------|------------|
| **DDP** | Models that fit on 1 GPU | Faster SFT/KD training | Low |
| **FSDP** | Models near GPU memory limit | VLM GRPO with more generations | Medium |
| **DeepSpeed ZeRO-2** | Same as FSDP, better tooling | VLM GRPO, large batch | Medium |
| **DeepSpeed ZeRO-3** | Models that don't fit on 1 GPU | Future: larger VLMs (7B+) | High |
| **Tensor Parallel** | Latency-sensitive generation | Not needed at 2B scale | High |

---

## Data Parallel (DDP)

Simplest multi-GPU strategy. Each GPU holds a full copy of the model, processes different data, and synchronizes gradients.

### When to Use

- Model fits on a single GPU (Forge's 2B VLM with LoRA does)
- You want faster throughput, not more memory

### How It Works

```
GPU 0: Full Model + Batch A -> Gradients A ─┐
                                             ├─ AllReduce -> Update
GPU 1: Full Model + Batch B -> Gradients B ─┘
```

### Forge Setup

```bash
# SLURM: request 2 GPUs
#SBATCH --gres=gpu:h200:2

# Launch with torchrun
torchrun --nproc_per_node=2 forge.py --config configs/...
```

HuggingFace Trainer detects multiple GPUs automatically and uses DDP.

### Memory Impact

No reduction in per-GPU memory. Each GPU still holds the full model, optimizer, and activations. But you process 2x more data per step, so you can halve `gradient_accumulation_steps`.

---

## FSDP (Fully Sharded Data Parallel)

Shards model parameters, gradients, and optimizer states across GPUs. Each GPU only holds a fraction of the model at rest, but gathers full layers on-demand for computation.

### When to Use

- Model + activations are tight on a single GPU
- You need both more memory AND more throughput
- This is the best fit for Forge's VLM GRPO OOM issues

### How It Works

```
At rest:
  GPU 0: Shard 0 of params/grads/optimizer
  GPU 1: Shard 1 of params/grads/optimizer

During forward/backward:
  1. AllGather: collect full layer params from all GPUs
  2. Compute: forward/backward on local data
  3. ReduceScatter: shard gradients back
  4. Discard non-local params (frees memory)
```

### Memory Savings

With N GPUs, each GPU holds approximately:
- **Params**: 1/N of model weights
- **Gradients**: 1/N
- **Optimizer states**: 1/N
- **Activations**: Still full (per-GPU batch), but batch can be smaller

For Forge's 2B model with LoRA on 2x H200:
- Optimizer/gradient memory: halved
- Activation headroom: significantly more room for GRPO generations

### Forge Setup

```yaml
# In config or as env/CLI args
training:
  fsdp: "full_shard auto_wrap"
  fsdp_config:
    fsdp_transformer_layer_cls_to_wrap: "Qwen2VLDecoderLayer"
```

```bash
#SBATCH --gres=gpu:h200:2
#SBATCH --ntasks-per-node=2

torchrun --nproc_per_node=2 forge.py \
  --config configs/ocr/olmocr-mix/qwen3vl-2b/rl/02_omnidoc.yaml
```

---

## DeepSpeed ZeRO

Similar concept to FSDP but with more granular control. Three stages of increasing memory savings.

### ZeRO Stages

```
Stage 1: Shard optimizer states only
  Memory savings: ~4x for Adam (2 states per param)
  Communication: Low overhead

Stage 2: Shard optimizer + gradients
  Memory savings: ~8x
  Communication: Moderate

Stage 3: Shard optimizer + gradients + parameters
  Memory savings: ~N× (linear in GPU count)
  Communication: High (AllGather for every forward/backward)
```

### Forge Setup

Create `ds_config.json`:

```json
{
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": { "device": "none" },
    "allgather_partitions": true,
    "reduce_scatter": true
  },
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto"
}
```

```bash
torchrun --nproc_per_node=2 forge.py \
  --config configs/... \
  --deepspeed ds_config.json
```

### ZeRO-Offload (CPU offloading)

For extreme memory pressure, offload optimizer states to CPU RAM:

```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    }
  }
}
```

Trades PCIe bandwidth for GPU memory. Only worth it when you're close to fitting but not quite.

---

## GRPO-Specific Considerations

GRPO is uniquely memory-hungry because of the generation phase:

```
Standard training:
  Forward(batch) -> Backward(batch) -> Update

GRPO training:
  Generate(batch x num_gen) -> Score(rewards) -> Forward(batch x num_gen) -> Backward -> Update
        |                                              |
   KV cache for all              All sequences need
   sequences in parallel         log-probs simultaneously
```

### The Generation Bottleneck

Multi-GPU helps GRPO in two ways:

1. **FSDP/ZeRO**: More memory per GPU for KV cache during generation
2. **DDP**: Each GPU generates for its own data subset (but doesn't reduce per-sample memory)

For Forge's OmniDocBench GRPO (batch=1, num_gen=4 OOM on H200):
- **FSDP with 2 GPUs**: Would free ~35GB of optimizer/param memory, likely enough for num_gen=4
- **DDP with 2 GPUs**: Would NOT help (same per-GPU memory pressure)

### Alternative: generation_batch_size

Before going multi-GPU, check if trl supports chunked generation:

```python
# If supported, generate 1-at-a-time but collect num_generations total
GRPOConfig(
    num_generations=4,
    generation_batch_size=1,  # Generate sequentially, lower peak memory
)
```

This avoids the multi-GPU complexity entirely and is the preferred first approach.

---

## SLURM Configuration

### Requesting Multiple GPUs

```bash
#SBATCH --gres=gpu:h200:2          # 2 GPUs on same node
#SBATCH --ntasks-per-node=2        # 1 task per GPU
#SBATCH --cpus-per-task=4          # CPUs per GPU process
#SBATCH --mem=128G                 # More memory for 2 processes
```

### Multi-Node (Future)

```bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:h200:1          # 1 GPU per node
#SBATCH --ntasks-per-node=1

# Requires NCCL setup across nodes
srun torchrun --nnodes=2 --nproc_per_node=1 forge.py --config ...
```

### GPU Limit Considerations

Explorer cluster limits: `QOSMaxJobsPerUserLimit` applies per-user. Using 2 GPUs per job means fewer concurrent jobs. For exploratory runs, single-GPU with reduced num_generations may be more practical than blocking 2 GPU slots.

---

## When Multi-GPU Is Absolutely Necessary

Multi-GPU is a last resort. It adds complexity (distributed comms, NCCL debugging, multi-GPU SLURM scheduling) and halves your concurrent job capacity. Only go multi-GPU when single-GPU optimizations are exhausted.

**You need multi-GPU when:**

1. **Model doesn't fit on 1 GPU at all** - Full fine-tune of 7B+ models (weights + optimizer + gradients > 140GB). LoRA avoids this for most models up to ~7B.
2. **Generation OOMs after all single-GPU fixes** - You've already applied `max_visual_tokens`, `generation_batch_size`, reduced `max_completion_length`, and it still doesn't fit. This means KV cache alone exceeds free memory.
3. **Training throughput is the bottleneck, not memory** - You need faster wall-clock time and are willing to trade GPU slots for it (DDP for 2x throughput).

**You do NOT need multi-GPU when:**

- OOM during generation - try `generation_batch_size: 1` first
- OOM from large images - try `max_visual_tokens` cap first
- OOM during forward/backward - try gradient checkpointing first
- GPU idle during reward scoring - fix with parallel reward computation
- Slow training - check if the bottleneck is actually CPU-side (reward scoring, data loading)

### What Forge Has Built to Avoid Multi-GPU

These single-GPU optimizations are applied before considering multi-GPU:

| Optimization | What It Does | Where |
|---|---|---|
| `max_visual_tokens` | Caps image resolution to limit visual tokens per image. Processor computes `max_pixels` from its own patch/merge sizes, so it's model-agnostic. | `configs/ocr/base.yaml`, wired in `forge.py` and `datasets.py` |
| `generation_batch_size` | Generates completions one at a time instead of all `num_generations` at once. Decouples generation memory from `num_generations`. trl-native. | `02_omnidoc.yaml`, passed through `GRPOConfig` |
| Parallel reward scoring | `ParallelRewardGroup` runs all reward functions concurrently (across-function). `_score_parallel` threads per-completion scoring (within-function). Cuts GPU idle during reward phase. | `src/training/rewards.py` |
| Eval disabled for GRPO | `eval_strategy: "no"` avoids 70+ min eval passes that idle the GPU. Monitor via W&B reward logs instead. | `02_omnidoc.yaml` |
| LoRA | Only trains ~0.1% of parameters, cutting optimizer/gradient memory by 99%+. | `configs/ocr/olmocr-mix/base.yaml` |
| Gradient accumulation | Simulates large effective batch (32) with small per-step batch (1). No extra memory. | Config `batch_size` + `gradient_accumulation_steps` |

### Decision Flowchart

```
OOM during GRPO training?
  |
  +-> Cap image tokens (max_visual_tokens) --> Still OOM?
       |
       +-> Chunk generation (generation_batch_size: 1) --> Still OOM?
            |
            +-> Reduce max_completion_length --> Still OOM?
                 |
                 +-> Enable gradient checkpointing --> Still OOM?
                      |
                      +-> FSDP with 2 GPUs (now it's necessary)
```

## Recommendations for Forge

### Current State (2B VLM, single H200)

| Workload | GPUs | Strategy | Why |
|----------|------|----------|-----|
| SFT (LoRA) | 1 | None needed | Fits easily |
| KD | 1 | None needed | Teacher frozen, fits |
| GRPO (synthmix) | 1 | None needed | Rendered PDFs, fits at batch=2 |
| GRPO (OmniDocBench) | 1 | `max_visual_tokens` + `generation_batch_size` | Single-GPU fixes sufficient |
| GRPO (OmniDocBench) | 2 | FSDP | Only if single-GPU fixes fail |

### Future (7B+ VLMs)

| Workload | GPUs | Strategy |
|----------|------|----------|
| SFT (LoRA) | 1-2 | DDP or FSDP |
| GRPO | 2-4 | FSDP + chunked generation |
| Full fine-tune | 4+ | DeepSpeed ZeRO-3 |

---

## References

- [HuggingFace FSDP Guide](https://huggingface.co/docs/transformers/fsdp)
- [DeepSpeed ZeRO Tutorial](https://www.deepspeed.ai/tutorials/zero/)
- [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)

---

*See also: [memory-math.md](memory-math.md), [utilization.md](utilization.md)*
