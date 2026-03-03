# Scaling Forge: From 1 GPU to 100,000

How to scale training from a single GPU debug run to a frontier-scale distributed system — what changes at each level, what stays the same, and why.

---

## Table of Contents

1. [Scale Spectrum](#scale-spectrum)
2. [Tier 0: Dev/Debug](#tier-0-devdebug)
3. [Tier 1: Single GPU](#tier-1-single-gpu)
4. [Tier 2: Single Node, Multi-GPU](#tier-2-single-node-multi-gpu)
5. [Tier 3: Multi-Node Cluster](#tier-3-multi-node-cluster)
6. [Tier 4: Large Cluster](#tier-4-large-cluster)
7. [Tier 5: Frontier Scale](#tier-5-frontier-scale)
8. [Cross-Cutting Concerns](#cross-cutting-concerns)
9. [Forge Support Matrix](#forge-support-matrix)

---

## Scale Spectrum

| Tier | GPUs | CPUs | Duration | Strategy | Forge Support |
|------|------|------|----------|----------|-------------------|
| **0** Dev/Debug | 1 | 1 | Minutes | Single process | Fully supported |
| **1** Single GPU | 1 | 8 | Hours | Single process, multi-worker data loading | Fully supported |
| **2** Single Node | 2–8 | 8–64 | Hours | DDP (data parallelism) | Works today, zero code changes |
| **3** Multi-Node | 16–100 | 100–800 | Days | DDP or FSDP across nodes | Works with launch command changes |
| **4** Large Cluster | 100–1K | 1K–10K | Weeks | FSDP / DeepSpeed ZeRO-3, tensor parallelism | Needs config additions |
| **5** Frontier | 10K–100K | 100K+ | Months | 3D parallelism (data + tensor + pipeline) | Out of scope (Megatron-LM territory) |

**Reference point:** Qwen3-VL-2B + LoRA on olmOCR-mix (5K samples), H200 GPU, ~6s/step.

---

## Tier 0: Dev/Debug

**Resources:** 1 CPU, 1 GPU, minutes

```bash
python forge.py --config configs/ocr/olmocr-mix/qwen3vl-2b/ablations/00_no-teacher_ce-only.yaml --debug
```

**What's happening:**
- Single Python process, single GPU
- `--debug` limits to small data subset, verbose logging, W&B Test project
- Validates: data loads, collator batches, forward pass runs, loss computes

**When to use:** First run of any new task/config. Catch shape mismatches, dtype errors, OOM before committing to a real run.

---

## Tier 1: Single GPU

**Resources:** 8 CPUs, 1 GPU (H200 140GB), hours

```bash
python forge.py --config configs/ocr/olmocr-mix/qwen3vl-2b/ablations/00_no-teacher_ce-only.yaml
```

**This is the current Forge default.** Everything runs in a single process.

### CPU Utilization

CPUs do two things:

**1. Data loading** — `dataloader_num_workers` (default: 4 on CUDA)

The HuggingFace Trainer spawns N worker processes to load and preprocess batches in parallel while the GPU trains on the current batch. This hides I/O latency:

```
Worker 0: loads batch 4 ──────────────────┐
Worker 1: loads batch 5 ──────────────────┤
Worker 2: loads batch 6 ──────────────────├─► ready in queue
Worker 3: loads batch 7 ──────────────────┘
GPU:      trains on batch 3 ──────────────────► grabs batch 4 from queue
```

Set `dataloader_num_workers` in your config's `training` section. Rule of thumb: 2–4 workers per GPU. More workers = more CPU RAM usage.

**2. Offline data prep** — `scripts/prepare_ocr_data.py --workers N`

PDF rendering is CPU-bound. The `--workers` flag uses `ProcessPoolExecutor` for parallel rendering. Defaults to `os.cpu_count()`.

```
8 workers, 5K PDFs: ~4 minutes  (18.93 pages/sec)
1 worker,  5K PDFs: ~25 minutes (3.3 pages/sec)
```

### GPU Memory Optimization

All of these work within a single GPU:

| Technique | What it does | Forge config |
|-----------|-------------|-----------------|
| **LoRA** | Train only ~1% of parameters | `training.use_lora: true` |
| **BF16** | Half-precision weights and activations | `training.fp16: false` + model loaded in bf16 |
| **Gradient accumulation** | Simulate large batches with small micro-batches | `training.gradient_accumulation_steps: 16` |
| **Gradient checkpointing** | Recompute activations during backward instead of storing them | `training.gradient_checkpointing: true` (TrainingArguments) |

**Memory budget (Qwen3-VL-2B + LoRA on H200):**

```
Model weights (bf16):     ~4 GB
LoRA parameters:          ~35 MB
Optimizer states (Adam):  ~140 MB  (2× LoRA params in fp32)
Activations + images:     ~20-40 GB  (depends on batch size, image resolution)
─────────────────────────────────────
Total:                    ~25-45 GB of 140 GB
```

### Fault Tolerance

Forge has built-in SLURM integration for long-running jobs:

**Checkpointing:** `save_steps: 500` saves model + optimizer + scheduler state. `save_total_limit: 3` keeps only the 3 most recent.

**SLURM signal handling:** When the job approaches its time limit, SLURM sends SIGUSR1 (configured via `#SBATCH --signal=B:SIGUSR1@120`). Forge's `SLURMSignalCallback` catches this and saves a checkpoint before the job is killed.

**Auto-resubmission:** `scripts/run_training.sh` reads `training_state.json` after each job. If training isn't complete, it resubmits with `resume_from_checkpoint` pointing to the latest checkpoint. This chains jobs across SLURM's time limits.

```
Job 1 (8 hrs) → checkpoint at step 4800 → SIGUSR1 → save → exit
Job 2 (8 hrs) → resume from step 4800 → checkpoint at step 9600 → ...
Job 3 (8 hrs) → resume from step 9600 → training complete → done
```

---

## Tier 2: Single Node, Multi-GPU

**Resources:** 8 CPUs, 2–8 GPUs (e.g., 8× H200), hours

```bash
# Instead of:
python forge.py --config ...

# Use:
torchrun --nproc_per_node=8 forge.py --config ...
```

**Zero code changes.** Here's why:

### What `torchrun` Does

`torchrun` spawns N identical copies of your script as separate OS processes. Before each process starts, it sets environment variables:

```
Process 0: RANK=0  LOCAL_RANK=0  WORLD_SIZE=8
Process 1: RANK=1  LOCAL_RANK=1  WORLD_SIZE=8
...
Process 7: RANK=7  LOCAL_RANK=7  WORLD_SIZE=8
```

Each process sees only its assigned GPU via `LOCAL_RANK`. That's all `torchrun` does — it doesn't modify your code.

### What HF Trainer Does Automatically

Trainer reads those environment variables and adapts:

**1. Device placement** — Each process puts the model on `cuda:{LOCAL_RANK}`:
```
Process 0 → cuda:0    Process 4 → cuda:4
Process 1 → cuda:1    Process 5 → cuda:5
Process 2 → cuda:2    Process 6 → cuda:6
Process 3 → cuda:3    Process 7 → cuda:7
```

**2. DDP wrapping** — Trainer wraps the model in `DistributedDataParallel`:
```python
# Trainer does this internally:
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
```

**3. Data splitting** — Trainer replaces the sampler with `DistributedSampler`:
```
5000 samples, 8 GPUs:
  GPU 0 sees: samples 0, 8, 16, 24, ...  (625 samples)
  GPU 1 sees: samples 1, 9, 17, 25, ...  (625 samples)
  ...
  GPU 7 sees: samples 7, 15, 23, 31, ... (625 samples)
```
Each GPU processes different data — no duplication.

**4. Gradient synchronization** — After each backward pass, DDP all-reduces (averages) gradients across all GPUs:
```
GPU 0: backward → ∇LoRA_0 ─┐
GPU 1: backward → ∇LoRA_1 ─┤
GPU 2: backward → ∇LoRA_2 ─┼─► all-reduce → ∇LoRA_avg → all GPUs update identically
...                         │
GPU 7: backward → ∇LoRA_7 ─┘
```

**5. Logging/saving** — Only rank 0 logs to W&B, saves checkpoints, prints progress. Processes 1–7 are silent.

### Effective Batch Size

`per_device_train_batch_size` is per GPU:

```
Effective batch = per_device_batch × num_GPUs × gradient_accumulation_steps

1 GPU:  2 × 1 × 16 = 32
4 GPUs: 2 × 4 × 16 = 128
8 GPUs: 2 × 8 × 16 = 256
```

To keep the same effective batch size when scaling GPUs, reduce `gradient_accumulation_steps`:

```yaml
# 1 GPU, effective batch = 32:
training:
  batch_size: 2
  gradient_accumulation_steps: 16

# 8 GPUs, effective batch = 32:
training:
  batch_size: 2
  gradient_accumulation_steps: 2     # 2 × 8 × 2 = 32
```

### Speed Scaling

Speedup is near-linear because all communication happens during the backward pass (overlapped with computation):

| GPUs | Steps/epoch | Time per step | Epoch time | Speedup |
|------|-------------|---------------|------------|---------|
| 1 | 157 | 6.0s | 15.7 min | 1× |
| 2 | 79 | 6.2s | 8.2 min | 1.9× |
| 4 | 40 | 6.5s | 4.3 min | 3.7× |
| 8 | 20 | 7.0s | 2.3 min | 6.8× |

Time per step increases slightly due to gradient sync overhead, but steps per epoch decreases proportionally with GPU count.

### Prerequisites

Your code must satisfy one rule: **the data collator must not hardcode device placement.** If the collator calls `.to("cuda:0")`, processes on GPUs 1–7 will crash. Return CPU tensors from the collator — Trainer moves them to the correct device.

Forge's `VLMDataCollator` already follows this rule (verified: no `.to("cuda")` calls).

### Checking SLURM Multi-GPU Access

Before requesting multiple GPUs, verify your cluster permissions:

```bash
# 1. What GPU nodes exist and how many GPUs per node?
sinfo -p gpu -o "%N %G %c %m"

# 2. What QOS (Quality of Service) do you have access to?
sacctmgr show assoc where user=$USER format=User,Account,QOS%30

# 3. What are the GPU limits for each QOS?
sacctmgr show qos format=Name%20,MaxTRESPerUser%30,MaxTRESPerJob%30

# 4. Quick test — does a multi-GPU request get accepted?
sbatch -p gpu --gres=gpu:h200:2 --wrap="nvidia-smi" --time=00:01:00
```

Common QOS patterns:
- `gpu`: Typically 1 GPU per job (general use)
- `multigpu`: Multiple GPUs per job (need to request access)
- `courses-gpu`: Academic course allocations (usually 1 GPU)

If your QOS limits you to 1 GPU per job, contact the HPC team or your advisor to request `multigpu` QOS access.

### SLURM Script for Multi-GPU

```bash
#!/bin/bash
#SBATCH --job-name=forge-ddp
#SBATCH --partition=gpu
#SBATCH --qos=multigpu            # Requires multigpu QOS access
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --signal=B:SIGUSR1@120

source ~/.zshrc && conda activate forge

# Detect number of GPUs allocated by SLURM
NGPUS=$(nvidia-smi -L | wc -l)

torchrun --nproc_per_node=$NGPUS \
    forge.py --config configs/ocr/olmocr-mix/qwen3vl-2b/ablations/00_no-teacher_ce-only.yaml
```

Note: `--ntasks-per-node=1` because `torchrun` handles process spawning — SLURM just launches the one `torchrun` process.

---

## Tier 3: Multi-Node Cluster

**Resources:** 100+ CPUs, 16–100 GPUs across multiple machines, days

```bash
# 4 nodes × 8 GPUs = 32 GPUs
torchrun \
    --nnodes=4 \
    --node_rank=$SLURM_NODEID \
    --nproc_per_node=8 \
    --master_addr=$MASTER_ADDR \
    --master_port=29500 \
    forge.py --config ...
```

### What Changes from Tier 2

**Same:** DDP mechanics, zero code changes, Trainer handles everything.

**New bottleneck: network.** Within a node, GPUs communicate via NVLink (900 GB/s on H200). Across nodes, communication goes over InfiniBand (typically 200–400 Gb/s) or Ethernet (25–100 Gb/s). Gradient sync becomes the bottleneck.

```
Intra-node (NVLink):   900 GB/s  ← almost free
Inter-node (IB HDR):   25 GB/s   ← 36× slower
Inter-node (Ethernet): 3 GB/s    ← 300× slower
```

### Gradient Sync Cost

For DDP, every backward pass all-reduces all trainable parameters. With LoRA:

```
LoRA trainable params: 17.4M × 4 bytes (fp32 grads) = ~70 MB per sync
At 25 GB/s InfiniBand: ~3 ms  ← negligible
```

LoRA makes multi-node DDP extremely efficient because the gradient payload is tiny. Full fine-tuning of a 2B model would be 2000M × 4 = ~8 GB per sync — much more painful.

### When DDP Isn't Enough

DDP replicates the full model on every GPU. If the model doesn't fit on a single GPU:

| Model Size | BF16 Memory | Fits on H200 (140GB)? | Strategy |
|-----------|-------------|----------------------|----------|
| 2B | ~4 GB | Yes | DDP |
| 7B | ~14 GB | Yes | DDP |
| 70B | ~140 GB | Barely (no room for activations) | FSDP / ZeRO-3 |
| 405B | ~810 GB | No | Must shard across GPUs |

**FSDP (Fully Sharded Data Parallel):** Shards model parameters, gradients, and optimizer states across GPUs. Each GPU holds only 1/N of the model. Parameters are gathered on-demand for forward/backward passes.

```bash
# FSDP via accelerate
accelerate launch \
    --config_file fsdp_config.yaml \
    --num_processes 32 \
    forge.py --config ...
```

**DeepSpeed ZeRO-2/3:** Similar sharding, config-only change:

```bash
torchrun --nproc_per_node=8 --nnodes=4 \
    forge.py --config ... --deepspeed ds_config.json
```

With a `ds_config.json`:
```json
{
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "overlap_comm": true,
        "reduce_scatter": true
    },
    "gradient_accumulation_steps": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto"
}
```

### SLURM Multi-Node Script

```bash
#!/bin/bash
#SBATCH --job-name=forge-multinode
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --signal=B:SIGUSR1@120

source ~/.zshrc && conda activate forge

# SLURM provides these
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29500

srun torchrun \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --nproc_per_node=8 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    forge.py --config ...
```

### Time Estimates (260K samples, Qwen3-VL-2B + LoRA)

| Setup | GPUs | Steps/epoch | Est. epoch time | Est. total (3 epochs) |
|-------|------|-------------|-----------------|----------------------|
| 1 node × 1 GPU | 1 | 8,125 | ~13.5 hrs | ~41 hrs |
| 1 node × 8 GPU | 8 | 1,016 | ~2 hrs | ~6 hrs |
| 2 nodes × 8 GPU | 16 | 508 | ~1 hr | ~3 hrs |
| 4 nodes × 8 GPU | 32 | 254 | ~35 min | ~1.7 hrs |

---

## Tier 4: Large Cluster

**Resources:** 1K CPUs, 100–1K GPUs, weeks

At this scale, you're training models with 70B–400B+ parameters. New concerns emerge.

### Tensor Parallelism

DDP and FSDP are **data parallelism** — each GPU processes different data through the same model. **Tensor parallelism** splits individual layers across GPUs:

```
Standard (1 GPU):
  Input → [Full Attention Layer] → [Full MLP Layer] → Output

Tensor Parallel (4 GPUs):
  Input → [Attn split across 4 GPUs] → [MLP split across 4 GPUs] → Output
```

Each GPU computes a portion of every matrix multiply. This is necessary when a single layer's activations exceed GPU memory.

Typically used **within a node** (requires NVLink bandwidth). Combined with data parallelism **across nodes**:

```
Node 0 (8 GPUs): TP=8, processes batch 0
Node 1 (8 GPUs): TP=8, processes batch 1
Node 2 (8 GPUs): TP=8, processes batch 2
...
```

### Pipeline Parallelism

Splits the model by layers across GPUs:

```
GPU 0: Layers 0–7    (forward, then idle, then backward)
GPU 1: Layers 8–15   (idle, then forward, then idle, then backward)
GPU 2: Layers 16–23  (idle, idle, then forward, then backward)
GPU 3: Layers 24–31  (idle, idle, idle, then forward+backward)
```

Naive pipeline parallelism has a "bubble" problem — GPUs sit idle waiting for their turn. Solutions like **GPipe** and **PipeDream** use micro-batching to fill the bubble:

```
GPU 0: [micro1 fwd] [micro2 fwd] [micro3 fwd] [micro4 fwd] [micro4 bwd] [micro3 bwd] ...
GPU 1:              [micro1 fwd] [micro2 fwd] [micro3 fwd] [micro4 fwd] [micro4 bwd] ...
GPU 2:                           [micro1 fwd] [micro2 fwd] [micro3 fwd] [micro4 fwd] ...
GPU 3:                                        [micro1 fwd] [micro2 fwd] [micro3 fwd] ...
```

### Elastic Training

At 1K GPUs, hardware failures are routine (mean time between failures drops from months to hours). Solutions:

- **TorchElastic** (`torchrun` already supports this): Automatically restarts failed processes, redistributes work
- **Checkpoint frequency**: Every N minutes, not every N steps — because a step takes longer but failures are time-based
- **Redundant checkpointing**: Save to distributed storage (not local SSD), replicate across nodes

### Data Pipeline

With 1K GPUs each consuming data, the data pipeline becomes a bottleneck:

- **Streaming datasets**: `datasets.load_dataset(..., streaming=True)` — don't download all data upfront
- **WebDataset**: Tar-based format designed for distributed training, sequential reads
- **Distributed file system**: Lustre, GPFS — parallel reads from shared storage

### Forge at This Scale

Forge's HF Trainer backbone supports FSDP and DeepSpeed out of the box, but you'd need:

1. **DeepSpeed ZeRO-3 config** for parameter sharding (config file only, no code changes)
2. **Tensor parallelism** via Megatron-DeepSpeed or HF's `device_map` with manual shard placement
3. **Streaming data loading** to replace `load_from_disk()` with streaming

---

## Tier 5: Frontier Scale

**Resources:** 10K–100K GPUs, 100K+ CPUs, months

This is where Google trains Gemini, Meta trains Llama, and Anthropic trains Claude. Included here for conceptual completeness — this is far beyond Forge's scope.

### 3D Parallelism

All three parallelism strategies combined:

```
                    ┌─────────────── Data Parallel ───────────────┐
                    │                                              │
          ┌────────┴────────┐                            ┌────────┴────────┐
          │   Replica 0     │                            │   Replica 1     │
          │                 │                            │                 │
    ┌─────┴─────┐    ┌─────┴─────┐                ┌─────┴─────┐    ┌─────┴─────┐
    │ Pipeline 0 │    │ Pipeline 1 │                │ Pipeline 0 │    │ Pipeline 1 │
    │ (Layers    │    │ (Layers    │                │ (Layers    │    │ (Layers    │
    │  0–15)     │    │  16–31)    │                │  0–15)     │    │  16–31)    │
    ├────────────┤    ├────────────┤                ├────────────┤    ├────────────┤
    │ TP across  │    │ TP across  │                │ TP across  │    │ TP across  │
    │ 8 GPUs     │    │ 8 GPUs     │                │ 8 GPUs     │    │ 8 GPUs     │
    │ (NVLink)   │    │ (NVLink)   │                │ (NVLink)   │    │ (NVLink)   │
    └────────────┘    └────────────┘                └────────────┘    └────────────┘
         Node 0            Node 1                        Node 2            Node 3
```

**Total GPUs in this example:** 4 nodes × 8 GPUs = 32 GPUs
- Tensor Parallel: 8-way (within each node)
- Pipeline Parallel: 2-way (across 2 nodes per replica)
- Data Parallel: 2-way (2 replicas processing different data)

Scale this up: TP=8, PP=8, DP=1562 → 8 × 8 × 1562 = 100,000 GPUs.

### Fault Tolerance at Scale

With 100K GPUs, at least one fails every few hours:

| GPUs | MTBF per GPU | Expected cluster failure rate |
|------|-------------|------------------------------|
| 8 | ~6 months | Once per ~22 days |
| 100 | ~6 months | Once per ~1.8 days |
| 1,000 | ~6 months | Once per ~4.4 hours |
| 100,000 | ~6 months | Once per ~2.6 minutes |

Solutions:
- **In-memory checkpointing**: Save to neighbor GPU's RAM (microseconds vs. minutes to disk)
- **Redundant computation**: Run the same data on 2 replicas, compare results
- **Graceful degradation**: Remove failed node, redistribute its shards, continue training
- **Async checkpointing**: Background checkpoint save while training continues

### Networking

At frontier scale, networking is the dominant engineering challenge:

- **InfiniBand NDR**: 400 Gb/s per port, typically 8 ports per node = 3.2 Tb/s per node
- **NVLink/NVSwitch**: 900 GB/s between GPUs within a node (DGX H200)
- **Rail-optimized topology**: Each GPU connects to a different network switch — avoids bottleneck at any single switch
- **Gradient compression**: Reduce communication volume by compressing gradients (top-k sparsification, quantization)

### Frameworks

At this scale, teams typically use specialized frameworks:

| Framework | Organization | Scale |
|-----------|-------------|-------|
| **Megatron-LM** | NVIDIA | 3D parallelism, up to 1T+ params |
| **FSDP2 / torchtitan** | Meta/PyTorch | Native PyTorch distributed |
| **Megascale** | ByteDance | 100K+ GPU training |
| **Pathways** | Google | TPU pods, multi-datacenter |
| **JAX/XLA** | Google | Compiler-driven parallelism on TPUs |

---

## Cross-Cutting Concerns

### Checkpointing

Checkpointing serves two purposes: **fault tolerance** (resume after crash) and **model selection** (pick best checkpoint).

**Forge config:**
```yaml
checkpointing:
  save_steps: 500        # Save every 500 optimizer steps
  save_total_limit: 3    # Keep only the 3 most recent checkpoints
```

**What gets saved:**
- Model weights (or LoRA adapter weights)
- Optimizer states (Adam momentum, variance)
- Learning rate scheduler state
- RNG states (for reproducibility)
- `training_state.json` (completed epochs/steps)

**Checkpoint size (Qwen3-VL-2B + LoRA):**
```
LoRA adapter:     ~70 MB
Optimizer states: ~140 MB
Scheduler + RNG:  ~1 MB
──────────────────────────
Total:            ~211 MB per checkpoint
```

Full fine-tuning would be ~4 GB per checkpoint (full model weights + gradients + optimizer).

**Resume from checkpoint:**
```bash
# Automatic: Forge checks for checkpoints in output_dir
python forge.py --config ... --resume_from_checkpoint outputs/run_name/checkpoint-500
```

### Gradient Checkpointing (Activation Recomputation)

Not to be confused with model checkpointing above. Gradient checkpointing trades compute for memory:

**Without gradient checkpointing:**
```
Forward:  store activations at every layer  → ~20 GB for 2B model
Backward: use stored activations to compute gradients
```

**With gradient checkpointing:**
```
Forward:  only store activations at checkpoint boundaries  → ~2 GB
Backward: recompute activations between checkpoints on-the-fly
```

**Tradeoff:** ~30% slower training, ~60-70% less activation memory.

Enable via TrainingArguments: `gradient_checkpointing=True`. Particularly useful for large batch sizes or long sequences where activations dominate memory.

### Mixed Precision

| Format | Bits | Range | Forge Usage |
|--------|------|-------|-----------------|
| **FP32** | 32 | Full | Optimizer states (Adam), loss computation |
| **BF16** | 16 | Wide range, lower precision | Model weights, forward/backward pass |
| **FP16** | 16 | Narrow range, higher precision | Alternative to BF16 (needs loss scaling) |
| **FP8** | 8 | Very narrow | Inference, some training (H200/H100 support) |

**BF16 is the default for modern training** on Ampere+ GPUs (A100, H100, H200). It halves memory for weights and activations with minimal accuracy loss. FP32 is still used for optimizer states and gradient accumulation to maintain numerical stability.

**FP8** is emerging for training (via Transformer Engine) but primarily used for inference today. The olmOCR teacher model loads as FP8 natively.

### CPU Scaling Summary

| What | How CPUs help | Forge config |
|------|--------------|-----------------|
| **Data loading** | Workers prefetch batches while GPU trains | `training.dataloader_num_workers: 4` |
| **Data prep** | Parallel PDF rendering | `scripts/prepare_ocr_data.py --workers N` |
| **CPU offloading** | Move optimizer states to CPU RAM (DeepSpeed ZeRO-Offload) | `ds_config.json: offload_optimizer.device: "cpu"` |
| **Tokenization** | HF `datasets.map()` uses multiple processes | Automatic (uses available CPUs) |

Rule of thumb: **4–8 CPUs per GPU** for data loading. More CPUs help with data prep but not training.

---

## Forge Support Matrix

| Feature | Status | What's needed |
|---------|--------|--------------|
| Single GPU training | **Supported** | Nothing — default mode |
| Multi-GPU DDP | **Supported** | Change `python` → `torchrun` |
| SLURM signal handling | **Supported** | Built-in `SLURMSignalCallback` |
| Checkpointing + resume | **Supported** | Built-in via `save_steps`, `training_state.json` |
| Auto-resubmission | **Supported** | Built-in via `run_training.sh` |
| Gradient checkpointing | **Supported** | Set `gradient_checkpointing: true` in TrainingArguments |
| BF16 mixed precision | **Supported** | Models load in bf16 by default |
| Multi-worker data loading | **Supported** | `dataloader_num_workers` config |
| Parallel data prep | **Supported** | `--workers` flag on prep scripts |
| DeepSpeed ZeRO-2/3 | **Config-only** | Add `--deepspeed ds_config.json` to launch command |
| FSDP | **Config-only** | Use `accelerate launch` with FSDP config |
| Multi-node DDP | **Launch command** | Use `torchrun --nnodes=N` or `srun` |
| Tensor parallelism | **Not supported** | Requires Megatron-LM or manual model sharding |
| Pipeline parallelism | **Not supported** | Requires Megatron-LM or DeepSpeed pipeline |
| 3D parallelism | **Not supported** | Frontier-scale frameworks (Megatron, Megascale) |
| Elastic training | **Partial** | `torchrun` supports it; Forge hasn't been tested |

### Decision Tree

```
Can your model fit on 1 GPU (with LoRA + bf16)?
├── Yes → How many GPUs do you have?
│         ├── 1    → python forge.py (Tier 1)
│         ├── 2-8  → torchrun --nproc_per_node=N forge.py (Tier 2)
│         └── 9+   → torchrun --nnodes=N (Tier 3)
│
└── No  → How big is the model?
          ├── 70B  → DeepSpeed ZeRO-3 or FSDP (Tier 3-4)
          └── 400B+→ Megatron-LM, 3D parallelism (Tier 4-5)
```

---

*Last updated: 2026-02-25*
