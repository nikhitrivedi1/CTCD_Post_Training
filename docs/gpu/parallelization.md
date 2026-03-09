# Parallelization Flows

How Forge workloads map to CPU, GPU, and mixed CPU-GPU execution. Covers ideal data flows, bottlenecks, and optimization strategies for each pattern.

---

## Workload Classification

Every Forge operation falls into one of three categories:

| Pattern | Examples | Bound By | Key Resource |
|---------|----------|----------|--------------|
| **CPU-only** | Data prep, scoring, text metrics | CPU cores, I/O | `--cpus-per-task` |
| **GPU-only** | Inference, generation, forward/backward | GPU compute/memory | `--gres=gpu` |
| **CPU-GPU mixed** | GRPO training, eval with metrics | Both, alternating | Balance matters |

---

## CPU-Only Workflows

### Data Preparation

```
┌──────────────────────────────────────────────────────────────┐
│                    DATA PREP FLOW                            │
│                                                              │
│  Source         Process              Output                  │
│  ──────         ───────              ──────                  │
│  HF Hub ──→ Download ──→ Convert ──→ data/ocr/{dataset}/    │
│  PDFs   ──→ Render   ──→ Resize ──→ images/                 │
│  JSON   ──→ Parse    ──→ Format ──→ markdowns/              │
│                                                              │
│  Parallelism: multiprocessing.Pool(num_workers)              │
│  Bottleneck: disk I/O (PDF rendering), network (downloads)   │
└──────────────────────────────────────────────────────────────┘
```

**Ideal settings:**
```bash
#SBATCH --partition=short
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
# No GPU needed
```

**Scripts:** `run_data_prep.sh {olmocr-mix|synthmix|omnidocbench|rl}`

### Scoring (Post-Inference)

```
┌──────────────────────────────────────────────────────────────┐
│                    SCORING FLOW                              │
│                                                              │
│  Model outputs         Score              Aggregate          │
│  ──────────────        ─────              ─────────          │
│  markdowns/*.md ──→ Per-page metrics ──→ results.json        │
│                        │                                     │
│                        ├─ Edit distance (Levenshtein)        │
│                        ├─ Unit tests (olmocr bench)          │
│                        ├─ Table cell F1                      │
│                        ├─ Formula similarity                 │
│                        └─ Reading order LCS                  │
│                                                              │
│  Parallelism: concurrent.futures (per-page scoring)          │
│  Bottleneck: CPU string operations, test execution           │
└──────────────────────────────────────────────────────────────┘
```

**Ideal settings:**
```bash
#SBATCH --partition=short
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
# No GPU needed
```

**Scripts:** `run_score.sh {olmocr|omnidoc} --output_dir ...`

**Optimization opportunities:**
- Scoring is embarrassingly parallel across pages
- `concurrent.futures.ProcessPoolExecutor` with `cpus-per-task` workers
- Each page is independent, no shared state needed

---

## GPU-Only Workflows

### Inference / Benchmarking

```
┌──────────────────────────────────────────────────────────────┐
│                    INFERENCE FLOW                            │
│                                                              │
│  Image ──→ Vision Encoder ──→ LLM Generate ──→ markdown      │
│            (GPU)               (GPU)            (disk)       │
│                                                              │
│  Per page:                                                   │
│    1. Load image from disk          (CPU, ~10ms)             │
│    2. Preprocess (resize, normalize) (CPU, ~20ms)            │
│    3. Vision encoder forward         (GPU, ~50ms)            │
│    4. Autoregressive generation      (GPU, ~2-5s)            │
│    5. Decode tokens to text          (CPU, ~1ms)             │
│    6. Write markdown to disk         (CPU, ~1ms)             │
│                                                              │
│  Bottleneck: Step 4 (generation), ~95% of time               │
│  GPU util: ~90%+ (generation dominates)                      │
└──────────────────────────────────────────────────────────────┘
```

**Ideal settings:**
```bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4     # Enough for dataloader prefetch
#SBATCH --mem=64G
```

**Parallelism via sharding** (see [Independent Sharding](multi-gpu.md#independent-sharding-simplest)):
```bash
# Split 1355 pages across 4 GPUs (separate jobs)
for i in 1 2 3 4; do
  sbatch run_bench.sh omnidoc --shard $i/4
done
# Each GPU processes ~339 pages independently
# Merge results with --score_only after all complete
```

**Scripts:** `run_bench.sh {olmocr|omnidoc} --model ... [--shard N/M]`

### SFT / Knowledge Distillation

```
┌──────────────────────────────────────────────────────────────┐
│                    SFT TRAINING FLOW                         │
│                                                              │
│  DataLoader (CPU workers)                                    │
│    │  Images + text loaded and collated                      │
│    │  num_workers=4, pin_memory=True                         │
│    ▼                                                         │
│  Forward Pass (GPU)                                          │
│    │  Vision encoder + LLM forward                           │
│    │  Activations stored for backward                        │
│    ▼                                                         │
│  Loss Computation (GPU)                                      │
│    │  CE loss (SFT) or CE + KL + feature/attention (KD)      │
│    ▼                                                         │
│  Backward Pass (GPU)                                         │
│    │  Gradients through LLM + LoRA adapters                  │
│    ▼                                                         │
│  Optimizer Step (GPU)                                        │
│    │  AdamW update on LoRA params only                       │
│    ▼                                                         │
│  Next batch                                                  │
│                                                              │
│  GPU util: ~95%+ (compute-bound, small LoRA optimizer)       │
│  Bottleneck: forward/backward (GPU), rarely data loading     │
└──────────────────────────────────────────────────────────────┘
```

---

## CPU-GPU Mixed Workflows

### GRPO Training (The Hard Case)

GRPO alternates between GPU-heavy generation and CPU-heavy reward computation. This is where GPU utilization suffers.

```
┌──────────────────────────────────────────────────────────────┐
│                    GRPO TRAINING FLOW                        │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Step 1: GENERATION (GPU-bound)                         │  │
│  │                                                        │  │
│  │ For each prompt in micro-batch:                        │  │
│  │   Repeat prompt x num_generations                      │  │
│  │   -> batch of (batch_size x num_gen) sequences         │  │
│  │   -> Autoregressive generation (GPU, 10-60s)           │  │
│  │                                                        │  │
│  │ Peak memory: KV cache x num_gen x max_completion_len   │  │
│  │ GPU util: ~100%                                        │  │
│  └────────────────────────────────────────────────────────┘  │
│                            ▼                                 │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Step 2: REWARD SCORING (CPU-bound)        <-- GPU IDLE │  │
│  │                                                        │  │
│  │ For each completion:                                   │  │
│  │   edit_similarity: Levenshtein distance (CPU)          │  │
│  │   format: regex pattern matching (CPU)                 │  │
│  │   table_cell_f1: cell extraction + matching (CPU)      │  │
│  │   formula_cdm: LaTeX extraction + distance (CPU)       │  │
│  │   reading_order: block extraction + LCS (CPU)          │  │
│  │                                                        │  │
│  │ GPU util: ~0% (sitting idle)                           │  │
│  │ Duration: depends on num rewards and text length       │  │
│  └────────────────────────────────────────────────────────┘  │
│                            ▼                                 │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Step 3: ADVANTAGE + POLICY UPDATE (GPU-bound)          │  │
│  │                                                        │  │
│  │ Compute advantages from rewards (CPU, fast)            │  │
│  │ Forward pass: log-probs for all completions (GPU)      │  │
│  │ Backward pass: GRPO policy gradient (GPU)              │  │
│  │ Optimizer step (every grad_accum steps)                │  │
│  │                                                        │  │
│  │ GPU util: ~100%                                        │  │
│  └────────────────────────────────────────────────────────┘  │
│                            ▼                                 │
│  Repeat for next micro-batch                                 │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### GRPO Timeline (Actual)

```
Time ──────────────────────────────────────────────────────────→

GPU:  ██████████████░░░░░░██████████████░░░░░░██████████████░░░
      ↑ Generation   ↑    ↑ Forward/     ↑    ↑ Generation
                   Idle    Backward    Idle

CPU:  ░░░░░░░░░░░░░░██████░░░░░░░░░░░░░░██████░░░░░░░░░░░░░░██
                     ↑ Reward              ↑ Reward
                     scoring               scoring
```

### GRPO Bottleneck Analysis

| Phase | Time (typical) | Bound | Parallelizable? |
|-------|---------------|-------|-----------------|
| Generation | 10-60s per micro-batch | GPU memory + compute | Across GPUs (sharding) |
| Reward scoring | 1-10s per micro-batch | CPU cores | Within batch (multiprocessing) |
| Policy update | 5-20s per micro-batch | GPU compute | Not easily |

The GPU idle time during reward scoring is the root cause of the "GPU idling 15 min" alerts. With 5 rewards and long documents, scoring adds up.

### Optimization Strategies for GRPO

#### Memory: Chunked Generation (Preferred)

The core OOM problem: trl expands the batch to `batch_size x num_generations` and generates all at once. With VLM document images producing thousands of visual tokens, even `batch=1, num_gen=2` can OOM on H200.

The fix is to decouple generation parallelism from `num_generations`:

```
Current (OOM):     Generate(batch x 4 completions at once)
                   -> all 4 sequences + KV caches in GPU memory

Chunked (fits):    Generate(batch x 1) -> store on CPU
                   Generate(batch x 1) -> store on CPU
                   Generate(batch x 1) -> store on CPU
                   Generate(batch x 1) -> store on CPU
                   -> only 1 sequence + KV cache at a time
```

trl's `GRPOConfig` may support `generation_batch_size` to control this. If not, subclass `GRPOTrainer` and override the generation loop. This lets you keep `num_generations=4` (better advantage estimation) without the memory cost.

The training forward/backward pass still processes all `num_generations` sequences, but gradient checkpointing keeps that manageable since there's no KV cache.

#### Memory: Image Resolution Cap

For VLMs, image resolution directly controls visual token count. Qwen3-VL's default `max_pixels` produces thousands of tokens per document scan. Capping resolution (e.g., halving `max_pixels`) can cut memory significantly, at the cost of OCR quality on fine-grained content.

#### Memory: Multi-GPU (FSDP)

When single-GPU optimizations aren't enough, FSDP across 2 GPUs frees ~35GB of optimizer/param memory. See [multi-gpu.md](multi-gpu.md) for setup details.

#### Throughput: Parallelize reward computation (CPU-side)

```python
from concurrent.futures import ProcessPoolExecutor

def score_batch(completions, ground_truth, reward_fns):
    with ProcessPoolExecutor(max_workers=num_cpus) as pool:
        futures = [pool.submit(fn, completions, ground_truth)
                   for fn in reward_fns]
        return [f.result() for f in futures]
```
All 5 reward functions can run in parallel on different CPU cores. This reduces reward time by up to 5x.

#### Throughput: Async reward computation (overlap with generation)

```
Micro-batch N:    Generate --> [Reward scoring in background]
Micro-batch N+1:  Generate --> [Reward scoring in background]
                                        |
                              Policy update uses rewards from N
```
Requires modifications to trl's GRPOTrainer loop but can nearly eliminate GPU idle time.

#### Throughput: Request more CPUs for GRPO jobs

```bash
#SBATCH --cpus-per-task=16    # More CPU cores for reward computation
```
Default of 4 CPUs limits parallel reward scoring.

---

## Ideal Resource Allocation

### By Job Type

| Job | SLURM Script | Partition | GPUs | CPUs | Memory | Time |
|-----|-------------|-----------|------|------|--------|------|
| Data prep | `run_data_prep.sh` | short | 0 | 32 | 64G | 8h |
| Scoring | `run_score.sh` | short | 0 | 32 | 64G | 2h |
| SFT training | `run_training.sh` | gpu | 1 | 4 | 64G | 8h |
| KD training | `run_training.sh` | gpu | 1 | 4 | 64G | 8h |
| GRPO training | `run_training.sh` | gpu | 1 | 8-16 | 64G | 8h |
| GRPO (large VLM) | `run_training.sh` | gpu | 2 ([FSDP](multi-gpu.md#fsdp-fully-sharded-data-parallel)) | 4 | 128G | 8h |
| Inference/bench | `run_bench.sh` | gpu | 1 | 4 | 64G | 8h |

### Scaling with Multiple GPUs

For details on each strategy, see [multi-gpu.md](multi-gpu.md).

```
Embarrassingly parallel (separate SLURM jobs):
  Benchmarking: 4 shards x 1 GPU = 4x faster wall-clock
  Scoring: 1 CPU job (already parallel internally)
  Hyperparameter search: N configs x 1 GPU each

Tightly coupled (single multi-GPU job):
  GRPO with large num_gen: FSDP across 2 GPUs
  Large model training: DeepSpeed ZeRO
```

### Cost-Efficiency Rules

1. **Never use a GPU for CPU work.** Scoring, data prep, and metric computation belong on `short` partition.
2. **Shard inference, don't multi-GPU it.** Independent SLURM jobs are simpler and more fault-tolerant than distributed inference.
3. **Match CPUs to workload.** SFT needs 4 CPUs (dataloader). GRPO benefits from 8-16 (reward computation).
4. **Use the right account.** Check `sshare -U` and pick the account with highest FairShare for faster scheduling.

---

## Pipeline Orchestration

### Full Training Pipeline

```
Phase 1: Data Prep (CPU)
  sbatch run_data_prep.sh olmocr-mix
  sbatch run_data_prep.sh rl
  → Wait for completion

Phase 2: SFT Training (GPU)
  sbatch run_training.sh configs/.../00_sft.yaml
  → Auto-resubmits if time limit hit
  → Wait for completion

Phase 3: GRPO Training (GPU + CPU)
  ADAPTER=outputs/sft/ sbatch run_training.sh grpo_run configs/.../01_synthmix.yaml
  → Auto-resubmits if time limit hit
  → Wait for completion

Phase 4: Evaluation (GPU, parallel shards)
  for i in 1 2 3 4; do
    sbatch run_bench.sh olmocr --checkpoint outputs/grpo_run/ --shard $i/4
  done
  → Wait for all shards

Phase 5: Scoring (CPU)
  sbatch run_score.sh olmocr --output_dir outputs/bench/olmocr/grpo_run
  → Final results in results.json
```

### Job Dependencies (SLURM)

```bash
# Score after all bench shards complete
BENCH1=$(sbatch --parsable run_bench.sh omnidoc --shard 1/4)
BENCH2=$(sbatch --parsable run_bench.sh omnidoc --shard 2/4)
BENCH3=$(sbatch --parsable run_bench.sh omnidoc --shard 3/4)
BENCH4=$(sbatch --parsable run_bench.sh omnidoc --shard 4/4)

sbatch --dependency=afterok:$BENCH1:$BENCH2:$BENCH3:$BENCH4 \
  run_score.sh omnidoc --output_dir outputs/bench/omnidoc/my_model
```

---

## Monitoring Mixed Workloads

### Detecting GPU Idle During GRPO

```bash
# Check GPU utilization on a running job's node
srun --jobid=JOBID --overlap \
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

# If GPU util is 0% but memory is high, the job is in reward scoring phase
# If GPU util is 0% and memory is low, something is wrong
```

### Checking CPU Utilization

```bash
# See if CPU cores are being used during reward computation
srun --jobid=JOBID --overlap top -bn1 | head -20
```

---

*See also: [multi-gpu.md](multi-gpu.md), [memory-math.md](memory-math.md), [utilization.md](utilization.md)*
