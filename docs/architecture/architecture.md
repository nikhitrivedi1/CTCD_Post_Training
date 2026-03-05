# Forge Architecture

Framework for knowledge distillation research across modalities.

---

## 1. Core: Training Modes

### 1.1 Supervised Fine-Tuning (SFT)

Standard CE loss on ground truth labels. The degenerate case of distillation where `alpha=0`.

```
Input → Student → CE(output, ground_truth)
```

- **Classification:** Full fine-tuning (66M–110M params fit on any GPU)
- **Generation (OCR, Translation):** LoRA fine-tuning (1B+ params need PEFT)

### 1.2 Knowledge Distillation (KD)

Teacher provides soft supervision. Loss: `(1-α)·CE + α·T²·KL(student/T ‖ teacher/T)`

```
Input → Teacher (frozen or co-training) → soft labels
Input → Student → KL(student, teacher) + CE(student, ground_truth)
```

**Distillation signals (FAR):**

| Signal | What transfers | Loss | Module |
|--------|---------------|------|--------|
| **Feature (F)** | Hidden states | MSE/Cosine | `losses.py` → `FeatureLossStrategy` |
| **Attention (A)** | Attention weights | KL | `losses.py` → `AttentionLossStrategy` |
| **Response (R)** | Output logits | KL | `losses.py` → `KLDistillationLoss` |

Signals compose via `CombinedDistillationLoss` for factorial experiments (8 combinations: CE-only, F, A, R, FA, FR, AR, FAR).

**Teacher strategies:**

| Strategy | `teacher.freeze` | Finding |
|----------|------------------|---------|
| Frozen SFT | `true` | Risk: overconfident teacher on imbalanced data |
| Frozen Base | `true` | Safer but limited knowledge |
| Co-train SFT | `false` | Self-corrects, best for binary tasks |
| Co-train Base | `false` | Best overall for most tasks |

### 1.3 Reinforcement Learning (Future)

Post-training alignment via TRL. Planned integration:

| Mode | Algorithm | Data | Status |
|------|-----------|------|--------|
| **Pure RL (DPO)** | Offline preference optimization | (prompt, chosen, rejected) pairs | Planned |
| **Pure RL (GRPO)** | Online group-relative optimization | Prompts + reward function | Planned |
| **KD+RL hybrid** | Distillation + preference simultaneously | Combined signals | Future |
| **KD → RL pipeline** | Sequential: distill then align | Two-stage | Future |

Entry point dispatch:
```python
mode = config["training"].get("mode", "kd")
# "kd" → DistillationTrainer (existing)
# "dpo" → TRL DPOTrainer
# "grpo" → TRL GRPOTrainer
```

See [rl-integration.md](../planning/rl-integration.md) for full design.

---

## 2. Module Reference

### 2.1 Entry Point (`forge.py`)

Orchestrates the full pipeline: CLI parsing → config loading → data → models → training → evaluation.

```
forge.py
  ├── Parse CLI (--config, --resume, --run_name, --seed, --max_steps, --epochs, --force, --timestamp, --debug)
  ├── Load config with inheritance
  ├── Load data (src/data/)
  ├── Load models (src/models/)
  ├── Build trainer (src/training/)
  ├── Train
  ├── Evaluate
  └── Save state (training_state.json, run_metadata.json)
```

### 2.2 Data (`src/data/`)

**Contract:** Every dataset loader returns `(train_dataset, eval_dataset, tokenizer)`

| Loader | Task | Architecture | Notes |
|--------|------|-------------|-------|
| Classification (MNLI, QQP, SST-2, CoLA) | Text classification | Encoder-only | Tokenized premise+hypothesis or sentence pairs |
| Translation (Europarl) | EN→FR | Decoder-only | Causal LM with custom end token |
| OCR (olmOCR-mix) | Image→Markdown | Vision-Language | Image tensors + text, LoRA fine-tuning |

**`collators.py`** — Batch collation for different architectures.
**`datasets.py`** — Dataset loading, preprocessing, and task registry.

### 2.3 Models (`src/models/loader.py`)

Loads teacher and student with appropriate initialization:

| Init Mode | Description | Use Case |
|-----------|-------------|----------|
| `pretrained` | Load pretrained weights | Default — faster convergence |
| `fresh` | Random initialization | True compression experiments |

VLMs load via `AutoModelForImageTextToText` + `AutoProcessor`. LoRA checkpoints load via `PeftModel.from_pretrained()` + `merge_and_unload()`.

### 2.4 Training (`src/training/`)

**`trainer.py`** — `DistillationTrainer(HF Trainer)`: overrides `compute_loss()` to combine CE + distillation signal.

**`losses.py`** — Strategy pattern for loss computation:

| Strategy | Formula | Key |
|----------|---------|-----|
| `kl_divergence` | `(1-α)·CE + α·T²·KL(s/T ‖ t/T)` | Standard Hinton |
| `mse_logits` | `(1-α)·CE + α·MSE(s, t)` | Direct logit matching |
| `feature_mse` | `(1-α)·CE + α·MSE(hidden)` | FitNets-style |
| `attention_kl` | `(1-α)·CE + α·KL(attn)` | Attention transfer |
| `CombinedDistillationLoss` | Weighted sum of strategies | FAR compositions |

**`layer_mapping.py`** — Maps teacher layers → student layers:

| Strategy | Example (12→6) |
|----------|----------------|
| `uniform` | [1,3,5,7,9,11] |
| `last_n` | [6,7,8,9,10,11] |
| `first_last` | [0,2,4,7,9,11] |

**`projections.py`** — Align dimensions when teacher/student differ:

| Strategy | Learnable |
|----------|-----------|
| `identity` | No |
| `linear` | Yes |
| `mlp` | Yes |

**`callbacks.py`** — Training callbacks (logging, checkpointing signals).

### 2.5 Evaluation (`src/evaluation/`)

**`metrics.py`** — Task-specific metrics:
- Classification: Accuracy, F1
- Translation: BLEU
- OCR: CER/WER (via `evaluate` library)

**`evaluate.py`** — Evaluation orchestration.

### 2.6 Utils (`src/utils/`)

| Module | Purpose |
|--------|---------|
| `config.py` | YAML loading with `inherit:` key resolution |
| `checkpoint.py` | Training state persistence + resume |
| `environment.py` | Device detection (CUDA/MPS/CPU) |

---

## 3. Configuration System

### Inheritance

Configs use `inherit: base.yaml` for cascading overrides. Resolution walks up the chain, merging at each level:

```
configs/base.yaml                        # Global defaults
  └── configs/ocr/olmocr-mix/base.yaml   # Task + dataset defaults
      └── configs/.../sizes/5k.yaml       # Data size settings
          └── configs/.../qwen3vl-2b/base.yaml  # Model-specific
              └── configs/.../ablations/01.yaml  # Experiment-specific
```

### Schema

```yaml
model:
  name: "model-name"             # HF model ID (student for KD, sole model for SFT)
  init: pretrained               # pretrained | fresh
  processor: "model-name"        # For VLMs
  teacher:                       # Optional, presence triggers KD
    name: "model-name"
    freeze: true                 # Frozen or co-trained teacher

data:
  task: ocr                      # classification | translation | ocr
  dataset: olmocr-mix
  data_dir: data/ocr/olmocr-mix-full
  max_length: 1024

training:
  num_epochs: 3
  batch_size: 16
  learning_rate: 2e-5
  use_lora: true
  lora:
    r: 16
    alpha: 16
    dropout: 0.05
    target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

distillation:
  temperature: 2.0
  alpha: 0.5                     # 0.0 = CE only (SFT), 1.0 = distillation only
  loss_type: kl_divergence
  layer_mapping:
    type: uniform
  projection:
    type: linear

checkpointing:
  save_steps: 100
  save_total_limit: 3

logging:
  report_to: wandb
  logging_steps: 10

evaluation:
  eval_steps: 100
  eval_strategy: steps
```

---

## 4. Orchestration

### 4.1 Local

```bash
python forge.py --config configs/.../ablation.yaml                     # auto-named
python forge.py --run_name my_run --config configs/.../ablation.yaml   # named
python forge.py --resume outputs/classification/mnli/.../20260304-1200Z  # resume
```

### 4.2 SLURM (Explorer HPC)

**Training:** `scripts/slurm/run_training.sh`

```bash
sbatch scripts/slurm/run_training.sh configs/.../ablation.yaml           # auto-named
sbatch scripts/slurm/run_training.sh my_run configs/.../ablation.yaml    # named run
```

Resume is automatic via `.incomplete_jobs` and `_FORGE_RESUME_PATH` env var.

Environment variables: `SEED`, `MAX_STEPS`, `EPOCHS`, `TIMESTAMP`, `FORCE`, `DEBUG`

**Evaluation:** `scripts/slurm/run_olmocr_bench.sh`

```bash
sbatch scripts/slurm/run_olmocr_bench.sh --model allenai/olmOCR-2-7B-1025-FP8
sbatch scripts/slurm/run_olmocr_bench.sh --checkpoint outputs/.../checkpoint-400
```

**Data Preparation:** `scripts/slurm/run_prepare_data.sh`

```bash
sbatch scripts/slurm/run_prepare_data.sh --output data/ocr/olmocr-mix-full
```

### 4.3 Graceful Shutdown & Auto-Resume

All SLURM scripts handle time limit interruptions. The pattern:

1. `#SBATCH --signal=SIG@N` sends a signal N seconds before the time limit
2. Python signal handler sets a shutdown flag
3. Main loop checks flag between units of work (training steps / eval pages)
4. State is saved, then auto-resubmission occurs

| Script | Signal | Buffer | State | Resume Mechanism |
|--------|--------|--------|-------|-----------------|
| **Training** (`slurm/run_training.sh`) | `SIGUSR1@120` | 2 min | HF checkpoints + `training_state.json` | `.incomplete_jobs` JSON → auto `sbatch` |
| **Evaluation** (`slurm/run_olmocr_bench.sh`) | `SIGUSR1@120` | 2 min | Per-page atomic writes (`.tmp` + rename) | `.incomplete_jobs` JSON → auto `sbatch` |

**Training auto-resume flow:**
```
Python detects SIGUSR1 → saves checkpoint → writes to .incomplete_jobs
Shell reads .incomplete_jobs → extracts resume_path → _FORGE_RESUME_PATH=... sbatch $0
New job runs forge.py --resume outputs/$_FORGE_RESUME_PATH
```

**Eval auto-resume flow:**
```
Shell detects SIGUSR1 → kills Python → reads .incomplete_jobs
Shell extracts output_dir → sbatch $0 --output_dir $OUTPUT_DIR
New job skips already-generated markdowns
```

**Atomic writes:** Eval uses `write to .tmp` → `os.rename()` to prevent corrupted partial files from looking complete on resume. Stale `.tmp` files are cleaned up at both startup and end of run.

### 4.4 Metadata & Job Tracking

**Training outputs** (`outputs/{run_name}/`):
```
config.yaml           # Frozen copy of resolved config
run_metadata.json     # Model info, timestamps, git hash
training_state.json   # Epochs completed, target, for resume decisions
output.log            # Training logs
checkpoint-N/         # HF Trainer checkpoints
```

Job tracking via `.incomplete_jobs` (project root): maps `SLURM_JOB_ID` → `{type, resume_path, output_dir}`.

**Eval outputs** (`outputs/bench/{model}/{timestamp}/`):
```
metadata.json         # Model info, generation params, slurm_jobs list
markdowns/            # Per-page generated markdown (atomic writes)
results.json          # Overall score, per-category breakdown
test_details.jsonl    # Per-test pass/fail with explanations
```

`metadata.json` tracks an ordered list of all SLURM jobs that touched the run:
```json
{
  "model_name": "Qwen3-VL-2B-Instruct",
  "slurm_jobs": [
    {"job_id": "4661666", "node": "d4052", "started": "2026-02-26T07:04:00+00:00"},
    {"job_id": "4664300", "node": "d4053", "started": "2026-02-26T08:30:00+00:00"}
  ]
}
```

On `--resume`, the model name is validated against `metadata.json` to prevent accidental model mixing across jobs.

When run outside SLURM (local, interactive), the `slurm_jobs` tracking is simply skipped — no error, no empty entry.

### 4.5 Colab

`scripts/run_colab.ipynb` — notebook for running on Google Colab with T4 GPU.

### 4.6 Environment

```bash
module load cuda/12.1
module load anaconda3/2024.06
conda activate forge
export HF_HOME=/scratch/$USER/.cache/huggingface  # Avoid home dir quota
```

All SLURM scripts set `HF_HOME` to redirect HuggingFace cache to scratch storage.

---

## 5. Evaluation

### 5.1 Training Metrics (Inline)

Computed during training via HF Trainer callbacks:

| Task | Metrics |
|------|---------|
| Classification | Accuracy, F1 |
| Translation | BLEU |
| OCR | CER, WER |

Logged to W&B (`report_to: wandb` in config).

### 5.2 olmOCR-Bench (Standalone)

**Script:** `scripts/eval/olmocr_bench.py`

Runs any VLM against the olmOCR-Bench benchmark (7,010 tests, 1,403 PDFs). Categories: text presence, text absence, reading order, tables, math.

**Flow:**
1. Load model (HF ID or LoRA checkpoint)
2. For each unique (PDF, page): render → generate markdown → atomic write
3. Score all tests using `olmocr.bench.tests`
4. Save results.json with per-category breakdown

**Features:** Resume support (`--resume`), sharding (`--shard N/M`), score-only mode (`--score-only`), model mismatch validation, tqdm progress, SIGTERM graceful shutdown.

**Score:** Average of per-JSONL-category pass rates (equal weight per category regardless of test count). Published top: ~82-83%.

### 5.3 Workbench (Post-Training Analysis)

`workbench/` — Analysis tools for examining learned representations:

| Tool | What it does |
|------|-------------|
| `attention.py` | Attention pattern analysis |
| `cka.py` | Centered Kernel Alignment (representation similarity) |
| `disagreement.py` | Where do student and teacher disagree? |
| `probing.py` | Linear probing of intermediate representations |
| `finetune_head.py` | Head-only fine-tuning experiments |

---

## 6. Extension Points

| Component | How to Extend | Register At |
|-----------|---------------|-------------|
| Dataset | Add loader function | `load_dataset_for_task()` |
| Loss Strategy | Subclass `DistillationLossStrategy` | `LOSS_REGISTRY` |
| Layer Mapping | Implement `map()` method | `LAYER_MAPPING_REGISTRY` |
| Projection | Subclass `nn.Module` | `PROJECTION_REGISTRY` |
| Metrics | Add `compute_metrics_xxx()` | `get_compute_metrics()` |
| Task | Register in task registry | `src/tasks/registry.py` |

### Adding a New Task

1. **Dataset** — Add loader in `src/data/datasets.py`
2. **Metrics** — Add compute function in `src/evaluation/metrics.py`
3. **Config** — Create `configs/{task}/...yaml`
4. **Labels** — Set `num_labels` if classification

### Adding a New SLURM Script

1. Add `#SBATCH --signal=SIGTERM@N` (N = buffer seconds; 120 for fast tasks, 300 for slow)
2. Register a signal handler in Python that sets a shutdown flag
3. Check the flag between units of work in the main loop
4. Save all state before exiting
5. Write a resume file or log the resume command
6. Use atomic writes for output files (`.tmp` + `os.rename`)

---

## 7. Design Patterns

| Pattern | Where | Purpose |
|---------|-------|---------|
| **Strategy** | Losses, layer mapping, projections | Swap components via config |
| **Factory + Registry** | `LOSS_REGISTRY`, `get_xxx()` | Instantiate by name |
| **Template Method** | `compute_loss()` | Algorithm skeleton delegates to strategy |
| **Config Inheritance** | YAML `inherit:` | DRY configs across experiments |
| **Atomic Write** | Eval markdown output | Corruption protection on interrupt |
| **Signal Handler** | SLURM shutdown | Graceful cleanup before time limit |

---

## 8. Tasks & Modalities

### Current

| Task | Architecture | Models | Status |
|------|-------------|--------|--------|
| **Classification** (MNLI, QQP, SST-2, CoLA) | Encoder-only | BERT → DistilBERT | Complete |
| **Translation** (Europarl EN→FR) | Decoder-only | Llama-3.2-3B → 1B | Complete |
| **OCR** (olmOCR-mix) | Vision-Language | Qwen3-VL-2B, Qwen2.5-VL-3B | Active |

### Planned

| Task | Architecture | Models | KD Approach |
|------|-------------|--------|-------------|
| **Diffusion** (CIFAR-10) | UNet | DDPM 1000-step → 4-step | Progressive distillation (step reduction) |
| **Voice** (StyleTTS2) | Multi-component | Full → compressed | Component-wise: text encoder (F/A/R) + diffusion (steps) + decoder (Feature KD) |
| **Code Generation** | Decoder-only | Large → small code LM | Token-level KL, same as translation |
| **Summarization** | Encoder-decoder | BART → DistilBART | Sequence-level KD |

### Research Directions

| Direction | Description | Reference |
|-----------|-------------|-----------|
| Adaptive distillation | Adjust α based on teacher confidence | Our finding: 82% confidence when wrong |
| Calibration-aware KD | Reweight/skip examples where teacher is miscalibrated | Per-example α scheduling |
| Progressive (chained) KD | Large → Medium → Small (smaller capacity gaps) | Mirzadeh et al., 2020 |
| Self-distillation | Model trains on its own soft labels | Furlanello et al., 2018 |
| Multi-teacher ensemble | Combine knowledge from multiple teachers | Learned weighting per example/class |
| Sequence-level KD | Distill on full sequence likelihood, not per-token | Kim & Rush, 2016 |
| KD+RL hybrid | Combine distillation with DPO/GRPO | FAR factorial extended to hybrid training |

---

## 9. Project Structure

```
forge.py                          # Entry point
src/
  data/datasets.py, collators.py    # Data loading
  models/loader.py                  # Model loading
  training/                         # Trainer, losses, mappings, projections
  evaluation/                       # Metrics, evaluation
  tasks/registry.py                 # Task registry
  utils/                            # Config, checkpoint, environment
scripts/
  slurm/                            # SLURM job scripts
    slurm_essentials.sh             # Shared utilities (trap, env, GPU detection, resubmit)
    run_training.sh                 # Training with auto-resubmit
    run_olmocr_bench.sh             # Benchmark eval with auto-resubmit
    run_prepare_data.sh             # Data preparation
  eval/olmocr_bench.py              # olmOCR-Bench scoring
  data/prepare_ocr_data.py           # OCR data pipeline
  models/                           # HF Hub push/pull
configs/
  base.yaml                         # Global defaults
  classification/                   # MNLI, QQP, SST-2, CoLA
  translation/                      # Europarl
  ocr/                              # olmOCR-mix (qwen3vl-2b, qwen25vl-3b)
workbench/                          # Post-training analysis tools
docs/                               # Documentation
```

---

*Last updated: 2026-03-04*
