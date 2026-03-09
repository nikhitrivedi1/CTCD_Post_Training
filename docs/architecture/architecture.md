# Forge Architecture

Open training framework for fine-tuning (FT), knowledge distillation (KD), and reinforcement learning (RL).

---

## Module Documentation

Each module has its own detailed doc. Click through for full API coverage, design decisions, and implementation details.

### Core

| Doc | Source | Description |
|-----|--------|-------------|
| [Entry Point](entry-point.md) | `forge.py` | CLI, pipeline orchestration, training mode dispatch |
| [Tasks](tasks.md) | `src/tasks/registry.py` | Task registry (MNLI, QQP, SST-2, CoLA, Europarl, OCR) |

### Training

| Doc | Source | Description |
|-----|--------|-------------|
| [Trainer](training/trainer.md) | `src/training/trainer.py` | FinetuningTrainer, DistillationTrainer |
| [Losses](training/losses.md) | `src/training/losses.py` | 9 loss strategies (KL, reverse-KL, MSE, Cosine, CE, Feature MSE, Feature Cosine, Attention KL, Staged) |
| [Callbacks](training/callbacks.md) | `src/training/callbacks.py` | Step metrics, SLURM signals, checkpoint marking, teacher eval, peak metrics |
| [Layer Mapping](training/layer-mapping.md) | `src/training/layer_mapping.py` | Teacher-student layer correspondence (uniform, last_n, first_last) |
| [Projections](training/projections.md) | `src/training/projections.py` | Dimension alignment (identity, linear, MLP) |
| [Rewards](training/rewards.md) | `src/training/rewards.py` | GRPO reward functions (BLEU, edit sim, unit test, format, table F1, formula, reading order) |

### Data

| Doc | Source | Description |
|-----|--------|-------------|
| [Data](data.md) | `src/data/` | Dataset loaders (GLUE, Europarl, OCR), VLM collator, RL datasets |

### Models

| Doc | Source | Description |
|-----|--------|-------------|
| [Models](models.md) | `src/models/loader.py` | Teacher-student pair loading, LoRA, attention impl selection |

### Evaluation & Inference

| Doc | Source | Description |
|-----|--------|-------------|
| [Evaluation](evaluation.md) | `src/evaluation/` | Direct eval loop, task metrics, transfer metrics |
| [Inference](inference.md) | `src/inference/engine.py` | Unified generation engine (VLM + causal LM) |
| [Save Structure](save-structure.md) | `forge.py` | Unified model save layout (merged model + adapter + base ref) |

### Utilities

| Doc | Source | Description |
|-----|--------|-------------|
| [Utils](utils.md) | `src/utils/`, `utils/` | Config inheritance, checkpointing, environment, job locking, TeeWriter |

---

## Training Modes

### Supervised Fine-Tuning (SFT)

Standard CE loss on ground truth labels. The degenerate case of distillation where `alpha=0`.

```
Input -> Student -> CE(output, ground_truth)
```

- **Classification:** full fine-tuning (66M-110M params fit on any GPU)
- **Generation (OCR, Translation):** LoRA fine-tuning (1B+ params need PEFT)

### Knowledge Distillation (KD)

Teacher provides soft supervision. Loss: `(1-alpha) * CE + alpha * T^2 * KL(student/T || teacher/T)`

```
Input -> Teacher (frozen or co-training) -> soft labels
Input -> Student -> KL(student, teacher) + CE(student, ground_truth)
```

**Distillation signals (FAR):**

| Signal | What transfers | Loss | Module |
|--------|---------------|------|--------|
| **Feature (F)** | Hidden states | MSE/Cosine | `losses.py` -> `FeatureLossStrategy` |
| **Attention (A)** | Attention weights | KL | `losses.py` -> `AttentionLossStrategy` |
| **Response (R)** | Output logits | KL | `losses.py` -> `KLDistillationLoss` |

Signals compose via `CombinedDistillationLoss` for factorial experiments (8 combinations: CE-only, F, A, R, FA, FR, AR, FAR).

**Teacher strategies:**

| Strategy | `teacher.freeze` | Finding |
|----------|------------------|---------|
| Frozen SFT | `true` | Risk: overconfident teacher on imbalanced data |
| Frozen Base | `true` | Safer but limited knowledge |
| Co-train SFT | `false` | Self-corrects, best for binary tasks |
| Co-train Base | `false` | Best overall for most tasks |

### Reinforcement Learning (GRPO)

Online group-relative policy optimization via trl's `GRPOTrainer`.

```
Prompt -> Model -> Completions -> Reward Functions -> Policy Update
```

Reward functions: BLEU (translation), edit similarity (OCR), unit tests (olmOCR-bench), format quality, table F1, formula CDM, reading order.

---

## Configuration System

Hierarchical YAML configs with `inherit:` key resolution:

```
configs/base.yaml                        # Global defaults
  -> configs/ocr/olmocr-mix/base.yaml    # Task + dataset defaults
    -> configs/.../sizes/5k.yaml         # Data size settings
      -> configs/.../qwen3vl-2b/base.yaml  # Model-specific
        -> configs/.../ablations/01.yaml   # Experiment-specific
```

See [Utils](utils.md) for full config system details.

---

## Orchestration

### Local

```bash
python forge.py --config configs/.../ablation.yaml
python forge.py --run_name my_run --config configs/.../ablation.yaml
python forge.py --resume outputs/classification/mnli/.../20260304-1200Z
```

### SLURM

```bash
sbatch scripts/slurm/run_training.sh configs/.../ablation.yaml
sbatch scripts/slurm/run_bench.sh olmocr --model allenai/olmOCR-2-7B-1025-FP8
```

### Graceful Shutdown & Auto-Resume

1. Shell sends `SIGUSR1` 120 seconds before time limit
2. `SLURMSignalCallback` sets shutdown flag
3. `on_step_end()` triggers `should_save=True` + `should_training_stop=True`
4. HF Trainer saves checkpoint normally
5. `CheckpointMarkerCallback` writes `latest_checkpoint` atomically
6. Shell reads `.incomplete_jobs` and resubmits with `--resume`

See [Callbacks](training/callbacks.md) and [Utils](utils.md) for details.

---

## Design Patterns

| Pattern | Where | Purpose |
|---------|-------|---------|
| **Strategy** | Losses, layer mapping, projections | Swap components via config |
| **Factory + Registry** | `LOSS_REGISTRY`, `REWARD_REGISTRY`, `get_xxx()` | Instantiate by name |
| **Template Method** | `compute_loss()` | Algorithm skeleton delegates to strategy |
| **Config Inheritance** | YAML `inherit:` | DRY configs across experiments |
| **Atomic Write** | Checkpoints, eval output, job registry | Corruption protection on interrupt |
| **Signal Handler** | SLURM shutdown | Graceful cleanup before time limit |

---

## Extension Points

| Component | How to Extend | Register At |
|-----------|---------------|-------------|
| Task | Add to `TASKS` dict | `src/tasks/registry.py` |
| Dataset | Add loader function | `src/data/datasets.py` |
| Loss Strategy | Subclass `DistillationLossStrategy` | `LOSS_REGISTRY` in `src/training/losses.py` |
| Layer Mapping | Implement `map()` method | `LAYER_MAPPING_REGISTRY` in `src/training/layer_mapping.py` |
| Projection | Subclass `nn.Module` | `PROJECTION_REGISTRY` in `src/training/projections.py` |
| Metrics | Add `compute_xxx_metrics()` | `get_compute_metrics()` in `src/evaluation/metrics.py` |
| Reward | Add function with trl signature | `REWARD_REGISTRY` in `src/training/rewards.py` |
| Callback | Subclass `TrainerCallback` | Add in `forge.py` |

---

## Project Structure

```
forge.py                          # Entry point
src/
  data/
    datasets.py                   # GLUE, Europarl, OCR loaders
    collators.py                  # VLMDataCollator
    ocr_utils.py                  # PDF rendering, shared prompts
    rl_datasets.py                # GRPO dataset loaders
  models/
    loader.py                     # Teacher-student pair loading
  training/
    trainer.py                    # FinetuningTrainer, DistillationTrainer
    losses.py                     # 9 loss strategies + registry
    callbacks.py                  # Step metrics, SLURM, checkpoints, teacher eval, peaks
    layer_mapping.py              # 4 mapping strategies + registry
    projections.py                # 3 projection types + registry
    rewards.py                    # 7 reward functions + registry
  evaluation/
    evaluate.py                   # Direct eval loop, transfer metrics
    metrics.py                    # Classification, translation, OCR metrics
  inference/
    engine.py                     # Unified generation engine
  tasks/
    registry.py                   # Task properties registry
  utils/
    config.py                     # YAML loading with inheritance
    checkpoint.py                 # Training state persistence
    environment.py                # Platform detection
    jobs.py                       # Job registry with file locking
utils/
  logging.py                      # TeeWriter for clean log files
scripts/
  slurm/                          # SLURM job scripts
  eval/                           # Benchmark evaluation scripts
  data/                           # Data preparation scripts
  models/                         # HF Hub push/pull
configs/                          # YAML config hierarchy
workbench/                        # Post-training analysis tools
docs/                             # Documentation
```

---

## Tasks & Modalities

### Current

| Task | Architecture | Models | Status |
|------|-------------|--------|--------|
| **Classification** (MNLI, QQP, SST-2, CoLA) | Encoder-only | BERT -> DistilBERT | Complete |
| **Translation** (Europarl EN-FR) | Decoder-only | Llama-3.2-3B -> 1B | Complete |
| **OCR** (olmOCR-mix, synthmix, OmniDocBench) | Vision-Language | Qwen3-VL-2B, Qwen2.5-VL-3B | Active |

### Planned

| Task | Architecture | KD Approach |
|------|-------------|-------------|
| **Diffusion** (CIFAR-10) | UNet | Progressive distillation (step reduction) |
| **Voice** (StyleTTS2) | Multi-component | Component-wise F/A/R + step reduction |
| **Code Generation** | Decoder-only | Token-level KL |
| **Summarization** | Encoder-decoder | Sequence-level KD |

---

*Last updated: 2026-03-09*
