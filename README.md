# Knowledge Distillation Framework

A universal, config-driven knowledge distillation framework for compressing any teacher-student model pair across
classification and translation tasks.

## Overview

This framework implements knowledge distillation using KL divergence + task loss to transfer knowledge from large "
teacher" models to smaller, faster "student" models. **Designed to be completely generic and extensible** - works with
any HuggingFace model pair.

### Key Features

- **Universal**: Works with any HuggingFace model pair (BERT, Llama, etc.)
- **Multi-task**: Classification, translation, or custom tasks
- **Production-ready**: SLURM integration, automatic job chaining, W&B tracking
- **Cross-platform**: Local (Apple Silicon MPS/CPU), Google Colab, HPC clusters
- **Config-driven**: Everything controlled via YAML configs

### Two Complementary Tools

| | **Distillation Framework** | **Workbench** |
|---|---|---|
| **Purpose** | Train distilled models | Analyze trained models |
| **Entry point** | `distill.py` | `workbench/analyze.py` |
| **Runtime** | Hours to days | Minutes |
| **Resources** | GPU cluster, full datasets | Local CPU, sampled data |

**Distillation Framework** (`distill.py`) is the heavy machine - it runs large-scale training experiments on GPU clusters, handles checkpointing, SLURM job chaining, and W&B logging. You point it at a config and let it run for hours.

**Workbench** (`workbench/analyze.py`) is the quick experimentation bench - after the heavy machine produces trained models, you use the workbench for rapid post-hoc analysis: linear probing, attention entropy, prediction disagreement, CKA similarity. These run in minutes on CPU with sampled data, letting you iterate quickly on understanding what the models learned.

```bash
# Heavy machine: train a model (hours)
python distill.py --config configs/classification/mnli/bert_distilbert/ablations/01_frozen_base-teacher_kl.yaml

# Workbench: analyze the result (minutes)
python workbench/analyze.py --test probe --teacher textattack/bert-base-uncased-MNLI --student outputs/mnli_00 --task mnli
python workbench/analyze.py --test disagreement --teacher ... --student outputs/mnli_00 --task mnli
```

## DistilKit vs Unsloth

DistilKit and [Unsloth](https://github.com/unslothai/unsloth) solve different problems in the model compression pipeline.

| | **DistilKit** | **Unsloth** |
|---|---|---|
| **Goal** | Produce a smaller model that learns from a larger one | Make fine-tuning the same model faster |
| **Method** | Knowledge distillation (teacher-student training) | Optimized CUDA kernels + memory tricks |
| **Output** | A different, smaller architecture (e.g., BERT → DistilBERT) | Same architecture with adapted weights |
| **Model types** | Any HuggingFace pair (encoders, decoders) | Decoder-only LLMs (Llama, Mistral, Gemma) |
| **Research question** | *What knowledge transfers and how?* | *How fast can we fine-tune?* |
| **Distillation signals** | Response (KL), Feature (MSE), Attention (KL) | N/A — no teacher model involved |
| **Analysis tooling** | Workbench: CKA, probing, disagreement, attention entropy | N/A |
| **Training mode** | Full fine-tune or LoRA, frozen or co-trained teacher | LoRA/QLoRA with fused kernels |

**When to use DistilKit:**
- You need a structurally smaller model (fewer layers, smaller hidden dim) for deployment
- You want to study *which* distillation signals matter (RFA factorial design)
- You're compressing across architectures (e.g., Llama-3B teaching a custom 500M model)
- You need post-hoc analysis of what knowledge transferred

**When to use Unsloth:**
- You want to fine-tune a single LLM as fast as possible
- You're doing LoRA/QLoRA adaptation and want 2-5x speedups
- You don't need a smaller architecture, just adapted weights

**Using them together:** Unsloth can accelerate the student's training loop inside DistilKit. Load the student via `FastLanguageModel`, apply LoRA, and pass it to `DistillationTrainer` — you get Unsloth's speed optimizations with DistilKit's knowledge transfer signals. This is most useful for LLM-scale distillation (e.g., Llama-70B → Llama-3B).

## Quick Start

### Installation

```bash
conda env create -f environment.yml
conda activate distillation
```

### Verify Installation

```bash
python test_setup.py
```

### Basic Usage

```bash
# New run (auto-named with timestamp)
python distill.py --config configs/classification/mnli/bert_distilbert/ablations/01_frozen_base-teacher_kl.yaml

# New run with custom name
python distill.py --run_name mnli_00 --config configs/classification/mnli/bert_distilbert/ablations/00_no-teacher_ce-only.yaml

# Resume existing run
python distill.py --run_name mnli_00

# Overwrite existing run
python distill.py --run_name mnli_00 --config configs/... --force

# Local testing (10 steps)
python distill.py --config configs/... --max_steps 10

# SLURM (with auto-requeue)
sbatch scripts/run_slurm.sh configs/your_config.yaml
```

### Logging

All logs use UTC timestamps for consistency across machines and clusters:

```
2026-02-09 19:30:00,123Z - INFO - Loading model...
```

Use `--debug` for verbose logging with module names:

```
2026-02-09 19:30:00,123Z - src.training.trainer - DEBUG - ...
```

Logs are written to both stdout and `outputs/<run_name>/output.log`. On SLURM, they also appear in `logs/<jobid>.out`.

See [docs/guides/run-options.md](docs/guides/run-options.md) for complete CLI reference.

## Using with Your Own Models

The framework is **completely generic**. Here's how to adapt it for your use case:

### 1. Create a Config File

```yaml
# configs/my_custom_distillation.yaml
inherit: base.yaml

model:
  teacher: your-org/your-teacher-model  # Any HuggingFace model
  student: your-org/your-student-model  # Any HuggingFace model
  student_init: fresh  # or 'pretrained'

data:
  task: your_task  # Define custom task
  # Add task-specific parameters

training:
  num_epochs: 3
  batch_size: 16
  learning_rate: 2e-5

distillation:
  temperature: 2.0
  alpha: 0.5  # Balance between KL and CE loss
```

### 2. Add Custom Dataset Loader (if needed)

If your task isn't classification or translation, add a loader to `src/data/datasets.py`:

```python
def load_your_task(tokenizer, **kwargs):
    """Load your custom dataset."""
    dataset = load_dataset("your_dataset")

    def preprocess(examples):
        return tokenizer(examples["input"], ...)

    tokenized = dataset.map(preprocess, batched=True)
    return tokenized["train"], tokenized["validation"], tokenizer
```

Then update `load_dataset_for_task()` to handle your task type.

### 3. Add Custom Metrics (optional)

If you need custom evaluation metrics, extend `src/evaluation/metrics.py`:

```python
def compute_metrics_your_task(eval_pred):
    """Compute metrics for your task."""
    predictions, labels = eval_pred
    # Your metric computation
    return {"your_metric": score}
```

## Supported Model Types

### Auto-detected

- **Sequence Classification**: BERT, RoBERTa, DistilBERT, ALBERT, etc.
- **Causal LM**: Llama, Mistral, etc.

The framework automatically detects model architecture from HuggingFace model names.

### LoRA Support

For large causal LM models (e.g., Llama translation), enable LoRA in config:

```yaml
training:
  use_lora: true
  lora:
    r: 16
    alpha: 16
    dropout: 0.05
    target_modules: [ "q_proj", "v_proj" ]  # Model-specific
```

## Example Use Cases

### Included Examples

1. **BERT → DistilBERT on MNLI**: Sentence entailment classification
2. **BERT → DistilBERT on QQP**: Question pair similarity
3. **BERT → DistilBERT on SST-2**: Sentiment analysis
4. **BERT → DistilBERT on CoLA**: Linguistic acceptability
5. **Llama 3.2 3B → 1B on EN-FR**: Neural machine translation (Europarl)

### Custom Domain Example

```yaml
model:
  teacher: your-finetuned-bert
  student: distilbert-base-uncased
data:
  task: your_classification_task
  dataset: your_dataset
```

## Architecture

```
distillation/
├── configs/              # YAML configurations
│   ├── base.yaml         # Shared defaults
│   ├── classification/   # MNLI, QQP, SST-2, CoLA configs
│   └── translation/      # Europarl EN-FR configs
├── src/
│   ├── data/            # Dataset loaders
│   ├── models/          # Model loading (generic)
│   ├── training/        # DistillationTrainer
│   ├── evaluation/      # Metrics
│   └── utils/           # Config, checkpointing, environment
├── workbench/           # Post-hoc analysis (the "quick bench")
│   ├── analyze.py       # CLI: --test probe|cka|attention-entropy|disagreement|all
│   ├── probing.py       # Linear probing per layer
│   ├── cka.py           # CKA similarity between layers
│   ├── attention.py     # Attention entropy analysis
│   ├── disagreement.py  # Teacher-student prediction differences
│   └── results/         # Analysis outputs (.json + .log)
├── utils/               # Shared utilities
│   └── logging.py       # TeeWriter for clean progress bar logging
├── scripts/             # Execution scripts
│   ├── run_local.sh
│   └── run_slurm.sh
├── test_setup.py        # Environment verification
└── distill.py           # Main entry point
```

## Key Components

### DistillationTrainer

Extends HuggingFace `Trainer` with:

- **RFA distillation signals**: Response (KL/MSE/Cosine), Feature (MSE/Cosine), Attention (KL)
- **Training modes**: Frozen teacher, co-trained teacher, CE-only baseline (no teacher)
- **Staged distillation**: Sequential signal phases (e.g., Feature → Attention → Response)
- Temperature-scaled distributions
- Automatic loss component logging to W&B

### Config System

- YAML-based with inheritance
- Environment-specific overrides
- All hyperparameters in one place

### SLURM Integration

- Automatic job chaining for >8hr runs
- Signal handling (SIGUSR1)
- W&B run continuation across restarts

## Extending the Framework

### Adding New Task Types

1. **Dataset loader**: Add to `src/data/datasets.py`
2. **Metrics**: Add to `src/evaluation/metrics.py`
3. **Config**: Create new config inheriting from `base.yaml`

### Modifying Distillation Loss

The core loss is in `src/training/trainer.py` (`compute_loss` method). Response, feature, and attention distillation are already implemented. You can:

- Change temperature scaling strategy
- Add new loss functions
- Modify alpha weighting dynamically

### Multi-GPU Support

Currently single-GPU. To add multi-GPU:

1. Use HuggingFace `accelerate`
2. Update `TrainingArguments` with distributed settings
3. Ensure teacher model is properly replicated

## Environment-Specific Notes

### Local (M2/MPS/CPU)

- Suitable for BERT-size models
- Apple Silicon (M1/M2/M3) automatically uses MPS GPU acceleration
- Use `batch_size=8` for memory constraints
- Test with `--max_steps 10` first

### Google Colab (A100)

- Full BERT experiments: ~3 hours
- Llama with LoRA: ~6-8 hours
- Run training scripts directly or adapt for notebook usage

### HPC/SLURM (H200/A100)

- **Recommended:** Use conda for HPC environments
- Production runs for Llama models
- Auto-requeue handles 8-hour limits
- Configure in `scripts/run_slurm.sh`

```bash
# Named run (new)
sbatch scripts/run_slurm.sh mnli_00 configs/classification/mnli/bert_distilbert/ablations/01_frozen_base-teacher_kl.yaml
# -> outputs/mnli_00/

# Resume existing run
sbatch scripts/run_slurm.sh mnli_00
# -> outputs/mnli_00/ (continues from checkpoint)

# Auto-named run (includes timestamp)
sbatch scripts/run_slurm.sh configs/classification/mnli/bert_distilbert/ablations/01_frozen_base-teacher_kl.yaml
# -> outputs/mnli_default_2026-01-31_03-15Z/

# Named run with timestamp (unique each submission)
TIMESTAMP=1 sbatch scripts/run_slurm.sh mnli_00 configs/...
# -> outputs/mnli_00_2026-01-31_03-15Z/

# Force overwrite existing run
FORCE=1 sbatch scripts/run_slurm.sh mnli_00 configs/...

# Testing on cluster (10 steps only)
MAX_STEPS=10 sbatch scripts/run_slurm.sh configs/...

# Custom seed
SEED=123 sbatch scripts/run_slurm.sh mnli_00 configs/...
```

## Running Included Examples

```bash
# BERT → DistilBERT on MNLI (sentence entailment)
python distill.py --run_name mnli_baseline --config configs/classification/mnli/bert_distilbert/ablations/00_no-teacher_ce-only.yaml

# BERT → DistilBERT on QQP (question pairs)
python distill.py --run_name qqp_baseline --config configs/classification/qqp/bert_distilbert/ablations/00_no-teacher_ce-only.yaml

# Auto-named runs (includes timestamp)
python distill.py --config configs/classification/mnli/bert_distilbert/ablations/01_frozen_base-teacher_kl.yaml
# Creates: outputs/mnli_default_2026-01-31_12-00Z/

# Resume a run
python distill.py --run_name mnli_baseline
```

## Development

### Code Quality

This project uses pre-commit hooks for code quality:

```bash
# Install hooks (one-time setup)
pre-commit install

# Hooks will run automatically on git commit
# To run manually:
pre-commit run --all-files
```

### Testing

```bash
# Verify environment
python test_setup.py

# Run tests (when implemented)
pytest tests/
```

## License

MIT License - see LICENSE file

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Install development dependencies: `conda env create -f environment.yml`
3. Set up pre-commit hooks: `pre-commit install`
4. Create a feature branch
5. Make your changes
6. Run tests and linting: `pytest tests/` and `pre-commit run --all-files`
7. Submit a pull request

Areas for improvement:

- Multi-GPU support
- Seq2Seq model support (T5, BART, mBART)
- More evaluation metrics
- Additional dataset loaders

## References

This framework implements standard knowledge distillation techniques:

- Hinton et al. (2015): "Distilling the Knowledge in a Neural Network"
- Sanh et al. (2019): "DistilBERT, a distilled version of BERT"

Example configs demonstrate results from:

- Yue & Venkatesan (2024): "Less is More: Achieving Smaller, Faster Models for Edge Computing with Knowledge
  Distillation"
