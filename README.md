# Knowledge Distillation Framework

A universal, config-driven knowledge distillation framework for compressing any teacher-student model pair across
classification, generation, and translation tasks.

## Overview

This framework implements knowledge distillation using KL divergence + task loss to transfer knowledge from large "
teacher" models to smaller, faster "student" models. **Designed to be completely generic and extensible** - works with
any HuggingFace model pair.

### Key Features

- **Universal**: Works with any HuggingFace model pair (BERT, GPT, Llama, T5, etc.)
- **Multi-task**: Classification, generation, translation, or custom tasks
- **Production-ready**: SLURM integration, automatic job chaining, W&B tracking
- **Cross-platform**: Local (Apple Silicon MPS/CPU), Google Colab, HPC clusters
- **Config-driven**: Everything controlled via YAML configs

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
# Local testing (10 steps)
python train.py --config configs/your_config.yaml --output_dir ./outputs --max_steps 10

# Full training
python train.py --config configs/your_config.yaml --output_dir ./outputs

# SLURM (with auto-requeue)
sbatch scripts/run_slurm.sh configs/your_config.yaml ./outputs
```

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
- **Causal LM**: GPT-2, Llama, Mistral, etc.
- **Seq2Seq**: T5, BART, mBART (requires minor extension)

The framework automatically detects model architecture from HuggingFace model names.

### LoRA Support

For large models, enable LoRA in config:

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
3. **Llama 3.2 3B → 1B on EN-FR**: Neural machine translation

### More Examples

#### GPT-2 Distillation

```yaml
model:
  teacher: gpt2-medium
  student: gpt2
data:
  task: text_generation
  dataset: wikitext
```

#### T5 Summarization

```yaml
model:
  teacher: t5-base
  student: t5-small
data:
  task: summarization
  dataset: cnn_dailymail
```

#### Custom Domain

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
│   ├── examples/         # Generic example configs
│   └── [task].yaml       # Task-specific configs
├── src/
│   ├── data/            # Dataset loaders
│   ├── models/          # Model loading (generic)
│   ├── training/        # DistillationTrainer
│   ├── evaluation/      # Metrics
│   └── utils/           # Config, checkpointing, environment
├── scripts/             # Execution scripts
│   ├── run_local.sh
│   └── run_slurm.sh
├── test_setup.py        # Environment verification
└── train.py            # Main entry point
```

## Key Components

### DistillationTrainer

Extends HuggingFace `Trainer` with:

- Frozen teacher model
- Combined KL divergence + CE loss
- Temperature-scaled distributions
- Automatic loss component logging

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

The core loss is in `src/training/trainer.py` (`compute_loss` method). You can:

- Change temperature scaling strategy
- Add additional loss terms (e.g., attention distillation)
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

### HPC/SLURM (Explorer H200/V100)

- **Recommended:** Use conda for HPC environments
- Production runs for Llama models
- Auto-requeue handles 8-hour limits
- Configure in `scripts/run_slurm.sh`

## Running Included Examples

```bash
# BERT on MNLI (sentence entailment)
python train.py --config configs/bert_distilbert_mnli.yaml --output_dir ./outputs/mnli

# BERT on QQP (question pairs)
python train.py --config configs/bert_distilbert_qqp.yaml --output_dir ./outputs/qqp

# Llama on translation (EN-FR)
sbatch scripts/run_slurm.sh configs/llama_3b_1b_translation.yaml ./outputs/translation
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
- Seq2Seq model support
- Additional distillation strategies (attention, hidden states)
- More evaluation metrics
- Additional dataset loaders

## References

This framework implements standard knowledge distillation techniques:

- Hinton et al. (2015): "Distilling the Knowledge in a Neural Network"
- Sanh et al. (2019): "DistilBERT, a distilled version of BERT"

Example configs demonstrate results from:

- Yue & Venkatesan (2024): "Less is More: Achieving Smaller, Faster Models for Edge Computing with Knowledge
  Distillation"
