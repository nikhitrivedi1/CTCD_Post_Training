# Forge

A universal, config-driven framework for fine-tuning, knowledge distillation, and reinforcement learning across any HuggingFace model pair.

## Overview

Forge unifies three training paradigms under a single config-driven interface:

- **Fine-tuning (SFT)**: Standard supervised training with cross-entropy loss
- **Knowledge Distillation (KD)**: Transfer knowledge from a teacher to a smaller student via KL divergence, feature matching, and attention transfer
- **Reinforcement Learning (RL)**: Post-training with GRPO, DPO, and other policy optimization methods

Every experiment is a YAML config change, not a code change. The framework handles model loading, training loops, checkpointing, SLURM job chaining, and W&B logging. You bring the config.

### Key Features

- **Universal**: Any HuggingFace model pair - encoders (BERT), decoders (Llama), VLMs (Qwen-VL)
- **Multi-task**: Classification, translation, OCR, or custom tasks
- **Three training modes**: SFT, KD (frozen/co-trained teacher), RL (GRPO/DPO)
- **Config-driven**: YAML configs with inheritance, no code changes between experiments
- **Production-ready**: SLURM integration, automatic job chaining, W&B tracking
- **Cross-platform**: Local (Apple Silicon MPS/CPU), Google Colab, HPC clusters

### Two Complementary Tools

| | **Forge** | **Workbench** |
|---|---|---|
| **Purpose** | Train models | Analyze trained models |
| **Entry point** | `forge.py` | `workbench/analyze.py` |
| **Runtime** | Hours to days | Minutes |
| **Resources** | GPU cluster, full datasets | Local CPU, sampled data |

**Forge** is the training engine - it runs large-scale experiments on GPU clusters with checkpointing, SLURM job chaining, and W&B logging. You point it at a config and let it run.

**Workbench** is the analysis bench - after training, use it for rapid post-hoc analysis: linear probing, attention entropy, prediction disagreement, CKA similarity. Runs in minutes on CPU with sampled data.

```bash
# Train a model (hours)
python forge.py --config configs/classification/mnli/bert_distilbert/ablations/01_frozen_base-teacher_kl.yaml

# Analyze the result (minutes)
python workbench/analyze.py --test probe --teacher textattack/bert-base-uncased-MNLI --student outputs/mnli_00 --task mnli
```

See also: [Forge vs Unsloth](docs/forge-vs-unsloth.md)

## Quick Start

### Installation

```bash
conda env create -f environment.yml
conda activate forge
source scripts/env/init.sh
```

`init.sh` installs conda hooks that automatically set `PATH`, `LD_LIBRARY_PATH`,
and `HF_HOME` on every `conda activate forge`. Run it whenever the
environment is created or recreated. See [scripts/README.md](scripts/README.md)
for details.

### Verify Installation

```bash
python test_setup.py
```

### Basic Usage

```bash
# New run (auto-named with timestamp)
python forge.py --config configs/classification/mnli/bert_distilbert/ablations/01_frozen_base-teacher_kl.yaml

# New run with custom name
python forge.py --run_name mnli_00 --config configs/classification/mnli/bert_distilbert/ablations/00_no-teacher_ce-only.yaml

# Resume existing run
python forge.py --run_name mnli_00

# Overwrite existing run
python forge.py --run_name mnli_00 --config configs/... --force

# Local testing (10 steps)
python forge.py --config configs/... --max_steps 10

# SLURM (with auto-requeue)
sbatch scripts/run_training.sh configs/your_config.yaml
```

See [docs/guides/run-options.md](docs/guides/run-options.md) for complete CLI reference.

## Training Modes

### Fine-Tuning (SFT)

Standard supervised training. The baseline for all experiments.

```yaml
model:
  name: distilbert-base-uncased
# No teacher, no RL - just CE loss
```

### Knowledge Distillation (KD)

Transfer knowledge from a larger teacher to a smaller student. Supports frozen or co-trained teachers, with three distillation signal types that can be combined:

- **Response (R)**: KL divergence on output logits
- **Feature (F)**: Hidden state alignment via MSE or cosine similarity
- **Attention (A)**: Attention weight transfer via KL divergence

```yaml
model:
  name: distilbert-base-uncased
  teacher:
    name: bert-base-uncased
    freeze: true          # frozen or co-trained

distillation:
  temperature: 2.0
  alpha: 0.5              # balance between KL and CE loss
  feature_weight: 0.1     # optional feature matching
  attention_weight: 0.1   # optional attention transfer
```

### Reinforcement Learning (RL)

Post-training with policy optimization. Supports multiple algorithms and reward sources.

```yaml
model:
  name: qwen3vl-2b
rl:
  algorithm: grpo          # grpo | dpo | kto
  reward: verifiable       # verifiable | teacher | judge
  reference: self          # KL anchor: self | teacher | path
  num_generations: 8
```

## Supported Tasks

| Task | Models | Fine-tuning | Eval Metric |
|------|--------|-------------|-------------|
| **Classification** (MNLI, QQP, SST-2, CoLA) | Encoder (BERT, DistilBERT) | Full params (66M-110M) | Accuracy / F1 / MCC |
| **Translation** (Europarl EN-FR) | Decoder (Llama 3.2) | LoRA (1B-3B) | BLEU |
| **OCR** (olmocr-mix) | VLM (Qwen3-VL) | LoRA (2B-7B) | Edit similarity |

### LoRA Support

For large models (1B+ parameters), LoRA keeps memory usage manageable:

```yaml
training:
  use_lora: true
  lora:
    r: 16
    alpha: 16
    dropout: 0.05
    target_modules: ["q_proj", "v_proj"]
```

## Architecture

```
forge/
├── configs/              # YAML configurations with inheritance
│   ├── base.yaml         # Shared defaults
│   ├── classification/   # MNLI, QQP, SST-2, CoLA configs
│   ├── translation/      # Europarl EN-FR configs
│   └── ocr/              # olmocr-mix configs
├── src/
│   ├── data/             # Dataset loaders
│   ├── models/           # Model loading (generic)
│   ├── training/         # Trainer classes
│   │   ├── trainer.py    # FinetuningTrainer, DistillationTrainer
│   │   ├── rl_trainer.py # RLTrainer (extends trl.GRPOTrainer)
│   │   └── losses.py     # KL, feature, attention losses
│   ├── evaluation/       # Metrics
│   └── utils/            # Config, checkpointing, environment
├── workbench/            # Post-hoc analysis
│   ├── analyze.py        # CLI: --test probe|cka|attention-entropy|disagreement|all
│   ├── probing.py        # Linear probing per layer
│   ├── cka.py            # CKA similarity between layers
│   ├── attention.py      # Attention entropy analysis
│   └── disagreement.py   # Teacher-student prediction differences
├── scripts/              # Execution scripts
│   ├── run_local.sh
│   └── run_training.sh   # SLURM with auto-requeue
├── test_setup.py         # Environment verification
└── forge.py            # Entry point
```

## Key Components

### Trainer Hierarchy

```
HuggingFace Trainer
  ├── FinetuningTrainer            # SFT: LoRA, VLM collation, eval, checkpointing
  │     └── DistillationTrainer    # + teacher, KL/feature/attention loss
  └── trl.GRPOTrainer
        └── RLTrainer              # + reward dispatch, teacher-as-reward-model
```

Config determines which trainer is used. No code changes needed.

### Config System

- YAML-based with inheritance (`inherit: base.yaml`)
- Environment-specific overrides
- All hyperparameters in one place

### SLURM Integration

- Automatic job chaining for runs exceeding time limits
- Signal handling (SIGUSR1) for clean checkpointing
- W&B run continuation across restarts

## Environment-Specific Notes

### Local (M2/MPS/CPU)

- Suitable for BERT-size models
- Apple Silicon automatically uses MPS GPU acceleration
- Use `batch_size=8` for memory constraints
- Test with `--max_steps 10` first

### Google Colab (A100)

- Full BERT experiments: ~3 hours
- Llama with LoRA: ~6-8 hours

### HPC/SLURM (H200/A100)

- Production runs for large models
- Auto-requeue handles time limits
- Configure in `scripts/run_training.sh`

```bash
# Named run
sbatch scripts/run_training.sh mnli_00 configs/classification/mnli/bert_distilbert/ablations/01_frozen_base-teacher_kl.yaml

# Resume existing run
sbatch scripts/run_training.sh mnli_00

# Force overwrite
FORCE=1 sbatch scripts/run_training.sh mnli_00 configs/...

# Testing on cluster (10 steps)
MAX_STEPS=10 sbatch scripts/run_training.sh configs/...
```

## Development

### Code Quality

```bash
# Install hooks (one-time setup)
pre-commit install

# Run manually
pre-commit run --all-files
```

## License

MIT License - see LICENSE file

## References

- Hinton et al. (2015): "Distilling the Knowledge in a Neural Network"
- Sanh et al. (2019): "DistilBERT, a distilled version of BERT"
- Shao et al. (2024): "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" (GRPO)
- Rafailov et al. (2023): "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
