# Configuration Directory Structure

```
configs/
├── base.yaml                              # Root defaults (training, logging, checkpointing)
├── classification/
│   ├── base.yaml                          # Classification defaults (T=2, α=0.5)
│   ├── mnli/
│   │   ├── base.yaml                      # MNLI task passthrough
│   │   └── bert_distilbert/
│   │       ├── base.yaml                  # Model pair: BERT→DistilBERT on MNLI
│   │       ├── rfa/
│   │       │   ├── base.yaml              # Passthrough
│   │       │   └── *.yaml                 # RFA factorial study (8 configs)
│   │       ├── staged/
│   │       │   ├── base.yaml              # Passthrough
│   │       │   └── *.yaml                 # Staged distillation (F→R, A→R, F→A→R)
│   │       ├── ablations/
│   │       │   ├── base.yaml              # Passthrough
│   │       │   └── *.yaml                 # Teacher/training ablations
│   │       └── gpu/
│   │           ├── base.yaml              # Passthrough
│   │           └── {gpu_type}/
│   │               ├── base.yaml          # Passthrough
│   │               └── *.yaml             # GPU-specific batch/LR configs
│   ├── sst2/                              # SST-2 sentiment classification
│   ├── cola/                              # CoLA linguistic acceptability
│   └── qqp/                               # QQP paraphrase detection
└── generation/
    ├── base.yaml                          # Generation defaults (T=15, α=0.2)
    └── translation/
        ├── base.yaml                      # Translation task (BLEU/COMET eval)
        └── llama3p2-3b_llama3p2-1b/
            ├── base.yaml                  # Model pair: Llama 3.2 3B→1B
            ├── rfa/
            │   ├── base.yaml              # Passthrough
            │   └── *.yaml                 # RFA factorial study
            ├── staged/
            │   ├── base.yaml              # Passthrough
            │   └── *.yaml                 # Staged distillation
            └── ablations/
                ├── base.yaml              # Passthrough
                └── *.yaml                 # Teacher/training ablations
```

## Inheritance Convention

**Rule: All configs inherit from `../base.yaml`** (always one level up).

Every directory has a `base.yaml` that chains to its parent. This creates a clean inheritance tree:

```
base.yaml                                    # Root defaults
    └── {type}/base.yaml                     # Classification (T=2) or Generation (T=15)
        └── {task}/base.yaml                 # Task passthrough
            └── {model_pair}/base.yaml       # Model/data config
                └── {experiment}/base.yaml   # Experiment passthrough
                    └── *.yaml               # Final configs (inherit: ../base.yaml)
```

## Key Parameters by Task Type

| Type           | Temperature | Alpha | Rationale                              |
|----------------|-------------|-------|----------------------------------------|
| Classification | 2.0         | 0.5   | Small output space (2-3 classes)       |
| Generation     | 15.0        | 0.2   | Large vocab (32k-128k tokens)          |

## Run Naming Convention

Format: `{task}_{experiment}_{config}_{timestamp}`

Examples:
- `mnli_rfa_01-r-only_20240115-1430Z`
- `translation_staged_02-a-r_20240115-1430Z`

Timestamp: `YYYYMMDD-HHMMZ` (UTC)

## Experiment Types

| Experiment   | Description                                           |
|--------------|-------------------------------------------------------|
| `rfa/`       | Factorial study: 8 combinations of R/F/A signals      |
| `staged/`    | Curriculum: F→R, A→R, F→A→R progressions              |
| `ablations/` | Teacher variants, training modes, loss configurations |
| `gpu/`       | Hardware-specific batch/LR scaling                    |

## Creating New Configs

1. Place config in appropriate directory
2. **Always inherit from `../base.yaml`**
3. Override only what differs from parent

```yaml
# configs/classification/mnli/bert_distilbert/ablations/my_experiment.yaml
inherit: ../base.yaml

distillation:
  alpha: 0.7  # Override only what changes
```

Adding a new experiment directory:
```yaml
# configs/classification/mnli/bert_distilbert/my_experiment/base.yaml
inherit: ../base.yaml
```
