# Configuration Directory Structure

**88 runnable configs** across classification and translation tasks.

```
configs/
├── base.yaml                              # Root defaults (training, logging, checkpointing)
├── classification/
│   ├── base.yaml                          # Classification defaults (T=2, α=0.5)
│   ├── mnli/
│   │   ├── base.yaml                      # MNLI task passthrough
│   │   └── bert_distilbert/
│   │       ├── base.yaml                  # Model pair: BERT→DistilBERT on MNLI
│   │       ├── ablations/                 # Teacher/training ablations (5 configs)
│   │       ├── far/                       # FAR factorial study (8 configs)
│   │       ├── staged/                    # Staged distillation (4 configs)
│   │       └── gpu/                       # GPU-specific batch/LR configs
│   ├── qqp/                               # QQP paraphrase detection (5 ablations)
│   ├── sst2/                              # SST-2 sentiment (5 ablations)
│   └── cola/                              # CoLA linguistic acceptability (5 ablations)
└── translation/
    ├── base.yaml                          # Translation defaults (T=15, α=0.2, BLEU eval)
    └── llama3p2-3b_llama3p2-1b/
        ├── base.yaml                      # Model pair: Llama 3.2 3B→1B
        ├── ablations/                     # Teacher/training ablations (5 configs)
        ├── far/                           # FAR factorial study (8 configs)
        └── staged/                        # Staged distillation (4 configs)
```

## Config Counts

| Category | Count |
|----------|-------|
| Classification Ablations | 20 (5 × 4 tasks) |
| Classification FAR | 8 (MNLI) |
| Classification Staged | 4 (MNLI) |
| Translation Ablations | 5 |
| Translation FAR | 8 |
| Translation Staged | 4 |
| **Total Runnable** | **49** |

Plus 39 base.yaml files for inheritance = 88 total YAML files.

## Inheritance Convention

**Rule: All configs inherit from `../base.yaml`** (always one level up).

Every directory has a `base.yaml` that chains to its parent:

```
base.yaml                                    # Root defaults
    └── {task_type}/base.yaml                # classification/ or translation/
        └── {task}/base.yaml                 # Task passthrough (mnli, qqp, etc.)
            └── {model_pair}/base.yaml       # Model/data config
                └── {experiment}/base.yaml   # Experiment passthrough
                    └── *.yaml               # Final configs (inherit: ../base.yaml)
```

## Key Parameters by Task Type

| Type           | Temperature | Alpha | Rationale                              |
|----------------|-------------|-------|----------------------------------------|
| Classification | 2.0         | 0.5   | Small output space (2-3 classes)       |
| Translation    | 15.0        | 0.2   | Large vocab (128k tokens)              |

## Experiment Types

| Experiment   | Description                                           |
|--------------|-------------------------------------------------------|
| `ablations/` | Teacher variants (frozen/co-train, base/SFT)          |
| `far/`       | FAR factorial: 8 combinations of F/A/R signals        |
| `staged/`    | Curriculum: F→R, A→R, F→A→R progressions              |
| `gpu/`       | Hardware-specific batch/LR scaling                    |

## FAR Factorial Design

**F**eature → **A**ttention → **R**esponse ordering:

| Config | F | A | R | Description |
|--------|---|---|---|-------------|
| 00 | 0 | 0 | 0 | CE only (baseline) |
| 01 | 1 | 0 | 0 | Feature only |
| 02 | 0 | 1 | 0 | Attention only |
| 03 | 0 | 0 | 1 | Response only (classic KL) |
| 04 | 1 | 1 | 0 | Feature + Attention |
| 05 | 1 | 0 | 1 | Feature + Response |
| 06 | 0 | 1 | 1 | Attention + Response |
| 07 | 1 | 1 | 1 | Full FAR |

## Ablation Study Design

| Config | Teacher | Mode | Description |
|--------|---------|------|-------------|
| 00 | None | - | CE only (no distillation) |
| 01 | Base | Frozen | Frozen base teacher |
| 02 | SFT | Frozen | Frozen fine-tuned teacher |
| 03 | Base | Co-train | Co-trained base teacher |
| 04 | SFT | Co-train | Co-trained SFT teacher |

## Run Naming Convention

Format: `{task}_{experiment}_{config}_{timestamp}`

Examples:
- `mnli_far_03-r-only_20240115-1430Z`
- `translation_staged_02-a-r_20240115-1430Z`

Timestamp: `YYYYMMDD-HHMMZ` (UTC)

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
