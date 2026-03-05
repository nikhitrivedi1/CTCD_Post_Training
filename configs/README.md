# Configuration Directory Structure

**113 config files** (70 runnable + 38 base.yaml + 5 meta).

```
configs/
├── base.yaml                              # Root defaults (training, logging, checkpointing)
├── TEMPLATE.yaml                          # Config parameter reference
├── examples/                              # Example configs (4)
├── classification/
│   ├── base.yaml                          # Classification defaults (T=2, α=0.5)
│   ├── mnli/
│   │   ├── base.yaml                      # MNLI task passthrough
│   │   └── distilbert/
│   │       ├── 00_sft.yaml               # SFT baseline (no teacher)
│   │       └── bert-base/
│   │           ├── base.yaml              # Model pair: BERT→DistilBERT on MNLI
│   │           ├── ablations/             # Teacher/training ablations (4 configs)
│   │           ├── far/                   # FAR factorial study (8 configs)
│   │           ├── staged/                # Staged distillation (4 configs)
│   │           └── gpu/                   # GPU-specific batch/LR configs (10)
│   ├── qqp/                               # QQP paraphrase detection (SFT + 4 ablations)
│   ├── sst2/                              # SST-2 sentiment (SFT + 4 ablations)
│   └── cola/                              # CoLA linguistic acceptability (SFT + 4 ablations)
├── translation/
│   ├── base.yaml                          # Translation defaults (T=15, α=0.2, BLEU eval)
│   └── europarl/
│       └── llama32-1b/
│           ├── 00_sft.yaml               # SFT baseline
│           └── llama32-3b/
│               ├── base.yaml              # Model pair: Llama 3.2 3B→1B
│               ├── ablations/             # Teacher/training ablations (4 configs)
│               ├── far/                   # FAR factorial study (8 configs)
│               └── staged/                # Staged distillation (4 configs)
└── ocr/
    └── olmocr-mix/
        ├── qwen3vl-2b/                    # SFT + 1 KD config
        └── qwen25vl-3b/                   # SFT + 2 KD configs
```

## Config Counts

| Category | Count |
|----------|-------|
| Classification SFT | 4 (1 per task) |
| Classification Ablations | 17 (4 per task + 1 obsolete) |
| Classification FAR | 8 (MNLI) |
| Classification Staged | 4 (MNLI) |
| Classification GPU | 10 (MNLI hardware variants) |
| Translation SFT | 1 |
| Translation Ablations | 4 |
| Translation FAR | 8 |
| Translation Staged | 4 |
| OCR SFT + KD | 6 |
| Examples | 4 |
| **Total Runnable** | **70** |

Plus 38 base.yaml files + TEMPLATE.yaml = 113 total YAML files.

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

## Design Rationale

**Path structure:** `configs/{type}/{task}/{model_pair}/{experiment}/{config}.yaml`

Two structures were considered:

1. `{task}/{model_pair}/{experiment}/` (current)
2. `{task}/{experiment}/{model_pair}/`

We chose (1) because it matches our research workflow:

1. **Fix model pair** → Run all experiments on BERT→DistilBERT + MNLI
2. **Validate findings** → Same model pair on other tasks (QQP, SST-2, CoLA)
3. **Scale model pair** → Try larger models (LLaMA 3B→1B)

The alternative (2) would be better if comparing model pairs within experiments, but our research fixes the model pair first and varies experiments.

## Research Workflow

### Phase 1: Ablation Study (Teacher Setup)
Establish baseline with BERT→DistilBERT on MNLI:
- `00_sft` - Student alone (lower bound)
- `01_kd_frozen-base` - Frozen pretrained teacher
- `02_kd_frozen-sft` - Frozen task-trained teacher
- `03_kd_cotrain-base` - Co-training from pretrained
- `04_kd_cotrain-sft` - Co-training from task-trained

### Phase 2: FAR Factorial Study
8 combinations of Feature/Attention/Response signals on best teacher setup.

### Phase 3: Staged Distillation
Progressive signal introduction:
- `00_r` - Response only
- `01_f-r` - Feature → Response
- `02_a-r` - Attention → Response
- `03_f-a-r` - Feature → Attention → Response

### Phase 4: Cross-Task Validation
Replicate ablations on QQP, SST-2, CoLA to verify findings generalize.

### Phase 5: Model Pair Scaling
Apply best practices to LLaMA 3B→1B on Europarl translation.

## W&B Organization

Runs are automatically grouped and tagged based on config path:
- **Group:** `{task}_{experiment}` (e.g., `mnli_ablations`)
- **Tags:** `[task, experiment, model_pair]` (e.g., `[mnli, ablations, distilbert, bert-base]`)

## Creating New Configs

1. Place config in appropriate directory
2. **Always inherit from `../base.yaml`**
3. Override only what differs from parent

```yaml
# configs/classification/mnli/distilbert/bert-base/ablations/my_experiment.yaml
inherit: ../base.yaml

distillation:
  alpha: 0.7  # Override only what changes
```

Adding a new experiment directory:
```yaml
# configs/classification/mnli/distilbert/bert-base/my_experiment/base.yaml
inherit: ../base.yaml
```
