# Configuration Directory Structure

```
configs/
├── base.yaml                              # Root defaults (training, logging, checkpointing)
├── CTCD_Experiments/                      # The paper's config suite — see below
└── classification/                        # Pre-paper exploratory suite — see note below
    ├── base.yaml                          # Classification defaults (T=2, α=0.5)
    ├── mnli/
    │   ├── base.yaml                      # MNLI task passthrough
    │   └── distilbert/
    │       ├── 00_sft.yaml               # SFT baseline (no teacher)
    │       └── bert-base/
    │           ├── base.yaml              # Model pair: BERT→DistilBERT on MNLI
    │           ├── ablations/             # Teacher/training ablations (4 configs)
    │           ├── far/                   # FAR factorial study (8 configs)
    │           ├── staged/                # Staged distillation (4 configs)
    │           └── gpu/                   # GPU-specific batch/LR configs (10)
    ├── qqp/                               # QQP paraphrase detection (SFT + 4 ablations)
    ├── sst2/                              # SST-2 sentiment (SFT + 4 ablations)
    ├── cola/                              # CoLA linguistic acceptability (SFT + 4 ablations)
    ├── mrpc/                              # MRPC paraphrase detection (SFT + 4 ablations)
    ├── qnli/                              # QNLI QA entailment (SFT + 4 ablations)
    ├── rte/                               # RTE textual entailment (SFT + 4 ablations)
    ├── wnli/                              # WNLI Winograd NLI (SFT + 4 ablations)
    └── clinc150/                          # CLINC150 intent classification (SFT baseline)
        ├── base.yaml                      # CLINC150 task passthrough
        └── distilbert/
            ├── base.yaml                  # DistilBERT model defaults
            └── 00_sft.yaml                # SFT baseline (no teacher)
```

## `CTCD_Experiments/` — the paper's config suite

This is the config tree that actually produced the paper's results (see root `README.md` and
`configs/CTCD_Experiments/CTCD_PART1_TEAM_RUNBOOK.md`). Six ablations per task, across mnli, qqp, qnli, cola,
mrpc, rte, sst2, and clinc150:

| Config | Teacher | Mode | Description |
|--------|---------|------|-------------|
| 00 | None | – | SFT, CE only (no distillation) |
| 01 | Base (non-SFT) | Frozen | Frozen pretrained teacher |
| 02 | SFT | Frozen | Frozen task-fine-tuned teacher |
| 03 | Base (non-SFT) | Co-train | Co-trained teacher, CE-only teacher loss (`beta_s=0`) |
| 04 | SFT | Co-train | Co-trained teacher, CE-only teacher loss (`beta_s=0`) |
| 05 | Base (non-SFT) | Co-train + co-distill | Full CTCD (`beta_h=1`, `beta_s=1`) |

`_shared.yaml` holds cross-task defaults (loss type, temperature, alpha/beta weights); each task's `base.yaml`
layers task-specific cadence/eval overrides on top. `best/` and `best_seed/` hold the winning hyperparameter
configs from the Part 1 grid search and multi-seed replication, respectively.

## `classification/` — pre-paper exploratory suite (not the paper's results)

**This directory predates `CTCD_Experiments/` and is not what the paper's numbers come from.** It's a broader
playground built while exploring the design space — more tasks (including WNLI, which isn't in the paper) and
more experiment types (FAR factorial, staged distillation, GPU hardware variants). It's kept for reference; use
`CTCD_Experiments/` to reproduce the paper.

## Inheritance Convention

**Rule: All configs inherit from `../base.yaml`** (always one level up).

Every directory has a `base.yaml` that chains to its parent:

```
base.yaml                                    # Root defaults
    └── classification/base.yaml             # Classification defaults
        └── {task}/base.yaml                 # Task passthrough (mnli, qqp, etc.)
            └── {model_pair}/base.yaml       # Model/data config
                └── {experiment}/base.yaml   # Experiment passthrough
                    └── *.yaml               # Final configs (inherit: ../base.yaml)
```

## Experiment Types (`classification/`)

| Experiment   | Description                                           |
|--------------|-------------------------------------------------------|
| `ablations/` | Teacher variants (frozen/co-train, base/SFT)          |
| `far/`       | FAR factorial: 8 combinations of F/A/R signals        |
| `staged/`    | Curriculum: F→R, A→R, F→A→R progressions              |
| `gpu/`       | Hardware-specific batch/LR scaling                    |

## Run Naming Convention

Format: `{task}_{experiment}_{config}_{timestamp}`

Example: `mnli_far_03-r-only_20240115-1430Z`

Timestamp: `YYYYMMDD-HHMMZ` (UTC)

## Design Rationale

**Path structure:** `configs/classification/{task}/{model_pair}/{experiment}/{config}.yaml`

Two structures were considered:

1. `{task}/{model_pair}/{experiment}/` (current)
2. `{task}/{experiment}/{model_pair}/`

We chose (1) because it matches the research workflow: fix the model pair (BERT→DistilBERT), validate an
experiment on one task (MNLI), then replicate across the other GLUE tasks (QQP, SST-2, CoLA, ...).

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
