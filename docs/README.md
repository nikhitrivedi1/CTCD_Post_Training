# Documentation

Documentation and reference materials for the Knowledge Distillation project.

---

## Structure

```
docs/
├── foundations/      # Foundational concepts (tasks, models, datasets)
├── guides/           # How-to & operational guides
├── gpu/              # GPU/hardware documentation
├── training/         # Training theory & dynamics
├── research/         # Research questions & findings
│   ├── framing/      # Hypotheses & study design
│   ├── references/   # Literature & paper critiques
│   └── findings/     # Raw experimental findings
├── analysis/         # W&B data analysis
│   ├── data/         # Downloaded metrics (CSV/JSON)
│   ├── scripts/      # Fetch & analysis scripts
│   ├── notebooks/    # Jupyter notebooks
│   └── summaries/    # Markdown summaries
├── paper/            # Publication materials
│   ├── paper-0-framework/
│   ├── paper-1-ablation/
│   ├── paper-2-far/
│   └── paper-3-translation/
├── planning/         # Direction & roadmaps
├── architecture/     # Technical reference
└── archive/          # Original documentation (preserved)
```

---

## Quick Navigation

### Foundations
Core concepts for understanding ML tasks, modalities, and knowledge distillation.

| Document | Description |
|----------|-------------|
| [tasks-models-datasets.md](foundations/tasks-models-datasets.md) | ML paradigms, task types, datasets, model selection |
| [transformer-architectures.md](foundations/transformer-architectures.md) | Transformer internals: MLP, attention, encoder/decoder |

### Guides
Operational how-to documentation.

| Document | Description |
|----------|-------------|
| [config-parameters.md](guides/config-parameters.md) | Config file parameter reference |
| [run-options.md](guides/run-options.md) | CLI reference and run management |
| [explorer-hpc.md](guides/explorer-hpc.md) | HPC cluster setup and job submission |

### GPU
Hardware and memory optimization.

| Document | Description |
|----------|-------------|
| [memory-math.md](gpu/memory-math.md) | GPU memory calculations and optimization |
| [comparison.md](gpu/comparison.md) | GPU hardware comparison (V100, A100, H100) |

### Training
Training theory and dynamics.

| Document | Description |
|----------|-------------|
| [optimizers-and-lr.md](training/optimizers-and-lr.md) | Optimizer deep dive (SGD, Adam, AdamW) |
| [batch-size-lr-scaling.md](training/batch-size-lr-scaling.md) | Linear scaling rule and warmup |
| [training-dynamics.md](training/training-dynamics.md) | Small vs large batch training |

### Research
Findings and methodology.

| Document | Description |
|----------|-------------|
| [framing/research-framing.md](research/framing/research-framing.md) | Research questions and key findings |
| [framing/ablation-study.md](research/framing/ablation-study.md) | 5-config ablation study design |
| [findings/experimental-findings.md](research/findings/experimental-findings.md) | Experiment results and analysis |
| [references/paper-evaluation.md](research/references/paper-evaluation.md) | Original paper analysis and critique |

### Analysis
W&B data analysis workflow.

| Document | Description |
|----------|-------------|
| [README.md](analysis/README.md) | Analysis workflow overview |
| [scripts/fetch_wandb.py](analysis/scripts/fetch_wandb.py) | Script to download W&B data |

### Workbench
Post-hoc analysis tools for trained models. See [workbench/results/probing_analysis.md](../workbench/results/probing_analysis.md) for results.

| Command | Description |
|---------|-------------|
| `probe` | Linear probing - measures task information in each layer |
| `cka` | CKA similarity - compares teacher/student representations |
| `attention-entropy` | Attention entropy - measures focus vs diffuse attention |
| `disagreement` | Prediction disagreement and per-class accuracy breakdown |

Usage: `python workbench/analyze.py <command> --teacher <model> --student <checkpoint> --task <task>`

### Paper
Publication materials.

| Document | Description |
|----------|-------------|
| [README.md](paper/README.md) | Paper series overview and narrative arc |
| [paper-0-framework/draft.md](paper/paper-0-framework/draft.md) | Unified Framework for KD Research |
| [paper-1-ablation/draft.md](paper/paper-1-ablation/draft.md) | When Teachers Fail: Frozen vs Co-training |
| [paper-2-far/draft.md](paper/paper-2-far/draft.md) | Simplicity Wins: R-only beats multi-signal |
| [paper-3-translation/draft.md](paper/paper-3-translation/draft.md) | Cross-task validation on generation |

### Planning
Direction and roadmaps.

| Document | Description |
|----------|-------------|
| [future-direction.md](planning/future-direction.md) | Task x Depth framework, experiment planning |
| [framework-direction.md](planning/framework-direction.md) | Mastery path and learning roadmap |
| [distillation-tasks.md](planning/distillation-tasks.md) | Task catalog (MNLI, QQP, Translation) |
| [experiment-decision-tree.md](planning/experiment-decision-tree.md) | L0/L1/L2 experiment hierarchy |

### Architecture
Technical implementation reference.

| Document | Description |
|----------|-------------|
| [architecture.md](architecture/architecture.md) | Framework modules and extension points |

---

## Key Concepts

### The Overconfident-Wrong Teacher Paradox
Our key finding: frozen SFT teachers can **hurt** student performance because confident wrong predictions fight ground truth gradients. See [research-framing.md](research/framing/research-framing.md).

### Ablation Results (MNLI)

| Rank | Config | Accuracy | Description |
|------|--------|----------|-------------|
| 1 | 04 | 82.62% | Co-train SFT teacher |
| 2 | 03 | 82.06% | Co-train base teacher |
| 3 | 01 | 81.25% | Frozen base teacher |
| 4 | 00 | 80.94% | No teacher (CE only) |
| 5 | 02 | 79.26% | Frozen SFT teacher |

### FAR Factorial Design
Testing Feature (F), Attention (A), Response (R) combinations. See [experiment-decision-tree.md](planning/experiment-decision-tree.md).

---

## Archive

All original documentation is preserved in `archive/`. These files are the unedited source materials that were condensed into the reorganized structure above.

See [archive/](archive/) for the 22 original documents.
