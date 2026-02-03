# Documentation

Documentation and reference materials for the Knowledge Distillation project.

---

## Structure

```
docs/
├── codebase-todo.md  # Code review items needing attention
├── foundations/      # Foundational concepts (tasks, models, datasets)
├── archive/          # Original documentation (preserved)
├── guides/           # How-to & operational guides
├── gpu/              # GPU/hardware documentation
├── training/         # Training theory & dynamics
├── research/         # Research findings & methodology
├── planning/         # Direction & roadmaps
├── architecture/     # Technical reference
└── paper/            # Publication materials
```

---

## Quick Navigation

### Foundations
Core concepts for understanding ML tasks, modalities, and knowledge distillation.

| Document | Description |
|----------|-------------|
| [tasks-models-datasets.md](foundations/tasks-models-datasets.md) | ML modalities, task types, datasets, architectures, model selection |

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
| [research-framing.md](research/research-framing.md) | Research questions and key findings |
| [experimental-findings.md](research/experimental-findings.md) | Experiment results and analysis |
| [ablation-study.md](research/ablation-study.md) | 5-config ablation study design |
| [paper-evaluation.md](research/paper-evaluation.md) | Original paper analysis and critique |

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
| [diffusion-distillation.md](architecture/diffusion-distillation.md) | Diffusion model distillation guide |

### Paper
Publication materials.

| Document | Description |
|----------|-------------|
| [paper-draft.md](paper/paper-draft.md) | Draft manuscript structure |

### Maintenance
Code quality and maintenance tracking.

| Document | Description |
|----------|-------------|
| [codebase-todo.md](codebase-todo.md) | Code review findings and items needing attention |

---

## Key Concepts

### The Overconfident-Wrong Teacher Paradox
Our key finding: frozen SFT teachers can **hurt** student performance because confident wrong predictions fight ground truth gradients. See [research-framing.md](research/research-framing.md).

### Ablation Results (MNLI)

| Rank | Config | Accuracy | Description |
|------|--------|----------|-------------|
| 1 | 04 | 82.62% | Co-train SFT teacher |
| 2 | 03 | 82.06% | Co-train base teacher |
| 3 | 01 | 81.25% | Frozen base teacher |
| 4 | 00 | 80.94% | No teacher (CE only) |
| 5 | 02 | 79.26% | Frozen SFT teacher |

### RFA Factorial Design
Testing Response (R), Feature (F), Attention (A) combinations. See [experiment-decision-tree.md](planning/experiment-decision-tree.md).

---

## Archive

All original documentation is preserved in `archive/`. These files are the unedited source materials that were condensed into the reorganized structure above.

See [archive/](archive/) for the 22 original documents.
