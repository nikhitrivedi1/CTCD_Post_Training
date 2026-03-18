# Documentation

Reference materials for Forge, an open training framework for fine-tuning, knowledge distillation, and reinforcement learning.

---

## Structure

```
docs/
├── foundations/      # Foundational ML concepts
├── guides/           # How-to and operational guides
├── gpu/              # GPU hardware and memory optimization
├── training/         # Training theory and dynamics
├── research/         # Research questions, findings, and references
├── analysis/         # W&B data, benchmarks, and summaries
├── paper/            # Publication drafts
├── planning/         # Direction, roadmaps, and design docs
└── architecture/     # Technical implementation reference
```

---

## Quick Navigation

### Foundations

Core concepts for understanding ML tasks, modalities, and knowledge distillation.

| Document | Description |
|----------|-------------|
| [tasks-models-datasets.md](foundations/tasks-models-datasets.md) | ML paradigms, task types, datasets, model selection |
| [transformer-architectures.md](foundations/transformer-architectures.md) | Transformer internals: MLP, attention, encoder/decoder |
| [quantization.md](foundations/quantization.md) | Weight quantization: INT8, INT4, GPTQ, AWQ |

### Guides

Operational how-to documentation.

| Document | Description |
|----------|-------------|
| [config-parameters.md](guides/config-parameters.md) | Config file parameter reference |
| [run-options.md](guides/run-options.md) | CLI reference and run management |
| [explorer-hpc.md](guides/explorer-hpc.md) | HPC cluster setup and job submission |
| [logging.md](guides/logging.md) | Logging architecture and best practices |
| [metrics.md](guides/metrics.md) | All training/eval metrics reference |
| [scaling.md](guides/scaling.md) | Scaling from 1 GPU to distributed training |
| [tmux_chain-orchestrator.md](guides/tmux_chain-orchestrator.md) | Running chain orchestrator in tmux |
| [ppo-dual-lora.md](guides/ppo-dual-lora.md) | PPO with dual LoRA adapter swapping (design doc) |
| [forge-vs-unsloth.md](guides/forge-vs-unsloth.md) | Forge vs Unsloth: when to use which |
| [mnli-model-label-mapping.md](guides/mnli-model-label-mapping.md) | MNLI label ordering across pretrained models |

### GPU

Hardware and memory optimization.

| Document | Description |
|----------|-------------|
| [memory-math.md](gpu/memory-math.md) | GPU memory calculations and optimization |
| [memory-landscape.md](gpu/memory-landscape.md) | Memory ceiling vs optimization trade-offs |
| [comparison.md](gpu/comparison.md) | GPU hardware comparison (V100, A100, H100, H200) |
| [utilization.md](gpu/utilization.md) | GPU utilization and training dynamics balance |
| [multi-gpu.md](gpu/multi-gpu.md) | Multi-GPU training strategies |
| [parallelization.md](gpu/parallelization.md) | CPU/GPU/mixed parallelization flows |

### Training

Training theory and dynamics.

| Document | Description |
|----------|-------------|
| [optimizers-and-lr.md](training/optimizers-and-lr.md) | Optimizer deep dive (SGD, Adam, AdamW) |
| [batch-size-lr-scaling.md](training/batch-size-lr-scaling.md) | Linear scaling rule and warmup |
| [training-dynamics.md](training/training-dynamics.md) | Small vs large batch training dynamics |

### Research

Findings, methodology, and literature.

| Document | Description |
|----------|-------------|
| [framing/research-framing.md](research/framing/research-framing.md) | Research questions and key findings |
| [framing/ablation-study.md](research/framing/ablation-study.md) | 5-config ablation study design |
| [findings/experimental-findings.md](research/findings/experimental-findings.md) | Experiment results and analysis |
| [distillation-factors.md](research/distillation-factors.md) | KD design space: all factors that matter |
| [layer-analysis.md](research/layer-analysis.md) | Layer-wise analysis for distillation |
| [loss_comparison.md](research/loss_comparison.md) | Loss functions: literature vs implementation |
| [references/paper-evaluation.md](research/references/paper-evaluation.md) | Original paper analysis and critique |

### Analysis

W&B data analysis, benchmarks, and summaries.

| Document | Description |
|----------|-------------|
| [README.md](analysis/README.md) | Analysis workflow overview |
| [bench/formatting.md](analysis/bench/formatting.md) | Formatting benchmark results |
| [bench/formatting_9b.md](analysis/bench/formatting_9b.md) | 9B model formatting benchmark |
| [bench/olmocr.md](analysis/bench/olmocr.md) | OLMoOCR benchmark results |
| [bench/omnidoc.md](analysis/bench/omnidoc.md) | OmniDocBench results |
| [summaries/classification_ablations.md](analysis/summaries/classification_ablations.md) | Classification ablation results |
| [summaries/training_dynamics.md](analysis/summaries/training_dynamics.md) | Training dynamics analysis |
| [summaries/fmt_bench_results.md](analysis/summaries/fmt_bench_results.md) | Formatting benchmark summary |
| [summaries/fmt_grpo_reward_experiments.md](analysis/summaries/fmt_grpo_reward_experiments.md) | GRPO reward experiment results |
| [summaries/formatting_bench_plan.md](analysis/summaries/formatting_bench_plan.md) | Formatting benchmark design |
| [summaries/kd_prerequisites.md](analysis/summaries/kd_prerequisites.md) | KD prerequisites analysis |
| [summaries/olmocr_bench_results.md](analysis/summaries/olmocr_bench_results.md) | OLMoOCR benchmark summary |
| [summaries/omnidoc_bench_results.md](analysis/summaries/omnidoc_bench_results.md) | OmniDocBench summary |
| [summaries/ocr_sft-rl_large-to-small-kd.md](analysis/summaries/ocr_sft-rl_large-to-small-kd.md) | OCR SFT/RL large-to-small KD analysis |
| [summaries/workbench_analysis.md](analysis/summaries/workbench_analysis.md) | Workbench probe/CKA analysis |
| [summaries/ablation_explainer.md](analysis/summaries/ablation_explainer.md) | Ablation methodology explainer |
| Task-specific: [mnli](analysis/summaries/mnli_ablation_analysis.md), [qqp](analysis/summaries/qqp_ablation_analysis.md), [sst2](analysis/summaries/sst2_ablation_analysis.md), [cola](analysis/summaries/cola_ablation_analysis.md) | Per-task ablation deep dives |

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

Publication drafts.

| Document | Description |
|----------|-------------|
| [README.md](paper/README.md) | Paper series overview and narrative arc |
| [paper-0-framework/draft.md](paper/paper-0-framework/draft.md) | Unified Framework for KD Research |
| [paper-1-ablation/draft.md](paper/paper-1-ablation/draft.md) | When Teachers Fail: Frozen vs Co-training |
| [paper-2-far/draft.md](paper/paper-2-far/draft.md) | Simplicity Wins: R-only beats multi-signal |
| [paper-3-translation/draft.md](paper/paper-3-translation/draft.md) | Cross-task validation on generation |

### Planning

Direction, roadmaps, and design docs.

| Document | Description |
|----------|-------------|
| [research-directions.md](planning/research-directions.md) | KD design space exploration and future directions |
| [distillation-tasks.md](planning/distillation-tasks.md) | Task catalog (MNLI, QQP, Translation, OCR) |
| [experiment-decision-tree.md](planning/experiment-decision-tree.md) | L0/L1/L2 experiment hierarchy |
| [chain.md](planning/chain.md) | Native SLURM job chaining design |
| [rl-integration.md](planning/rl-integration.md) | RL integration with TRL (GRPO, PPO) |
| [ocr-distillation.md](planning/ocr-distillation.md) | OCR distillation three-phase plan |
| [kd_formatting.md](planning/kd_formatting.md) | Formatting KD: 9B teacher to 2B student |
| [model_improvements_formatting.md](planning/model_improvements_formatting.md) | Improving inline formatting (bold, italic) |
| [diffusion-distillation.md](planning/diffusion-distillation.md) | Diffusion model compression plan |
| [voice-distillation.md](planning/voice-distillation.md) | StyleTTS2 voice compression plan |
| [refactor.md](planning/refactor.md) | Framework refactor plan |
| [individual-repos-plan.md](planning/individual-repos-plan.md) | Code/docs repo split strategy |

### Architecture

Technical implementation reference.

| Document | Description |
|----------|-------------|
| [architecture.md](architecture/architecture.md) | Framework modules and extension points |
| [core-and-extensions.md](architecture/core-and-extensions.md) | Core vs extension boundary definitions |
| [entry-point.md](architecture/entry-point.md) | forge.py entry point walkthrough |
| [data.md](architecture/data.md) | Data loading and preprocessing |
| [models.md](architecture/models.md) | Model loading and configuration |
| [tasks.md](architecture/tasks.md) | Task registry and dispatch |
| [evaluation.md](architecture/evaluation.md) | Evaluation pipeline |
| [inference.md](architecture/inference.md) | Inference engine |
| [save-structure.md](architecture/save-structure.md) | Checkpoint and output structure |
| [utils.md](architecture/utils.md) | Utility modules |
| [future.md](architecture/future.md) | Future architecture: training platform refactor |
| [diffusion-distillation.md](architecture/diffusion-distillation.md) | Diffusion distillation architecture |
| [training/trainer.md](architecture/training/trainer.md) | DistillationTrainer internals |
| [training/losses.md](architecture/training/losses.md) | Loss strategies (KL, MSE, Cosine, Attention) |
| [training/layer-mapping.md](architecture/training/layer-mapping.md) | Teacher-student layer mapping |
| [training/projections.md](architecture/training/projections.md) | Hidden state projection layers |
| [training/rewards.md](architecture/training/rewards.md) | RL reward functions |
| [training/callbacks.md](architecture/training/callbacks.md) | Training callbacks |

---

## Meta

Project-level docs at the docs root.

| Document | Description |
|----------|-------------|
| [DRY_assessment.md](DRY_assessment.md) | Code duplication audit with 26 identified issues |
| [codebase-todo.md](codebase-todo.md) | Remaining code quality items and testing gaps |

---

## Key Concepts

### The Overconfident-Wrong Teacher Paradox
Frozen SFT teachers can **hurt** student performance because confident wrong predictions fight ground truth gradients. See [research-framing.md](research/framing/research-framing.md).

### Ablation Results (MNLI)

| Rank | Config | Accuracy | Description |
|------|--------|----------|-------------|
| 1 | 04 | 82.69% | Co-train SFT teacher |
| 2 | 03 | 82.62% | Co-train base teacher |
| 3 | 00 | 82.19% | No teacher (CE only) |
| 4 | 01 | 81.58% | Frozen base teacher |
| 5 | 02 | 79.82% | Frozen SFT teacher |

### FAR Factorial Design
Testing Feature (F), Attention (A), Response (R) combinations. See [experiment-decision-tree.md](planning/experiment-decision-tree.md).
