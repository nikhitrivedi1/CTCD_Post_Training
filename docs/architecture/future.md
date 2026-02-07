# Future Architecture: Training Platform Refactor

## Current State

`distill.py` serves as the entry point for knowledge distillation training. It works well for its purpose but is tightly coupled to distillation-specific workflows.

## Proposed Restructure

### Rename and Specialize

```
Current:
  distill.py → distillation training

Proposed:
  distill.py → distillation training (renamed from distill.py)
  distill.py   → general training platform (new)
```

### New `distill.py` Capabilities

A unified training platform supporting multiple modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| `sft` | Supervised fine-tuning from scratch | Baseline training |
| `head-only` | Freeze encoder, train classification head | Post-distillation refinement |
| `full` | Full model fine-tuning | Transfer learning |
| `adapter` | Train adapters with frozen base | Parameter-efficient tuning |

### Usage Examples

```bash
# Supervised fine-tuning
python distill.py --mode sft --model bert-base-uncased --task mnli

# Head-only fine-tuning (post-distillation)
python distill.py --mode head-only --model outputs/ablation_04/checkpoint --task mnli

# Full fine-tuning
python distill.py --mode full --model distilbert-base-uncased --task mnli
```

### Shared Infrastructure

Both `distill.py` and `distill.py` would share:

- `src/training/base_trainer.py` - Common training loop, logging, checkpointing
- `src/data/` - Data loading and preprocessing
- `src/utils/` - Shared utilities
- Config inheritance system

### Class Hierarchy

```
BaseTrainer
├── DistillationTrainer (distill.py)
│   └── Handles teacher-student training, loss computation
└── StandardTrainer (distill.py)
    └── Handles SFT, head-only, full fine-tuning modes
```

## Why Deferred

1. **Scope**: This is a coursework project with limited timeline
2. **Working System**: Current `distill.py` works well for distillation experiments
3. **Risk**: Refactoring introduces bugs; don't fix what isn't broken
4. **Workaround**: Quick scripts in `workbench/` suffice for immediate experiments

## Immediate Alternative

For head-only fine-tuning experiments, add a lightweight command to `workbench/`:

```bash
python workbench/analyze.py finetune-head \
    --model outputs/ablation_04/checkpoint \
    --task mnli \
    --epochs 3
```

This avoids architectural changes while enabling the experiment.

## Future Work

If this project extends beyond coursework:

1. Extract `BaseTrainer` from `DistillationTrainer`
2. Implement `StandardTrainer` with mode support
3. Rename `distill.py` → `distill.py`
4. Create new `distill.py` using `StandardTrainer`
5. Update configs to support both entry points

---

*Documented: 2026-02-05*
