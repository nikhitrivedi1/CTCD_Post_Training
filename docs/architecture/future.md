# Future Architecture: Training Platform Refactor

## Current State

`forge.py` serves as the entry point for knowledge distillation training. It works well for its purpose but is tightly coupled to distillation-specific workflows.

## Proposed Restructure

### Rename and Specialize

```
Current:
  forge.py → distillation training

Proposed:
  forge.py → distillation training (renamed from forge.py)
  forge.py   → general training platform (new)
```

### New `forge.py` Capabilities

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
python forge.py --mode sft --model bert-base-uncased --task mnli

# Head-only fine-tuning (post-distillation)
python forge.py --mode head-only --model outputs/ablation_04/checkpoint --task mnli

# Full fine-tuning
python forge.py --mode full --model distilbert-base-uncased --task mnli
```

### Shared Infrastructure

Both `forge.py` and `forge.py` would share:

- `src/training/base_trainer.py` - Common training loop, logging, checkpointing
- `src/data/` - Data loading and preprocessing
- `src/utils/` - Shared utilities
- Config inheritance system

### Class Hierarchy

```
BaseTrainer
├── DistillationTrainer (forge.py)
│   └── Handles teacher-student training, loss computation
└── StandardTrainer (forge.py)
    └── Handles SFT, head-only, full fine-tuning modes
```

## Why Deferred

1. **Scope**: Limited timeline for initial development
2. **Working System**: Current `forge.py` works well for distillation experiments
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

Future steps:

1. Extract `BaseTrainer` from `DistillationTrainer`
2. Implement `StandardTrainer` with mode support
3. Rename `forge.py` → `forge.py`
4. Create new `forge.py` using `StandardTrainer`
5. Update configs to support both entry points

---

*Documented: 2026-02-05*
