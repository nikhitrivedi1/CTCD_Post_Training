# Framework Refactor Plan

## Design Principles

1. **Fine-tuning first, distillation second** — the framework is primarily a fine-tuning framework with distillation support
2. **Stem-based identity** — one shared path stem drives configs, outputs, and W&B naming
3. **Config path = experiment identity** — no-teacher configs at student root, teacher-specific configs nested under teacher
4. **Explicit over implicit** — no auto-picking, no guessing; errors state the problem, user decides

## Pipeline

```
DataLoader(config) → (train, eval)
    ↓
Trainer(model, train, eval)
    ├── training loop
    ├── built-in eval (loss + task metrics) @ eval_steps     [seconds]
    ↓
Eval Hooks (end-of-training)                                 [minutes–hours]
    ├── olmocr-bench
    ├── glue-test
    └── custom benchmarks
```

## Config Reorg

### Current Structure
```
configs/{type}/{task}/{teacher_student}/{experiment}/{config}.yaml
configs/classification/mnli/bert_distilbert/ablations/02_frozen_sft-teacher_kl.yaml
```

### New Structure
```
configs/{dataset}/{student}/{config}.yaml                          # student root (no teacher)
configs/{dataset}/{student}/{teacher}/{experiment}/{config}.yaml   # teacher-specific

configs/mnli/distilbert/no_teacher_ce_only.yaml
configs/mnli/distilbert/bert-base/ablations/02_frozen_sft.yaml
configs/mnli/distilbert/bert-base/far/07_far.yaml
configs/mnli/distilbert/bert-base/staged/03_f-a-r.yaml

configs/olmocr-mix/qwen3vl-2b/no_teacher_ce_only.yaml
configs/olmocr-mix/qwen3vl-2b/olmocr-2-7b/ablations/frozen_base.yaml
configs/olmocr-mix/qwen3vl-2b/qwen3vl-7b/ablations/frozen_base.yaml

configs/europarl/llama3p2-1b/no_teacher_ce_only.yaml
configs/europarl/llama3p2-1b/llama3p2-3b/ablations/02_frozen_sft.yaml
configs/europarl/llama3p2-1b/llama3p2-3b/far/07_far.yaml
```

### Inheritance Chain
```
configs/base.yaml                                    # global defaults
configs/mnli/base.yaml                               # dataset defaults
configs/mnli/distilbert/base.yaml                    # student defaults
configs/mnli/distilbert/bert-base/base.yaml          # teacher defaults
configs/mnli/distilbert/bert-base/ablations/base.yaml
configs/mnli/distilbert/bert-base/ablations/02_frozen_sft.yaml
```

### Key Changes
- Drop `classification/`, `translation/`, `ocr/` type prefix — dataset name implies task type
- No-teacher configs live at student root (not inside a teacher dir)
- Teacher identity is a directory level, not encoded in filenames
- GPU configs live under student level (hardware is student-specific, not teacher-specific):
  `configs/mnli/distilbert/gpu/h200/batch256.yaml`
- Scale configs live at dataset level (any student can inherit):
  `configs/olmocr-mix/scales/5k.yaml`, `configs/olmocr-mix/scales/full.yaml`
- Scale configs set `data.max_train_samples` (truncates at load time via `Dataset.select()`)
  and size-dependent hyperparams (warmup, save/eval steps, logging frequency)
- Student model bases inherit from a scale config (default: `full.yaml`);
  switch to a different size by changing the inherit path

## Stem-Based CLI

### Current
```bash
python distill.py --config configs/classification/mnli/bert_distilbert/ablations/02_frozen_sft-teacher_kl.yaml
python distill.py --run_name mnli_ablations_02_frozen_sft-teacher_kl_20260225-1234Z
```

### New
```bash
# New run: --config is stem relative to configs/
python distill.py --config mnli/distilbert/bert-base/ablations/02_frozen_sft

# Named run:
python distill.py --config mnli/distilbert/bert-base/ablations/02_frozen_sft --run_name seed42

# Resume: full path relative to outputs/
python distill.py --resume mnli/distilbert/bert-base/ablations/02_frozen_sft/20260225-1234Z
```

### Resolution Rules
- `--config {stem}` → `configs/{stem}.yaml` (config file)
- Output dir → `outputs/{stem}/{timestamp}/`
- `--config ... --run_name X` → `outputs/{stem}/X_{timestamp}/`
- `--resume {path}` → `outputs/{path}/` (must be exact run directory)
- If `--config` starts with `configs/`, strip it (backward compat)
- If `--config` ends with `.yaml`, keep it; otherwise append

### Output Directory
```
outputs/mnli/distilbert/bert-base/ablations/02_frozen_sft/
  20260225-1234Z/               # run 1
    config.yaml                 # resolved config snapshot
    run_metadata.json
    output.log
    checkpoint-400/
    eval/                       # eval hook results
      olmocr-bench/
        results.json
        markdowns/
  20260226-0900Z/               # run 2
  seed42_20260225-1500Z/        # named run
```

Config stem is a directory. Runs are timestamped subdirectories within it.
Eval hook results live inside the run that produced them.

### Resume Error Messages

Path doesn't resolve to a run:
```
Error: Run not found: outputs/mnli/distilbert/bert-base/ablations/02_frozen_sft/20260225-1234Z
```

Path is a directory, not a specific run:
```
Error: Not a run directory: outputs/mnli/distilbert/bert-base/ablations/02_frozen_sft/
```

## Eval Hooks

End-of-training only. Config-driven, script-based.

```yaml
eval_hooks:
  - script: scripts/eval_olmocr_bench.py
    args:
      bench_dir: data/olmocr-bench
```

Contract: any script that accepts `--checkpoint <path>` and writes results.
Evaluating pretrained baselines (no checkpoint) is a manual run, not a hook.
The trainer runs hooks after training completes:

```python
for hook in eval_hooks:
    subprocess.run(["python", hook.script, "--checkpoint", checkpoint_path, *hook.args])
```

No tight coupling — trainer doesn't know what the benchmarks are.
Results written to `outputs/.../run/eval/{hook_name}/`.

## W&B Naming

Derived from the stem:

| Aspect | Value |
|--------|-------|
| Run ID | `mnli-distilbert-bert-base-ablations-02_frozen_sft-20260225-1234Z` (stem with `/` → `-`, truncated to 64 chars) |
| Group | `mnli/distilbert/bert-base/ablations` (parent of config) |
| Tags | `[mnli, distilbert, bert-base, ablations]` (path segments) |

Note: W&B run IDs don't support `/` — replace with `-` at serialization time.
No more hardcoded index lookups (`parts[2]`, `parts[4]`). Just path operations on the stem.

## Trainer Class Hierarchy

### Current
```
HuggingFace Trainer
  └── DistillationTrainer    # handles both fine-tuning (α=0) and distillation
```

### New
```
HuggingFace Trainer
  └── FinetuningTrainer       # LoRA, VLM collation, custom eval, checkpointing
        └── DistillationTrainer  # + teacher loading, KL/feature/attention loss
```

### FinetuningTrainer Responsibilities
- LoRA adapter setup
- VLM data collation (OCR tasks)
- Custom evaluation metrics
- Checkpointing and resume
- compute_loss: cross-entropy only
- Eval hook execution (end-of-training)

### DistillationTrainer Additions
- Teacher model loading (from config `teacher` section)
- Freeze/co-train teacher logic
- compute_loss override: CE + KL divergence
- Feature matching loss (hidden states)
- Attention matching loss
- Temperature scaling
- Vocab alignment: slice logits to shared vocab size for cross-tokenizer pairs
  (e.g., Qwen3-VL has 4 extra special tokens vs Qwen2.5-VL)

### Trainer Selection
```python
has_teacher = "teacher" in config.get("model", {})
TrainerClass = DistillationTrainer if has_teacher else FinetuningTrainer
```

Config determines trainer — no-teacher configs at student root use FinetuningTrainer,
teacher-nested configs use DistillationTrainer.

## Implementation Order

### Phase 1: Config reorg (file moves only)
1. Move config files to new structure
2. Update `inherit:` paths
3. Update SLURM scripts to use new config paths
4. Verify inheritance chains resolve correctly

### Phase 2: CLI + output dir changes
1. Rewrite `generate_config_id` → simple stem extraction from path
2. Rewrite `generate_wandb_group` / `generate_wandb_tags` → path operations
3. Update argparse: `--config` (stem-based), `--resume` (replaces `--run_name`)
4. Update output dir: `outputs/{stem}/{timestamp}/`
5. Implement resume error messages
6. Backward compat: handle `configs/` prefix, `.yaml` suffix

### Phase 3: Trainer split
1. Extract `FinetuningTrainer` from current `DistillationTrainer`
2. Make `DistillationTrainer` subclass of `FinetuningTrainer`
3. Move teacher-specific logic to `DistillationTrainer`
4. Update entry point to select trainer class based on config
5. Test all existing config types still work

### Phase 4: Eval hooks
1. Define hook config schema (`eval_hooks` section in YAML)
2. Implement hook runner in FinetuningTrainer (end-of-training callback)
3. Write results to `outputs/.../run/eval/{hook_name}/`
4. Update existing eval scripts to accept `--checkpoint` uniformly

### Phase 5: Cleanup
1. Rename `distill.py` → `train.py` (or keep for backward compat)
2. Update README, docs
3. Archive old flat outputs (leave as-is, won't match new convention)

## Migration Notes

- Existing runs in `outputs/` (flat names) are completed — leave them as-is
- Existing W&B runs keep their old group/tag structure
- New runs use new naming — won't mix with old runs in W&B groups
- All SLURM scripts need config path updates
