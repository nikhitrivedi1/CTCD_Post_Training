# Framework Refactor Plan

## Design Principles

1. **Training framework** — fine-tuning first, with distillation and RL as optional layers on top
2. **Stem-based identity** — one shared path stem drives configs, outputs, and W&B naming
3. **Config path = experiment identity** — base configs at model root, teacher/RL configs nested under teacher
4. **Model, not student** — the primary entity is "the model you're training". It only becomes a "student" when a teacher exists. Config key is `model:` with optional `model.teacher:`.
5. **Explicit over implicit** — no auto-picking, no guessing; errors state the problem, user decides

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

### Task Types

The framework handles three task types. The core training loop is the same
(forward pass → loss → backprop), but the output space, models, and evaluation differ.

| | Classification (MNLI) | Translation (Europarl) | OCR (olmocr-mix) |
|---|---|---|---|
| Output space | 3 labels | 128k vocab per position | 152k vocab per position |
| Model arch | Encoder (BERT) | Decoder (Llama) | VLM (Qwen-VL) |
| Fine-tuning | Full params (66M-110M) | LoRA (1B-3B) | LoRA (2B-7B) |
| Loss | CE over class logits | CE over token logits (causal LM) | CE over token logits (causal LM) |
| KD temperature | 2.0 | 15.0 (larger output space) | 15.0 |
| KD alpha | 0.5 | 0.2 (high T already softens KL) | 0.2 |
| Eval metric | Accuracy/F1/MCC | BLEU (via generation callback) | Edit similarity |
| Eval method | Forward pass → argmax | Generate translation → compare | Generate markdown → compare |

For generative tasks (translation, OCR), there is an eval asymmetry:
- **Training loss** uses teacher forcing (model sees correct previous tokens, predicts next)
- **Eval metrics** require free-form generation (model generates from scratch)
- These can diverge, so `TranslationMetricsCallback` runs actual generation on a subset per epoch
- `prediction_loss_only=True` prevents generating full outputs for every eval step (would be too slow)

## Config Reorg

### Current Structure
```
configs/{type}/{task}/{teacher_student}/{experiment}/{config}.yaml
configs/classification/mnli/bert_distilbert/ablations/02_frozen_sft-teacher_kl.yaml
```

### New Structure
```
configs/{dataset}/{model}/{config}.yaml                          # model root (SFT, no teacher)
configs/{dataset}/{model}/{teacher}/{experiment}/{config}.yaml   # with teacher (KD/RL)

configs/mnli/distilbert/sft.yaml
configs/mnli/distilbert/bert-base/ablations/02_frozen_sft.yaml
configs/mnli/distilbert/bert-base/far/07_far.yaml
configs/mnli/distilbert/bert-base/staged/03_f-a-r.yaml

configs/olmocr-mix/qwen3vl-2b/sft.yaml
configs/olmocr-mix/qwen3vl-2b/olmocr-2-7b/ablations/frozen_base.yaml
configs/olmocr-mix/qwen3vl-2b/qwen3vl-7b/ablations/frozen_base.yaml

configs/europarl/llama3p2-1b/sft.yaml
configs/europarl/llama3p2-1b/llama3p2-3b/ablations/02_frozen_sft.yaml
configs/europarl/llama3p2-1b/llama3p2-3b/far/07_far.yaml

configs/mnli/distilbert/rl/grpo_verifiable.yaml                    # RL without teacher
configs/mnli/distilbert/bert-base/rl/grpo_teacher_reward.yaml      # RL with teacher as reward
```

### Inheritance Chain
```
configs/base.yaml                                    # global defaults
configs/mnli/base.yaml                               # dataset defaults
configs/mnli/distilbert/base.yaml                    # model defaults
configs/mnli/distilbert/bert-base/base.yaml          # teacher defaults
configs/mnli/distilbert/bert-base/ablations/base.yaml
configs/mnli/distilbert/bert-base/ablations/02_frozen_sft.yaml
```

### Key Changes
- Drop `classification/`, `translation/`, `ocr/` type prefix — dataset name implies task type
- No-teacher configs live at model root (SFT, RL without teacher)
- Teacher identity is a directory level, not encoded in filenames
- GPU configs live under model level (hardware is model-specific, not teacher-specific):
  `configs/mnli/distilbert/gpu/h200/batch256.yaml`
- Scale configs live at dataset level (any model can inherit):
  `configs/olmocr-mix/scales/5k.yaml`, `configs/olmocr-mix/scales/full.yaml`
- Scale configs set `data.max_train_samples` (truncates at load time via `Dataset.select()`)
  and size-dependent hyperparams (warmup, save/eval steps, logging frequency)
- Model bases inherit from a scale config (default: `full.yaml`);
  switch to a different size by changing the inherit path

## Stem-Based CLI

### Current
```bash
python forge.py --config configs/classification/mnli/bert_distilbert/ablations/02_frozen_sft-teacher_kl.yaml
python forge.py --run_name mnli_ablations_02_frozen_sft-teacher_kl_20260225-1234Z
```

### New
```bash
# New run: --config is stem relative to configs/
python forge.py --config mnli/distilbert/bert-base/ablations/02_frozen_sft

# Named run:
python forge.py --config mnli/distilbert/bert-base/ablations/02_frozen_sft --run_name seed42

# Resume: full path relative to outputs/
python forge.py --resume mnli/distilbert/bert-base/ablations/02_frozen_sft/20260225-1234Z
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
    extra_args = [f"--{k}", str(v) for k, v in hook.args.items()]
    subprocess.run(["python", hook.script, "--checkpoint", checkpoint_path, *extra_args])
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
  ├── FinetuningTrainer            # LoRA, VLM collation, custom eval, checkpointing
  │     └── DistillationTrainer    # + teacher, KL/feature/attention loss
  └── trl.GRPOTrainer
        └── RLTrainer              # + reward dispatch, teacher-as-RM
```

Supervised training (SFT, KD, DPO) shares one branch.
RL (GRPO, PPO) uses a fundamentally different training loop (generate → score → update)
and lives on the trl branch. Shared utilities (model loading, eval hooks) are functions,
not inherited methods.

### FinetuningTrainer Responsibilities
- LoRA adapter setup
- VLM data collation (OCR tasks)
- Custom evaluation metrics
- Checkpointing and resume
- compute_loss: cross-entropy only
- Eval hook execution (end-of-training)

### DistillationTrainer Additions
- Teacher model loading (from config `model.teacher` section)
- Freeze/co-train teacher logic
- compute_loss override: CE + KL divergence
- Feature matching loss (hidden states)
- Attention matching loss
- Temperature scaling
- Vocab alignment: slice logits to shared vocab size for cross-tokenizer pairs
  (e.g., Qwen3-VL has 4 extra special tokens vs Qwen2.5-VL)

### RLTrainer Additions
- Extends `trl.GRPOTrainer` (separate branch from FinetuningTrainer)
- Reward function dispatch (teacher logprob, verifiable check, judge LLM)
- KL penalty against reference policy
- Can load teacher as reward model (reuses `load_models()` utility)
- Eval hooks called via shared utility function (not inherited from FinetuningTrainer)

### Config Schema

```yaml
# SFT — no teacher, no RL
model:
  name: distilbert-base-uncased

# Distillation — teacher nested under model
model:
  name: distilbert-base-uncased
  teacher:
    name: bert-base-uncased
    freeze: true

# RL — DPO (supervised, no generation loop)
model:
  name: distilbert-base-uncased
rl:
  algorithm: dpo                   # dpo | kto | spin
  data: path/to/pairs.jsonl       # pre-generated preference pairs
  reference: self                  # KL anchor (see reference values below)
  teacher:                         # only for SPIN (teacher generates "chosen")
    name: bert-base-uncased

# RL — GRPO with teacher as reward (online generation)
model:
  name: distilbert-base-uncased
  teacher:
    name: bert-base-uncased
    freeze: true
rl:
  algorithm: grpo                  # grpo | ppo
  reward: teacher                  # teacher | verifiable | judge:{model_name}
  reference: self                  # KL anchor (see reference values below)
  num_generations: 8               # G in GRPO

# RL — GRPO without teacher (verifiable reward)
model:
  name: qwen3vl-2b
rl:
  algorithm: grpo
  reward: verifiable               # uses task's primary metric from registry
  reference: self
  num_generations: 8
```

#### Reference values
- `self` (default): Snapshot model weights at training start. KL against that frozen copy.
- `teacher`: Use the loaded teacher as reference. Only valid when `model.teacher` exists.
- Any path/model name: Load as a separate model. e.g. `outputs/mnli/distilbert/sft/20260225-1234Z/`

#### Reward: verifiable dispatch
`reward: verifiable` looks up the task's primary metric from `src/tasks/registry.py`
(accuracy for MNLI, F1 for QQP, CER for OCR) and uses it as the reward function.
Override with explicit reward config if needed:
```yaml
rl:
  reward:
    type: verifiable
    fn: exact_match              # override task default
```

### Trainer Selection
```python
config_model = config.get("model", {})
has_teacher = "teacher" in config_model
has_rl = "rl" in config
rl_algorithm = config.get("rl", {}).get("algorithm")

if has_rl and rl_algorithm in ("grpo", "ppo"):
    TrainerClass = RLTrainer             # extends trl.GRPOTrainer, online generation
elif has_rl and rl_algorithm in ("dpo", "kto", "spin"):
    TrainerClass = FinetuningTrainer     # supervised preference loss, no generation
elif has_teacher:
    TrainerClass = DistillationTrainer
else:
    TrainerClass = FinetuningTrainer
```

Config determines trainer. The model being trained is always `model.name`.
Teacher is an optional auxiliary under `model.teacher`.
RL config is a top-level section. The algorithm determines whether the training
loop needs online generation (GRPO/PPO → RLTrainer) or stays supervised
(DPO/KTO/SPIN → FinetuningTrainer with preference loss).

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

### Phase 5: Repo Split + Cleanup

#### Repo Structure
Two repos under [github.com/NU-Deep-Learning](https://github.com/NU-Deep-Learning):
- **Forge** (public) — code, configs, scripts, public docs
- **Forge-Docs** (private) — analysis, results, paper drafts, research notes

Forge-Docs is a git submodule mounted at `docs/internal/`. Contributors with
access to both repos get everything with a single clone:
```bash
git clone --recurse-submodules git@github.com:NU-Deep-Learning/Forge.git
```
Those without Forge-Docs access still clone Forge fine — `docs/internal/` is
just an empty directory.

#### Directory Layout (post-split)
```
Forge/
  forge.py                    # entry point
  src/
  configs/
  scripts/
  docs/
    README.md                 # public docs index
    guides/                   # config params, metrics, run options, HPC
    architecture/             # architecture docs
    foundations/              # tasks-models-datasets, transformer architectures
    planning/                 # refactor plan, roadmap
    internal/                 # ← git submodule → Forge-Docs
      analysis/               #   ablation data, summaries, scripts
      paper/                  #   paper drafts
      research/               #   research framing, findings, references
      archive/                #   old docs
      wip_docs/               #   work in progress
```

#### Steps
1. Create `Forge` repo on GitHub, push current code
2. Create `Forge-Docs` repo on GitHub (private)
3. Move internal docs: `analysis/`, `paper/`, `research/`, `archive/`, `wip_docs/`,
   `references/` into Forge-Docs
4. Add Forge-Docs as submodule at `docs/internal/`
   ```bash
   git submodule add git@github.com:NU-Deep-Learning/Forge-Docs.git docs/internal
   ```
5. Add `docs/internal` to `.gitignore` fallback (so Forge works without the submodule)
6. ~~Rename entry point: `distill.py` → `forge.py`~~ (done)
7. Update README, import paths, SLURM scripts
8. Archive old flat outputs (leave as-is, won't match new convention)
9. Update internal doc cross-references (`docs/analysis/` → `docs/internal/analysis/`, etc.)

## Phase 6: RL Integration

### Conceptual Map

Every RL-for-LLMs method is a point on two orthogonal axes:

**Algorithm** (how you optimize): Supervised (CE/KL/DPO) → REINFORCE → GRPO → PPO
**Reward source** (where signal comes from): Teacher | Verifiable check | AI judge | Human | Learned RM

The named methods are just reward sources, not algorithms:
- **RLHF** = human preferences → reward model → PPO/GRPO
- **RLAIF** = AI judge preferences → reward model → PPO/GRPO
- **RLVR** = programmatic correctness check (math, code, OCR) → GRPO (no RM needed)

### KD → RL Transition

Each step generalizes exactly one thing from the previous:

```
SFT           CE(student, labels)                        fixed data, hard labels
    ↓ add teacher signal
Offline KD    α·KL(student, teacher) + (1-α)·CE          fixed data, soft labels     ← freeze_teacher: true
    ↓ teacher co-trains
Online KD     same loss, teacher weights update           fixed data, evolving labels ← freeze_teacher: false
    ↓ data source: fixed → student-generated
On-policy KD  student generates → teacher scores          generated data, KL loss
    ↓ loss: KL → policy gradient
REINFORCE     student generates → reward = teacher logprob → ∇J = r·∇log π
    ↓ variance reduction
GRPO          generate N per prompt → rank by reward → group-relative advantage
    ↓ swap reward source
RLVR/RLAIF/RLHF   same algorithm, different reward function
    ↓ add value network
PPO           GRPO + learned critic (usually unnecessary, GRPO is winning)
```

The KL term persists across the boundary but changes meaning:
- In KD: KL(student ‖ teacher) = "match the teacher"
- In RL: KL(student ‖ reference) = "don't drift from starting policy"
- Same math, different purpose. Setting reference = teacher unifies them.

### Preference-Based RL (DPO family)

DPO/KTO/SPIN use the `rl:` config but don't need online generation.
They train on fixed preference data with a supervised loss derived from the RL objective.

- **DPO**: preference pairs (chosen/rejected) → supervised loss. Closed-form solution
  to the RLHF objective, mathematically equivalent under Bradley-Terry assumptions.
- **KTO**: unpaired, just (output, good/bad). Easier data collection.
- **SPIN**: teacher output = chosen, student output = rejected. KD recast as preference optimization.
- **Online DPO**: generate on-policy → score → form pairs → DPO loss. Bridges back to the generation-based RL path.

Uses `rl:` config with `algorithm: dpo|kto|spin`. Trainer selection routes to
FinetuningTrainer (preference loss strategy), not RLTrainer.

### Framework Impact

```
HuggingFace Trainer
  ├── FinetuningTrainer                 # CE, fixed data
  │     └── DistillationTrainer         # + teacher (offline/online KD)
  └── trl.GRPOTrainer
        └── RLTrainer                   # + generation loop, reward
              └── teacher as reward_model (on-policy KD / GRPO)
              └── teacher as kl_reference (regularization)

DPO/KTO/SPIN → rl: config, but routed to FinetuningTrainer (no generation needed)
```

What reuses the existing training loop (new loss strategy only):
- DPO, KTO, SPIN — preference loss strategies + offline data prep step

What needs a new trainer class (extends trl.GRPOTrainer):
- GRPO, PPO — online generation loop, reward computation, KL penalty
- Shared utilities (model loading, eval hooks) are functions, not inherited

### Implementation Order

1. **Phase 6a — DPO/SPIN** (low effort): Add preference loss strategy + data prep script
   that generates (chosen, rejected) pairs using teacher. No new trainer needed.
2. **Phase 6b — GRPO** (medium effort): Add RLTrainer wrapping `trl.GRPOTrainer`.
   Teacher-as-reward-model support. KL reference = teacher or SFT checkpoint.
3. **Phase 6c — Reward sources** (config-only after 6b): Swap reward function via config.
   RLVR for OCR/translation (verifiable), RLAIF for open-ended (judge LLM).

## Migration Notes

- Existing runs in `outputs/` (flat names) are completed — leave them as-is
- Existing W&B runs keep their old group/tag structure
- New runs use new naming — won't mix with old runs in W&B groups
- All SLURM scripts need config path updates
- Internal doc references (in code comments, CLAUDE.md, memory files) need `docs/X` → `docs/internal/X` updates
- Forge-Docs submodule requires `--recurse-submodules` on clone (or `git submodule update --init` after)
