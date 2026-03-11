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

## Config Reorg (DONE)

### Old Structure
```
configs/{type}/{dataset}/{student}_{teacher}/ablations/{config}.yaml
configs/classification/mnli/bert_distilbert/ablations/02_frozen_sft.yaml
```

### New Structure
```
configs/{type}/{dataset}/{model}/{config}.yaml                          # model root (SFT, no teacher)
configs/{type}/{dataset}/{model}/{teacher}/{experiment}/{config}.yaml   # with teacher (KD/RL)

configs/classification/mnli/distilbert/00_sft.yaml
configs/classification/mnli/distilbert/bert-base/ablations/01_kd_frozen-base.yaml
configs/classification/mnli/distilbert/bert-base/ablations/02_kd_frozen-sft.yaml
configs/classification/mnli/distilbert/bert-base/far/07_far.yaml
configs/classification/mnli/distilbert/bert-base/staged/03_f-a-r.yaml

configs/ocr/olmocr-mix/qwen3vl-2b/00_sft.yaml
configs/ocr/olmocr-mix/qwen3vl-2b/olmocr2-7b/03_kd_frozen.yaml
configs/ocr/olmocr-mix/qwen25vl-3b/00_sft.yaml
configs/ocr/olmocr-mix/qwen25vl-3b/olmocr2-7b/01_kd_frozen.yaml

configs/translation/europarl/llama32-1b/00_sft.yaml
configs/translation/europarl/llama32-1b/llama32-3b/ablations/02_kd_frozen-sft.yaml
configs/translation/europarl/llama32-1b/llama32-3b/far/07_far.yaml

configs/classification/mnli/distilbert/rl/grpo_verifiable.yaml                    # RL without teacher
configs/classification/mnli/distilbert/bert-base/rl/grpo_teacher_reward.yaml      # RL with teacher as reward
```

### Naming Convention
```
00_sft.yaml                  # SFT baseline (no teacher)
01_kd_frozen-base.yaml       # KD, frozen teacher, pretrained student
02_kd_frozen-sft.yaml        # KD, frozen teacher, SFT-init student
03_kd_cotrain-base.yaml      # KD, co-training teacher, pretrained student
04_kd_cotrain-sft.yaml       # KD, co-training teacher, SFT-init student
02_reasoning_sft.yaml        # reasoning SFT (OCR-specific)
```

### Inheritance Chain
```
configs/base.yaml                                                    # global defaults
configs/classification/base.yaml                                     # classification defaults
configs/classification/mnli/base.yaml                                # dataset defaults (task, max_length)
configs/classification/mnli/distilbert/base.yaml                     # model defaults
configs/classification/mnli/distilbert/bert-base/base.yaml           # teacher defaults
configs/classification/mnli/distilbert/bert-base/ablations/base.yaml # experiment group
configs/classification/mnli/distilbert/bert-base/ablations/02_kd_frozen-sft.yaml
```

Each level provides exactly one concern. SFT configs (no teacher) live at model root
and inherit from the model base directly. Teacher-specific settings (teacher name,
distillation temperature/alpha) live at the teacher level.

### Key Changes
- Keep `classification/`, `translation/`, `ocr/` type prefix (dataset name alone is ambiguous)
- Split model-pair directories: `bert_distilbert/` becomes `distilbert/bert-base/` (model/teacher)
- Model directory uses short names: `llama32-1b` not `llama3p2-1b`, `qwen25vl-3b` not `qwen2.5-vl-3b`
- SFT configs (no teacher) live at model root, not inside ablations/
- Teacher identity is a directory level, not encoded in filenames
- GPU configs live under model level (hardware is model-specific, not teacher-specific):
  `configs/classification/mnli/distilbert/gpu/h200/batch256.yaml`
- Scale configs (OCR) live at dataset level:
  `configs/ocr/olmocr-mix/sizes/5k.yaml`, `configs/ocr/olmocr-mix/sizes/full.yaml`
- Scale configs set `data.max_train_samples` (truncates at load time via `Dataset.select()`)
  and size-dependent hyperparams (warmup, save/eval steps, logging frequency)
- Model bases inherit from a scale config (default: `full.yaml`);
  switch to a different size by changing the inherit path

## Stem-Based CLI

### Current
```bash
python forge.py --config configs/classification/mnli/distilbert/bert-base/ablations/02_kd_frozen-sft.yaml
python forge.py --run_name mnli_ablations_02_kd_frozen-sft_20260225-1234Z
```

### New
```bash
# Auto-named run (hierarchical output, tab-completable):
python forge.py --config configs/classification/mnli/distilbert/bert-base/ablations/02_kd_frozen-sft.yaml
# -> outputs/classification/mnli/distilbert/bert-base/ablations/02_kd_frozen-sft/20260225-1234Z/

# Named run (flat output):
python forge.py --config configs/classification/mnli/distilbert/00_sft.yaml --run_name seed42
# -> outputs/seed42/

# Named + timestamp:
python forge.py --config configs/classification/mnli/distilbert/00_sft.yaml --run_name seed42 --timestamp
# -> outputs/seed42_20260225-1234Z/

# Resume (always explicit, tab-completable):
python forge.py --resume outputs/classification/mnli/distilbert/00_sft/20260225-1234Z
python forge.py --resume outputs/seed42
```

Users type real filesystem paths (tab-completable). `configs/` prefix and `.yaml` suffix
are stripped internally to derive the stem. `outputs/` prefix is stripped from `--resume`.

### Resolution Rules
- `--config configs/{stem}.yaml` → resolves config file, extracts stem internally
- Auto-named output dir → `outputs/{stem}/{timestamp}/`
- Named output dir → `outputs/{run_name}/` or `outputs/{run_name}_{timestamp}/`
- `--resume outputs/{path}` → resumes `outputs/{path}/` (must have run_metadata.json)
- Named run that already exists without `--force` is an error, not implicit resume

### Output Directory

Auto-named (hierarchical):
```
outputs/classification/mnli/distilbert/bert-base/ablations/02_kd_frozen-sft/
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
```

Named (flat):
```
outputs/seed42/                 # named run
  config.yaml
  run_metadata.json
  output.log
  checkpoint-400/
```

### Resume Error Messages

Path doesn't resolve to a run:
```
Error: Run not found: outputs/classification/mnli/distilbert/bert-base/ablations/02_kd_frozen-sft/20260225-1234Z
```

Path is a stem directory, not a specific run:
```
Error: Not a run directory: outputs/classification/mnli/distilbert/bert-base/ablations/02_kd_frozen-sft/
Specify the full run path: --resume outputs/.../02_kd_frozen-sft/TIMESTAMP
```

## Eval Hooks

End-of-training only. Config-driven, script-based.

```yaml
eval_hooks:
  - script: scripts/benchmarks/olmocr_bench.py
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

| Aspect | Auto-named | Named |
|--------|-----------|-------|
| Run name | `{stem}/{timestamp}` | `{run_name}` or `{run_name}_{timestamp}` |
| Run ID | `{stem[:x]}/{timestamp}` (truncate stem to fit 64 chars) | `{run_name}` or `{run_name}_{timestamp}` |
| Group | `{stem_parent}` (e.g. `classification/mnli/distilbert/bert-base/ablations`) | same |
| Tags | path segments of stem (e.g. `[classification, mnli, distilbert, bert-base, ablations]`) | same |

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
- Any path/model name: Load as a separate model. e.g. `outputs/classification/mnli/distilbert/00_sft/20260225-1234Z/`

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

### Phase 1: Trainer split (DONE)
1. ~~Extract `FinetuningTrainer` from current `DistillationTrainer`~~
2. ~~Make `DistillationTrainer` subclass of `FinetuningTrainer`~~
3. ~~Move teacher-specific logic to `DistillationTrainer`~~
4. ~~Update entry point to select trainer class based on config~~
5. ~~Test all existing config types still work~~

### Phase 2: RL Integration

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

#### Sub-phases
1. **Phase 2a - DPO/SPIN** (deferred): Preference loss strategy for encoder models is an
   awkward fit with trl. Deferred until there's a clear use case.
2. **Phase 2b - GRPO** (IN PROGRESS): GRPO integration via `trl.GRPOTrainer`.
   - ~~Add trl dependency~~
   - ~~Reward functions: BLEU (translation), edit similarity (OCR), format reward~~
   - ~~RL dataset loaders: prompt-based format for GRPO~~
   - ~~Config mapping: `rl:` YAML section to GRPOConfig~~
   - ~~Trainer selection: RL path in forge.py main()~~
   - ~~Example configs: translation RLVR, OCR RLVR~~
   - Teacher-as-reward-model support (TODO)
   - End-to-end GPU testing (TODO)
3. **Phase 2c - Reward sources** (config-only after 2b): Swap reward function via config.
   RLVR for OCR/translation (verifiable), RLAIF for open-ended (judge LLM).

### Phase 3: Config reorg (DONE)
1. ~~Config schema migration: `model.student` → `model.name`, `model.teacher` as nested dict~~
2. ~~Split model-pair dirs into model/teacher hierarchy~~
3. ~~Move SFT configs to model root, rename with new convention~~
4. ~~Create layered base.yaml files (dataset → model → teacher)~~
5. ~~Update `inherit:` paths and doc references~~
6. ~~Verify inheritance chains resolve correctly~~

### Phase 4: CLI + output dir changes (DONE)
1. ~~Replace `generate_config_id` / `generate_wandb_*` / `truncate_for_wandb` with stem-based functions~~
2. ~~Add `--resume` arg (explicit resume, tab-completable `outputs/` paths)~~
3. ~~Auto-named runs: `outputs/{stem}/{timestamp}/` (hierarchical)~~
4. ~~Named runs: `outputs/{run_name}/` (flat, keep `--timestamp`/`--force`)~~
5. ~~No implicit resume - named run exists without `--force` is an error~~
6. ~~Update SLURM script resubmit to use `--resume`~~
7. ~~Update run-options.md docs~~

### Phase 5: Eval hooks (DEFERRED)

Deferred. The subprocess-based hook pattern doesn't fit the SLURM workflow well:
- Heavy benchmarks (olmocr-bench, ~8h) need their own SLURM jobs, not in-process hooks
- Lightweight post-training eval (accuracy, loss) is already handled by `run_evaluation()`
- SIGUSR1 during hook execution would orphan subprocesses
- Adds machinery for a use case that doesn't exist yet

**Future direction:** Auto-submit a separate eval SLURM job after training completes.
This avoids time-limit issues and gives eval full GPU allocation. Deferred due to
implementation complexity.

### Phase 6: Repo Split + Cleanup

#### Repo Structure
Two repos under [github.com/NU-Deep-Learning](https://github.com/NU-Deep-Learning):
- **Forge** (public) - code, configs, scripts, public docs
- **Forge-Docs** (private) - analysis, results, paper drafts, research notes

Forge-Docs is a git submodule mounted at `docs/internal/`. Contributors with
access to both repos get everything with a single clone:
```bash
git clone --recurse-submodules git@github.com:NU-Deep-Learning/Forge.git
```
Those without Forge-Docs access still clone Forge fine - `docs/internal/` is
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

## Migration Notes

- Existing runs in `outputs/` (flat names) are completed - leave them as-is
- Existing W&B runs keep their old group/tag structure
- New runs use new naming - won't mix with old runs in W&B groups
- All SLURM scripts need config path updates
- Internal doc references (in code comments, CLAUDE.md, memory files) need `docs/X` → `docs/internal/X` updates
- Forge-Docs submodule requires `--recurse-submodules` on clone (or `git submodule update --init` after)

---

## Phase 4: Stem-Based CLI + Output Directory

Config IDs are generated via brittle index-based lookups (`parts[2]`, `parts[4]`). Phase 3 changed the directory depth, so these lookups are now wrong. Phase 4 replaces them with path-based operations on a "stem" and restructures auto-named output dirs to mirror config paths.

### Core Concept: Stem

The stem is derived internally from the config path. Users always type full tab-completable paths.

```
--config configs/classification/mnli/distilbert/bert-base/ablations/02_kd_frozen-sft.yaml
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  internal stem: classification/mnli/distilbert/bert-base/ablations/02_kd_frozen-sft
```

### CLI (tab-completable paths)

```bash
# Auto-named (hierarchical output):
python forge.py --config configs/classification/mnli/distilbert/bert-base/ablations/02_kd_frozen-sft.yaml
# -> outputs/classification/mnli/distilbert/bert-base/ablations/02_kd_frozen-sft/20260304-1200Z/

# Named run (flat output):
python forge.py --config configs/classification/mnli/distilbert/00_sft.yaml --run_name seed42
# -> outputs/seed42/

# Named + timestamp:
python forge.py --config configs/classification/mnli/distilbert/00_sft.yaml --run_name seed42 --timestamp
# -> outputs/seed42_20260304-1200Z/

# Resume (always explicit, tab-completable):
python forge.py --resume outputs/classification/mnli/distilbert/00_sft/20260304-1200Z
python forge.py --resume outputs/seed42
```

Users type real filesystem paths (tab-completable). `configs/` and `outputs/` prefixes + `.yaml` suffix are stripped internally to get the stem.

### Output Dir Modes

**Create** (requires `--config`):

| `--config` | `--run_name` | `--timestamp` | Output dir |
|:---:|:---:|:---:|---|
| Y | - | - | `outputs/{stem}/{timestamp}/` |
| Y | Y | - | `outputs/{run_name}/` |
| Y | Y | Y | `outputs/{run_name}_{timestamp}/` |

**Resume** (always via `--resume`, never implicit):

`--resume outputs/{path}` -> resumes `outputs/{path}/` (must have run_metadata.json)

Named run that already exists without `--force` is an error, not implicit resume.

### Changes

#### 1. Replace utility functions (forge.py:243-357)

Delete: `generate_config_id`, old `generate_wandb_group`, old `generate_wandb_tags`, `truncate_for_wandb`

Add:

- `resolve_config_path(config_arg)` - validate path exists as given. If not, try adding `configs/` prefix and/or `.yaml` suffix as fallback. Returns Path. Primary expectation is full tab-completable paths; fallback normalization is a convenience, not the design intent.
- `extract_stem(config_path)` - strip `configs/` prefix and `.yaml` suffix from an already-resolved config path to get stem
- `generate_wandb_run_id(prefix, timestamp)` - format: `{prefix[:x]}-{timestamp}`, truncate prefix to fit 64-char limit while preserving full timestamp. Uses `-` not `/` as separator (W&B run IDs must not contain `/`). Always includes timestamp for uniqueness (W&B tombstones deleted run IDs, so --force re-creates need unique IDs).
  - Auto-named: `generate_wandb_run_id(stem, timestamp)` -> `classification-mnli-...-{timestamp}`
  - Named: `generate_wandb_run_id(run_name, timestamp)` -> `seed42-{timestamp}`
  - Named+ts: `generate_wandb_run_id(run_name, timestamp)` -> `seed42-{timestamp}` (prefix is `seed42`, not `seed42_{ts}`)
- `generate_wandb_group(stem)` - parent dir of stem
- `generate_wandb_tags(stem)` - path segments except leaf

#### 2. Update argparse (forge.py:688-702)

- Add `--resume` (path to output dir, e.g. `outputs/...`)
- `--config` stays as-is (full path, tab-completable)
- `--run_name` is create-only (label for flat output dir)
- Keep `--timestamp`, `--force` (apply to named runs)

#### 3. Rewrite output dir logic (forge.py:710-773)

```
if args.resume:
    # RESUME MODE
    validate: no --config, no --run_name, no --force, no --timestamp
    resume_path = strip outputs/ prefix from args.resume
    output_dir = outputs/{resume_path}
    must exist + have run_metadata.json
    if dir exists but no run_metadata.json -> error:
      "No run_metadata.json found in {output_dir}. Run may have crashed before
       setup completed. Start fresh with: python forge.py --config CONFIG"
    is_resume = True

elif args.config:
    # CREATE MODE
    config_path = resolve_config_path(args.config)
    stem = extract_stem(config_path)
    warn if --timestamp without --run_name ("redundant, auto-named always includes timestamp")
    warn if --force without --run_name ("redundant, auto-named names are always unique")

    if args.run_name:
        # Named (flat)
        name = run_name or run_name_timestamp
        output_dir = outputs/{name}
        if exists and not --force -> error ("use --resume outputs/{name} or --force")
        if exists and --force -> delete W&B run + shutil.rmtree + create fresh
    else:
        # Auto-named (hierarchical)
        output_dir = outputs/{stem}/{timestamp}
    is_resume = False

elif args.run_name:
    # --run_name without --config
    error: "--config required for new runs"

else:
    error: no --config or --resume
```

#### 4. Config loading (forge.py:780-803)

- Create: `load_config(config_path)` (already resolved)
- Resume: load from stored `config.yaml`, load `stem` from metadata
- Backward compat: if metadata exists but lacks `stem`, regenerate it from stored `config` path field (which all existing metadata has). Use old `run_name` field as `display_name` (format won't match new scheme but it's the best available). Log a warning and re-save metadata with the new `stem` and `display_name` fields.

#### 5. Metadata + W&B (forge.py:873-957)

Metadata schema (auto-named):
```json
{
  "stem": "classification/mnli/distilbert/bert-base/ablations/02_kd_frozen-sft",
  "display_name": "classification/mnli/distilbert/bert-base/ablations/02_kd_frozen-sft/20260304-1200Z",
  "wandb_run_id": "classification-mnli-distilbert-...-20260304-1200Z",
  "config": "configs/classification/.../02_kd_frozen-sft.yaml",
  "created": "20260304-1200Z",
  "seed": 42
}
```

Metadata schema (named):
```json
{
  "stem": "classification/mnli/distilbert/00_sft",
  "display_name": "seed42",
  "wandb_run_id": "seed42-20260304-1200Z",
  "config": "configs/classification/mnli/distilbert/00_sft.yaml",
  "created": "20260304-1200Z",
  "seed": 42
}
```

`display_name` = the W&B display name and what's passed to `_train(run_name=...)`. This replaces the old ambiguous `run_name` metadata field.

- Auto-named: `{stem}/{timestamp}`
- Named: `{run_name}`
- Named + ts: `{run_name}_{timestamp}`

W&B group/tags derive from `stem`. W&B run name and group support `/` characters. W&B run IDs do not (restricted chars: `/ \ # ? % :`).

#### 6. Update `_train` call site (forge.py:496, ~997)

`_train(run_name=...)` passes to `TrainingArguments(run_name=...)` which is the W&B display name.
Pass `metadata["display_name"]` (works for both create and resume).

#### 7. Update `add_incomplete_job` (forge.py:408-432)

Store `resume_path` (output dir relative to `outputs/`). SLURM resubmit uses `--resume outputs/{resume_path}`.

#### 8. Update `scripts/slurm/run_training.sh`

The script keeps positional args for initial submission (tab-completable, familiar). Resubmit switches to `--resume`.

**Initial submission (positional, unchanged UX):**
```bash
sbatch scripts/slurm/run_training.sh configs/classification/mnli/distilbert/00_sft.yaml          # auto-named
sbatch scripts/slurm/run_training.sh seed42 configs/classification/mnli/distilbert/00_sft.yaml   # named
```

Script detects `*.yaml` as config (same logic as today), builds `python forge.py --config ... [--run_name ...] $ARGS`.

**Resubmit (from incomplete jobs):**

Current resubmit does `sbatch "$0" "$ACTUAL_RUN_NAME"`. New resubmit passes `_FORGE_RESUME_PATH` env var:

```bash
# At end of job, script reads resume_path from .incomplete_jobs:
_FORGE_RESUME_PATH="$ACTUAL_RESUME_PATH" PREV_JOB_ID="$SLURM_JOB_ID" \
  SEED="$SEED" MAX_STEPS="$MAX_STEPS" EPOCHS="$EPOCHS" DEBUG="$DEBUG" \
  sbatch "$0"
```

At script startup, before positional arg parsing, check for `_FORGE_RESUME_PATH`:
```bash
if [ -n "$_FORGE_RESUME_PATH" ]; then
    # Resubmit resume mode - skip positional arg parsing
    # Build ARGS from env vars (--seed, --max_steps, --epochs, --debug)
    # Run: python forge.py --resume "outputs/$_FORGE_RESUME_PATH" $ARGS
fi
```

Does NOT forward TIMESTAMP or FORCE (irrelevant for resume).

- Remove old implicit resume logic (dir-exists check that auto-resumed)
- Keep the pre-completion check for named runs (prevents wasteful GPU allocation). Change behavior: instead of silently resuming, exit with "Training already complete for: $RUN_NAME"
- Keep TIMESTAMP, FORCE env vars for initial named create runs only

#### 9. ~~Update `scripts/run_local.sh`~~ (REMOVED)

Script deleted during restructure. Local runs use `python forge.py` directly.

#### 10. Update `docs/guides/run-options.md`

Rewrite: create vs resume modes, error messages, output structure.

### Files

- `forge.py` - utility functions (243-357), argparse (688-702), output dir logic (710-803), metadata/W&B (873-957)
- `scripts/slurm/run_training.sh` - arg parsing, resubmit
- `docs/guides/run-options.md` - documentation

### Verification

1. Auto-named: `python forge.py --config configs/classification/mnli/distilbert/00_sft.yaml --max_steps 5 --debug`
   - Output at `outputs/classification/mnli/distilbert/00_sft/{timestamp}/`
   - Metadata has `stem` field
2. Named: `python forge.py --config configs/classification/mnli/distilbert/00_sft.yaml --run_name test1 --max_steps 5 --debug`
   - Output at `outputs/test1/`
3. Named + timestamp: add `--timestamp` -> `outputs/test1_{timestamp}/`
4. Named exists error: repeat #2 without --force -> error with helpful message
5. Resume auto-named: `python forge.py --resume outputs/classification/mnli/distilbert/00_sft/{timestamp} --max_steps 10 --debug`
6. Resume named: `python forge.py --resume outputs/test1 --max_steps 10 --debug`
7. Error cases: `--config` + `--resume`, `--resume` nonexistent, `--resume` pointing to stem dir (no run_metadata.json)
