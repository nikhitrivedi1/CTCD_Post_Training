# Forge: Core and Extensions

Forge is a config-driven framework for fine-tuning, knowledge distillation, and reinforcement learning. This doc defines what belongs in the **Core** (the platform) and what belongs in **Extensions** (lightweight additions that plug into Core without modifying it).

**Core** defines *how* Forge trains. We modify Core when adding fundamental capabilities: new training paradigms, new infrastructure, new abstract interfaces.

**Extensions** define *what* it trains on, *what* it optimizes for, and *how* it's evaluated. Configs, datasets, losses, rewards, tasks, bench scripts, SLURM wrappers, analysis tools. These are lightweight and never require Core changes.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                              EXTENSIONS                             │
│                       (lightweight, pluggable)                      │
│                                                                     │
│  ┌─────────────┐ ┌──────────────┐ ┌──────────────┐ ┌─────────────┐  │
│  │   Configs   │ │   Datasets   │ │    Losses    │ │   Rewards   │  │
│  │    (YAML)   │ │  (handlers)  │ │ (strategies) │ │ (functions) │  │
│  │             │ │              │ │              │ │             │  │
│  │ base.yaml   │ │ glue.py      │ │ ResponseCE   │ │ edit_sim    │  │
│  │ 00_sft.yaml │ │ europarl.py  │ │ ResponseKD   │ │ inline_fmt  │  │
│  │ 01_grpo.yaml│ │ formatting.py│ │ FeatureKD    │ │ unit_test   │  │
│  │ ...         │ │ omnidoc.py   │ │ AttentionKD  │ │ bleu        │  │
│  │             │ │ olmocr_mix.py│ │ RevKD        │ │ ...         │  │
│  └──────┬──────┘ └───────┬──────┘ └───────┬──────┘ └──────┬──────┘  │
│         │                │                │               │         │
│  ┌──────┴──────┐ ┌───────┴──────┐ ┌───────┴──────┐ ┌──────┴──────┐  │
│  │    Tasks    │ │    Bench     │ │     Data     │ │    SLURM    │  │
│  │  (registry) │ │  (scripts)   │ │ (pipelines)  │ │  (scripts)  │  │
│  │             │ │              │ │              │ │             │  │
│  │ mnli        │ │ olmocr_bench │ │ prepare.py   │ │ run_train   │  │
│  │ qqp         │ │ omnidoc_bench│ │ fmt_generator│ │ run_bench   │  │
│  │ sst2        │ │ fmt_bench    │ │ olmocr_pipe  │ │ run_score   │  │
│  │ cola        │ │              │ │              │ │             │  │
│  └─────────────┘ └──────────────┘ └──────────────┘ └─────────────┘  │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                     Workbench (analysis)                    │   │
│   │  probing │ cka │ attention │ disagreement │ audit           │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
├═════════════════════════════════════════════════════════════════════┤
│                                                                     │
│         DATASET_REGISTRY    LOSS_REGISTRY    REWARD_REGISTRY        │
│    TASK_REGISTRY      DatasetHandler   DistillationLossStrategy     │
│                                                                     │
├═════════════════════════════════════════════════════════════════════┤
│                                                                     │
│                            CORE PLATFORM                            │
│                      (infrastructure, evolves)                      │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    forge.py (entry point)                   │   │
│   │         Config parsing -> Model loading -> Training         │   │
│   └──────────────────────────────┬──────────────────────────────┘   │
│                                  │                                  │
│  ┌──────────┐  ┌─────────────────┴─────┐  ┌──────────────────────┐  │
│  │  Models  │  │        Training       │  │         Data         │  │
│  │          │  │                       │  │                      │  │
│  │ loader   │  │ trainer    callbacks  │  │ collators  formatters│  │
│  │          │  │ projections           │  │ handlers/base.py     │  │
│  │          │  │ layer_mapping         │  │ ocr_utils            │  │
│  └──────────┘  └───────────────────────┘  └──────────────────────┘  │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────────┐  │
│  │  Inference   │  │  Evaluation  │  │           Utils           │  │
│  │              │  │              │  │                           │  │
│  │ engine       │  │ evaluate     │  │ config     checkpoint     │  │
│  │              │  │ metrics      │  │ jobs       environment    │  │
│  └──────────────┘  └──────────────┘  └───────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Extensions plug INTO Core through registries (arrows flow down).
Core never imports Extensions. Extensions import Core interfaces.
```

---

## Core Platform

Core owns the **infrastructure**: how losses are combined, how rewards are dispatched, how datasets are loaded, how configs are parsed.

### Modules

| Module | Path | Responsibility |
|--------|------|----------------|
| Entry point | `forge.py` | Config parsing, model loading, training orchestration |
| Trainer | `src/training/trainer.py` | Training loop, loss/reward dispatch |
| Projections | `src/training/projections.py` | Dimension alignment for KD |
| Layer mapping | `src/training/layer_mapping.py` | Student-teacher layer correspondence |
| Callbacks | `src/training/callbacks.py` | SLURM signal handling, W&B integration |
| Model loader | `src/models/loader.py` | HF Hub loading, LoRA attachment |
| Collators | `src/data/collators.py` | Batch assembly for different model types |
| Formatters | `src/data/formatters.py` | SFT vs RL data formatting pipelines |
| Config system | `src/utils/config.py` | YAML loading with inheritance |
| Checkpointing | `src/utils/checkpoint.py` | Save/resume state |
| Inference engine | `src/inference/engine.py` | Model inference (used by RL + bench) |
| Evaluation | `src/evaluation/evaluate.py` | Eval loop, best model selection |
| Metrics | `src/evaluation/metrics.py` | Task-specific metric functions |
| Job tracking | `src/utils/jobs.py` | Incomplete job registry |
| Environment | `src/utils/environment.py` | Device, SLURM, Colab detection |
| OCR utils | `src/data/ocr_utils.py` | Image loading, PDF rendering |
| Handler base | `src/data/handlers/base.py` | Abstract interface for all datasets |

`losses.py`, `rewards.py`, and `tasks/registry.py` contain both infrastructure and individual entries. The infrastructure (registries, dispatch logic, ABCs) is Core. The individual functions and entries are Extensions.

### When to modify Core

- **New training paradigm**: Adding DPO alongside GRPO changes the trainer's dispatch logic.
- **New distillation signal type**: Beyond Feature/Attention/Response changes the trainer's forward pass.
- **New model type**: Special collation or formatting needs for a new model family.
- **Registry infrastructure**: A new category of pluggable components (like adding a registry itself).
- **Config schema**: New top-level config sections or inheritance rules.

### When NOT to modify Core

- Adding a new dataset -> write a handler
- Adding a new loss function -> define strategy, register it
- Adding a new RL reward -> define function, register it
- Adding a new task -> add a registry entry
- Adding a new benchmark -> write a bench script
- Running a new experiment -> write a config YAML
- Adding cluster automation -> write a SLURM script

If any of these requires a Core change, the Core's abstractions need improvement.

---

## Extensions

Extensions own the **individual entries**: specific loss functions, specific reward functions, specific dataset handlers, specific configs.

### 1. Configs (YAML)

Pure data. Define *what* to run by referencing extensions by name.

```
configs/base.yaml                     Root defaults
  configs/ocr/base.yaml              OCR defaults
    configs/ocr/.../base.yaml        Model pair defaults
      configs/ocr/.../01_grpo.yaml   Specific experiment
```

Configs reference extensions by name. Core resolves through registries at runtime:

```yaml
rl:
  reward: [edit_similarity, inline_formatting]   # -> REWARD_REGISTRY
  reward_weights: [1.0, 3.0]
data:
  dataset: formatting                             # -> DATASET_REGISTRY
```

### 2. Data Handlers

Each dataset implements the `DatasetHandler` interface:

```python
class DatasetHandler(ABC):
    def prepare(data_dir, **kwargs) -> None         # Download/generate (idempotent)
    def validate(data_dir) -> dict                   # Check integrity
    def load(data_dir, ...) -> (Dataset, Dataset)    # HF Datasets (for training)
    def load_paths(data_dir, split) -> list[dict]    # File paths (for bench)
```

| Handler | File | Domain |
|---------|------|--------|
| GLUE | `src/data/handlers/glue.py` | Classification (MNLI, QQP, SST-2, CoLA) |
| Europarl | `src/data/handlers/europarl.py` | Translation (EN-DE) |
| OmniDocBench | `src/data/handlers/omnidocbench.py` | Document OCR |
| olmOCR-mix | `src/data/handlers/olmocr_mix.py` | OCR training data |
| Formatting | `src/data/handlers/formatting.py` | Synthetic formatting |
| Synthmix | `src/data/handlers/synthmix.py` | Synthetic OCR mixtures |

**Adding a new dataset:** Create a handler file, register in `__init__.py`. Zero Core changes.

### 3. Loss Functions

`src/training/losses.py` has a `LOSS_REGISTRY` and `get_loss_strategy()` factory:

```python
LOSS_REGISTRY = {
    "response_kl": KLDistillationLoss,
    "response_ce": SoftTargetDistillationLoss,
    "feature_mse": FeatureLossStrategy,
    "attention_kl": AttentionLossStrategy,
    ...
}
```

**Adding a new loss:** Define the strategy class, register it. Config references it by name.

### 4. Reward Functions

`src/training/rewards.py` has a `REWARD_REGISTRY`:

```python
REWARD_REGISTRY = {
    "edit_similarity": ocr_reward,
    "inline_formatting": inline_formatting_reward,
    "unit_test": unit_test_reward,
    "bleu": translation_reward,
    "format": format_reward,
    "table_cell_f1": table_cell_f1_reward,
    "formula_cdm": formula_cdm_reward,
    "reading_order": reading_order_reward,
}
```

**Adding a new reward:** Define the function, add to registry. Config references it by name.

### 5. Task Registry

`src/tasks/registry.py` stores task metadata (labels, metrics, type):

```python
TASK_REGISTRY = {
    "mnli": {"num_labels": 3, "type": "classification", "primary_metric": "accuracy", ...},
    "qqp":  {"num_labels": 2, "type": "classification", "primary_metric": "f1", ...},
    "ocr":  {"num_labels": None, "type": "ocr", "primary_metric": "edit_similarity", ...},
}
```

**Adding a new task:** Add an entry to the `TASK_REGISTRY` dict.

### 6. Bench Scripts

Bench scripts interface with Core through two abstractions:

- **`DatasetHandler`** (`src/data/handlers/base.py`) - data lifecycle: prepare, validate, load paths
- **`InferenceEngine`** (`src/inference/engine.py`) - model loading and generation

Every bench script should follow the same interface contract:

```
handler = get_handler(dataset_name)     # Core: registry lookup
handler.prepare(data_dir)               # Interface: ensure data exists
paths = handler.load_paths(data_dir)    # Interface: file paths (memory-efficient)

engine = InferenceEngine.load(model_id) # Core: model loading
for page in paths:                      # Extension: bench-specific scoring
    output = engine.generate(page)
    score(output, page["gt_path"])
```

| Script | Path | Handler |
|--------|------|---------|
| olmOCR-Bench | `scripts/benchmarks/olmocr_bench.py` | None (own format, violation #4) |
| OmniDoc-Bench | `scripts/benchmarks/omnidoc_bench.py` | `omnidocbench` |
| Formatting-Bench | `scripts/benchmarks/formatting_bench.py` | `formatting` |

**Adding a new bench:** Write a script following this interface. No Core changes.

### 7. Data Pipelines

Data generation and preparation (`scripts/data/`). Invoked by handlers or run standalone:

- `prepare.py` - unified CLI for all dataset prep
- `formatting_generator.py` - synthetic formatting data
- `olmocr_mix_pipeline.py` - olmOCR-mix download + render

### 8. SLURM Scripts

Cluster automation (`scripts/slurm/`). Wraps Core entry points with job scheduling, auto-requeue, and signal handling. Fully optional for local runs.

### 9. Analysis Tools

Post-training analysis (`workbench/`). Probing, CKA similarity, attention entropy, formatting audits. Independent of Core.

---

## Open-Closed Principle

> "Open for extension, closed for modification."

**Core is open for extension through new capabilities.** But Core's interfaces (how configs are parsed, how trainers iterate, how handlers are called) should remain stable.

**Extensions are open for addition without modification.** Adding a new dataset, loss, reward, task, or bench never touches Core files.

### Extension Points

| What you're adding | Interface | Registry | Core changes? |
|---------------------|-----------|----------|---------------|
| Dataset | `DatasetHandler` | `DATASET_REGISTRY` | No |
| Loss function | `DistillationLossStrategy` | `LOSS_REGISTRY` | No |
| Reward function | `Callable[[str, str], float]` | `REWARD_REGISTRY` | No |
| Task | Dict entry | `TASK_REGISTRY` | No |
| Bench script | `DatasetHandler` + `InferenceEngine` | Standalone | No |
| Config | YAML inheritance | File system | No |
| Training paradigm | Added to `trainer.py` | Config `algorithm` field | **Yes (Core)** |
| Distillation signal | Added to trainer forward pass | Config flags | **Yes (Core)** |

### The Test

When adding something, ask: "Does this require changing `forge.py`, `trainer.py`, or any Core infrastructure?"

- **No** -> Extension. Write the plugin, register it, done.
- **Yes** -> Core improvement. Treat as a platform change with review and testing.

---

## Directory Ownership

```
Core Platform (infrastructure)
  forge.py                       Entry point, config orchestration
  src/training/trainer.py        Training loop, loss/reward dispatch
  src/training/projections.py    Dimension alignment
  src/training/layer_mapping.py  Layer correspondence
  src/training/callbacks.py      Training callbacks
  src/models/loader.py           Model loading
  src/data/collators.py          Batch assembly
  src/data/formatters.py         Data formatting pipelines
  src/data/handlers/base.py      Handler interface (abstract)
  src/data/ocr_utils.py          Image loading, PDF rendering
  src/inference/engine.py        Inference engine
  src/evaluation/                Eval loop, metrics
  src/utils/                     Config, checkpoint, jobs, environment

Extensions (pluggable entries)
  src/data/handlers/*.py         Dataset plugins (one file per dataset)
  src/training/losses.py         Loss strategies + registry
  src/training/rewards.py        Reward functions + registry
  src/tasks/registry.py          Task metadata entries
  configs/                       Experiment definitions (YAML)
  scripts/benchmarks/                  Bench scripts
  scripts/data/                  Data pipelines
  scripts/slurm/                 Cluster automation
  workbench/                     Analysis tools
```

---

## Violations

Active deviations from Core/Extension separation, found via codebase audit.

**1. `TASK_REWARDS` hardcoded in rewards.py**

`src/training/rewards.py` lines 711-714 hardcode task-to-reward mappings:

```python
TASK_REWARDS = {"europarl": "bleu", "ocr": "edit_similarity"}
```

This belongs in the task registry as a `default_reward` field per task. Adding a new task with verifiable rewards currently requires modifying this file.

**2. `TranslationMetricsCallback` hardcodes dataset name**

`src/training/callbacks.py` ~line 343 hardcodes `"Helsinki-NLP/europarl"` instead of reading from config. Should accept the dataset name as a parameter.

**3. Europarl handler applies SFT formatting in `load()`**

`src/data/handlers/europarl.py` lines 92-106 build prompt structure (system message, source/target formatting) inside `load()`. The handler contract says "unformatted samples." This formatting belongs in `src/data/formatters.py`.

**4. olmOCR-Bench bypasses handler interface**

`scripts/benchmarks/olmocr_bench.py` lines 139-188 has its own `load_bench_data()` with custom JSONL parsing. Should have a registered handler like omnidoc and formatting benches do.

---

## Structural Gaps

**5. Mixed Core/Extension code in single files**

`losses.py`, `rewards.py`, and `tasks/registry.py` each contain both infrastructure (registries, dispatch, ABCs) and individual entries (specific strategies, functions, metadata). This makes it hard to distinguish Core from Extensions at the file level. The loss registry was missed during initial documentation because it shares a file with 800+ lines of individual loss strategies.

Possible improvement:

```
src/training/losses/
  base.py         DistillationLossStrategy ABC, LOSS_REGISTRY, get_loss_strategy()
  strategies.py   Individual strategy classes (ResponseCE, FeatureKD, ...)
```

Same pattern for rewards.

**6. olmOCR-Bench has no handler**

Its data loading is self-contained in the bench script. A registered handler would make it consistent with omnidoc/formatting.

**7. Bench scripts don't share a base class**

All three follow the same pattern (prepare -> load_paths -> inference -> score) but implement it independently. A `BaseBench` could formalize this.

**8. Data generators called via subprocess**

Handlers invoke generators (e.g., `formatting_generator.py`) via `subprocess.run()`, which makes error propagation fragile. Direct imports would surface failures better.

**9. Inconsistent registry naming**

The four registries don't follow a consistent naming convention:

| Component | Registry name | Factory function |
|-----------|--------------|-----------------|
| Datasets | `DATASET_REGISTRY` | `get_handler()` |
| Losses | `LOSS_REGISTRY` | `get_loss_strategy()` |
| Rewards | `REWARD_REGISTRY` | `get_reward_functions()` |
| Tasks | `TASK_REGISTRY` | (no factory, just accessors) |

Registry names are now consistent. Factory functions should follow a consistent pattern (e.g., `get_dataset()`, `get_loss()`, `get_reward()`, `get_task()`).

Similarly, some reward registry keys don't match their function names: `"edit_similarity"` maps to `ocr_reward`, `"bleu"` maps to `translation_reward`. Other entries are consistent (e.g., `"unit_test"` -> `unit_test_reward`). The keys should either match the function name pattern or all use a domain-based naming scheme, not a mix of both.

---

## Fail-Fast, Fail-Cheap

> Validate everything you can before doing anything expensive.

GPU hours are the scarcest resource. A training run that OOMs 10 minutes in, after downloading a 7B model, wastes time and compute that a 2-second config check could have prevented. The same applies to bench scripts that load a model before realizing the evaluation data doesn't exist.

**Principle:** Every entry point (training, bench, data prep) should run a lightweight preflight check that catches misconfigurations before any heavyweight work begins. Config parsing, registry lookups, file existence checks, and memory estimates are all cheap. Model downloads, data loading, and GPU allocation are expensive. Separate them.

### Current State

Today, validation is "fail on first use." Missing config keys raise `KeyError` partway through setup. Registry names are resolved only when the trainer or bench script first needs them. There is no config schema - a typo in a YAML field silently becomes `None` and surfaces as a confusing error much later.

### Preflight Checks

Each entry point should validate in this order, before any model or data loading:

**1. Config structure**

- Required fields present (`model.student_name`, `training.num_epochs`, `data.dataset`, etc.)
- Field types correct (numeric values are numbers, lists are lists)
- No unknown top-level keys (catches typos like `trainng` instead of `training`)
- Inheritance chain resolves without cycles (already checked, depth limit = 10)

**2. Registry references**

- `data.dataset` exists in `DATASET_REGISTRY`
- `training.loss_type` exists in `LOSS_REGISTRY` (if specified)
- `rl.reward` names all exist in `REWARD_REGISTRY` (if RL mode)
- `data.task` exists in `TASK_REGISTRY` (if classification)

**3. Resource availability**

- Model IDs are valid HuggingFace Hub references (HEAD request, no download)
- Checkpoint paths exist on disk (if `--resume` or `--checkpoint`)
- Data directories exist or can be created
- Output directory is writable

**4. Hardware fit**

- Estimated GPU memory for student + teacher + optimizer fits available VRAM
- Batch size is feasible given sequence length and model size
- Multi-GPU config matches available device count

Only after all preflight checks pass should the run proceed to model downloads, data loading, and training.

### Where This Lives

Preflight validation is **Core infrastructure**. It belongs in a dedicated module that entry points call before doing real work:

```
src/utils/preflight.py       Validation functions
  validate_config(config)    Schema + required fields
  validate_references(config) Registry lookups
  validate_resources(config)  File/model existence
  validate_hardware(config)   Memory estimates
  preflight(config)           Runs all of the above
```

Entry points call `preflight(config)` right after config loading, before any model or data work:

```python
# forge.py
config = load_config(config_path)
preflight(config)       # Fail here, not 10 minutes later
# ... proceed to model loading, training
```

Bench scripts do the same:

```python
# scripts/benchmarks/olmocr_bench.py
config = parse_args()
preflight_bench(config)  # Check model path, data dir, output dir
# ... proceed to inference
```

### Failure Messages

Preflight failures should be actionable. Not just "KeyError: 'student_name'" but:

```
Preflight failed:
  - config.model.student_name: missing (required for all training modes)
  - config.rl.reward[1]: "inline_format" not in REWARD_REGISTRY
    Available: edit_similarity, inline_formatting, unit_test, bleu
  - config.data.data_dir: "data/ocr/formatting" does not exist
    Run: python scripts/data/prepare.py --dataset formatting
```

All errors collected and reported at once, not one at a time.

### Dry Run Mode

A `--preflight` flag on every entry point runs all validation, reports results, and exits without doing any real work. This lets you verify a config before committing GPU time:

```bash
# Training - validate config, registries, model refs, hardware fit
python forge.py --config configs/ocr/qwen3.5/01_grpo.yaml --preflight

# Bench - validate model path, data dir, output dir
python scripts/benchmarks/olmocr_bench.py --model Qwen/Qwen2.5-VL-3B --preflight

# Data prep - validate handler exists, output dir writable
python scripts/data/prepare.py --dataset formatting --preflight
```

```
Preflight passed (3 checks in 1.2s):
  [ok] Config structure: all required fields present
  [ok] Registry references: dataset=formatting, rewards=[edit_similarity, inline_formatting]
  [ok] Resources: model exists on HF Hub, data_dir writable
  [warn] Hardware: estimated 34GB VRAM, available 40GB (85% utilization)
```

This is especially valuable for cluster workflows where jobs queue for hours before running. A preflight check in the SLURM script's preamble catches bad configs before the job even reaches the GPU node:

```bash
# scripts/slurm/run_training.sh
python forge.py --config $CONFIG --preflight || exit 1
# ... proceed to actual training
```
