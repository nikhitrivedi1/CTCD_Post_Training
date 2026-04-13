# GLUE checkpoint policy

This document describes **when** the trainer writes checkpoints for GLUE classification runs in Forge, as defined by YAML configs and `TrainingArguments` in `forge.py`.

## What counts as a “training step”

Checkpoints are driven by Hugging Face `TrainingArguments.save_steps`. In Forge, that value comes from `checkpointing.save_steps` in the merged config (see `forge.py`).

That number is the **global optimizer step** (the trainer’s `global_step`): one step per optimizer update, after any gradient accumulation. It is **not** “micro-batch” or “epoch” unless your run happens to align that way.

With `save_strategy` left at Hugging Face defaults (`"steps"` when saving on a schedule), the trainer saves a checkpoint **every `save_steps` steps** (e.g. 2000 → steps 2000, 4000, 6000, …), and still performs the usual **final save** at the end of training.

## Design rule used in configs

For runs that use `load_best_model_at_end`, **`eval_steps` should match `save_steps`** so each evaluation lines up with a checkpoint that can be selected as “best.” This is called out explicitly in `configs/CTCD_Experiments/_shared.yaml`; the DistilBERT `00_sft.yaml` GLUE configs follow the same pairing.

## Repository default (any task)

If a config inherits only through `configs/base.yaml` and does **not** override `checkpointing`, you get:

| Setting | Value |
|--------|--------|
| `checkpointing.save_steps` | **500** |
| `evaluation.eval_steps` | **500** |

Path: `configs/base.yaml`.

Classification task roots (`configs/classification/<task>/base.yaml`) only set `data`; they inherit this **500** cadence unless a deeper config overrides it.

## Canonical cadence by GLUE task (DistilBERT)

The usual **DistilBERT supervised baseline** for each GLUE task is `configs/classification/<task>/distilbert/00_sft.yaml`. Those files set both `save_steps` and `eval_steps` to the same value:

| GLUE task | `save_steps` / `eval_steps` | Primary config reference |
|-----------|----------------------------|---------------------------|
| **MNLI** | **2000** | `configs/classification/mnli/distilbert/00_sft.yaml` |
| **QQP** | **2000** | `configs/classification/qqp/distilbert/00_sft.yaml` |
| **SST-2** | **2000** | `configs/classification/sst2/distilbert/00_sft.yaml` |
| **CoLA** | **500** | `configs/classification/cola/distilbert/00_sft.yaml` |
| **MRPC** | **200** | `configs/classification/mrpc/distilbert/00_sft.yaml` |
| **QNLI** | **2000** | `configs/classification/qnli/distilbert/00_sft.yaml` |
| **RTE** | **100** | `configs/classification/rte/distilbert/00_sft.yaml` |
| **WNLI** | **40** | `configs/classification/wnli/distilbert/00_sft.yaml` |

**Expected checkpoint steps** for one of these runs: all positive multiples of the table value that occur **before or at** the last optimizer step of the run, plus the **final** checkpoint when training completes.

Smaller tasks use a smaller `save_steps` so you get several saves within a typical 3-epoch run (WNLI/RTE/MRPC), while large-data tasks use 2000 to limit disk and evaluation cost.

## CTCD experiment tree (`configs/CTCD_Experiments/`)

The CTCD study shares defaults in `configs/CTCD_Experiments/_shared.yaml` (**`save_steps`: 2000**, **`eval_steps`: 2000**, **`save_total_limit`: 1**). Per-task `base.yaml` files override cadence where needed:

| Task | `save_steps` / `eval_steps` in task `base.yaml` |
|------|--------------------------------------------------|
| MNLI, QQP, QNLI | **2000** |
| CoLA | **500** |
| MRPC | **200** |
| RTE | **100** |

**SST-2** and **WNLI** do not have folders under `configs/CTCD_Experiments/`; use the DistilBERT `00_sft.yaml` paths in the table above for the same cadence.

## BERT-base KD ablations under `bert-base/ablations/`

For each GLUE task, the ablation YAMLs (`01_kd_frozen-base`, `02_kd_frozen-sft`, `03_kd_cotrain-base`, `04_kd_cotrain-sft`, etc.) use the **same `save_steps` as the DistilBERT `00_sft.yaml`** for that task (e.g. MNLI/QQP/SST-2/QNLI at **2000**, CoLA **500**, MRPC **200**, RTE **100**, WNLI **40**).

## Other variants (not the default per-task policy)

Some **MNLI** GPU / batch-size configs under `configs/classification/mnli/distilbert/gpu/` and staged/FAR experiments use different `save_steps` (e.g. **250**, **500**, or **2000**) for throughput or staging—always read the `checkpointing:` block in the exact YAML you launch.

## Related settings

| Key | Role |
|-----|------|
| `checkpointing.save_total_limit` | Max number of `checkpoint-*` directories to keep rotating (does not change *when* saves trigger). |
| `checkpointing.no_weights` | Forge can skip writing full weight checkpoints when configured (see `forge.py`); most GLUE configs use normal saves. |
| `evaluation.eval_steps` | Should match `save_steps` when using best-checkpoint selection at end of training. |

Supported GLUE classification tasks in the Forge task registry are: MNLI, QQP, SST-2, CoLA, MRPC, QNLI, RTE, WNLI (`src/tasks/registry.py`). **STS-B** and other GLUE subsets are not registered as tasks in this repo.
