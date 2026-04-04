# CTCD Experiments — Required Configurations

Cross-reference against `CTCD_Experiment_Plan.md` (April 2, 2026).

The 36 ablation configs (6 tasks × 6 ablations) are baseline configs. The grid search parameters
(LR, batch size, gradient clipping for Part 1; loss weights for Part 2; temperature for Part 3)
are swept by the bash script via runtime overrides.

---

## Config Coverage

All 36 required baseline configs exist:

| Task | 00 | 01 | 02 | 03 | 04 | 05 |
|---|---|---|---|---|---|---|
| mnli | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| qqp | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| qnli | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| cola | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| mrpc | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| rte | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

---

## Missing Fields in Existing Configs

The following required fields from the plan are absent from every task `base.yaml` and `_shared.yaml`.
These should be added to the appropriate base configs — they are fixed across all parts of the grid,
so the bash script should not need to override them.

### Add to each task `base.yaml`

| Field | Required value | Plan note |
|---|---|---|
| `evaluation.metric_for_best_model` | Task-specific (see below) | Must key early stopping to task metric, not eval loss |
| `evaluation.greater_is_better` | `true` | HF Trainer defaults to lower-is-better; this overrides it |
| `training.load_best_model_at_end` | `true` | Required for best checkpoint saving |
| `training.early_stopping_patience` | `5` | Gives small tasks (RTE, MRPC, CoLA) sufficient runway |

Per-task `metric_for_best_model`:

| Task | `metric_for_best_model` |
|---|---|
| mnli | `eval_accuracy` |
| qqp | `eval_f1` |
| qnli | `eval_accuracy` |
| cola | `eval_matthews_correlation` |
| mrpc | `eval_f1` |
| rte | `eval_accuracy` |

### `max_grad_norm` in `_shared.yaml`

`_shared.yaml` hardcodes `max_grad_norm: 2.0`. The Part 1 grid sweeps `{None, 1.0}` — **2.0 is not in the plan's sweep values**. Remove it from `_shared.yaml` and let the bash script pass the correct value per run.

---

## Additional Observations

### Ablations 03/04 — co-train vs frozen teacher
**03/04 use `model.teacher.freeze: false`** (trainable teacher = co-training). **Frozen** teachers are **01/02** (`freeze: true`).

Each `03_cotrain_base.yaml` and `04_cotrain_sft.yaml` sets **`distillation.beta_h: 1.0`** and **`beta_s: 0.0`** so the teacher is explicitly **CE-only** (no KD term on the teacher). **05** sets `beta_s: 1.0` for full CTCD co-distillation.

### T² correction factor (Part 3)
The plan warns: "All three T terms must use the same value: `s/T`, `t/T`, and the `T²` correction factor." Verify in the trainer before running Part 3 temperature sweeps.

### MRPC `eval_split`
`mrpc/base.yaml` sets `eval_split: validation`. Confirm the other tasks default to the correct split without needing this field.
