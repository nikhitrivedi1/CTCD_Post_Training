# CTCD Experiments — Notes for Next Run

## Early Stopping / Eval Cadence

### QQP and MNLI: increase eval_steps
The current `eval_steps: 2000` is calibrated for small tasks but creates a severe mismatch
for large datasets. With `patience=5`, the effective patience in epochs is:

| Task | Steps/epoch | eval_steps | Evals/epoch | Effective patience |
|------|-------------|------------|-------------|-------------------|
| MNLI | 24,544      | 2,000      | 12.3        | 0.4 epochs        |
| QQP  | 22,741      | 2,000      | 11.4        | 0.4 epochs        |
| QNLI | 6,547       | 2,000      | 3.3         | 1.5 epochs ✓      |
| CoLA | 535         | 500        | 1.1         | 4.7 epochs ✓      |
| MRPC | 230         | 200        | 1.2         | 4.3 epochs ✓      |
| RTE  | 156         | 100        | 1.6         | 3.2 epochs ✓      |

QQP ablation 02 (`kd_frozen_sft`) was confirmed to stop at epoch ~0.97 due to early
stopping firing after only 10,000 steps (5 evals × 2,000 steps), never completing a
full epoch. MNLI is equally affected.

**Fix**: Set `eval_steps: 8000` (and matching `save_steps: 8000`) in `qqp/base.yaml` and
`mnli/base.yaml`. This targets ~3 evals/epoch, consistent with QNLI.

```yaml
# qqp/base.yaml and mnli/base.yaml
checkpointing:
  save_steps: 8000

evaluation:
  eval_steps: 8000
```
