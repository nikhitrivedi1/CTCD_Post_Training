<div align="center">
  <h1>Beyond Frozen Teachers: Co-Training and Co-Distillation for Task-Specific Fine-Tuning</h1>
  <p>
    <strong>Nikhil Trivedi&nbsp;·&nbsp;Karl Ni&nbsp;·&nbsp;Karthik Ravi&nbsp;·&nbsp;Mohammed Ali Sehral</strong><br>
    <em>Northeastern University</em>
  </p>
  <p>
    <a href="[PASTE ARXIV LINK HERE]">arXiv</a>&nbsp;|&nbsp;
    <a href="[PASTE PROJECT PAGE LINK HERE]">Project Page</a>&nbsp;|&nbsp;
    <a href="LICENSE">License</a>
  </p>
</div>

---

## CTCD

Standard knowledge distillation freezes a teacher and treats it as a fixed source of soft labels for a student. **Co-Training and Co-Distillation (CTCD)** generalizes this setup along two independent axes:

- **Co-training** — the teacher trains alongside the student instead of staying frozen.
- **Co-distillation** — the teacher itself receives a distillation signal back from the student.

This repository provides the reference implementation: `DistillationTrainer` (`src/training/trainer.py`), which implements all three regimes below behind a single YAML switch, the config suite used to sweep them across 8 classification tasks, and the SLURM scripts used to produce the paper's results.

**Highlights**

- Every point on the spectrum from plain fine-tuning to full CTCD is a YAML change, not a code change.
- The teacher is a first-class trainable model, not a fixed label source — `create_optimizer()` adds its parameters to the optimizer whenever it's unfrozen.
- Reproduces the paper's full experiment pipeline: hyperparameter grid search → loss-coefficient search → temperature scaling → multi-seed replication.
- Covers 7 GLUE tasks (CoLA, MNLI, MRPC, QNLI, QQP, RTE, SST-2) plus CLINC150 intent classification, DistilBERT student / BERT-base teacher.
- `tests/unit/test_trainer.py` verifies the mechanism directly, e.g. that CTCD loss exceeds pure co-training loss, and that co-distillation gradients don't leak across the student/teacher boundary.

---



## Problem Formulation

Let `y` be the ground-truth label, `y_S` the student's prediction, and `y_T` the teacher's prediction. Given student loss weights `alpha_h, alpha_s` and teacher loss weights `beta_h, beta_s`:

```text
L_student = alpha_h * CE(y, y_S) + alpha_s * KD(y_T || y_S)
L_teacher = beta_h  * CE(y, y_T) + beta_s  * KD(y_S || y_T)
```

The `alpha_h, alpha_s, beta_h, beta_s` combination selects the training regime:

| Regime          | `alpha_h` | `alpha_s` | `beta_h` | `beta_s` | Description                                                                   |
| --------------- | --------- | --------- | -------- | -------- | ----------------------------------------------------------------------------- |
| SFT (no teacher)| 1         | 0         | –        | –        | Student trained alone on ground-truth labels, no teacher involved            |
| Traditional KD  | 1         | 1         | 0        | 0        | Teacher is frozen — a fixed source of soft labels                             |
| Co-training     | 1         | 1         | 1        | 0        | Teacher trains on its own hard labels, but not from the student               |
| **CTCD** (full) | 1         | 1         | 1        | 1        | Teacher trains on both signals — it learns from the student while teaching it |

The `0`/`1` values above indicate which loss terms are active, not fixed magnitudes — any nonzero `alpha`/`beta` coefficient is itself a tunable hyperparameter (e.g. `alpha_s=0.5` to down-weight the KD term relative to the hard-label CE term).


---



## Quick Start

```bash
conda env create -f environment.yml
conda activate forge
source scripts/env/init.sh

cp .env.example .env   # fill in WANDB_API_KEY, HF_TOKEN, etc.
python test_setup.py   # verify the environment
```

`init.sh` installs conda activate/deactivate hooks (`PATH`, `LD_LIBRARY_PATH`, `HF_HOME`) and Python dependencies from `requirements.txt` (a lockfile generated via `uv` from `pyproject.toml` — don't hand-edit it). Run `init.sh` once per environment creation.

To regenerate the lockfile after changing `pyproject.toml`, always pass `--universal`: `uv pip compile pyproject.toml -o requirements.txt --universal`. Without it, `uv` only resolves for the machine it's run on and silently drops Linux-only CUDA runtime packages (`nvidia-*-cu12`, `triton`) that torch needs on the cluster — since `init.sh` installs with `--no-deps`, pip won't pull them in as transitive deps either.

Then launch a single experiment:

```bash
python forge.py --config configs/CTCD_Experiments/sst2/05_ctcd.yaml
```

Useful `forge.py` flags: `--run_name` (custom output dir), `--max_steps` (smoke test), `--debug` (routes W&B to a `Forge-Test` project instead of `CTCD_Experiments`), `--force` (overwrite an existing named run), `--resume <output_dir>`.

---



## Repository Architecture

```text
forge.py               entry point: YAML config -> training run
configs/                base task configs + CTCD ablations, best/best_seed configs, runbooks
src/training/           DistillationTrainer (core CTCD mechanism), loss strategies
scripts/slurm/          SLURM sweep scripts for the paper's experiment pipeline
tests/                  pytest suite, including CTCD-specific correctness tests
```

---



## Testing

```bash
pytest tests/unit/test_trainer.py
```

This is the correctness reference for the CTCD mechanism itself — dedicated test classes cover the co-training teacher CE path (`TestCoTrainingTeacherCE`), that CTCD loss exceeds pure co-training loss (`test_ctcd_loss_larger_than_cotraining_only`), and that gradients from co-distillation don't leak across the student/teacher boundary incorrectly (`test_codistillation_student_kl_does_not_backprop_into_teacher`).

---



## Citation

```bibtex
@inproceedings{trivedi2026ctcd,
  title     = {Beyond Frozen Teachers: Co-Training and Co-Distillation for Task-Specific Fine-Tuning},
  author    = {Trivedi, Nikhil and Ni, Karl and Ravi, Karthik and Shehral, Mohammad Ali},
  booktitle = {[PASTE VENUE HERE]},
  year      = {2026},
  note      = {[PASTE DOI/LINK HERE]}
}
```

---



## License

MIT License — see [LICENSE](LICENSE).
