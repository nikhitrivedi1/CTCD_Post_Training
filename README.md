# Beyond Frozen Teachers: Co-Training and Co-Distillation for Task-Specific Fine-Tuning

**Train students and teachers together, with and without co-distillation.**

Nikhil Trivedi · Karl Ni · Karthik Ravi · Mohammed Ali Sehral
[PASTE AFFILIATION HERE]

[Project Page]([PASTE PROJECT PAGE LINK HERE])
[SPIE Paper]([PASTE PAPER LINK HERE])
[MIT License](LICENSE)

Standard knowledge distillation freezes a teacher and treats it as a fixed source of soft labels for a student. This repository generalizes that setup: the teacher can train alongside the student (**co-training**), and can itself receive a distillation signal back from the student (**co-distillation**). Given student loss weights `alpha_h, alpha_s` and teacher loss weights `beta_h, beta_s`:

```text
L_student = alpha_h * CE(y, y_S) + alpha_s * KD(y_T || y_S)
L_teacher = beta_h  * CE(y, y_T) + beta_s  * KD(y_S || y_T)
```

Traditional distillation sets `beta_h = beta_s = 0` (frozen teacher). Co-training sets `beta_h = 1, beta_s = 0` (the teacher only trains on its own labels). **CTCD** sets `beta_h = beta_s = 1` — the teacher trains on both, so it learns from the student while teaching it. This repository contains the `DistillationTrainer` (`src/training/trainer.py`) that implements all three regimes, the config suite used to sweep them across 8 classification tasks, and the SLURM scripts used to produce the paper's results.

## Highlights

- Every point on the spectrum from plain fine-tuning to full CTCD is a YAML change, not a code change.
- The teacher is a first-class trainable model, not a fixed label source — `create_optimizer()` adds its parameters to the optimizer whenever it's unfrozen.
- Reproduces the paper's full experiment pipeline: hyperparameter grid search → loss-coefficient search → temperature scaling → multi-seed replication.
- Covers 7 GLUE tasks (CoLA, MNLI, MRPC, QNLI, QQP, RTE, SST-2) plus CLINC150 intent classification, DistilBERT student / BERT-base teacher.
- `tests/unit/test_trainer.py` verifies the mechanism directly, e.g. that CTCD loss exceeds pure co-training loss, and that co-distillation gradients don't leak across the student/teacher boundary.



## Quick Start

```bash
conda env create -f environment.yml
conda activate forge-ctcd
source scripts/env/init.sh

cp .env.example .env   # fill in WANDB_API_KEY, HF_TOKEN, etc.
python test_setup.py   # verify the environment
```

`init.sh` installs conda activate/deactivate hooks (`PATH`, `LD_LIBRARY_PATH`, `HF_HOME`) and Python dependencies from `requirements.txt` (a lockfile generated via `uv` from `pyproject.toml` — don't hand-edit it). Run `init.sh` once per environment creation.

Then launch a single experiment:

```bash
python forge.py --config configs/CTCD_Experiments/sst2/05_ctcd.yaml
```



## Running CTCD Experiments



### Ablation ladder

Every task in `configs/CTCD_Experiments/{task}/` defines the same six-point spectrum from SFT baseline to full CTCD:


| Config                   | Description                                                       |
| ------------------------ | ----------------------------------------------------------------- |
| `00_sft.yaml`            | Student fine-tuned alone, no teacher                              |
| `01_kd_frozen_base.yaml` | Frozen, pretrained-only teacher (traditional KD)                  |
| `02_kd_frozen_sft.yaml`  | Frozen, task-fine-tuned teacher                                   |
| `03_cotrain_base.yaml`   | Co-training with a pretrained-only teacher (`beta_h=1, beta_s=0`) |
| `04_cotrain_sft.yaml`    | Co-training with a task-fine-tuned teacher                        |
| `05_ctcd.yaml`           | Full CTCD: co-training + co-distillation (`beta_h=1, beta_s=1`)   |


Supported tasks: `cola`, `mnli`, `mrpc`, `qnli`, `qqp`, `rte`, `sst2` (GLUE) and `clinc150`. **Coverage is not fully symmetric** — `clinc150` only has ablations `00`, `01`, `03`, `05` (no fine-tuned-teacher variants), and `sst2` has no `best_seed/` replication configs.

Useful `forge.py` flags: `--run_name` (custom output dir), `--max_steps` (smoke test), `--debug` (routes W&B to a `Forge-Test` project instead of `CTCD_Experiments`), `--force` (overwrite an existing named run), `--resume <output_dir>`.

### Paper experiment pipeline

The full study is a sequence of SLURM sweeps in `scripts/slurm/`, each a 4-way job array with checkpoint/resubmit on SLURM's `SIGUSR1` signal:

1. **Part 1 — hyperparameter grid** (`run_ctcd_part1_sweep.sh`): all 6 tasks × 6 ablations × learning rate × batch size × gradient-clip mode.
2. **Part 2 — loss-coefficient search** (`run_ctcd_part2_search.sh`): sweeps `alpha`/`beta` presets over each task's Part-1 winner, constrained per ablation (frozen-teacher ablations only sweep `alpha`; CTCD sweeps both). `run_ctcd_part2_best_weights.sh` retrains the winning cells.
3. **Temperature scaling** (`run_ctcd_temp_scale.sh`): sweeps softmax temperature (2/3/4/5) on frozen-teacher best configs.
4. **Seed replication**: `run_ctcd_best_seed_sweep.sh` replays each `best_seed/*.yaml` using the seed baked into that file; `run_ctcd_best_seeds_sweep.sh` instead takes each `best/*.yaml` (Part-1 winner) and sweeps a configurable list of seeds across it. These are deliberately different sweeps — don't confuse the two.

```bash
mkdir -p logs
sbatch scripts/slurm/run_ctcd_part1_sweep.sh
```

See `[configs/CTCD_Experiments/CTCD_PART1_TEAM_RUNBOOK.md](configs/CTCD_Experiments/CTCD_PART1_TEAM_RUNBOOK.md)` for a full walkthrough (branch setup, environment refresh, smoke test, splitting a sweep across collaborators).

## Reproducing Paper Results

- `configs/CTCD_Experiments/{task}/best/` — the winning hyperparameter cell per task/ablation from the Part 1 grid search.
- `configs/CTCD_Experiments/{task}/best_seed/` — those winning configs replayed across multiple random seeds, for the variance numbers reported in the paper.
- `configs/CTCD_Experiments/_shared.yaml` — defaults common to every task (model pair, training cadence, $\alpha_h=1.0, \alpha_s=1.0$).



## Repository Layout

```text
forge.py               entry point: YAML config -> training run
configs/                base task configs + CTCD ablations, best/best_seed configs, runbooks
src/training/           DistillationTrainer (core CTCD mechanism), loss strategies
scripts/slurm/          SLURM sweep scripts for the paper's experiment pipeline
tests/                  pytest suite, including CTCD-specific correctness tests
```



## Testing

```bash
pytest tests/unit/test_trainer.py
```

This is the correctness reference for the CTCD mechanism itself — dedicated test classes cover the co-training teacher CE path (`TestCoTrainingTeacherCE`), that CTCD loss exceeds pure co-training loss (`test_ctcd_loss_larger_than_cotraining_only`), and that gradients from co-distillation don't leak across the student/teacher boundary incorrectly (`test_codistillation_student_kl_does_not_backprop_into_teacher`).

## Citation

```bibtex
@inproceedings{trivedi2026ctcd,
  title     = {Beyond Frozen Teachers: Co-Training and Co-Distillation for Task-Specific Fine-Tuning},
  author    = {Trivedi, Nikhil and Ni, Karl and Ravi, Karthik and Sehral, Mohammed Ali},
  booktitle = {[PASTE VENUE HERE]},
  year      = {2026},
  note      = {[PASTE DOI/LINK HERE]}
}
```



## License

MIT License — see [LICENSE](LICENSE).