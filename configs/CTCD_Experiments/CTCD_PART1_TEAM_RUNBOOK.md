# CTCD Part 1 — team runbook (Slurm)

Branch: **`Codistillation`**. Assumes the **`forge`** conda environment already exists and **conda is on your PATH** (no `module load anaconda`).

---

## Step 1 — Switch to the branch

```bash
cd /path/to/Forge
git switch Codistillation
git pull
```

**Branch checkout:** Use **`git switch Codistillation`** (Git 2.23+) to move to that branch, then **`git pull`** to update from its upstream. If Git reports that the branch does not exist locally yet, create it tracking the remote with **`git switch -c Codistillation origin/Codistillation`** (check **`git branch -r`** lists `origin/Codistillation`—it will after a normal clone that included that branch, or ask the person who pushed it to confirm the remote name).

---

## Step 2 — Activate conda

```bash
conda activate forge-ctcd
```

If `conda` is not defined in a fresh shell (e.g. some `srun` sessions), run your site’s usual conda initialization (e.g. `eval "$(conda shell.bash hook)"`) once before `conda activate`.

---

## Step 3 — Refresh dependencies (`source scripts/env/init.sh` **or** `pip install …`)

After pulling, align installed packages with this branch’s **`requirements.txt`** (and thus **`pyproject.toml`**).

**Why this matters**

- The lockfile pins versions for a reproducible env. If someone else’s `forge` was built from an older commit, versions can drift.
- **ACE** is imported as `from calibration_toolbox import ACE` (`src/evaluation/metrics.py`). That package is installed as **`calibration-toolbox`** from a **pinned Git URL** in `requirements.txt` (not on PyPI). A fresh or updated install needs **network access to GitHub** during `pip install`.

**Option A — full env hook + lockfile (recommended if you use Forge’s HPC hooks)**

```bash
source scripts/env/init.sh
```

This installs from `requirements.txt` (including `calibration-toolbox`), may install Playwright Chromium, and installs conda activate/deactivate hooks that mirror `scripts/env/activate.sh` (e.g. `HF_HOME` on `/scratch`, `LD_PRELOAD` for libstdc++).

**Option B — dependencies only (lighter)**

```bash
pip install --no-deps -r requirements.txt
```

Use this when you only need package pins updated and already manage `HF_HOME` / `.env` yourself.

**Quick check**

```bash
python -c "from calibration_toolbox import ACE; print('ACE OK')"
```

---

## Step 4 — Smoke test on a GPU (`srun` + `forge.py`, MNLI ablation `05`)

Ablation **05** is **`05_ctcd.yaml`** (full CTCD). Config:

`configs/CTCD_Experiments/mnli/05_ctcd.yaml`

**Interactive shell on a GPU node** (adjust `--partition`, `--gres`, time, and memory for your cluster):

```bash
cd /path/to/Forge
srun --partition=gpu --gres=gpu:1 --cpus-per-task=16 --mem=64G --time=01:00:00 --pty bash
```

**On the allocated node**, activate the env, optionally load Forge’s shell helpers, then run a **10-step** train. Use **`--debug`**: in `forge.py`, debug mode sets **`WANDB_PROJECT` to `Forge-Test`**, so runs do **not** use the production **`CTCD_Experiments`** project from `configs/CTCD_Experiments/_shared.yaml`.

```bash
cd /path/to/Forge
conda activate forge-ctcd
source scripts/env/activate.sh

python forge.py \
  --config configs/CTCD_Experiments/mnli/05_ctcd.yaml \
  --run_name smoke_mnli_ctcd_p1 \
  --max_steps 10 \
  --debug \
  --force
```

Notes:

- Use **`--max_steps 10`** (space-separated; do not put spaces around `=`).
- **`--force` (named runs only):** If you pass **`--run_name`** and **`outputs/<run_name>/` already exists**, Forge would normally exit with an error (to avoid clobbering a run). **`--force` deletes that directory** (`shutil.rmtree`) and starts clean. If the previous run stored a W&B run id in metadata, Forge **attempts to delete that W&B run** as well. **`--force` does nothing special for auto-named runs** (paths include a timestamp, so they are already unique); Forge logs a warning if you use `--force` without `--run_name`.
- To **avoid Weights & Biases entirely** for the smoke test, run with **`WANDB_MODE=disabled`** (Forge treats this as `report_to: none` and skips W&B init).

---

## Step 5 — Split the task list (**required**)

**Everyone must set `TASKS` before submitting the sweep** so two people do not both run the full six-task grid. Edit **`scripts/slurm/run_ctcd_part1_sweep.sh`** around the **`TASKS=(...)`** line (currently ~line 50).

| Who     | `TASKS` assignment        |
|---------|---------------------------|
| Karthik | `TASKS=(mnli rte)`        |
| Akshay  | `TASKS=(qnli mrpc)`       |

**No other edits are required for the split:** the script sets `NUM_TASKS=${#TASKS[@]}` and uses fixed **`NUM_ABLATIONS=6`** so `TOTAL` and `decode_tid` stay consistent when you only change `TASKS`.

**Coverage:** Karthik (**mnli + rte**) and Akshay (**qnli + mrpc**) cover four tasks; **qqp** and **cola** still need an owner (third checkout with `TASKS=(qqp cola)` or a dedicated full-grid runner—coordinate so the union of all `TASKS` assignments covers every task you need without duplicate GPU spend).

If two people run different `TASKS` at the same time, use **separate repo checkouts** or different **`CTCD_COMPLETION_DIR`** so `.done` markers and `outputs/` paths do not fight over the same run names.

Coordinate **`#SBATCH` resources** (`partition`, `gres`, time, memory) with your queue policy if defaults are wrong for your account.

---

## Step 6 — Launch the full Part 1 sweep

Confirm **Step 5** (`TASKS=...`) matches **your** assignment before submitting.

From the **repo root** (so `SLURM_SUBMIT_DIR` is correct):

```bash
mkdir -p logs
sbatch scripts/slurm/run_ctcd_part1_sweep.sh
```

- Submits a **4-way job array**; each rank walks its slice of the grid until done.
- **Time limits:** SLURM sends **SIGUSR1** before the hard kill; training saves a checkpoint, the sweep can **resubmit** and **skip** finished work using `.done` files under `completed_runs/ctcd_part1/` (or `CTCD_COMPLETION_DIR`).
- **Disk / artifacts:** each run merges overrides that set **`checkpointing.no_weights: true`**. After post-training eval, Forge **removes model weight files** from the run directory to save **scratch/disk** space while keeping metadata such as `training_state.json` and logs (see `forge.py` — `no_weights` cleanup). This is not GPU RAM management; it reduces stored checkpoint size on disk.

Optional environment variables forwarded to training: `SEED`, `MAX_STEPS`, `EPOCHS`, `DEBUG`, `FORCE`, `TIMESTAMP`. Optional sweep env: `CTCD_COMPLETION_DIR`.
