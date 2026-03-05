# Tests

## Current Tests

### SLURM Integration (`tests/slurm/`)

Shell-based tests for the SIGUSR1 signal handling and auto-resubmit chain. These verify the SLURM plumbing without needing GPU or real training.

```bash
# Unit-style: trap survival, Python signal handler, .incomplete_jobs parsing
bash tests/slurm/test_sigusr1_resubmit.sh

# End-to-end: 3-iteration resubmit chain (works on OOD sessions too)
bash tests/slurm/test_slurm_resubmit.sh

# Or submit to SLURM directly
sbatch tests/slurm/test_slurm_resubmit.sh
```

## Running Python Tests

pytest is configured in `pyproject.toml`. Once Python tests exist:

```bash
pytest                          # all tests
pytest tests/unit/              # unit tests only
pytest tests/integration/       # integration tests only
pytest -m "not gpu"             # skip GPU-dependent tests
pytest -m "not slow"            # skip slow tests
```

## TDD Roadmap

### Tier 1: Pure Logic (no model/data deps)

These are fast, isolated, and should be written first.

| Module | What to test |
|--------|-------------|
| `src/utils/config.py` | YAML loading, `inherit:` chain resolution, `_deep_merge()` |
| `src/training/losses.py` | Each loss strategy with synthetic logits/hidden states |
| `src/training/layer_mapping.py` | Uniform, LastN, FirstLast mapping index computation |
| `src/training/projections.py` | Identity, Linear, MLP forward pass shapes |
| `src/utils/environment.py` | `detect_environment()`, `get_device()` |
| `src/tasks/registry.py` | Task lookup, `is_translation_task()`, `is_ocr_task()` |

### Tier 2: Mocked Dependencies

Need lightweight mocks for HuggingFace models/datasets.

| Module | What to test |
|--------|-------------|
| `src/data/datasets.py` | Each loader returns expected column names and types |
| `src/models/loader.py` | Model + tokenizer loading, LoRA application |
| `src/utils/checkpoint.py` | `TrainingState` save/load round-trip |
| `src/training/callbacks.py` | `SLURMSignalCallback` writes `.incomplete_jobs` on signal |

### Tier 3: Integration

Slower tests that exercise real components together.

| Scope | What to test |
|-------|-------------|
| Single-step training | Load config + models + data, run 1 training step, verify loss decreases |
| Evaluation pipeline | `run_evaluation()` with a tiny model on a few samples |
| End-to-end | Config YAML in, trained checkpoint out (mark `@pytest.mark.slow`) |

## Conventions

- **Markers**: Use `@pytest.mark.gpu` for GPU tests, `@pytest.mark.slow` for tests > 30s
- **Fixtures**: Shared fixtures go in `tests/conftest.py`
- **Naming**: `test_{module}_{behavior}.py` (e.g., `test_losses_kl_divergence.py`)
- **Structure**: Mirror `src/` layout under `tests/unit/` and `tests/integration/`

```
tests/
  conftest.py                      # shared fixtures
  slurm/                           # shell-based SLURM tests
    test_sigusr1_resubmit.sh
    test_slurm_resubmit.sh
  unit/                            # fast, isolated
    test_config.py
    test_losses.py
    test_layer_mapping.py
    ...
  integration/                     # slower, real components
    test_training_step.py
    test_evaluation.py
    ...
```
