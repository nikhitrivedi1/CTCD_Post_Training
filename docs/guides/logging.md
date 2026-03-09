# Logging Best Practices

How logging works in Forge and how to get the most out of it.

---

## Architecture

Forge uses a dual-channel logging system:

```
Python logger (logging.info/warning/error)
    └── StreamHandler -> stdout -> TeeWriter -> output.log + terminal

Print / tqdm / HF Trainer
    └── stdout/stderr -> TeeWriter -> output.log + terminal

W&B
    └── wandb.log() -> W&B dashboard (parallel, not file-dependent)
```

**TeeWriter** (`utils/logging.py`) intercepts stdout and stderr, writing to both the terminal and `output.log`. It collapses `\r` carriage returns so tqdm progress bars produce clean single lines in the log file instead of thousands of overwrite lines.

### Log Levels

| Level | When to use | Goes to output.log? |
|-------|-------------|---------------------|
| `logger.debug()` | Verbose diagnostics (only with `--debug`) | Only with `--debug` |
| `logger.info()` | Normal operational messages | Yes |
| `logger.warning()` | Recoverable problems | Yes |
| `logger.error()` | Failures that need attention | Yes |

Default level is INFO. Use `--debug` flag to enable DEBUG level, which also routes to the `Test` W&B project.

---

## What Gets Logged Where

### output.log (file)

Everything from the Python logger plus stdout/stderr, including:

- Config dump at startup
- Model/tokenizer loading messages
- Training progress (via `StepMetricsCallback`)
- Evaluation results
- Signal handling events
- Errors and warnings

### W&B (dashboard)

All metrics from `trainer.log()`:

- Step-level training loss, learning rate, grad norm
- Evaluation metrics (accuracy, F1, MCC, BLEU, etc.)
- GRPO reward values (per-function breakdown)
- Peak metric summaries
- Pre/post training comparisons

### Both

`StepMetricsCallback` bridges the gap by logging key metrics to the Python logger (and therefore output.log) in a grep-friendly format whenever HF Trainer emits a log event.

---

## StepMetricsCallback Output Format

The callback produces clean, structured log lines:

**SFT/Distillation:**
```
step=100  epoch=0.52  loss=1.234567  learning_rate=0.000010  grad_norm=0.456789
```

**GRPO (with reward breakdown):**
```
step=10  epoch=0.07  reward=0.654321  reward_std=0.123456  kl=0.001234  learning_rate=0.000001
  rewards: edit_similarity/mean=0.789012  format/mean=0.654321  table_cell_f1/mean=0.543210
```

This format is designed for easy grepping:

```bash
# Latest reward values
grep "reward=" output.log | tail -5

# Track a specific reward function
grep "edit_similarity" output.log

# Loss trend
grep "step=" output.log | grep "loss="

# Learning rate schedule
grep "learning_rate=" output.log
```

---

## Monitoring Running Jobs

### From the log file (no W&B needed)

```bash
# How many logging steps completed
grep "step=" output.log | wc -l

# Latest metrics
grep "step=" output.log | tail -1

# Reward trend (GRPO)
grep "reward=" output.log | tail -10

# Check for warnings or errors
grep -E "WARNING|ERROR" output.log

# Check for signal handling
grep "signal" output.log
```

### From SLURM (cluster)

```bash
# Real-time tail of the log
tail -20 logs/JOBID.out

# Step count
grep "step=" logs/JOBID.out | wc -l

# Latest reward
grep "reward=" logs/JOBID.out | tail -1
```

---

## Best Practices for Adding Logging

### 1. Use the Python logger, not print

```python
# Good
logger.info(f"Loaded {len(dataset)} examples")

# Bad
print(f"Loaded {len(dataset)} examples")
```

`print()` still reaches output.log via TeeWriter, but logger lines include timestamps and source info, making them much more useful for debugging.

### 2. Log at boundaries, not inside loops

```python
# Good - log once per phase
logger.info(f"Starting reward scoring ({len(completions)} completions)")
rewards = score_all(completions)
logger.info(f"Reward scoring complete: mean={np.mean(rewards):.4f}")

# Bad - log inside the loop
for c in completions:
    logger.info(f"Scoring completion: {c[:50]}...")  # thousands of lines
    score(c)
```

### 3. Use structured key=value format for metrics

```python
# Good - grep-friendly
logger.info(f"step={step}  loss={loss:.6f}  lr={lr:.8f}")

# Bad - human-readable but hard to parse
logger.info(f"At step {step}, the loss is {loss} with learning rate {lr}")
```

### 4. Log timing for expensive operations

```python
import time

start = time.time()
results = expensive_operation()
elapsed = time.time() - start
logger.info(f"Operation complete ({elapsed:.1f}s)")
```

### 5. Use warning/error for actionable problems

```python
# Warning: something unexpected but recoverable
logger.warning(f"Image {path} missing, skipping")

# Error: something failed that needs attention
logger.error(f"Reward function {name} raised {e}, returning 0.0")
```

### 6. Never silently swallow exceptions

```python
# Bad - hides bugs (cost us 9 hours of GPU time once)
try:
    result = compute_reward(...)
except:
    pass

# Good - log the failure so it's visible
try:
    result = compute_reward(...)
except Exception as e:
    logger.debug(f"Reward computation failed: {e}")
    result = 0.0
```

### 7. Log config/setup info at startup

When adding a new component, log its configuration once at initialization:

```python
logger.info(f"ParallelRewardGroup: {len(funcs)} functions, pool_size={pool_size}")
```

This makes it easy to verify what's running without reading the config file.

### 8. Don't log secrets or large data

```python
# Bad
logger.info(f"API key: {api_key}")
logger.info(f"Full completion: {completion}")  # could be 1000+ tokens

# Good
logger.info(f"Completion length: {len(completion)} tokens")
```

---

## Configuration Reference

### Logging frequency

```yaml
logging:
  logging_steps: 10     # Log metrics every N optimizer steps
  report_to: wandb      # 'wandb', 'tensorboard', or 'none'
```

### Debug mode

```bash
python forge.py --config ... --debug
```

Enables DEBUG level logging and routes W&B to the `Test` project.

### Disabling W&B

```bash
WANDB_MODE=disabled python forge.py --config ...
```

Metrics still appear in output.log via `StepMetricsCallback`.

---

## Callback Order

Callbacks fire in registration order. Forge registers them in this sequence:

| Order | Callback | Purpose |
|-------|----------|---------|
| 1 | `StepMetricsCallback` | Log metrics to output.log |
| 2 | `SLURMSignalCallback` | Handle SIGUSR1/SIGTERM for graceful stop |
| 3 | `TeacherEvalCallback` | Evaluate teacher at each eval step (KD only) |
| 4 | `PeakMetricsCallback` | Track best metric values |
| 5 | `TranslationMetricsCallback` | BLEU/COMET at epoch end (translation only) |
| 6 | `CheckpointMarkerCallback` | Mark checkpoint as complete (always last) |

`StepMetricsCallback` fires first so metrics are logged before any other callback processing.
