# Codebase TODO

Items identified during code review that need attention.

**Last reviewed:** 2026-02-03

---

## Fixed (2026-02-03)

### ~~1. Silent W&B Error Swallowing~~ FIXED
**File:** [forge.py:770-773](forge.py#L770-L773)

Now catches `ImportError` separately and logs other exceptions with `logger.debug()`.

---

### ~~2. Device Placement Bug~~ FIXED
**File:** [forge.py:283-284](forge.py#L283-L284)

Changed from `.cuda()` to `.to(device)` for cross-platform support (CUDA, MPS, CPU).

---

### ~~3. Broad Exception Handling in Callbacks~~ FIXED
**File:** [src/training/callbacks.py:126](src/training/callbacks.py#L126), [line 203](src/training/callbacks.py#L203)

Now catches `ImportError` for metrics loading and `(ValueError, RuntimeError)` for computation.

---

### ~~4. Layer Mapping Factory Missing Validation~~ FIXED
**File:** [forge.py:264-268](forge.py#L264-L268)

Added `mapper.validate()` call before `mapper.map()` with descriptive error message.

---

### ~~6. Config Inheritance Without Depth Limit~~ FIXED
**File:** [src/utils/config.py](src/utils/config.py)

Added `MAX_INHERITANCE_DEPTH = 10` constant and depth tracking to prevent circular inheritance.

---

### ~~7. Specific Exception Types Needed~~ FIXED
**File:** [src/evaluation/metrics.py:100](src/evaluation/metrics.py#L100)

Now catches `(ImportError, FileNotFoundError)` instead of broad `Exception`.

---

## Remaining Items

### 5. Projector Device Handling Incomplete
**Files:** [forge.py](forge.py), [src/training/losses.py](src/training/losses.py)

Projector created on GPU but device handling during eval or with different device configs is inconsistent. FeatureLossStrategy.compute() assumes projector and tensors are on same device.

**Fix:** Ensure projector follows model device in all scenarios.

---

### 8. Magic Number: W&B Run ID Truncation
**File:** [forge.py](forge.py)

64-character limit for W&B run names not documented. Add constant with comment explaining HuggingFace's limitation.

---

### 9. Model Architecture Validation Missing
**File:** [src/models/loader.py](src/models/loader.py)

No early validation for incompatible teacher-student pairs. Mismatches only detected at forward pass time.

**Fix:** Add validation in `load_teacher_student()` for obvious incompatibilities.

---

### 10. Translation Metrics Callback Efficiency
**File:** [src/training/callbacks.py](src/training/callbacks.py)

Generates full translations at every eval step, which is expensive with large models.

**Fix:** Consider periodic evaluation or sampling subset during training.

---

## Testing Gaps

| Area | Status | Notes |
|------|--------|-------|
| Integration tests (full training pipeline) | Missing | Would catch device/config issues early |
| Config inheritance tests | Missing | Verify circular reference handling |
| Dataset loader tests | Missing | Test MNLI/QQP/Europarl loading |
| Trainer unit tests | Missing | Test `compute_loss`, resume logic |
| Layer mapping validation | ~~Partial~~ Fixed | Now validated at use time in forge.py |

---

## Documentation Gaps

| Topic | Priority | Description |
|-------|----------|-------------|
| Troubleshooting guide | High | Common errors (OOM, NaN loss, slow training) |
| Quickstart notebook | Medium | 5-minute start with example |
| Loss strategy selection | Medium | Decision tree for KL vs MSE vs Feature vs Attention |
| TrainingState.json format | Low | Document schema for SLURM resumption |
| .incomplete_jobs registry | Low | Document format and cleanup |

---

## Code Quality Metrics

| Metric | Value |
|--------|-------|
| Python source lines | ~2,600 |
| Test lines | 585 |
| Config files | 54 |
| Doc files | 47 |
| Loss strategies | 8 |
| ValueError raises | 13 (added layer mapping validation) |
| Exception handling gaps | ~~3 critical, 2 medium~~ 0 critical, 2 medium |
| Pre-commit hooks | 8 |

---

## Summary

**Fixed in this review:**
- W&B error handling now logs failures
- Device placement works on MPS/CPU/CUDA
- Exception handling uses specific types
- Layer mapping validates before use
- Config inheritance has depth limit

**Still needs attention:**
- Projector device handling edge cases
- Test coverage for integration scenarios
- Troubleshooting documentation for users
