# DRY Assessment - Forge Codebase

*2026-03-08*

Thorough scan of the codebase identifying repeated patterns, duplicated logic, and opportunities to make the framework more modular and lean. Violations are numbered for reference. The summary table at the end orders them by impact.

---

## 1. GLUE Dataset Loaders Are Copy-Paste Clones

**Files:** `src/data/datasets.py:9-149`
**Severity:** High

`load_mnli`, `load_qqp`, `load_sst2`, `load_cola` all follow the exact same structure:

```python
def load_<task>(tokenizer, max_length=128):
    dataset = load_dataset("glue", "<subset>")
    def preprocess(examples):
        return tokenizer(examples["col1"], examples["col2"], ...)  # only diff
    tokenized = dataset.map(preprocess, batched=True, remove_columns=[...])
    tokenized = tokenized.rename_column("label", "labels")
    train_dataset = tokenized["train"]
    eval_dataset = tokenized["<validation_split>"]  # only diff
    return train_dataset, eval_dataset, tokenizer
```

The only differences are:
- Text column names (`premise`/`hypothesis` vs `question1`/`question2` vs `sentence`)
- Eval split name (`validation_matched` for MNLI, `validation` for others)
- QQP's -1 label filtering

**Fix:** Single `load_glue_task()` function with a config dict per task specifying columns, eval split, and optional filters. The TASKS registry already has `dataset` and `subset` fields that could drive this. This also eliminates `load_dataset_for_task()`'s if/elif dispatch chain (`datasets.py:478-495`), which becomes a natural registry lookup.

---

## 2. Europarl Constants Duplicated in 3 Files

**Files:**
- `src/data/datasets.py:194-207` (lang_names, system_message, split logic)
- `src/data/rl_datasets.py:41-59` (lang_names, system_message, split logic)
- `src/training/callbacks.py:271-273, 333-340` (split logic, lang_names, system_message)

**Severity:** High

Three things are copy-pasted across these files:

**a) Language name mapping dict** - `lang_names` dict in all three locations, but they've **diverged silently**: datasets.py has 6 languages, rl_datasets.py has 16, callbacks.py has 8. None match.

**b) Translation system prompt** - the exact same string:
> "You are a professional translator. Translate the provided text from {source} to {target}..."

**c) 70/20/10 split logic** - `train_test_split(test_size=0.3, seed=42)` then `.train_test_split(test_size=0.333, seed=42)` repeated in all three files.

**Fix:** Create `src/data/translation_utils.py` with:
- `LANG_NAMES` constant dict
- `translation_system_prompt(source_lang, target_lang)` function
- `split_europarl(dataset, max_samples)` returning (train, val, test)

---

## 3. OCR Dataset Loading Logic Duplicated

**Files:**
- `src/data/datasets.py:332-349` (in `load_ocr`)
- `src/data/rl_datasets.py:102-116` (in `load_ocr_rl`)

**Severity:** Medium

The multi-subset directory loading pattern is nearly identical:

```python
if (data_path / "train").exists():
    train = load_from_disk(str(data_path / "train"))
    eval = load_from_disk(...) if (data_path / "eval").exists() else None
else:
    for subset_dir in sorted(data_path.iterdir()):
        if not subset_dir.is_dir() or not (subset_dir / "train").exists():
            continue
        train_parts.append(load_from_disk(...))
        ...
    train = concatenate_datasets(train_parts)
```

**Fix:** Extract to `src/data/ocr_utils.py`:
```python
def load_ocr_splits(data_dir: str) -> tuple[Dataset, Dataset | None]:
```

---

## 4. OCR System Prompt Defined Twice

**Files:**
- `src/data/ocr_utils.py:10-14` - `OCR_SYSTEM_PROMPT` constant (canonical)
- `src/data/datasets.py:360-364` - hardcoded duplicate string in `load_ocr`

**Severity:** Medium

`load_ocr` defines its own `system_prompt` variable with identical text instead of importing `OCR_SYSTEM_PROMPT` from `ocr_utils.py`. The RL loaders correctly import from `ocr_utils.py`.

**Fix:** Replace the local string in `load_ocr` with `from src.data.ocr_utils import OCR_SYSTEM_PROMPT`.

---

## 5. Feature Loss Strategies Share 80% of Their Logic

**Files:** `src/training/losses.py:402-601`

**Severity:** Medium

`FeatureLossStrategy.compute()` and `CosineFeatureLoss.compute()` are nearly identical. Both:
1. Extract layer_mapping and projector from kwargs
2. Validate hidden_states exist
3. Loop over layer pairs: extract, project, align dtype
4. Aggregate losses and build LossOutput

The only difference is the per-layer loss function: `F.mse_loss(s, t)` vs `(1 - F.cosine_similarity(s, t)).mean()`.

**Fix:** Create a base `_FeatureDistillationBase` that implements the shared loop and calls an abstract `_layer_loss(s_hidden, t_hidden)` method. Each subclass implements only the 1-line loss computation.

---

## 6. Completion Unwrapping Has One Remaining Inline Copy

**Files:** `src/training/rewards.py`

**Severity:** Low

A `_unwrap_completion()` helper already exists (lines 23-27) and is used via `_score_parallel()` (line 41), but `unit_test_reward()` (lines 121-122) still has an inline copy of the pattern. One cleanup remains.

**Fix:** Replace the inline unwrapping in `unit_test_reward()` with a call to `_unwrap_completion()`.

---

## 7. Edit Similarity Calculation Duplicated 4 Times

**Files:** `src/training/rewards.py` - lines 78-80, 309-311, 416-418, 493-495

**Severity:** Medium

The pattern `1.0 - (Levenshtein.distance(a, b) / max(len(a), len(b)))` is repeated in:
- `ocr_reward`
- `_compute_table_similarity`
- `_match_block_order`
- `_latex_similarity`

**Fix:** Extract to a shared helper:
```python
def _edit_similarity(a: str, b: str) -> float:
    from rapidfuzz.distance import Levenshtein
    dist = Levenshtein.distance(a, b)
    max_len = max(len(a), len(b))
    return 1.0 - (dist / max_len) if max_len > 0 else 1.0
```

---

## 8. Model Loading Dispatch Repeated 3 Times

**Files:** `src/models/loader.py`
- `load_models()` lines 238-256 (student + teacher)
- `load_models_from_checkpoint()` lines 301-306 (student), lines 320-325 (teacher)

**Severity:** Medium

The `if ocr: _load_vlm(...) elif translation: _load_causal_lm(...) else: _load_classifier(...)` dispatch appears three times.

**Fix:** Extract a unified dispatcher:
```python
def _load_model(task_type: str, name: str, init: str, device: str, **kwargs):
    if task_type == "ocr":
        return _load_vlm(name, init, device, **kwargs)
    elif task_type == "translation":
        return _load_causal_lm(name, init, device, **kwargs)
    else:
        return _load_classifier(name, init, kwargs["num_labels"], device)
```

---

## 9. W&B Logging Boilerplate

**Files:**
- `src/evaluation/evaluate.py:141-147` (`log_pre_post_metrics`)
- `src/evaluation/evaluate.py:285-289` (`log_transfer_metrics`)
- `src/training/callbacks.py:423-431` (`PeakMetricsCallback`)
- `src/training/callbacks.py:298-308` (`TranslationMetricsCallback`)

**Severity:** Low

The try/except wandb import + `if wandb.run` guard is repeated in 4 places.

**Fix:** A small helper in `src/utils/`:
```python
def log_to_wandb(metrics: dict, commit: bool = False):
    try:
        import wandb
        if wandb.run:
            wandb.log(metrics, commit=commit)
    except Exception:
        pass
```

---

## 10. `max_train_samples` Truncation Pattern

**Files:**
- `src/data/datasets.py:351-352`
- `src/data/rl_datasets.py:118-119`
- `src/data/rl_datasets.py:242-243`
- `src/data/rl_datasets.py:331-332`

**Severity:** Low

Same 2-line pattern repeated 4 times:
```python
if max_train_samples and len(train_dataset) > max_train_samples:
    train_dataset = train_dataset.select(range(max_train_samples))
```

**Fix:** Tiny helper, but this is borderline. Could be left as-is since it's only 2 lines. Best addressed naturally if the OCR loading logic (#3) is consolidated.

---

## 11. `FinetuningTrainer` Is an Empty Pass-Through

**Files:** `src/training/trainer.py:17-26`

**Severity:** Low

```python
class FinetuningTrainer(Trainer):
    pass
```

This adds an unnecessary indirection layer. `DistillationTrainer` could extend `Trainer` directly.

**Fix:** Either give `FinetuningTrainer` real functionality (LoRA setup, custom eval hooks) or remove it and have `DistillationTrainer` inherit from `Trainer` directly.

---

## 13. Benchmark Scripts Are 70% Identical

**Files:**
- `scripts/eval/olmocr_bench.py` (880 lines)
- `scripts/eval/omnidoc_bench.py` (897 lines)

**Severity:** Critical

These two scripts share roughly 600 lines of nearly identical infrastructure. The scoring logic differs (olmocr uses unit tests, omnidoc uses edit distance across 4 dimensions), but everything else is copy-paste:

| Section | Identical? |
|---------|-----------|
| Imports, logging setup, constants | ~95% |
| Shutdown flag + SIGTERM handler | 100% |
| `_register_bench_job()` / `_remove_bench_job()` | 100% (only `"bench"` vs `"omnidocbench"` type string) |
| `_model_name_from_args()` | 100% |
| `derive_output_dir()` | 100% (only `DEFAULT_OUTPUT_BASE` differs) |
| `main()` argument parsing | ~95% |
| Resume/metadata detection (30 lines) | ~95% |
| Sharding logic | ~90% |
| Generation loop (temp files, progress bar, `first_generate` trick) | ~85% |
| SLURM job tracking in metadata | 100% |
| CPU worker count for scoring | 100% |

Additionally, `omnidoc_bench.py` re-implements several helpers that already exist in `src/training/rewards.py`:
- `_normalized_edit_distance()` = same as `_latex_similarity()` pattern in rewards.py (#7)
- `_extract_markdown_tables()` = identical to `rewards.py:_extract_markdown_tables()`
- `_extract_latex()` / `_clean_latex()` = identical to `rewards.py:_extract_latex()` / `_clean_latex()`
- `_extract_text_blocks()` = identical to `rewards.py:_extract_text_blocks()`
- `score_tables()` cell-matching loop = identical to `rewards.py:_compute_table_similarity()`
- `score_reading_order()` LCS + block matching = identical to `rewards.py:reading_order_reward()` internals

**Fix:** Two-part refactor:
1. Extract shared bench infrastructure into `scripts/eval/bench_base.py`: SIGTERM handling, job registry wrappers, model name derivation, output dir logic, arg parsing, generation loop, metadata management.
2. Move the markdown analysis helpers (`_extract_markdown_tables`, `_extract_latex`, `_clean_latex`, `_extract_text_blocks`, edit similarity) into a shared module (e.g. `src/evaluation/markdown_utils.py`) so both bench scripts and reward functions import from one place.

---

## 14. Auto-Submit Scripts Are Copy-Paste Variants

**Files:**
- `auto-submit/submit.sh` (78 lines)
- `auto-submit/submit-local.sh` (109 lines)
- `auto-submit/submit-bench.sh` (89 lines)

**Severity:** Medium

All three follow the exact same structure:

```bash
get_open_jobs()    # squeue | wc -l (local vs ssh variant)
get_next_config()  # read file, skip comments/submitted
submit_job()       # sbatch + log to .submitted_*

while true; do
    open_jobs=$(get_open_jobs)
    slots=$((MAX_OPEN - open_jobs))
    while [ $slots -gt 0 ]; do
        config=$(get_next_config) || break
        submit_job "$config"
        slots=$((slots - 1))
    done
    sleep $CHECK_INTERVAL
done
```

Differences: `submit.sh` runs on-cluster, `submit-local.sh` runs via SSH, `submit-bench.sh` submits bench jobs instead of training jobs. The core polling loop and config tracking is identical.

**Fix:** Extract `auto-submit/submit_lib.sh` with `poll_and_submit()` function that accepts a `submit_fn` callback. Each script sources the lib and provides its own `submit_job()` implementation.

---

## 15. GLUE Task Config Duplicated in Workbench

**Files:**
- `workbench/utils.py:21-50` - `TASK_CONFIG` dict (text_cols, eval_split, num_labels per task)
- `workbench/finetune_head.py:72-99` - same info as if/elif chain
- `src/tasks/registry.py:6-52` - `TASKS` dict (overlapping fields)

**Severity:** Medium

The workbench has its own `TASK_CONFIG` dict with `text_cols`, `eval_split`, `num_labels` per GLUE task. `finetune_head.py` duplicates the same information as a hardcoded if/elif chain. Meanwhile, `src/tasks/registry.py` already has `num_labels` and `type` per task.

This means adding a new classification task requires updating three places.

**Fix:** Extend `src/tasks/registry.py` with `text_cols` and `eval_split` fields. Workbench imports from the registry instead of maintaining its own copy.

---

## 16. Workbench Hidden State / Attention Extraction 95% Shared

**Files:** `workbench/utils.py:128-290`

**Severity:** Medium

`extract_hidden_states()` (lines 128-206) and `extract_attention_weights()` (lines 209-290) are nearly identical:
1. Same tokenization (single vs pair texts)
2. Same `inspect.signature` check for `token_type_ids`
3. Same `TensorDataset` + `DataLoader` setup
4. Same batched inference loop with `torch.no_grad()`
5. Only differ in: `output_hidden_states=True` vs `output_attentions=True`, and which output field to collect

**Fix:** A single `_extract_representations()` that takes an `output_key` parameter (`"hidden_states"` or `"attentions"`) and corresponding model kwarg.

---

## 17. `render_pdf_page()` Duplicated

**Files:**
- `src/data/ocr_utils.py:17-38` (canonical)
- `scripts/eval/olmocr_bench.py:115-133` (copy)

**Severity:** Low

`olmocr_bench.py` has its own `render_pdf_page()` that is functionally identical to `ocr_utils.py`, just missing the `.convert("RGB")` call (which is inconsequential for pypdfium2 output).

**Fix:** Import from `src.data.ocr_utils`.

---

## 18. SLURM CPU Worker Count Resolution

**Files:**
- `scripts/eval/olmocr_bench.py:866-869`
- `scripts/eval/omnidoc_bench.py:883-887`
- `scripts/data/convert_synthmix.py` (similar pattern)
- `scripts/data/prepare_ocr_data.py` (similar pattern)

**Severity:** Low

```python
slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
available_cpus = int(slurm_cpus) if slurm_cpus else (os.cpu_count() or 1)
num_workers = max(1, available_cpus // 2)
```

Repeated in 4 scripts.

**Fix:** Add to `src/utils/environment.py`:
```python
def get_num_workers(fraction: float = 0.5) -> int:
```

---

## 19. Metrics Prediction Decoding Pipeline Duplicated

**Files:** `src/evaluation/metrics.py`
- `compute_translation_metrics()` (lines 73-100)
- `compute_ocr_metrics()` (lines 163-196)

**Severity:** Medium

Both functions repeat the same 4-step pipeline:

```python
# Step 1: Shape check + argmax (identical)
if len(logits.shape) == 3:
    predictions = np.argmax(logits, axis=-1)

# Step 2: Padding replacement (near-identical, OCR adds fallback to 0)
labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

# Step 3: Batch decode (identical)
decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

# Step 4: Strip whitespace (identical)
decoded_preds = [p.strip() for p in decoded_preds]
decoded_labels = [l.strip() for l in decoded_labels]
```

The OCR version also computes edit similarity inline (same pattern as #7).

**Fix:** Extract `_decode_predictions(logits, labels, tokenizer)` returning `(decoded_preds, decoded_labels)`. Each metric function then only contains its task-specific scoring.

---

## 20. InferenceEngine Reimplements Model Loading

**Files:**
- `src/inference/engine.py:179-299` (`_load_hf_vlm()`, `_load_hf_causal()`)
- `src/models/loader.py:69-145` (`_load_vlm()`, `_load_causal_lm()`)

**Severity:** High

The inference engine has its own VLM and causal LM loaders that duplicate ~150 lines from `loader.py`:

| Pattern | engine.py | loader.py |
|---------|-----------|-----------|
| Attention impl detection | `_detect_attn_impl()` helper | `_detect_attn_implementation()` helper |
| dtype selection (cuda→bf16, cpu→fp32) | Lines 186-187 | Lines 126-127 |
| Load kwargs assembly | Lines 198-210 | Lines 130-140 |
| Parameter count logging | Lines 225-229 | Lines 100-102 |
| Fresh init handling | Lines 215-220 | Lines 87-93 |

The engine adds LoRA checkpoint handling that loader.py lacks, but the core loading is copy-paste.

**Fix:** Have `InferenceEngine` use `loader.py` primitives internally, adding LoRA support to the shared loaders.

---

## 21. Data Scripts Share Parallel Processing Pattern

**Files:**
- `scripts/data/convert_synthmix.py:55-70, 125-134`
- `scripts/data/prepare_ocr_data.py:54-78, 387-402`

**Severity:** Medium

Both scripts implement the same pattern:

1. Top-level picklable render function (for ProcessPoolExecutor)
2. Tuple arg unpacking with try/except returning None on failure
3. ProcessPoolExecutor + as_completed loop
4. Progress tracking (manual counter vs tqdm - inconsistent)

```python
# Both scripts:
def _render_one(args: tuple) -> ... | None:
    try:
        pdf_path, page_num, output_path, ... = args
        # render logic
    except Exception:
        return None

with ProcessPoolExecutor(max_workers=workers) as pool:
    futures = {pool.submit(_render_one, job): job for job in jobs}
    for future in as_completed(futures):
        result = future.result()
        ...
```

**Fix:** Extract to `scripts/data/utils.py`:
```python
def parallel_render(jobs, render_fn, workers, desc="Processing"):
```

---

## 22. PROJECT_ROOT + sys.path Boilerplate

**Files:**
- `scripts/eval/olmocr_bench.py:42-43`
- `scripts/eval/omnidoc_bench.py:42-43`
- `scripts/data/convert_omnidocbench.py:21-22`
- `scripts/data/convert_synthmix.py:21-22`

**Severity:** Low

All four scripts compute project root and inject into sys.path identically:

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
```

**Fix:** Not worth a utility for 2 lines, but could be eliminated by making `src/` an installable package (even just `pip install -e .`), which removes the need for sys.path hacking entirely.

---

## 23. Workbench Command Handler Boilerplate

**Files:** `workbench/analyze.py:135-295`

**Severity:** Medium

Four command handlers (`cmd_probe`, `cmd_cka`, `cmd_attention_entropy`, `cmd_disagreement`) follow identical structure:

```python
def cmd_X(args):
    output_path = args.output or get_default_output(...)
    log_path = setup_logging(output_path)
    results = {}
    logger.info("=" * 60)
    logger.info("ANALYSIS NAME")

    results["teacher"] = {"model": args.teacher, "data": run_X(...)}
    if args.student:
        results["student"] = {"model": args.student, "data": run_X(...)}
        results["comparison"] = compare_X(...)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")
    print_X_summary(results)
```

~70 lines of repeated scaffolding. Each print summary function also follows the same "header, extract, iterate" template.

**Fix:** Extract a generic `run_analysis()` that takes the analysis function and summary printer as callbacks.

---

## 24. Workbench Has 3 Separate Model Loaders

**Files:**
- `workbench/utils.py:53-85` - `load_model()` (AutoModel for analysis)
- `workbench/disagreement.py:23-56` - `load_classification_model()` (AutoModelForSequenceClassification)
- `workbench/finetune_head.py:159-160` - inline model loading

**Severity:** Medium

All three do: check if path exists, load config, load model via `from_pretrained`, load tokenizer, move to device, call `.eval()`, count parameters. The only difference is which Auto class is used.

A third copy of the tokenization-then-batch-inference pattern also lives in `disagreement.py:get_predictions()` (identical to the loops in `utils.py` but uses `padding=True` instead of `padding="max_length"` - a subtle inconsistency that could cause shape issues).

**Fix:** Extend `workbench/utils.py:load_model()` with a `model_class` parameter defaulting to `AutoModel`. All workbench files import from one place.

---

## 25. Device Detection Inconsistent

**Files:**
- `src/utils/environment.py:48` - `get_device()` (canonical)
- `workbench/finetune_head.py:173` - inline with MPS fallback
- `scripts/eval/olmocr_bench.py:699` - inline
- `scripts/eval/omnidoc_bench.py:730` - inline

**Severity:** Low

`get_device()` exists but several files don't use it, opting for inline `torch.cuda.is_available()` checks. `finetune_head.py` includes an MPS fallback that the canonical function lacks, creating inconsistent behavior on Apple Silicon.

**Fix:** Update `get_device()` to include MPS fallback. Replace inline checks with imports.

---

## 26. Logging Setup Fragmented Across 3 Functions

**Files:**
- `forge.py:26-45` - `setup_logging()`
- `workbench/analyze.py:96-132` - separate `setup_logging()`
- `utils/logging.py:53-89` - `setup_file_logging()`
- 8+ scripts use independent `logging.basicConfig()` with inconsistent formats

**Severity:** Medium

Three different logging setup functions exist, each with slightly different format strings, date formats, and file handler strategies. Scripts that don't import any of them roll their own `basicConfig()` calls.

**Fix:** Consolidate into `utils/logging.py` with a single `setup_logging(log_path=None, level=INFO, console=True)` that handles both file and console output. All entry points import from there.

---

## Summary Table

| # | Issue | Files | Impact | Effort |
|---|-------|-------|--------|--------|
| 13 | Bench scripts 70% identical | olmocr_bench.py, omnidoc_bench.py | Critical | High |
| 20 | InferenceEngine reimplements loader.py | engine.py, loader.py | High | Medium |
| 1 | GLUE loaders are clones | datasets.py | High | Low |
| 2 | Europarl constants 3x (diverged) | datasets.py, rl_datasets.py, callbacks.py | High | Low |
| 19 | Metrics decoding pipeline 2x | metrics.py | Medium | Low |
| 21 | Data scripts parallel processing | convert_synthmix.py, prepare_ocr_data.py | Medium | Low |
| 23 | Workbench cmd handler boilerplate 4x | analyze.py | Medium | Low |
| 24 | Workbench model loaders 3x | utils.py, disagreement.py, finetune_head.py | Medium | Low |
| 26 | Logging setup fragmented 3x | forge.py, analyze.py, utils/logging.py | Medium | Low |
| 14 | Auto-submit scripts copy-paste | auto-submit/*.sh | Medium | Low |
| 15 | GLUE task config 3x | registry.py, workbench/utils.py, finetune_head.py | Medium | Low |
| 16 | Hidden state / attention extraction 95% | workbench/utils.py | Medium | Low |
| 3 | OCR loading logic 2x | datasets.py, rl_datasets.py | Medium | Low |
| 4 | OCR prompt defined 2x | datasets.py, ocr_utils.py | Medium | Trivial |
| 5 | Feature loss strategies 80% shared | losses.py | Medium | Medium |
| 7 | Edit similarity calc 4x + bench scripts | rewards.py, omnidoc_bench.py | Medium | Low |
| 8 | Model loading dispatch 3x | loader.py | Medium | Low |
| 9 | W&B logging boilerplate 4x | evaluate.py, callbacks.py | Low | Low |
| 25 | Device detection inconsistent 4x | environment.py, finetune_head.py, eval scripts | Low | Trivial |
| 22 | PROJECT_ROOT sys.path 4x | eval + data scripts | Low | Trivial |
| 6 | Completion unwrapping (1 inline copy left) | rewards.py | Low | Trivial |
| 10 | max_train_samples 4x | datasets.py, rl_datasets.py | Low | Trivial |
| 11 | Empty FinetuningTrainer | trainer.py | Low | Trivial |
| 17 | render_pdf_page duplicated | olmocr_bench.py, ocr_utils.py | Low | Trivial |
| 18 | SLURM worker count 4x | bench + data scripts | Low | Trivial |

**Recommended priority:**
1. **#13** is the biggest win (500+ LOC saved), but high effort. Start by extracting the shared infra.
2. **#20** is high-value - unifying InferenceEngine with loader.py removes ~150 lines and prevents the two from diverging further (LoRA support only in one, attention impl detection duplicated).
3. **#1 and #2** are the highest-value src/ changes, both low effort. #2 is urgent because the copies have already diverged.
4. **#7** now has even more impact since both omnidoc_bench.py and metrics.py (#19) also duplicate the same patterns.
5. **#4, #17, #25** are one-line import/fix changes.
6. **#26** logging consolidation prevents the fragmentation from getting worse as more scripts are added.
7. **#24** and **#16** together clean up the entire workbench model/inference layer.

---

## Existing Patterns Worth Preserving

These are things the codebase already does well. They serve as reference models for fixing the violations above.

### A. Registry + Factory Pattern

The codebase consistently uses `REGISTRY` dicts paired with `get_*()` factory functions. This is the dominant extensibility pattern and should be the template for any new pluggable component.

| Registry | File | Factory |
|----------|------|---------|
| `LOSS_REGISTRY` | `src/training/losses.py:837` | `get_loss_strategy()` |
| `REWARD_REGISTRY` | `src/training/rewards.py:573` | `get_reward_funcs()` |
| `PROJECTION_REGISTRY` | `src/training/projections.py` | `get_projection()` |
| `LAYER_MAPPING_REGISTRY` | `src/training/layer_mapping.py` | `get_layer_mapping()` |
| `TASKS` | `src/tasks/registry.py:6` | `get_num_labels()`, `get_task_type()`, etc. |
| `LABEL_MAP_REGISTRY` | `utils/labels.py` | `get_label_map()` |

When consolidating the GLUE loaders (#1), follow this same pattern: define a `GLUE_TASKS` config dict and a single `load_glue_task()` that looks up the config.

### B. Shared Utils Module (`ocr_utils.py`)

`src/data/ocr_utils.py` is the exact pattern that should be replicated for translation (#2). It extracts shared constants (`OCR_SYSTEM_PROMPT`) and helpers (`render_pdf_page`, `pdf_image_name`) into one place that both supervised and RL loaders import from. The translation equivalent would be `src/data/translation_utils.py` holding `LANG_NAMES`, the system prompt builder, and the split logic.

### C. Strategy Pattern for Losses

`DistillationLossStrategy` ABC with `compute()` + `requires_hidden_states`/`requires_attentions` flags is clean. The trainer never knows loss internals, just calls `.compute()`. This is the right abstraction. The feature loss fix (#5) should preserve this pattern by adding a base class *within* the strategy hierarchy, not by breaking the interface.

### D. TASKS Registry as Single Source of Truth

`src/tasks/registry.py` centralizes task metadata (num_labels, type, primary_metric) with thin accessor functions (`is_classification_task()`, `get_primary_metric()`). This avoids scattering task knowledge across the codebase. When unifying GLUE loaders (#1), extend this registry with the extra fields needed (text columns, eval split name, label filter) rather than creating a parallel config.

### E. `compute_kl_divergence()` as Shared Primitive

`src/training/losses.py:12-59` extracts the KL computation into a standalone function used by both `KLDistillationLoss` (response-level) and `AttentionLossStrategy` (attention-level, with `input_is_probs=True`). This is exactly how edit similarity (#7) should be handled - one function, multiple callers.

### F. `evaluate_models()` Consolidation

`src/evaluation/evaluate.py:167-228` wraps the student/teacher eval loop into a single function with shared kwargs dict, avoiding duplicated `evaluate_model()` calls. The model loading dispatch (#8) should follow the same consolidation approach.

### G. Atomic File Operations

`src/utils/checkpoint.py:mark_checkpoint_complete()` uses write-to-temp + `os.replace()` for crash-safe checkpointing. `src/utils/jobs.py:locked_incomplete_jobs()` uses `fcntl.flock` for concurrent SLURM safety. Both are good examples of correctness-first utility design.

### H. Config Inheritance via YAML

`src/utils/config.py:load_config()` implements `inherit: base.yaml` with recursive merging and circular reference detection (max depth 10). This keeps configs DRY - task-specific configs only override what differs from the base. A good model for how the GLUE task configs (#1) could work if driven from YAML rather than code.

### I. `get_loss_strategy()` Param Filtering

`src/training/losses.py:886-903` uses an `ACCEPTED_PARAMS` dict to filter kwargs per strategy class, preventing `TypeError` when a shared config passes `temperature` to a strategy that doesn't accept it. This is a practical pattern for factory functions that instantiate heterogeneous classes from a shared config.

### J. `slurm_essentials.sh` as Sourced Shell Library

`scripts/slurm/slurm_essentials.sh` is the right approach for SLURM script DRY. It provides `slurm_setup_env`, `slurm_print_header/footer`, `slurm_read_incomplete`, and the SIGUSR1 trap, all via `source`. Every `run_*.sh` script sources it and gets the shared setup for free. The auto-submit scripts (#14) should follow this same pattern with their own `submit_lib.sh`. The bench scripts (#13) could benefit from a similar `bench_common.sh` or, better, a shared Python base class since most of the duplication is in Python.

### K. Workbench `TASK_CONFIG` as Data-Driven Dispatch

`workbench/utils.py:21-50` uses a `TASK_CONFIG` dict to drive `load_task_data()` generically. This is the correct pattern for GLUE loading, and it already solves the problem that `src/data/datasets.py` (#1) has. The fix for #1 should adopt this approach in `src/`, and the workbench should then import from `src/` instead of maintaining its own copy (#15).

