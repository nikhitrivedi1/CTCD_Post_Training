# Training: Reward Functions

Reward functions for GRPO (Group Relative Policy Optimization) training.

**Source:** `src/training/rewards.py` (~640 lines)

---

## Overview

Each reward function follows trl's signature: `def reward_fn(completions, **kwargs) -> list[float]`. The trainer passes all dataset columns as kwargs (including `ground_truth`, `tests`, `prompts`). Rewards return values in [0, 1] where higher is better.

---

## Reward Functions

### translation_reward (bleu)

Sentence-level BLEU score between generated translation and reference.

```python
def translation_reward(completions, ground_truth, **kwargs) -> list[float]
```

Uses sacrebleu's `sentence_bleu` for per-example scoring. Normalized to [0, 1] by dividing by 100. Scored in parallel via `ThreadPoolExecutor`.

### ocr_reward (edit_similarity)

Edit similarity between generated markdown and reference.

```python
def ocr_reward(completions, ground_truth, **kwargs) -> list[float]
```

Computes `1 - (levenshtein_distance / max_len)` using rapidfuzz's C implementation. Higher score means the generated text is closer to the reference.

### unit_test_reward

Fraction of olmOCR-bench unit tests passed per completion.

```python
def unit_test_reward(completions, tests, **kwargs) -> list[float]
```

Each sample has a list of test dicts (text presence, absence, reading order, math, table). If olmocr is installed, uses `olmocr.bench.tests.load_single_test` for exact scoring. Falls back to a simple string-matching implementation otherwise.

**Important detail:** Strips `None` values from test dicts before passing to `load_single_test`. HF `Dataset.from_list` schema-unifies heterogeneous dicts by adding `None` for missing keys, and olmocr's test loader rejects unexpected kwargs.

### format_reward

Partial credit for well-structured markdown output.

```python
def format_reward(completions, **kwargs) -> list[float]
```

Scoring rubric (0.2 points each, max 1.0):
- Non-empty output
- Contains markdown headings (`# ...`)
- Contains lists (`- ...` or `1. ...`)
- Contains tables (`|...|` with separator row)
- Output longer than 50 characters

### table_cell_f1_reward

Cell-level F1 similarity for markdown tables.

```python
def table_cell_f1_reward(completions, ground_truth, **kwargs) -> list[float]
```

Extracts markdown tables from both completion and ground truth, then computes cell-level precision/recall/F1:
1. Parse markdown tables into row-cell grids
2. Skip separator rows (`| --- | --- |`)
3. Flatten to cell lists
4. For each GT cell, find best matching pred cell (by Levenshtein similarity, threshold > 0.3)
5. Compute precision = matched / pred_cells, recall = matched / gt_cells
6. F1 = harmonic mean

Returns 0.5 for documents with no tables in ground truth (neutral score).

### formula_cdm_reward

Formula similarity based on normalized Levenshtein distance on cleaned LaTeX.

```python
def formula_cdm_reward(completions, ground_truth, **kwargs) -> list[float]
```

Extracts LaTeX formulas (`$$...$$` and `$...$`), cleans them (remove `\left`, `\right`, `\displaystyle`, normalize whitespace), and computes per-formula edit similarity. Averaged across all GT formulas.

A practical proxy for full CDM (computational distance metric) without needing a symbolic math library.

### reading_order_reward

Reading order correctness based on longest common subsequence.

```python
def reading_order_reward(completions, ground_truth, **kwargs) -> list[float]
```

1. Split text into blocks by double newline (paragraphs)
2. Keep blocks > 10 chars, truncate to first 50 chars for matching
3. Match pred blocks to GT blocks by Levenshtein similarity (threshold > 0.3)
4. Compute LCS length of matched GT indices in pred order
5. Score = LCS length / number of GT blocks

Measures whether content blocks appear in the correct order, regardless of exact text matching.

---

## Parallel Scoring

### `_score_parallel(score_fn, completions, references=None)`

Scores completions in parallel using a shared `ThreadPoolExecutor(max_workers=8)`. The pool is lazy-initialized on first use to avoid creating threads for non-RL training paths.

Thread-based parallelism works here because the heavy operations (rapidfuzz Levenshtein, sacrebleu) are C extensions that release the GIL.

### `_unwrap_completion(completion)`

Extracts text from conversational format. If completion is a list of messages (trl format), returns the last message's content. Otherwise returns the string as-is.

---

## ParallelRewardGroup

Runs multiple reward functions concurrently to minimize GPU idle time.

```python
class ParallelRewardGroup:
    def __init__(self, funcs):
        self._funcs = funcs
        self._cache = {}
        self._last_completions = None
```

**Problem:** trl calls each reward function sequentially. With multiple rewards, the GPU sits idle while each reward computes.

**Solution:** On the first call per batch (detected by object identity via `is`), runs ALL reward functions in parallel via `ThreadPoolExecutor`. Caches results. Subsequent calls for the same batch return cached results.

`wrap()` returns a list of wrapper functions that trl calls individually, each checking the cache.

---

## Registry

```python
REWARD_REGISTRY = {
    "bleu":            translation_reward,
    "edit_similarity": ocr_reward,
    "format":          format_reward,
    "unit_test":       unit_test_reward,
    "table_cell_f1":   table_cell_f1_reward,
    "table_teds":      table_cell_f1_reward,   # backward compat alias
    "formula_cdm":     formula_cdm_reward,
    "reading_order":   reading_order_reward,
}

TASK_REWARDS = {
    "europarl": "bleu",
    "ocr":      "edit_similarity",
}
```

### `get_reward_funcs(config) -> list[callable]`

Builds reward function list from config:

```yaml
# Option 1: Default verifiable reward for task
rl:
  reward: verifiable   # Looks up TASK_REWARDS[task]

# Option 2: Single named reward
rl:
  reward: edit_similarity

# Option 3: Multiple rewards (wrapped in ParallelRewardGroup)
rl:
  reward:
    - edit_similarity
    - format
    - table_cell_f1
```

When multiple rewards are specified, they are wrapped in `ParallelRewardGroup` for concurrent scoring.

---

*Source: `src/training/rewards.py`*
