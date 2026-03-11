# Model Improvements: Formatting (Bold, Italic, Strikethrough)

**Date:** 2026-03-09
**Goal:** Improve OCR model's ability to preserve inline formatting (bold, italic, strikethrough, underline, etc.) in markdown output.

## Current State

The model (Qwen3-VL-2B + LoRA, SFT on olmocr-mix) extracts text content reasonably well (0.680 text score on OmniDocBench, 67.1% on olmOCR-Bench) but has no explicit training signal for inline formatting preservation. The GRPO run on OmniDocBench regressed slightly (0.351 -> 0.338 overall), revealing a reward-bench disconnect.

**Neither benchmark currently measures inline formatting.** We have no data on how well the model preserves bold/italic/strikethrough today.

---

## 1. Dataset

### Problem

**olmocr-mix ground truth has near-zero inline formatting.** The training labels (GPT-4o generated) contain ~0.6% bold markers, 0% italic, 0% strikethrough. The model never sees formatting-rich examples during SFT, so it has no reason to produce them.

**OmniDocBench conversion strips formatting.** The `convert_omnidocbench.py` script converts `text_block` annotations to plain text, discarding any bold/italic that might exist in the source. The `html_table_to_markdown` function also strips HTML `<b>`, `<i>`, `<em>`, `<strong>` tags during table conversion.

### Actions

**D1. Audit olmocr-mix for formatting content.**
Run a quick scan of the 260K ground truth labels to quantify how many contain `**`, `*text*`, `~~`, `__` markers. This establishes the baseline. If it's truly <1%, we need external data.

```bash
# On cluster, after pulling data
grep -c '\*\*' data/ocr/olmocr-mix-full/*/ground_truth.md | awk -F: '{s+=$2} END {print s}'
```

**D2. Curate formatting-rich training data.**
Options ranked by effort:
1. **Filter olmocr-mix for formatting-rich pages.** If even 1-2% of 260K pages have formatting, that's 2-5K samples. Filter by presence of `**`, `*`, `~~` in ground truth. Zero marginal cost.
2. **Re-annotate OmniDocBench with formatting.** The source images contain bold/italic visually. Re-generate ground truth using a VLM (GPT-4o or Qwen-VL-72B) with a formatting-aware prompt. ~1,352 pages.
3. **Synthetic formatting pages.** Generate PDFs with known formatting (bold headings, italic captions, strikethrough edits) and pair with ground truth. Full control over distribution. Use tools like reportlab or LaTeX.
4. **arXiv papers.** Rich in bold/italic from LaTeX source. Pair rendered PDFs with LaTeX-derived markdown. Large scale but noisy alignment.

**D3. Update OmniDocBench conversion to preserve formatting.**
Modify `convert_omnidocbench.py` to check if source annotations contain formatting hints (some OmniDocBench blocks have `<b>`, `<i>` in their text/html fields). Preserve these as markdown `**bold**` and `*italic*`.

**D4. Formatting-aware system prompt for data generation.**
When generating new ground truth, use a prompt that explicitly requests formatting:
```
Extract all text content and output it in clean markdown format.
Preserve the document's structure (headings, lists, tables) AND
inline formatting (bold, italic, strikethrough, underline) exactly
as they appear in the document.
```

---

## 2. Rewards

### Problem

**`format_reward` checks structural elements only.** It scores: non-empty (0.2), headings (0.2), lists (0.2), tables (0.2), length (0.2). No points for bold, italic, strikethrough, or any inline formatting.

**`edit_similarity` captures formatting implicitly** but poorly. The `**` markers in `**bold**` vs `bold` contribute ~4 characters of edit distance across a potentially 1000+ character document. The signal is drowned out.

**No reference-aware formatting reward exists.** There's no function that compares formatting markers between prediction and ground truth.

### Actions

**R1. Add `inline_formatting_reward` function.**
A new reward that scores inline formatting preservation against ground truth. Design:

```python
def _score_inline_formatting(completion, reference):
    """Score preservation of inline formatting markers."""
    # Extract formatting spans from both texts
    pred_bold = set(re.findall(r'\*\*(.+?)\*\*', completion))
    gt_bold = set(re.findall(r'\*\*(.+?)\*\*', reference))
    pred_italic = set(re.findall(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', completion))
    gt_italic = set(re.findall(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', reference))
    pred_strike = set(re.findall(r'~~(.+?)~~', completion))
    gt_strike = set(re.findall(r'~~(.+?)~~', reference))

    # F1 across all formatting types
    all_pred = pred_bold | pred_italic | pred_strike
    all_gt = gt_bold | gt_italic | gt_strike

    if not all_gt:
        return 0.5  # No formatting in GT, neutral score
    if not all_pred:
        return 0.0

    # Fuzzy match (formatting text may differ slightly)
    matched = _fuzzy_set_match(all_pred, all_gt)
    precision = matched / len(all_pred) if all_pred else 0.0
    recall = matched / len(all_gt) if all_gt else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
```

Register as `"inline_formatting"` in `REWARD_REGISTRY`.

**R2. Update `format_reward` to include inline formatting.**
Add a 6th criterion (re-weight to keep max=1.0):
- Bold markers present: check `**...**` in completion
- Italic markers present: check `*...*` in completion

This gives a non-reference signal that at least encourages the model to produce formatting markers.

**R3. Consider a formatting-weighted edit similarity.**
A variant of `ocr_reward` that upweights formatting markers. For example, duplicate `**` and `*` markers 3x before computing Levenshtein distance, so formatting errors cost more.

---

## 3. Training Setup

### Problem

**GRPO showed reward-bench disconnect.** Training rewards improved (table 0->0.418, formula 0->0.303) but bench scores regressed. The rewards score isolated completions, the bench scores full pages.

**Sparse reward signal at low baselines.** With table F1 at 0.048 and formula CDM at 0.017, most completions score near zero, giving GRPO little gradient.

**Formatting will have the same cold-start problem.** If the model doesn't produce formatting at all, a formatting reward will return 0.0 for every completion, providing no learning signal.

### Actions

**T1. SFT-first strategy.**
Don't jump straight to GRPO for formatting. First fine-tune on formatting-rich data (from D2) with CE loss. This teaches the model what formatting looks like. Then use GRPO to refine.

Pipeline: `base -> SFT (olmocr-mix) -> SFT (formatting-rich) -> GRPO (formatting rewards)`

**T2. Warm-start GRPO with formatting examples.**
When running GRPO, seed the dataset with pages that definitely contain formatting. This ensures the reward function sees non-zero signals and can provide gradients.

**T3. Curriculum: start with bold-only.**
Bold is the most common inline formatting. Start with bold-only data and rewards, get that working, then expand to italic and strikethrough. Avoids diluting the signal across rare formatting types.

**T4. Higher temperature for formatting exploration.**
Current GRPO uses temperature=0.8. For formatting discovery, consider 1.0-1.2 to encourage the model to explore producing `**` markers it might not produce at lower temperatures.

**T5. Separate formatting LoRA adapter.**
Train a formatting-specific LoRA adapter that can be merged with the base OCR adapter. This avoids catastrophic forgetting of text extraction quality while adding formatting capability.

**T6. Increase `num_generations` for formatting GRPO.**
With 4 generations per prompt, the model may never produce a formatting-rich completion to learn from. Increase to 8-16 to improve the chance of at least one good formatting sample in the group.

---

## 4. Evals / Benchmarks

### Problem

**No existing benchmark measures inline formatting.** olmOCR-Bench checks content presence/absence/order. OmniDocBench checks text similarity, tables, formulas, reading order. Neither specifically evaluates whether bold/italic/strikethrough are preserved.

**We can't improve what we can't measure.**

### Actions

**E1. Build a formatting-specific benchmark.**
Create a targeted eval set:
- 100-200 document pages with known formatting (bold headings, italic captions, strikethrough edits, underlined terms)
- Ground truth markdown with formatting markers
- Score: F1 on formatting spans (similar to R1 reward design)

Sources:
- Academic papers (LaTeX -> bold/italic in headings, emphasis)
- Financial reports (bold totals, italic footnotes)
- Edited documents (strikethrough for revisions)
- Textbooks (bold key terms, italic definitions)

**E2. Standalone formatting benchmark.**
Do NOT modify `omnidoc_bench.py` or `olmocr_bench.py` - those are standardized benchmarks with defined metrics. Instead, use the standalone `scripts/benchmarks/formatting_bench.py` which scores formatting F1 independently. It can re-use existing OmniDocBench predictions (score-only mode) or run live inference.

**E3. Quick formatting probe on current model.**
Before building anything, run a quick manual check: give the current model 10-20 pages with obvious formatting and see what it produces. This tells us if the model already has latent formatting capability from pre-training that just needs activation, or if it's truly absent.

**E4. Add formatting stats to existing eval outputs.**
Even without a dedicated metric, count formatting markers in model output during eval runs: number of `**...**`, `*...*`, `~~...~~` spans. Track this across runs to see if training changes formatting behavior.

---

## 5. Additional Considerations

### System Prompt

**Current prompt doesn't mention inline formatting.** The OCR system prompt in `ocr_utils.py` says:
> "preserving the document's structure (headings, lists, tables, etc.)"

The "etc." is vague. Update to explicitly mention formatting:

```python
OCR_SYSTEM_PROMPT = (
    "Below is an image of a document page. Extract all text content "
    "and output it in clean markdown format, preserving the document's "
    "structure (headings, lists, tables) and inline formatting "
    "(bold, italic, strikethrough) exactly as shown."
)
```

This is the simplest possible intervention. Test it first before any training changes.

### Visual Grounding

The model needs to visually detect formatting from pixel-level cues (thicker strokes for bold, slanted for italic, line-through for strikethrough). Qwen3-VL's ViT may already encode these features but the LM head doesn't produce formatting markers because it was never trained to.

**Hypothesis:** SFT on formatting-rich data + updated system prompt may be sufficient to unlock latent formatting capability. GRPO is secondary.

### Formatting Detection Difficulty

Not all formatting is equally detectable from images:
- **Bold**: High contrast difference, relatively easy to detect visually
- *Italic*: Slant is visible but subtle, moderate difficulty
- ~~Strikethrough~~: Line through text is distinctive, moderate difficulty
- <u>Underline</u>: Often confused with links or table borders, hard
- `Code`: Monospace font difference, moderate difficulty

Prioritize bold first (most common, easiest to detect, strongest visual signal).

### Risk: Hallucinated Formatting

If the reward encourages formatting markers, the model might hallucinate formatting on text that isn't actually bold/italic. The reference-based reward (R1) mitigates this, but the structural reward (R2) doesn't. Monitor precision alongside recall.

---

## Proposed Execution Order

| Step | Action | Type | Effort | Status |
|------|--------|------|--------|--------|
| 1 | **E3**: Quick formatting probe on current model | Eval | Low | Tool ready (`probe_formatting.py`) |
| 2 | **System prompt update** (section 5) | Prompt | Low | Done |
| 3 | **D1**: Audit olmocr-mix for formatting content | Data | Low | Tool ready (`audit_formatting.py`) |
| 4 | **E4**: Add formatting stats to eval outputs | Eval | Low | Pending |
| 5 | **D2**: Curate formatting-rich training data | Data | Medium | Done (synthetic generator) |
| 6 | **R1**: Build `inline_formatting_reward` | Reward | Medium | Done + tested (60/60) |
| 7 | **E2**: Standalone formatting benchmark | Eval | Medium | Done (`formatting_bench.py`) |
| 8 | **T1**: SFT on formatting-rich data | Training | Medium | Pending |
| 9 | **T3**: GRPO with bold-first curriculum | Training | Medium | Pending |
| 10 | **E1**: Full formatting benchmark | Eval | High | Pending |

Steps 1-4 are low-effort diagnostics. Steps 5-7 build infrastructure. Steps 8-10 are training and evaluation.

---

## 6. Engineering Audit (2026-03-09)

Full audit of the GRPO training pipeline, reward system, data loaders, eval scripts, and configs. **Result: production-ready, no blockers.**

### Tests

60/60 passing. Coverage: rewards.py includes `inline_formatting_reward`, `_extract_formatting_spans`, `_fuzzy_set_f1`, `_score_inline_formatting`. Callbacks at 16%, data loaders at 0% (no test files yet).

### Config Consistency - Clean

All reward names in GRPO configs match `REWARD_REGISTRY` keys exactly:

| Config | Rewards | Status |
|--------|---------|--------|
| `00_grpo.yaml` | `[edit_similarity, format]` | OK |
| `01_synthmix.yaml` | `[unit_test]` | OK |
| `02_omnidoc.yaml` | `[edit_similarity, format, table_cell_f1, formula_cdm, reading_order]` | OK |

Dataset names route correctly: `synthmix` -> `load_synthmix_rl()`, `omnidocbench` -> `load_omnidocbench_rl()`, default -> `load_ocr_rl()`.

### Import Chains - Clean

All lazy imports (rapidfuzz, sacrebleu, olmocr, trl, peft) are properly guarded. olmocr is gracefully optional with fallback scoring in `unit_test_reward`. Playwright/Chromium auto-installs or skips math tests.

### Data Pipeline - Clean

Column names consistent: `ground_truth` for OCR/OmniDocBench, `tests` for synthmix. Directory structures match between conversion scripts and dataset loaders. Image extension fallbacks (.jpg/.png/.jpeg) work in both eval and data loading.

### Reward Edge Cases - Well-Handled

| Scenario | Behavior |
|----------|----------|
| Empty reference + empty completion | Returns 1.0 (correct match) |
| Empty reference + non-empty completion | Returns 0.0 |
| No GT tables/formulas | Returns 0.5 (neutral) |
| Conversational format input | Unwrapped automatically |
| HF Dataset None values in test dicts | Stripped before passing to olmocr |

### Eval Scripts - Robust

Both `omnidoc_bench.py` and `olmocr_bench.py` handle missing images, empty model output, missing Playwright, and missing olmocr gracefully. Sharding and resume support work correctly.

### Minor Notes (not blockers)

1. **`load_ocr_rl` silent None eval set.** If no `eval/` directory exists alongside `train/`, returns `None` for eval dataset without logging. Should add a warning.

2. **`omnidoc_bench.py` image format log message.** Tries `.jpg/.png/.jpeg` fallbacks correctly, but log message assumes `.jpg`. Cosmetic only.

### Coverage Gaps to Address Later

- No tests for `rl_datasets.py` (0% coverage). Dataset loaders are tested only implicitly through training runs.
- No tests for eval scripts. Would catch regressions in scoring logic.
- `callbacks.py` at 16% coverage. StepMetricsCallback reward logging untested.

---

## 7. Synthetic Data Generator (2026-03-09)

Built a full pipeline to generate formatting-rich training data: `scripts/data/generate_formatting_data.py`.

### Pipeline

```
Markdown GT -> HTML+CSS -> Playwright/Chromium render -> PNG -> Augmentation -> Final image
```

### Key Design

- **Single-pass span-based formatting injection**: picks word-boundary spans, assigns formatting types independently per span, applies right-to-left. No overlapping markers by construction.
- **Combined formatting**: spans can receive any combination of bold, italic, strikethrough. Explicit 3% chance of triple-combo, 5% chance of double-combo, plus natural co-occurrence from independent rolls.
- **10 CSS templates**: academic, textbook, technical, financial, modern, compact, book, slides, news, forms. Random per-page selection for visual diversity.
- **18 built-in paragraphs** across domains (academic, financial, legal, technical, medical, educational). Optional wikitext content source.
- **Augmentation**: rotation (0-2 deg), brightness/contrast variation, Gaussian blur, noise, JPEG compression, grayscale conversion. Simulates real scanned documents.

### Quality Metrics (10-page test run)

| Metric | Per Page Avg | Total (10 pages) |
|--------|-------------|-------------------|
| Bold spans | 14.8 | 148 |
| Italic spans | 7.1 | 71 |
| Strikethrough spans | 6.9 | 69 |
| Bold+Italic combined | 1.7 | 17 |
| Bold+Strike combined | 1.8 | 18 |
| Italic+Strike combined | 0.8 | 8 |
| All three combined | 1.2 | 12 |

Zero malformed markdown (all markers balanced). Sample renders in `wip_docs/formatting_samples/`.

### Usage

```bash
# Generate 1000 pages with augmentation
python scripts/data/generate_formatting_data.py --output_dir data/ocr/formatting-synth --num_pages 1000 --augment

# Quick test
python scripts/data/generate_formatting_data.py --output_dir /tmp/fmt_test --num_pages 10

# With wikitext content
python scripts/data/generate_formatting_data.py --output_dir data/ocr/formatting-synth --num_pages 500 --content_source wikitext --augment
```

---

## Key Files

| File | Role | Changes Needed |
|------|------|----------------|
| [rewards.py](src/training/rewards.py) | Reward functions | Done: `inline_formatting_reward`, updated `format_reward` |
| [ocr_utils.py](src/data/ocr_utils.py) | System prompt | Done: updated `OCR_SYSTEM_PROMPT` |
| [formatting_bench.py](scripts/benchmarks/formatting_bench.py) | Formatting benchmark | Standalone, do NOT modify omnidoc_bench.py |
| [formatting_generator.py](scripts/data/formatting_generator.py) | Synthetic data gen | Done: full pipeline |
| [audit_formatting.py](workbench/audit_formatting.py) | Data auditing | Done: scans GT for formatting stats |
| [probe_formatting.py](workbench/probe_formatting.py) | Model probing | Done: tests latent formatting capability |
| [src/data/handlers/](src/data/handlers/) | Dataset handlers | Done: unified prepare/validate/load per dataset |
| [src/data/formatters.py](src/data/formatters.py) | SFT/RL formatting | Done: format_for_sft(), format_for_rl() |
| [scripts/data/prepare.py](scripts/data/prepare.py) | Unified data CLI | Done: single entry point for all data prep |
| configs/ocr/.../rl/ | GRPO configs | New config with formatting rewards |

---

## 8. Data Pipeline Standardization (2026-03-09)

### Current State

Every dataset has a different prep workflow, different scripts, and different loader conventions:

| Dataset | Type | Download | Convert | Load | Scripts |
|---------|------|----------|---------|------|---------|
| olmOCR-mix | Public | HF Hub | `prepare_ocr_data.py` (PDF to PNG + HF) | `load_ocr_rl()` | 1 |
| synthmix | Public | `download_rl_datasets.sh` | `convert_synthmix.py` (PDF to JPG) | `load_synthmix_rl()` | 2 |
| OmniDocBench | Public | `download_rl_datasets.sh` | `convert_omnidocbench.py` (JSON to MD) | `load_omnidocbench_rl()` | 2 |
| Europarl | Public | Inline at load time | None | `load_europarl_rl()` | 0 |
| Formatting | Synthetic | N/A | N/A | Not written yet | 1 |

### Problems

1. **No unified entry point.** Each dataset has ad-hoc scripts in `scripts/data/`.
2. **Download is fragmented.** Shell script for some, Python for others, inline HF Hub for Europarl.
3. **`_sft` and `_rl` suffixes on loaders conflate two concerns.** A dataset is just data. How it's shaped for training (SFT vs RL) is a separate concern. "synthmix" isn't an "RL dataset" - it's a dataset that can be used for RL or SFT.
4. **Two separate loader files for no reason.** `datasets.py` (SFT) and `rl_datasets.py` (RL) duplicate logic. Both load the same underlying data, just format it differently. With the handler pattern, both files become unnecessary - handlers load raw data, formatting is a separate concern.
5. **The dispatcher** (`load_rl_dataset()`) is a growing if/elif chain. Each new dataset touches 3+ places.
6. **Validation is ad-hoc.** Inline Python snippets inside the shell script, nothing reusable.
7. **Output structure varies.** olmOCR-mix uses HF Dataset on disk, synthmix uses raw files, OmniDocBench uses raw files, formatting uses category subdirs.

### Two-Track Model

Data flows into training through two tracks. The distinction is about how data is sourced, not how it's consumed:

| Track | Pipeline | Datasets |
|-------|----------|----------|
| **Public** | Download -> Convert -> Validate | olmOCR-mix, synthmix, OmniDocBench, Europarl |
| **Synthetic** | Generate -> Validate | Formatting (and future synthetic datasets) |

Both produce the same thing: samples on disk. The loader decides whether to format for SFT or RL based on the training config.

### Solution: Handler Pattern

Standardize the **process** (prepare -> validate -> load), not the **data format** (which is task-dependent: OCR = image + markdown, translation = source + target, classification = text + label).

Each dataset gets a handler that implements a common interface:

```python
class DatasetHandler(ABC):
    def prepare(self, data_dir: str):
        """Download/generate + convert. Idempotent.

        Public datasets: download from source + convert to standard format.
        Synthetic datasets: invoke generator. If data already exists at
        data_dir, skip (idempotent). If not, generate automatically.
        """

    def validate(self, data_dir: str) -> bool:
        """Check integrity: files exist, stems match, counts are sane."""

    def load(self, data_dir: str, **kwargs) -> tuple[Dataset, Dataset]:
        """Read raw samples from disk. Returns (train, eval).

        Returns raw, unformatted samples. SFT vs RL formatting is NOT
        the handler's concern - that's done downstream.
        """
```

The interface is identical for public and synthetic datasets. The only difference is what `prepare()` does internally:

```
Public handler:    prepare() = download + convert
Synthetic handler: prepare() = generate (if data_dir doesn't exist or is empty)
```

For synthetic datasets, `prepare()` auto-invokes generation when the target directory is missing. No separate manual step needed. If pointed to an existing generation, it skips.

Registry replaces the if/elif dispatcher:

```python
DATASET_REGISTRY = {
    "omnidocbench": OmniDocBenchHandler,
    "synthmix":     SynthmixHandler,
    "formatting":   FormattingHandler,
    "olmocr-mix":   OlmOCRMixHandler,
    "europarl":     EuroparlHandler,
    "mnli":         GLUEHandler,
    "qqp":          GLUEHandler,
    "sst2":         GLUEHandler,
    "cola":         GLUEHandler,
}
```

SFT vs RL formatting is a separate step, not baked into the handler or the loader. Two thin formatting functions handle the conversion:

```
Handler.prepare()  ->  data on disk (download/generate + convert)
Handler.load()     ->  raw samples (image + text, or text + text, etc.)
format_for_sft()   ->  chat-templated, tokenizer-ready
format_for_rl()    ->  prompt dicts + ground_truth/tests
```

Unified CLI:

```bash
python scripts/data/prepare.py --dataset omnidocbench   # download + convert + validate
python scripts/data/prepare.py --dataset formatting      # generate + validate
python scripts/data/prepare.py --dataset all             # everything
```

### File Changes

```
NEW:
  src/data/handlers/
    __init__.py          # DATASET_REGISTRY, get_handler()
    base.py              # DatasetHandler ABC: prepare(), validate(), load()
    omnidocbench.py      # absorbs convert_omnidocbench.py + load_omnidocbench_rl
    synthmix.py          # absorbs convert_synthmix.py + load_synthmix_rl
    formatting.py        # absorbs generate_formatting_data.py + new loader
    olmocr_mix.py        # absorbs prepare_ocr_data.py + load_ocr/load_ocr_rl
    europarl.py          # absorbs load_europarl + load_europarl_rl
    glue.py              # absorbs load_mnli, load_qqp, load_sst2, load_cola
  src/data/formatters.py   # format_for_sft(), format_for_rl() - thin wrappers
  scripts/data/prepare.py  # unified CLI: --dataset X --action prepare|validate|all

MODIFIED:
  forge.py                 # simplified: registry lookup replaces if/elif chains
  src/data/__init__.py     # export registry + formatters

DELETED (absorbed into handlers):
  src/data/datasets.py           # -> handler.load() + formatters.format_for_sft()
  src/data/rl_datasets.py        # -> handler.load() + formatters.format_for_rl()
  scripts/data/convert_omnidocbench.py   # -> OmniDocBenchHandler.prepare()
  scripts/data/convert_synthmix.py       # -> SynthmixHandler.prepare()
  scripts/data/prepare_ocr_data.py       # -> OlmOCRMixHandler.prepare()
  scripts/data/download_rl_datasets.sh   # -> prepare.py --dataset all
  scripts/data/generate_formatting_data.py  # -> FormattingHandler.prepare()
```

### Benefits

- **Adding a new dataset**: write one handler file, register it. One place, one interface.
- **No `_sft`/`_rl` naming**: the dataset is the dataset. Training mode determines formatting.
- **No two loader files**: `datasets.py` and `rl_datasets.py` merge into handlers + formatters.
- **Synthetic auto-generation**: `prepare()` handles it. No separate manual invoke needed.
- **Unified CLI**: one script to prepare any dataset, instead of remembering per-dataset commands.
- **Consistent validation**: every handler implements `validate()`, no more inline Python in shell scripts.
- **Clean `scripts/data/`**: standalone scripts absorbed into handlers. No dead code lingering.
