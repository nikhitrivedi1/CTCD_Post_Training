# Formatting Benchmark & Training Plan (2026-03-06)

## Problem

olmOCR-mix training data has minimal formatting (0.6% bold, 0% headings, 0% markdown tables). olmOCR-Bench tests extraction correctness (text present/absent/order/math) but does not test formatting quality at all. We need both a formatting benchmark and formatting-rich training data.

## Current State

**SFT model (ablations_00):**
- Model: Qwen3-VL-2B + LoRA (r=16, alpha=16)
- Data: 5K olmOCR-mix, CE-only
- olmOCR-Bench: 67.1% overall (7-category average)
- Gap to olmOCR-7B: 10.2pp (extraction ~5pp, math ~3.3pp, convention ~1.9pp)
- Checkpoint: `outputs/olmocr-mix_ablations_00_no-teacher_ce-only_20260225-0952Z/`
- Config: `configs/ocr/olmocr-mix/qwen3vl-2b/ablations/00_no-teacher_ce-only.yaml`

**GRPO infrastructure (built, not yet run):**
- Config: `configs/ocr/olmocr-mix/qwen3vl-2b/rl/00_grpo.yaml`
- Reward functions: `edit_similarity` + `format` in `src/training/rewards.py`
- Dataset loader: `load_ocr_rl()` in `src/data/rl_datasets.py`
- Adapter merging: `--adapter` CLI arg in `forge.py` loads SFT LoRA, merges into base, trains fresh LoRA with GRPO
- trl version: 0.29.0 (confirmed compatible with VLM multimodal + Qwen3-VL)

---

## Implementation Details

### Adapter Merging (forge.py `_train_rl`)

The `--adapter` CLI arg enables the SFT-to-RL pipeline:
1. Loads base model (`Qwen/Qwen3-VL-2B-Instruct`) via `AutoModelForImageTextToText`
2. Loads SFT LoRA adapter via `PeftModel.from_pretrained(base_model, adapter_path)`
3. Merges adapter into base: `model.merge_and_unload()`
4. Saves merged model to `<output_dir>/base/` (full weights + processor/tokenizer)
5. Passes merged model object to `GRPOTrainer` with a fresh `peft_config` (new LoRA)

Without `--adapter`, GRPOTrainer receives the model name string and trains from pretrained.

**Adapter strategy: merge-and-save after each stage.** The output dir IS the final model:

```
outputs/<run>/
  model*.safetensors  # final merged model (base + LoRA merged, loadable via from_pretrained)
  base/               # this stage's starting model (full weights)
  checkpoint-XXX/     # LoRA adapter checkpoints
  run_metadata.json   # lineage: base_model, adapter source
```

`run_metadata.json` tracks lineage: `base_model` (HF model ID), `adapter` (path to previous stage's output). Each stage is self-contained - `outputs/<run>/` works directly as a model path.

### GRPO Config (00_grpo.yaml)

```yaml
inherit: ../base.yaml  # -> Qwen3-VL-2B, LoRA r=16, batch=2, grad_accum=16

data:
  task: ocr
  data_dir: data/ocr/olmocr-mix-full
  max_train_samples: 5000
  max_length: 1024

training:
  num_epochs: 1
  learning_rate: 1e-6

rl:
  algorithm: grpo
  reward: [edit_similarity, format]
  num_generations: 4
  max_completion_length: 1024
  temperature: 0.8
  beta: 0.0
  loss_type: grpo
  log_completions: true
```

Inherited from base: LoRA (r=16, alpha=16, dropout=0.05, targets=qkvo+gate+up+down), warmup_steps=50, save_steps=50, eval_steps=50, save_total_limit=3, W&B logging.

### Reward Functions (src/training/rewards.py)

| Function | Name | Input | Signal |
|----------|------|-------|--------|
| `ocr_reward` | `edit_similarity` | completions + ground_truth | 1 - CER (Levenshtein distance / max_len) |
| `format_reward` | `format` | completions only | 0.2 each for: has content, has headings, has lists, has tables, length > 50 chars |

Registry: `REWARD_REGISTRY = {"bleu": ..., "edit_similarity": ocr_reward, "format": format_reward}`

`get_reward_funcs(config)` handles list specs: `reward: [edit_similarity, format]` returns both functions.

### Dataset Loader (src/data/rl_datasets.py `load_ocr_rl`)

- Loads from `data/ocr/olmocr-mix-full` (Arrow dataset with train/eval splits)
- Keeps `image` column (PIL), creates `prompt` (conversational format) and `ground_truth` (reference markdown)
- System prompt: "Below is an image of a document page. Extract all text content and output it in clean markdown format, preserving the document's structure (headings, lists, tables, etc.)."
- GRPOTrainer auto-detects the `image` column and processes via AutoProcessor

### trl 0.29.0 Compatibility (verified)

- GRPOTrainer has explicit VLM support: detects `image`/`images` columns, calls `prepare_multimodal_messages`
- Qwen-specific handling for `image_grid_thw` and `pixel_values` in batch construction
- Accepts pre-loaded model objects (merged adapter) directly
- `processing_class` accepts `AutoProcessor` (not just tokenizers)
- `logits_to_keep` gracefully handled if model doesn't support it

### Key File Paths

| File | Purpose |
|------|---------|
| `forge.py` | Entry point, `_train_rl` with adapter merging (~line 769) |
| `src/data/ocr_utils.py` | Shared: `OCR_SYSTEM_PROMPT`, `render_pdf_page`, `pdf_image_name` |
| `src/training/rewards.py` | All 7 reward functions + `REWARD_REGISTRY` |
| `src/data/rl_datasets.py` | `load_ocr_rl`, `load_synthmix_rl`, `load_omnidocbench_rl` |
| `scripts/data/convert_synthmix.py` | PDF-to-image conversion for synthmix (cluster) |
| `scripts/data/convert_omnidocbench.py` | JSON-to-markdown GT conversion for OmniDocBench (cluster) |
| `configs/ocr/olmocr-mix/qwen3vl-2b/rl/01_synthmix.yaml` | GRPO config: synthmix + unit_test reward |
| `configs/ocr/olmocr-mix/qwen3vl-2b/rl/02_omnidocbench.yaml` | GRPO config: OmniDocBench + 5 rewards |
| `configs/ocr/olmocr-mix/qwen3vl-2b/base.yaml` | Model config (Qwen3-VL-2B) |
| `configs/ocr/olmocr-mix/base.yaml` | Dataset + training config (LoRA, batch, etc.) |
| `scripts/eval/olmocr_bench.py` | olmOCR-Bench eval script |
| `src/inference/engine.py` | InferenceEngine for eval (also does adapter merge) |

---

## Pipeline

### Data Sources

| Source | Size | Format | Reward signal | Integration effort |
|--------|:----:|--------|---------------|:------------------:|
| **olmOCR2-synthmix** | 2,186 pages | PDF + HTML renders + 30K JSONL unit tests | Unit test pass rate (binary per test) | Low |
| **OmniDocBench** | 1,355 pages | JPG images + structured JSON annotations | Edit distance, TEDS, CDM | Medium |

olmOCR 2 used only 2,186 pages for their GRPO stage (28 generations per doc) and got significant gains. OmniDocBench's 1,355 pages is in the same ballpark.

**Strategy:** Start with synthmix. Lower integration effort, proven reward signal from olmOCR 2, and we already have the olmocr test runner pattern. Add OmniDocBench later for formatting-specific rewards and as the eval benchmark.

#### olmOCR2-synthmix (`allenai/olmOCR-synthmix-1025`)

HuggingFace dataset (`repo_type="dataset"`). Structure:
```
bench_data/
  *.jsonl              # Unit tests: arxiv_cs, arxiv_math, arxiv_physics, general, tables
  pdfs/                # PDFs organized by category (2,187 files)
    arxiv_cs/          (337), arxiv_math/ (342), arxiv_physics/ (102),
    general/ (952), tables/ (454)
  claude_original/     # Original Claude OCR outputs (2,197 files)
html/                  # HTML renders from Claude Sonnet (2,197 files)
pdfs/                  # Side-by-side original + rerendered PDFs (4,394 files)
training/              # Markdown + PDF pairs for SFT (4,394 files)
metadata/              # Original URLs (5 JSONL files)
```

Tests use the **exact same format as olmOCR-bench** (4 types: `absent`, `order`, `math`, `table`):
```json
{"pdf": "arxiv_physics/pdf_00054_page46.pdf", "page": 1, "type": "absent", "text": "19", "max_diffs": 0}
{"pdf": "arxiv_physics/pdf_00054_page46.pdf", "page": 1, "type": "order", "before": "text A", "after": "text B", "max_diffs": 4}
{"pdf": "arxiv_physics/pdf_00055_page14.pdf", "page": 1, "type": "math", "math": "O(L^5)", "max_diffs": 0}
{"pdf": "arxiv_physics/pdf_00024_page35.pdf", "page": 1, "type": "table", "cell": "GAN Loss", "up": "L1 Loss", "down": "Cross Entropy Loss", "max_diffs": 0}
```

This means we can reuse `olmocr.bench.tests.load_single_test` directly for scoring.

**GRPO reward**: Unit test pass rate per document (fraction of tests passed). This is the same reward olmOCR 2 used. Binary, verifiable, no learned reward model needed.

#### OmniDocBench (`opendatalab/OmniDocBench`)

HuggingFace dataset with:
- **Images** (`images/`): 1,355 JPG document page images
- **Annotations** (`OmniDocBench.json`): Structured JSON with block-level and span-level annotations

The GT is NOT raw markdown. It's structured JSON with bounding boxes, block types, and content:
```json
{
  "page_info": {"image_path": "xxx.jpg", "width": 2480, "height": 3508},
  "layout_dets": [
    {"category_type": "title", "poly": [...], "text": "Introduction", "order": 0},
    {"category_type": "text_block", "poly": [...], "text": "Lorem ipsum...", "order": 1},
    {"category_type": "table", "poly": [...], "html": "<table>...</table>", "order": 2}
  ]
}
```

15 block types (title, text, table, formula, figure, header, footer, etc.) + 4 span types (bold, italic, inline_formula, footnote_mark).

---

### Phase 1: Data Prep

Goal: Get both datasets downloaded, converted, and loadable so we can train and eval.

#### 1a. synthmix (start here)

| # | Step | Details | Files to create/modify | Status |
|:-:|------|---------|----------------------|:------:|
| 1 | Download synthmix | `snapshot_download('allenai/olmOCR-synthmix-1025', local_dir='data/ocr/synthmix', repo_type='dataset')` (on cluster) | - | TODO |
| 2 | Inspect data | Check PDF count, JSONL test format, test type distribution, pages per doc | - | TODO |
| 3 | Convert PDFs to images | `python scripts/data/convert_synthmix.py --data_dir data/ocr/synthmix` (on cluster) | `scripts/data/convert_synthmix.py` | Done |
| 4 | Build RL dataset loader | Group tests by (pdf, page). Produce `{prompt, image, tests}` per page. Uses `dataset: synthmix` in config to route. | `src/data/rl_datasets.py` (add `load_synthmix_rl`) | Done |
| 5 | Add `unit_test` reward | Reuses `olmocr.bench.tests.load_single_test` for exact scoring. Fallback for absent/order/math/table when olmocr not installed. Score = fraction of tests passed. | `src/training/rewards.py` (add `unit_test_reward`, register as `unit_test`) | Done |
| 6 | Validate end-to-end | Load 10 samples, run SFT model inference, compute reward scores. Sanity check: SFT model should score > 0 on most tests. (on cluster) | - | TODO |

**Expected output:** `data/ocr/synthmix/` with PDFs, images, and JSONL tests. New `load_synthmix_rl()` loader. New `unit_test` reward function.

#### 1b. OmniDocBench

| # | Step | Details | Files to create/modify | Status |
|:-:|------|---------|----------------------|:------:|
| 1 | Download OmniDocBench | `snapshot_download('opendatalab/OmniDocBench', local_dir='data/ocr/omnidocbench', repo_type='dataset')` (on cluster) | - | TODO |
| 2 | Inspect annotations | 1,355 pages, 20 block types. Tables have `html` field, formulas have `latex` field. | - | Done |
| 3 | Convert JSON to markdown | `python scripts/data/convert_omnidocbench.py --data_dir data/ocr/omnidocbench` (on cluster). Maps: title to `#`, text_block to paragraph, table HTML to markdown table, formula latex preserved as-is. Sorted by `order` field. | `scripts/data/convert_omnidocbench.py` | Done |
| 4 | Build RL dataset loader | Reads `markdown/` dir, pairs with `images/`. Uses `dataset: omnidocbench` in config. Produces `{prompt, ground_truth, image}`. | `src/data/rl_datasets.py` (add `load_omnidocbench_rl`) | Done |
| 5 | Validate GT quality | `python scripts/data/convert_omnidocbench.py --data_dir data/ocr/omnidocbench --verify` to spot-check conversions. (on cluster) | - | TODO |

**Expected output:** `data/ocr/omnidocbench/` with images and `markdown/` GT. New `load_omnidocbench_rl()` loader.

---

### Phase 2: RL Training

Goal: Train GRPO on top of SFT checkpoint, measure improvement on olmOCR-Bench.

#### 2a. GRPO on synthmix (first run)

| # | Step | Details | Files to create/modify | Status |
|:-:|------|---------|----------------------|:------:|
| 1 | Create config | `reward: [unit_test]`, `dataset: synthmix`, `data_dir: data/ocr/synthmix`, inherit from `base.yaml`. Same hyperparams as `00_grpo.yaml` (lr=1e-6, 4 generations, temp=0.8). | `configs/ocr/olmocr-mix/qwen3vl-2b/rl/01_synthmix.yaml` | Done |
| 2 | Dry run | `python forge.py --config configs/.../01_synthmix.yaml --adapter outputs/olmocr-mix_ablations_00_.../  --debug` - verify adapter loads, data loads, reward computes, one step runs. | - | TODO |
| 3 | Full training | Run on H200. ~2K pages, 4 generations each, 1 epoch. Monitor W&B: reward curves, completion quality, gradient norms. | - | TODO |
| 4 | Eval on olmOCR-Bench | `python scripts/eval/olmocr_bench.py --model outputs/<grpo_run>/`. Compare to SFT baseline (67.1%). | - | TODO |
| 5 | Eval on synthmix holdout | Run unit tests on held-out pages (if any) to check generalization vs overfitting to test set. | - | TODO |

**Expected outcome:** olmOCR-Bench improvement over SFT baseline. olmOCR 2 saw meaningful gains from GRPO on synthmix.

**Run command:**
```bash
python forge.py \
  --config configs/ocr/olmocr-mix/qwen3vl-2b/rl/01_synthmix.yaml \
  --adapter outputs/olmocr-mix_ablations_00_no-teacher_ce-only_20260225-0952Z/ \
  --run-name synthmix_grpo_01
```

#### 2b. GRPO on OmniDocBench (formatting-focused)

| # | Step | Details | Files to create/modify | Status |
|:-:|------|---------|----------------------|:------:|
| 1 | Add TEDS reward | Cell-level F1 on markdown tables. Uses rapidfuzz for cell matching. Returns 0.5 if no tables in GT. | `src/training/rewards.py` (`table_teds_reward`) | Done |
| 2 | Add CDM reward | Normalized edit distance on cleaned LaTeX ($...$ and $$...$$). Strips \left, \right, \displaystyle. Returns 0.5 if no formulas in GT. | `src/training/rewards.py` (`formula_cdm_reward`) | Done |
| 3 | Create config | `reward: [edit_similarity, format]` (add TEDS/CDM when ready), `dataset: omnidocbench`, `data_dir: data/ocr/omnidocbench` | `configs/ocr/olmocr-mix/qwen3vl-2b/rl/02_omnidocbench.yaml` | Done |
| 4 | Run GRPO | Train from synthmix GRPO output (stacked: Base -> SFT -> GRPO-synthmix -> GRPO-omnidoc). Or from SFT directly for comparison. | - | TODO |
| 5 | Ablate rewards | Compare reward combinations: `[edit_similarity]` only, `[edit_similarity, format]`, all four. Track which rewards drive which improvements. | - | TODO |
| 6 | Eval on both benchmarks | olmOCR-Bench (extraction) + OmniDocBench (formatting). Check if formatting GRPO hurts extraction or helps both. | - | TODO |

**Expected outcome:** Formatting quality improvement (headings, tables, formulas) without regressing extraction accuracy.

#### Reward Status

| Reward | Signal | Needed for | Status |
|--------|--------|-----------|:------:|
| `edit_similarity` | Levenshtein distance to ground truth | Phase 2b | Done |
| `format` | Markdown heading/list/bold/italic/table scoring | Phase 2b | Done |
| `unit_test` | olmOCR-style assertion pass rate (text present/absent/order/math) | Phase 2a | Done |
| `table_teds` | Cell-level F1 on extracted markdown tables | Phase 2b | Done |
| `formula_cdm` | Normalized edit distance on cleaned LaTeX formulas | Phase 2b | Done |
| `reading_order` | LCS ratio on matched text block sequences | Phase 2b | Done |

**Related work - FireRed-OCR** uses RL with: Formula Syntax, Table Integrity, Hierarchical Closure, Text Accuracy.

---

### Phase 3: Formatting Bench (OmniDocBench Eval)

Goal: Build an OmniDocBench eval script so we can measure formatting quality, not just extraction. This is one of the three IDP Leaderboard benchmarks.

| # | Step | Details | Files to create/modify | Status |
|:-:|------|---------|----------------------|:------:|
| 1 | Port OmniDocBench eval | Follow `olmocr_bench.py` pattern: load model via InferenceEngine, run inference on all 1,355 pages, save predictions as markdown. Support `--shard` for parallel runs. | `scripts/eval/omnidocbench.py` | TODO |
| 2 | Implement text scoring | Normalized edit distance between predicted text blocks and GT text blocks. Match blocks by reading order. | `scripts/eval/omnidocbench.py` or `src/eval/metrics.py` | TODO |
| 3 | Implement table scoring (TEDS) | Parse predicted markdown tables to HTML, compute tree edit distance against GT HTML tables. Need `apted` library. Two variants: TEDS (content + structure) and TEDS-Struct (structure only). | Same as above | TODO |
| 4 | Implement formula scoring (CDM) | Computational distance metric for LaTeX formulas. Parse to symbol tree, compare. Existing OmniDocBench repo has reference implementation. | Same as above | TODO |
| 5 | Implement reading order scoring | Extract block sequence from prediction (by position or explicit markers), compute edit distance against GT block order. | Same as above | TODO |
| 6 | Run SFT baseline | Evaluate `ablations_00` on OmniDocBench. This gives us the starting point before any GRPO. | - | TODO |
| 7 | Run GRPO models | Evaluate Phase 2a and 2b outputs. Compare extraction vs formatting tradeoffs. | - | TODO |
| 8 | Run API baselines | Evaluate GPT-5.2, Claude Sonnet 4.6, Gemini-3-Pro via API. Cross-reference with IDP Leaderboard numbers for validation. | - | TODO |

**OmniDocBench v1.5** ([CVPR 2025](https://github.com/opendatalab/OmniDocBench)): 1,355 PDF pages, 9 document types, 15 block-level types + 4 span-level types.

| Component | Metrics | What it measures | Dependency |
|-----------|---------|-----------------|------------|
| **Text** | Normalized Edit Distance | Plain text extraction accuracy | Built-in (Levenshtein) |
| **Formulas** | CDM, Edit Distance, BLEU | Structural equivalence | `latex2sympy` or port from OmniDocBench repo |
| **Tables** | TEDS, TEDS-Struct, Edit Distance | Table structure + content | `apted` (tree edit distance) |
| **Reading Order** | Normalized Edit Distance | Correct sequencing of text blocks | Built-in (Levenshtein) |

Three eval levels: end-to-end (our focus), recognition-only, detection-only.

**Integration path:** Keep our architecture (local inference via InferenceEngine, sharded runs). Adopt OmniDocBench metrics. New eval script `scripts/eval/omnidocbench.py` following `olmocr_bench.py` pattern. Both scripts share the same model loading and inference code, only dataset loading and scoring differ.

**Run command:**
```bash
# Local model
python scripts/eval/omnidocbench.py --model outputs/<run>/ --shard 1/4

# API model (for baselines)
python scripts/eval/omnidocbench.py --model claude-sonnet-4-6-20250514
```

**Success criteria:** Our eval numbers for API models should be within 1-2pp of IDP Leaderboard numbers. If not, debug scoring implementation.

---

## IDP Leaderboard Context

The [IDP Leaderboard](https://idp-leaderboard.org/) combines three benchmarks (unweighted mean):

| Benchmark | What it tests | Eval code |
|-----------|--------------|-----------|
| **olmOCR-Bench** | Text extraction: present/absent/order/math (7,019 pass/fail tests) | [allenai/olmocr](https://github.com/allenai/olmocr) |
| **OmniDocBench v1.5** | Document parsing: text, formulas (CDM), tables (TEDS), reading order | [opendatalab/OmniDocBench](https://github.com/opendatalab/OmniDocBench) |
| **IDP Core Bench** | Real-world tasks: KIE, handwriting OCR, VQA, table understanding | [NanoNets/docext](https://github.com/NanoNets/docext) |

### Current leaderboard (March 2026)

| Rank | Model | Overall | olmOCR | OmniDoc | IDP Core |
|------|-------|:-------:|:------:|:-------:|:--------:|
| 1 | Nanonets OCR2+ | 81.8 | 82.2 | 89.5 | 73.8 |
| 2 | Gemini-3-Pro | 81.4 | 73.5 | 88.8 | 81.8 |
| 3 | Claude Sonnet 4.6 | 80.8 | 74.4 | 86.9 | 81.2 |
| 4 | Claude Opus 4.6 | 80.3 | 73.9 | 85.9 | 81.1 |
| 5 | Gemini-3-Flash | 79.9 | 69.2 | 90.1 | 80.5 |
| 6 | GPT-5.2 | 79.2 | 72.2 | 88.0 | 77.4 |
| 7 | GPT-5-Mini | 70.8 | 56.7 | 82.5 | 73.3 |
| 8 | GPT-4.1 | 70.0 | 55.5 | 79.9 | 74.7 |
| 9 | Claude Haiku 4.5 | 69.6 | 56.2 | 79.6 | 72.9 |
| 10 | Ministral-8B | 69.3 | 57.8 | 78.3 | 71.7 |
| 11 | GLM-OCR | 63.6 | 66.7 | 69.2 | 54.9 |
| 12 | GPT-5-Nano | 50.7 | 22.8 | 63.4 | 65.8 |
| 13 | Llama-3.2-Vision-11B | 50.1 | 47.2 | 44.6 | 58.6 |
| 14 | Pixtral-12B | 46.0 | 36.8 | 42.3 | 59.0 |

**Our position:** ablations_00 at 67.1% olmOCR-Bench places between GPT-5.2 (72.2%) and GLM-OCR (66.7%). No small open-source models (2-3B) on the leaderboard yet.

---

## Code Comparison: docext vs Forge

| Aspect | docext (NanoNets) | Forge (ours) |
|--------|-------------------|--------------|
| **Model interface** | API-based via litellm | Local HuggingFace models via InferenceEngine |
| **Inference** | ThreadPoolExecutor, parallel API calls | Sequential GPU inference, sharded runs |
| **Caching** | SHA256 hash of messages | Markdown files on disk, resume by existence |
| **Table scoring** | GriTS (content, location, topology) | olmocr pass/fail on table text presence |
| **Parallelism** | Multi-threaded API calls | Sharded runs (--shard 1/4), parallel scoring |

**What docext has that we don't:** GriTS table scoring, cost tracking, multi-task eval (KIE, VQA, classification), litellm API abstraction.

**What we have that docext doesn't:** Local model inference (any HF VLM or LoRA), sharded execution, KaTeX math scoring (DOM-level LaTeX comparison), training integration (SFT + KD + GRPO in same codebase).

---

## References

- [IDP Leaderboard](https://idp-leaderboard.org/)
- [OmniDocBench GitHub](https://github.com/opendatalab/OmniDocBench)
- [OmniDocBench HuggingFace](https://huggingface.co/datasets/opendatalab/OmniDocBench)
- [OmniDocBench paper (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Ouyang_OmniDocBench_Benchmarking_Diverse_PDF_Document_Parsing_with_Comprehensive_Annotations_CVPR_2025_paper.pdf)
- [docext repo](https://github.com/NanoNets/docext)
- [FireRed-OCR](https://huggingface.co/FireRedTeam/FireRed-OCR)
- ["OmniDocBench is Saturated"](https://www.llamaindex.ai/blog/omnidocbench-is-saturated-what-s-next-for-ocr-benchmarks)
