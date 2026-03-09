# OCR Distillation: Datasets

Planning doc for OCR as a new task bucket in Forge.

---

## Task Overview

**Task type:** Vision-Language Generation (image → structured markdown)

**Reference model:** [Nanonets OCR-2](https://nanonets.com/research/nanonets-ocr-2/) — Qwen2.5-VL-3B fine-tuned on 3M+ document pages.

**Applicable distillation signals:**
- **Response (R):** Token-level KL on autoregressive markdown output (primary)
- **Attention (A):** Cross-attention maps between vision and text tokens (spatial grounding)
- **Feature (F):** Hidden state alignment with projection (optional, lower priority)

**LoRA in Forge:** For generation tasks, LMs are fine-tuned with LoRA (PEFT) as the preferred method — models at 1B+ parameters need parameter-efficient tuning to fit on a single GPU. The framework is modular: the distillation loss follows Hinton et al. (2015) as `(1-α)·CE + α·KL`, and SFT is simply the degenerate case where α=0 (no teacher). Classification models (66M–110M params) are small enough for full fine-tuning without LoRA.

---

## Public Datasets

### Ready-to-Use (Image → Markdown Pairs)

| Dataset | Size | Format | Notes |
|---------|------|--------|-------|
| [davanstrien/vlm-ocr](https://huggingface.co/datasets/davanstrien/vlm-ocr) | TBD | image + markdown | Generated using Nanonets-OCR-s; closest to target task |
| [lightonai/fc-amf-ocr](https://huggingface.co/datasets/lightonai/fc-amf-ocr) | TBD | PDF + JSON annotations | OCR-heavy pretraining data |

### Evaluation Benchmarks

| Dataset | Task | Used By |
|---------|------|---------|
| [OmniDocBench](https://github.com/opendatalab/OmniDocBench) | Document parsing (CVPR 2025) | Comprehensive benchmark |
| [DocVQA](https://www.docvqa.org/) | Visual question answering | Nanonets eval (85.15% on OCR2+) |
| [ChartQA](https://github.com/vis-nlp/ChartQA) | Chart understanding | Nanonets eval (79.20% on OCR2+) |

### Build-Your-Own (Synthetic)

**Approach:** Render documents from known markdown source → perfect ground truth.

- Author markdown with tables, LaTeX, headings, lists
- Render to PDF/image (e.g., via Pandoc, browser, LaTeX compiler)
- Pair: rendered image → source markdown

**Advantages:** Infinite scale, perfect labels, full control over complexity.

**Alternatively:** Use a large teacher (Qwen2.5-VL-72B, Gemini) to annotate real PDFs → silver labels for distillation.

---

## Framework Requirements

What's already supported vs what needs work:

| Component | Status | Notes |
|-----------|--------|-------|
| Causal LM pipeline | Supported | Translation task uses this |
| LoRA / PEFT | Supported | `use_lora: true` in config |
| SFT (CE-only) | Supported | Drop distillation signals |
| R distillation (KL on logits) | Supported | Token-level KL |
| F distillation (hidden states) | Supported | With projection layers |
| A distillation (attention maps) | Supported | Cross-attention TBD |
| Vision encoder input | **Not supported** | Qwen2.5-VL needs image tensor handling |
| Image data collator | **Not supported** | Data loading needs image preprocessing |
| New configs (`configs/ocr/`) | **Needed** | Model paths, LoRA params, dataset config |

**Key gap:** The vision encoder pathway. Current causal LM pipeline is text-in/text-out. VLMs like Qwen2.5-VL require image inputs, which means the data collator and model input preparation need to handle image tensors. The distillation logic itself (losses, optimizer, training loop) is task-agnostic and works as-is.

---

## Data Preparation: `scripts/data/prepare_ocr_data.py`

**Date:** 2026-02-25

### Problem

olmOCR-mix-1025 stores PDFs in tarballs, not images. We need to render 260K PDF pages to images and build an HF dataset with `Image()` + `text` columns for training. Naive approaches OOM:

| Approach | Peak Memory | Why |
|----------|------------|-----|
| `Dataset.from_dict()` with PIL images | ~890 GB | 231K × 1288×996×3 = ~3.85 MB/image uncompressed |
| Per-tarball `from_dict` + concatenate | ~890 GB | Arrow tables still accumulate in-memory |
| Per-tarball `save_to_disk` + reload | OOM at tarball 8 | `from_dict()` holds PIL + Arrow simultaneously |

### Solution: Two-Phase Pipeline

**Phase 1 — Render PDFs to PNG files on disk** (parallel, O(1) memory)
- Workers call `pdftoppm` → base64 PNG → decode → write bytes to disk
- No PIL objects ever created. Peak memory: ~400 MB (Python + HF metadata)

**Phase 2 — Build HF dataset** (streaming, O(1) memory)
- `Dataset.from_generator()` reads JSONL + PNG files one at a time
- Writes Arrow incrementally. Never holds more than one image in memory.

### Pipeline Optimization (v2)

The serial version spent ~60% of wall time on I/O (download, extract, cleanup) with CPUs idle. The pipelined version overlaps I/O with compute:

```
Serial (v1):          Download → Extract → Render → Cleanup → Download → Extract → ...
                      [idle CPUs]          [busy]  [idle]

Pipelined (v2):       Download ALL (threaded)
                      Extract[0] → Render[0] + Extract[1] → Render[1] + Extract[2] → ...
                                   Cleanup[0] (bg thread)   Cleanup[1] (bg thread)
```

Three optimizations:

1. **Pre-download** — `ThreadPoolExecutor(4)` downloads all tarballs upfront via `hf_hub_download`. Saturates network bandwidth. Saves ~26 min (52 tarballs × 30s each, 4 threads).

2. **Double-buffer extract/render** — Two alternating extraction dirs (`_extract_a`, `_extract_b`). While `ProcessPoolExecutor(64)` renders tarball N from buffer A, a background thread extracts tarball N+1 into buffer B. Saves ~26 min.

3. **Background cleanup** — `shutil.rmtree` runs in a thread per buffer, never blocks the next render. Per-buffer tracking prevents race conditions between cleanup and prefetch. Saves ~13 min.

| Metric | Serial (v1) | Pipelined (v2) | Speedup |
|--------|-------------|----------------|---------|
| Phase 1 time (est.) | ~109 min | ~46 min | 2.4x |
| Phase 2 time | ~11 min | ~11 min | 1x |
| Peak memory | ~400 MB | ~400 MB | Same |
| CPU utilization | ~40% | ~95% | 2.4x |

### Complexity

| | Time | Memory |
|--|------|--------|
| Phase 1 | O(N/W) where N=pages, W=workers | O(1) — one tarball's PDFs + PNGs on disk |
| Phase 2 | O(N) — streaming I/O | O(1) — one image at a time |
| Disk | | O(N) during staging, cleaned up after |

### Usage

```bash
# Full 260K run (all 4 subsets)
sbatch scripts/slurm/run_data_prep.sh olmocr-mix --output data/ocr/olmocr-mix-full

# Test run (50 samples per subset)
sbatch scripts/slurm/run_data_prep.sh olmocr-mix --output data/ocr/olmocr-mix-test --max_samples 50

# Single subset
python scripts/data/prepare_ocr_data.py --subset 00_documents --output data/ocr/olmocr-mix-5k --max_samples 5000
```

---

## Evaluation: olmOCR-Bench

**Date:** 2026-02-25

### Context

Phase 1 SFT training (Qwen3-VL-2B on olmOCR-mix-1025, 5K samples) converged — loss 0.189→0.08 over 3 epochs. But eval is loss-only. We need a proper benchmark to measure actual OCR output quality and compare student vs teacher with all variables under our control.

### Benchmark

[olmOCR-Bench](https://huggingface.co/datasets/allenai/olmOCR-bench) (Allen AI): 7,010 pass/fail unit tests across 1,403 PDFs. Five categories:

| Category | Tests | What it checks |
|----------|-------|----------------|
| Text Presence | 721 | Are key sentences in the output? |
| Text Absence | 823 | Are headers/footers/page numbers excluded? |
| Reading Order | 1,061 | Is paragraph A before paragraph B? |
| Table Accuracy | 1,020 | Are cell values + neighbors correct? |
| Math Formulas | 3,385 | Are equations rendered correctly? |

Score = % of tests passed. Published top scores: ~82-83%.

### Models to Evaluate

| Model | HF ID | Architecture | Why |
|-------|-------|-------------|-----|
| olmOCR-2 7B (teacher) | `allenai/olmOCR-2-7B-1025-FP8` | Qwen2.5-VL-7B | Generated our training data. Our own measurement, not published numbers. |
| Qwen3-VL-2B pretrained | `Qwen/Qwen3-VL-2B-Instruct` | Qwen3-VL-2B | Student baseline before any training |
| Qwen3-VL-2B fine-tuned | Checkpoint from training run | Qwen3-VL-2B + LoRA | Student after Phase 1 SFT on 5K samples |

All three run through the same pipeline — same rendering, prompting, scoring.

### Implementation

**Script:** `scripts/eval/olmocr_bench.py`

```
python scripts/eval/olmocr_bench.py --model allenai/olmOCR-2-7B-1025-FP8
python scripts/eval/olmocr_bench.py --model Qwen/Qwen3-VL-2B-Instruct
python scripts/eval/olmocr_bench.py --checkpoint outputs/.../checkpoint-400
```

**Args:**
- `--model`: HuggingFace model ID (any VLM)
- `--checkpoint`: Local LoRA checkpoint path (mutually exclusive with --model)
- `--output-dir`: Override (default: `outputs/bench/olmocr/{name}/{YYYYMMDD-HHMMZ}/`)
- `--bench-dir`: Path to olmOCR-bench data (default: `data/olmocr-bench`)
- `--max-new-tokens`: Generation limit (default: 2048)
- `--dpi`: PDF render DPI (default: 150)

**Output dir naming:** `outputs/bench/olmocr/{name}/{timestamp}/`
- `--model allenai/olmOCR-2-7B-1025-FP8` → `outputs/bench/olmocr/olmOCR-2-7B-1025-FP8/20260225-1430Z/`
- `--model Qwen/Qwen3-VL-2B-Instruct` → `outputs/bench/olmocr/Qwen3-VL-2B-Instruct/20260225-1445Z/`
- `--checkpoint outputs/run_name_0952Z/checkpoint-400` → `outputs/bench/olmocr/run_name_0952Z/20260225-1500Z/`

**Output structure:**
```
outputs/bench/olmocr/{name}/{timestamp}/
  metadata.json    # model_id, hf_commit_hash, generation_params, prompt, dpi, timestamp
  markdowns/       # {pdf_stem}_{page}.md per (pdf, page) pair
  results.json     # overall score, per-category breakdown
```

**Flow:**
1. Parse args, derive output dir with UTC timestamp
2. Load olmOCR-bench data (JSONL test cases + PDF paths)
3. Load model + processor (AutoModel or PeftModel from checkpoint)
4. Capture HF commit hash for reproducibility
5. For each unique (pdf, page): render PDF → image → generate markdown → save
6. Score all tests using `olmocr.bench.tests` classes
7. Save results.json + metadata.json, print summary

**Dependencies:**
```bash
pip install olmocr  # for bench scoring classes
# Already have: pypdfium2, torch, transformers, peft
```

**Data setup:**
```bash
huggingface-cli download --repo-type dataset allenai/olmOCR-bench --local-dir data/olmocr-bench
```

**SLURM wrapper:** `scripts/slurm/run_bench.sh olmocr`

### Runtime Estimate
- 1,403 pages × ~5s/page = ~2 hours per model on H200
- Scoring: minutes (CPU-only)
- Total for 3 models: ~6 hours (parallelizable as 3 SLURM jobs)

### Verification
1. olmOCR-2-7B → should land near published 82.4 (validates pipeline)
2. Pretrained Qwen3-VL-2B → student baseline
3. Fine-tuned Qwen3-VL-2B → should score higher than pretrained
4. Transfer efficiency = (student_post - student_pre) / (teacher - student_pre)
5. Per-category breakdown → identify where distillation helps most
