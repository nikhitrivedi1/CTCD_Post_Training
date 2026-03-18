# olmOCR-Bench Results

Evaluation on [olmOCR-Bench](https://huggingface.co/datasets/allenai/olmOCR-bench): 7,019 tests across 7 categories, 1,403 unique PDF pages.

## Methodology

Each test checks one of 4 types against the model's markdown output:

| Test type | What it checks | Used in |
|-----------|---------------|---------|
| `present` | Required text IS in output | long_tiny_text, old_scans |
| `absent` | Text should NOT be in output (headers/footers) | headers_footers, old_scans |
| `order` | Text A appears before text B (reading order) | multi_column, old_scans |
| `math` | LaTeX matches via KaTeX DOM comparison | arxiv_math, old_scans_math |

Math scoring renders both reference and model LaTeX with KaTeX in headless Chromium, then compares relative symbol-level bounding box positions. Not exact string matching.

**Aggregate metrics:**
- **Overall**: average of all 7 category pass rates (primary metric)
- **Non-math Avg**: average of 5 non-math categories
- **Extraction Avg (w/o H/F)**: average of 4 extraction categories, excluding headers_footers (which measures a convention, not extraction capability)

**Script**: `scripts/benchmarks/olmocr_bench.py`

## Results by Category

> **NOTE**: Q3.5-* columns (0.8B, 2B, 4B, 9B) are **INVALID** - pending re-run.
> Qwen3.5 models generate `<think>` blocks; `<think>` is stripped as a special token but the
> thinking content and `</think>` tag survive as plain text in the output, corrupting scoring.
> All other columns (Q2.5-*, Nano-3B, olmOCR-7B, Q3-*, ocr_sft, fmt_sft) are clean.

| Category | Tests | Q2.5-3B | Nano-3B | Q2.5-7B | olmOCR-7B | Q3-2B | ocr_sft | fmt_sft | Q3-4B | Q3-8B | ~~Q3.5-0.8B~~ | ~~Q3.5-2B~~ | ~~Q3.5-4B~~ | ~~Q3.5-9B~~ |
|----------|:-----:|:-------:|:-------:|:-------:|:---------:|:-----:|:-------:|:-------:|:-----:|:-----:|:---------:|:-------:|:-------:|:-------:|
| arxiv_math | 2,927 | 72.4 | 55.6 | 74.2 | **85.8** | 77.3 | 77.9 | 77.3 | 79.6 | 82.9 | ~~69.8~~ | ~~76.6~~ | ~~61.0~~ | ~~66.7~~ |
| headers_footers | 760 | 65.5 | 21.3 | 52.9 | **92.6** | 46.3 | 79.6 | 75.9 | 37.5 | 39.5 | ~~64.6~~ | ~~40.3~~ | ~~33.2~~ | ~~29.6~~ |
| long_tiny_text | 442 | 46.8 | 65.6 | 65.8 | **69.7** | 49.5 | 59.3 | 57.2 | 52.7 | 59.3 | ~~34.2~~ | ~~57.9~~ | ~~36.0~~ | ~~44.3~~ |
| multi_column | 884 | 62.9 | 80.2 | 75.5 | **82.6** | 70.1 | 76.4 | 74.4 | 74.5 | 78.6 | ~~56.9~~ | ~~73.9~~ | ~~64.0~~ | ~~67.4~~ |
| old_scans | 526 | 35.5 | 38.0 | 44.9 | **48.1** | 38.0 | 39.2 | 35.7 | 38.6 | 45.2 | ~~30.8~~ | ~~37.8~~ | ~~36.3~~ | ~~40.1~~ |
| old_scans_math | 458 | 70.5 | 34.5 | 71.2 | **78.6** | 49.8 | 65.1 | 67.7 | 67.2 | 73.6 | ~~62.7~~ | ~~71.4~~ | ~~62.0~~ | ~~64.8~~ |
| table_tests | 1,022 | 69.7 | 82.5 | 77.1 | **83.5** | 61.5 | 72.2 | 75.5 | 80.5 | 80.0 | ~~54.4~~ | ~~61.1~~ | ~~57.0~~ | ~~55.8~~ |

### Aggregates

| Metric | Q2.5-3B | Nano-3B | Q2.5-7B | olmOCR-7B | Q3-2B | ocr_sft | fmt_sft | Q3-4B | Q3-8B | ~~Q3.5-0.8B~~ | ~~Q3.5-2B~~ | ~~Q3.5-4B~~ | ~~Q3.5-9B~~ |
|--------|:-------:|:-------:|:-------:|:---------:|:-----:|:-------:|:-------:|:-----:|:-----:|:---------:|:-------:|:-------:|:-------:|
| **Overall** | 60.5 | 54.0 | 65.9 | **77.3** | 56.1 | 67.1 | 66.3 | 61.5 | 65.6 | ~~53.3~~ | ~~59.9~~ | ~~49.9~~ | ~~52.7~~ |
| **Non-math Avg** | 56.1 | 57.5 | 63.2 | **75.3** | 53.1 | 65.3 | 63.7 | 56.8 | 60.5 | ~~48.2~~ | ~~54.2~~ | ~~45.3~~ | ~~47.5~~ |
| **Extraction Avg** | 53.7 | **66.6** | 65.8 | **71.0** | 54.8 | 61.8 | 60.7 | 61.6 | 65.8 | ~~44.1~~ | ~~57.7~~ | ~~48.3~~ | ~~51.9~~ |

## Models

| Short name | Full model | Backbone | Base | Size | Training |
|------------|-----------|----------|------|:----:|----------|
| Q2.5-3B | Qwen2.5-VL-3B-Instruct | Qwen2.5-VL | - | 3B | Base |
| Nano-3B | Nanonets-OCR2-3B | Qwen2.5-VL | Q2.5-3B | 3B | SFT |
| Q2.5-7B | Qwen2.5-VL-7B-Instruct | Qwen2.5-VL | - | 7B | Base |
| olmOCR-7B | olmOCR-2-7B-1025-FP8 | Qwen2.5-VL | Q2.5-7B | 7B | SFT + RL |
| Q3-2B | Qwen3-VL-2B-Instruct | Qwen3-VL | - | 2B | Base |
| ocr_sft | ablations_00 (CE-only) | Qwen3-VL | Q3-2B | 2B | SFT (5K olmocr-mix) |
| fmt_sft | fmt_sft | Qwen3-VL | ocr_sft | 2B | SFT (CE) on formatting-rich synthetic pages |
| Q3-4B | Qwen3-VL-4B-Instruct | Qwen3-VL | - | 4B | Base |
| Q3-8B | Qwen3-VL-8B-Instruct | Qwen3-VL | - | 8B | Base |
| Q3.5-0.8B | Qwen3.5-0.8B | Qwen3.5 | - | 0.8B | Base |
| Q3.5-2B | Qwen3.5-2B | Qwen3.5 | - | 2B | Base |
| Q3.5-4B | Qwen3.5-4B | Qwen3.5 | - | 4B | Base |
| Q3.5-9B | Qwen3.5-9B | Qwen3.5 | - | 9B | Base |

## Key Observations

**Training beats scale.** ocr_sft (2B, 5K CE, 67.1%) beats Q3-VL-8B base (65.6%) and Q2.5-VL-7B base (65.9%). 5K samples on the smallest model outperforms untrained models with 4x the params.

**CE recovered 52% of the base-to-olmOCR gap.** Q3-VL-2B base (56.1%) to ocr_sft (67.1%) = +11.0pp. Remaining 10.2pp gap to olmOCR-7B (77.3%) is the target for RL/KD.

**Qwen3.5 scaling is non-monotonic** *(pending re-run - current numbers contaminated)*. Peaks at 2B (59.9%), regresses at 4B (49.9%), partially recovers at 9B (52.7%). Mixed attention architecture trades visual quality for efficiency, worsening with scale.

**headers_footers shows inverse scaling.** Larger base models score worse (absent tests - passing means NOT including metadata). Fine-tuning reverses this: olmOCR-7B (92.6%), ocr_sft (79.6%).

**Nanonets is extraction-strong but overall-weak.** Best extraction avg (66.6%) but weak math (55.6% arxiv, 34.5% old_scans_math) and no header filtering (21.3%) drag overall down.

**Formatting SFT trades content for structure.** fmt_sft (66.3%) regresses -0.8pp overall from ocr_sft (67.1%). The tradeoff is uneven: table_tests improves +3.3pp (72.2% to 75.5%) and old_scans_math +2.6pp (65.1% to 67.7%), while old_scans regresses -3.5pp and headers_footers -3.7pp. The formatting training teaches structured output (tables, math) but slightly hurts raw text extraction on degraded inputs.

## Pending

- **fmt_grpo**: GRPO with formatting rewards, olmOCR-Bench scores pending (3 shards planned)
