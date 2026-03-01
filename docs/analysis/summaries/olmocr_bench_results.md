# olmOCR-Bench Results (2026-02-28)

Evaluation on [olmOCR-Bench](https://huggingface.co/datasets/allenai/olmOCR-bench): 7,019 tests across 7 categories, 1,403 unique PDF pages. See [olmOCR GitHub](https://github.com/allenai/olmocr) and [olmOCR demo](https://olmocr.allenai.org/) for benchmark details.

## Models

| Model | Role | Size | Base VLM | Source |
|-------|------|------|----------|--------|
| olmOCR-2-7B-1025-FP8 | Baseline | 7B (FP8) | Qwen2.5-VL-7B | `allenai/olmOCR-2-7B-1025-FP8` |
| Nanonets-OCR2-3B | Baseline | 3B | Qwen2.5-VL-3B (likely) | `Nanonets/Nanonets-OCR-2-3B` |
| ablations_00 (CE-only) | Fine-tuned (CE loss only) | 2B | Qwen3-VL-2B | Fine-tuned, 5K samples (of 260K) |
| Qwen3-VL-2B-Instruct | Base model (pre-fine-tune) | 2B | Qwen3-VL-2B | `Qwen/Qwen3-VL-2B-Instruct` |

## ablations_00 Fine-Tuning Config

| Parameter | Value |
|-----------|-------|
| Base model | `Qwen/Qwen3-VL-2B-Instruct` |
| LoRA rank (r) | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| LoRA targets | q/k/v/o_proj, gate/up/down_proj |
| Learning rate | 5e-5 |
| Epochs | 3 |
| Batch size | 2 (x16 grad accum = 32 effective) |
| Warmup steps | 50 |
| Dataset | [olmocr-mix](https://huggingface.co/datasets/allenai/olmOCR-mix-0225) (PDF→markdown, GPT-4o-generated labels) |
| Training samples | 5,000 (of ~260K available) |
| Max sequence length | 1024 |
| Distillation alpha | 0.0 (CE loss only, no KD) |

## Scoring Methodology

Each test in olmOCR-Bench checks one of 4 test types against the model's markdown output:

| Test type | What it checks | Used in |
|-----------|---------------|---------|
| `present` | Required text IS in output | long_tiny_text, old_scans |
| `absent` | Text should NOT be in output (e.g. headers/footers) | headers_footers, old_scans |
| `order` | Text A appears before text B (reading order) | multi_column, old_scans |
| `math` | LaTeX formula matches exactly | arxiv_math, old_scans_math |

Scoring uses `olmocr.bench.tests` from the [olmocr](https://github.com/allenai/olmocr) library. Each test returns pass/fail with an explanation.

**Two aggregate metrics:**
- **Aggregate**: total passes / total tests (7,019). Penalized heavily by the 2 math categories (3,385 tests, all 0%).
- **Per-JSONL Average**: average of per-category pass rates. Each category weighted equally regardless of test count.

## Results by Category

| Category | Tests | olmOCR-7B | Nanonets-3B | ablations_00 | Qwen3-VL-2B (base) |
|----------|:-----:|:---------:|:-----------:|:------------:|:--------------------:|
| arxiv_math | 2,927 | 0.0% | 0.0% | 0.0% | 0.0% |
| headers_footers | 760 | **92.6%** | 21.3% | 79.6% | 46.3% |
| long_tiny_text | 442 | **69.7%** | 65.6% | 59.3% | 49.5% |
| multi_column | 884 | **82.6%** | 80.2% | 76.4% | 70.1% |
| old_scans | 526 | **48.1%** | 38.0% | 39.2% | 38.0% |
| old_scans_math | 458 | 0.0% | 0.0% | 0.0% | 0.0% |
| table_tests | 1,022 | **83.5%** | 82.5% | 72.2% | 61.5% |
| **Aggregate (all)** | 7,019 | **40.6%** | 31.4% | 35.4% | 28.8% |
| **Per-JSONL Average (all)** | 7 | **53.8%** | 41.1% | 46.7% | 37.9% |
| **Per-JSONL Average (non-math)** | 5 | **75.3%** | 57.5% | 65.3% | 53.1% |

## Per-Category Analysis

**arxiv_math + old_scans_math (0% all models):** Tests expect exact LaTeX strings like `\mathcal{V}(\psi_m)\rightarrow +\infty`. Models output Unicode/plain text like `V(ψₘ) → +∞`. This is a format mismatch, not an extraction failure. All 3,385 math tests fail for every model.

**headers_footers (92.6% olmOCR vs 21.3% Nanonets):** These are `absent` tests, meaning pass = text NOT in output. olmOCR-7B was trained to strip headers/footers as metadata. Nanonets extracts everything verbatim, so it includes headers and fails. ablations_00 (79.6%) learned partial header filtering from training data. Qwen3-VL-2B base (46.3%) has some filtering from general instruction tuning.

**multi_column (82.6% olmOCR vs 70.1% base):** `order` tests checking that reading order is preserved across columns. All models do reasonably well since modern VLMs handle column reading order natively.

**table_tests (83.5% olmOCR vs 61.5% base):** olmOCR-7B was specifically trained on table extraction. Fine-tuning on 5K samples brought base from 61.5% to 72.2% (+10.7pp). Nanonets (82.5%) is also strong here.

**long_tiny_text (69.7% olmOCR vs 49.5% base):** `present` tests checking exact text extraction from small-font or long passages. Larger model + OCR-specific training helps. ablations_00 (59.3%) improved +9.8pp over base.

**old_scans (48.1% olmOCR vs 38.0% base):** Mixed test types on degraded/noisy scans. Low scores across the board. OCR-specific training provides the biggest advantage here.

## Gap Analysis: CE-only Recovery vs KD Opportunity

How much of the base-to-olmOCR gap did CE-only fine-tuning close, and how much remains for KD?

| Category | Base | ablations_00 | olmOCR-7B | CE Gain | Remaining Gap | Recovery |
|----------|:----:|:------------:|:---------:|:-------:|:-------------:|:--------:|
| headers_footers | 46.3% | 79.6% | 92.6% | +33.3pp | 13.0pp | 71.9% |
| table_tests | 61.5% | 72.2% | 83.5% | +10.7pp | 11.3pp | 48.6% |
| long_tiny_text | 49.5% | 59.3% | 69.7% | +9.8pp | 10.4pp | 48.5% |
| multi_column | 70.1% | 76.4% | 82.6% | +6.3pp | 6.2pp | 50.4% |
| old_scans | 38.0% | 39.2% | 48.1% | +1.2pp | 8.9pp | 11.9% |
| **Non-math avg** | 53.1% | 65.3% | 75.3% | +12.2pp | 10.0pp | 55.0% |

**Recovery** = CE Gain / (olmOCR-7B - Base). How much of the 7B's advantage CE-only training already captured.

**Where CE worked well:**
- headers_footers (72% recovery): Header/footer stripping is a formatting convention. The training data's prompt+completion format teaches this directly through hard labels.

**Where CE barely helped:**
- old_scans (12% recovery): Only +1.2pp over base. Degraded image understanding requires richer signal than token-level CE. The 7B model's internal representations for handling noise, faded text, and irregular layouts are not transferred through hard labels alone.

**Where KD has the most room:**
- old_scans (8.9pp remaining, 12% recovery) - strongest case for soft-label transfer
- table_tests (11.3pp remaining, 49% recovery) - structured output benefits from distribution-level guidance
- long_tiny_text (10.4pp remaining, 49% recovery) - fine-grained text recognition

Overall, CE-only recovered 55% of the base-to-7B gap. The remaining 10pp is the target for KD.

## Experimental Axes

Four axes define the comparison space:

| Axis | Values | Coverage |
|------|--------|----------|
| **1. Base VLM** | Qwen2.5-VL, Qwen3-VL | Architecture generation |
| **2. Param count** | 2B, 3B, 7B | Model capacity |
| **3. Dataset size** | 0, 5K, full (~260K) | Data scaling |
| **4. Training method** | None, CE (SFT), CE + KD | Distillation signal |

### Current data points

| Experiment | Base | Params | Data | Method | Non-math Avg |
|------------|------|:------:|:----:|--------|:------------:|
| Qwen3-VL-2B (base) | Qwen3-VL | 2B | 0 | None | 53.1% |
| ablations_00 | Qwen3-VL | 2B | 5K | CE | 65.3% |
| Qwen2.5-VL-3B (base) | Qwen2.5-VL | 3B | 0 | None | *pending* |
| Nanonets-3B | Qwen2.5-VL | 3B | ? | CE | 57.5%\* |
| Qwen2.5-VL-7B (base) | Qwen2.5-VL | 7B | 0 | None | *pending* |
| olmOCR-7B | Qwen2.5-VL | 7B | full | CE + RL | 75.3% |

\* Nanonets non-math average is deflated by headers_footers (21.3%) due to different design goal (verbatim extraction).

### What the pending baselines will isolate

Once Qwen2.5-VL-3B and 7B base results come in:

- **Base VLM effect** (axis 1): Qwen3-VL-2B base vs Qwen2.5-VL-3B base. Is Qwen3-VL's 2B competitive with Qwen2.5-VL's 3B despite fewer params?
- **Param count effect** (axis 2): Qwen2.5-VL-3B base vs Qwen2.5-VL-7B base. How much does 3B→7B gain without any OCR training?
- **Training effect** (axis 4): Qwen2.5-VL-7B base vs olmOCR-7B. How much of olmOCR's score comes from the base VLM vs OCR-specific training?
- **Nanonets decomposition**: Qwen2.5-VL-3B base vs Nanonets-3B. Nanonets' training gain, isolated from the base model.

### Planned experiments

| Experiment | Base | Params | Data | Method | What it tests |
|------------|------|:------:|:----:|--------|---------------|
| KD ablation | Qwen3-VL | 2B | 5K | CE + KD | Axis 4: does KD close the remaining 10pp gap? |
| Data scaling | Qwen3-VL | 2B | 25K, 50K | CE | Axis 3: diminishing returns on more data? |
| KD + more data | Qwen3-VL | 2B | 25K+ | CE + KD | Axes 3+4: interaction of data size and KD |

## Key Takeaways

1. **Fine-tuning impact**: Just 5K samples with CE-only improved non-math average from 53.1% to 65.3% (+12.2pp). Biggest gains in headers_footers (+33.3pp) and table_tests (+10.7pp).

2. **ablations_00 vs olmOCR-7B**: Non-math average gap is 10.0pp (65.3% vs 75.3%). The 2B model retains ~87% of olmOCR-7B quality with 3.5x fewer parameters.

3. **Math categories are not useful for comparison**: All models score 0% due to format mismatch (confirmed). Tests expect raw LaTeX like `\mathcal{V}`, models output Unicode like `V`. The non-math Per-JSONL Average (5 categories) is the fairer metric.

4. **Nanonets design trade-off**: Strong on content extraction (tables 82.5%, multi_column 80.2%) but doesn't filter metadata (headers_footers 21.3%). Different design goal than olmOCR.

5. **Base VLM lineage**: olmOCR-7B and Nanonets-3B are both built on Qwen2.5-VL. Our ablations_00 uses the newer Qwen3-VL-2B, making cross-architecture comparison informative.

## References

- [olmOCR GitHub](https://github.com/allenai/olmocr) - toolkit, scoring code, benchmark definitions
- [olmOCR paper](https://arxiv.org/abs/2502.18443) - "olmOCR: Unlocking Trillions of Tokens in PDFs with Vision Language Models"
- [olmOCR 2 paper](https://arxiv.org/abs/2510.19817) - "olmOCR 2: Unit Test Rewards for Document OCR"
- [olmOCR-Bench dataset](https://huggingface.co/datasets/allenai/olmOCR-bench) - 7,019 tests across 1,403 PDF pages
- [olmocr-mix training data](https://huggingface.co/datasets/allenai/olmOCR-mix-0225) - ~260K PDF pages with GPT-4o-generated labels

## Next Steps

- Run KD-trained model (with distillation loss from olmOCR-7B) and compare against ablations_00 (CE-only)
- Benchmark Qwen2.5-VL-3B and Qwen2.5-VL-7B as additional baselines (in progress, 4 shards each)
- Add Qwen2.5-VL results to this document once scoring completes
