# olmOCR-Bench Results (2026-03-04)

Evaluation on [olmOCR-Bench](https://huggingface.co/datasets/allenai/olmOCR-bench): 7,019 tests across 7 categories, 1,403 unique PDF pages. See [olmOCR GitHub](https://github.com/allenai/olmocr) and [olmOCR demo](https://olmocr.allenai.org/) for benchmark details.

## Models

| Model | Role | Size | Base VLM | Source |
|-------|------|------|----------|--------|
| Qwen2.5-VL-3B-Instruct | Base model | 3B | Qwen2.5-VL-3B | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Nanonets-OCR2-3B | Fine-tuned | 3B | Qwen2.5-VL-3B | `Nanonets/Nanonets-OCR-2-3B` |
| Qwen2.5-VL-7B-Instruct | Base model | 7B | Qwen2.5-VL-7B | `Qwen/Qwen2.5-VL-7B-Instruct` |
| olmOCR-2-7B-1025-FP8 | Fine-tuned | 7B (FP8) | Qwen2.5-VL-7B | `allenai/olmOCR-2-7B-1025-FP8` |
| Qwen3-VL-2B-Instruct | Base model (pre-fine-tune) | 2B | Qwen3-VL-2B | `Qwen/Qwen3-VL-2B-Instruct` |
| ablations_00 (CE-only) | Fine-tuned (CE loss only) | 2B | Qwen3-VL-2B | Fine-tuned, 5K samples (of 260K) |
| Qwen3-VL-4B-Instruct | Base model | 4B | Qwen3-VL-4B | `Qwen/Qwen3-VL-4B-Instruct` |
| Qwen3-VL-8B-Instruct | Base model | 8B | Qwen3-VL-8B | `Qwen/Qwen3-VL-8B-Instruct` |
| Qwen3.5-0.8B | Base model | 0.8B | Qwen3.5 | `Qwen/Qwen3.5-0.8B` |
| Qwen3.5-2B | Base model | 2B | Qwen3.5 | `Qwen/Qwen3.5-2B` |
| Qwen3.5-4B | Base model | 4B | Qwen3.5 | `Qwen/Qwen3.5-4B` |
| Qwen3.5-9B | Base model | 9B | Qwen3.5 | `Qwen/Qwen3.5-9B` |

**Qwen3.5 architecture notes:** Qwen3.5 models are natively multimodal VLMs (released 2026-03-02) with built-in Vision Transformers. Key differences from Qwen3-VL: mixed attention (linear + full attention layers) for efficiency, 248K vocab (vs 152K), 262K context (vs 32K), no DeepStack. ViT depths: 0.8B=12, 2B=24, 4B=24, 9B=27. Note that 2B and 4B share the same ViT depth despite different LM sizes.

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

**Three aggregate metrics:**
- **Aggregate**: total passes / total tests (7,019). Penalized heavily by the 2 math categories (3,385 tests, all 0%).
- **Per-JSONL Average (non-math)**: average of 5 non-math category pass rates. Fairer than Aggregate since all models score 0% on math (format mismatch).
- **Extraction Avg. (w/o H/F)**: average of 4 extraction categories (long_tiny_text, multi_column, old_scans, table_tests), excluding headers_footers which measures a convention (strip metadata) rather than extraction capability.

## Results by Category

| Category | Tests | Qwen2.5-VL-3B | Nanonets-3B | Qwen2.5-VL-7B | olmOCR-7B | Qwen3-VL-2B | ablations_00 | Qwen3-VL-4B | Qwen3-VL-8B | Qwen3.5-0.8B | Qwen3.5-2B | Qwen3.5-4B | Qwen3.5-9B |
|----------|:-----:|:-------------:|:-----------:|:-------------:|:---------:|:-----------:|:------------:|:-----------:|:-----------:|:------------:|:----------:|:----------:|:----------:|
| arxiv_math | 2,927 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| headers_footers | 760 | 65.5% | 21.3% | 52.9% | **92.6%** | 46.3% | 79.6% | 37.5% | 39.5% | 64.6% | 40.3% | 33.2% | 29.6% |
| long_tiny_text | 442 | 46.8% | 65.6% | 65.8% | **69.7%** | 49.5% | 59.3% | 52.7% | 59.3% | 34.2% | 57.9% | 36.0% | 44.3% |
| multi_column | 884 | 62.9% | 80.2% | 75.5% | **82.6%** | 70.1% | 76.4% | 74.5% | 78.6% | 56.9% | 73.9% | 64.0% | 67.4% |
| old_scans | 526 | 35.6% | 38.0% | 44.9% | **48.1%** | 38.0% | 39.2% | 38.6% | 45.2% | 30.8% | 37.8% | 36.3% | 40.1% |
| old_scans_math | 458 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| table_tests | 1,022 | 69.7% | 82.5% | 77.1% | **83.5%** | 61.5% | 72.2% | 80.5% | 80.0% | 54.4% | 61.2% | 57.0% | 55.8% |
| **Aggregate (all)** | 7,019 | 30.8% | 31.4% | 34.0% | **40.6%** | 28.8% | 35.4% | 31.4% | 33.0% | 26.5% | 29.0% | 24.9% | 25.6% |
| **Per-JSONL Average (all)** | 7 | 40.1% | 41.1% | 45.2% | **53.8%** | 37.9% | 46.7% | 40.5% | 43.2% | 34.4% | 38.7% | 32.4% | 33.9% |
| **Per-JSONL Average (non-math)** | 5 | 56.1% | 57.5% | 63.2% | **75.3%** | 53.1% | 65.3% | 56.8% | 60.5% | 48.2% | 54.2% | 45.3% | 47.4% |
| **Extraction Avg. (w/o H/F)** | 4 | 53.8% | **66.6%** | 65.8% | **71.0%** | 54.8% | 61.8% | 61.6% | 65.8% | 44.1% | 57.7% | 48.3% | 51.9% |

## Per-Category Analysis

**arxiv_math + old_scans_math (0% all models):** Tests expect exact LaTeX strings like `\mathcal{V}(\psi_m)\rightarrow +\infty`. Models output Unicode/plain text like `V(ψₘ) → +∞`. This is a format mismatch, not an extraction failure. All 3,385 math tests fail for every model.

**headers_footers (92.6% olmOCR vs 21.3% Nanonets):** These are `absent` tests, meaning pass = text NOT in output. olmOCR-7B was trained to strip headers/footers as metadata. Nanonets extracts everything verbatim, so it includes headers and fails. ablations_00 (79.6%) learned partial header filtering from training data. Qwen3-VL-2B base (46.3%) has some filtering from general instruction tuning.

**multi_column (82.6% olmOCR vs 70.1% base):** `order` tests checking that reading order is preserved across columns. All models do reasonably well since modern VLMs handle column reading order natively.

**table_tests (83.5% olmOCR vs 61.5% base):** olmOCR-7B was specifically trained on table extraction. Fine-tuning on 5K samples brought base from 61.5% to 72.2% (+10.7pp). Nanonets (82.5%) is also strong here.

**long_tiny_text (69.7% olmOCR vs 49.5% base):** `present` tests checking exact text extraction from small-font or long passages. Larger model + OCR-specific training helps. ablations_00 (59.3%) improved +9.8pp over base.

**old_scans (48.1% olmOCR vs 38.0% base):** Mixed test types on degraded/noisy scans. Low scores across the board. OCR-specific training provides the biggest advantage here.

## Scaling Analysis

Per-category scaling reveals that different capabilities respond differently to model size and architecture.

### Per-category scaling by architecture

| Category | Q2.5-VL 3B→7B | Delta | Q3-VL 2B→4B→8B | 2B→4B | 4B→8B |
|----------|:------------:|:-----:|:--------------:|:-----:|:-----:|
| headers_footers | 65.5%→52.9% | **-12.6pp** | 46.3%→37.5%→39.5% | **-8.8pp** | +2.0pp |
| long_tiny_text | 46.8%→65.8% | +19.0pp | 49.5%→52.7%→59.3% | +3.2pp | +6.6pp |
| multi_column | 62.9%→75.5% | +12.6pp | 70.1%→74.5%→78.6% | +4.4pp | +4.1pp |
| old_scans | 35.6%→44.9% | +9.3pp | 38.0%→38.6%→45.2% | +0.6pp | **+6.6pp** |
| table_tests | 69.7%→77.1% | +7.4pp | 61.5%→80.5%→80.0% | **+19.0pp** | -0.5pp |
| **Non-math avg** | 56.1%→63.2% | +7.1pp | 53.1%→56.8%→60.5% | +3.7pp | +3.7pp |
| **Extraction Avg.** | 53.8%→65.8% | +12.0pp | 54.8%→61.6%→65.8% | +6.8pp | +4.2pp |

### Key patterns

**Inverse scaling on headers_footers.** Both architecture families show the same pattern: larger base models score *worse*. These are `absent` tests, so passing means NOT including header/footer text. Larger models extract more thoroughly, capturing metadata that the benchmark penalizes. Q3-VL drops from 46.3% (2B) to 37.5% (4B), then partially recovers to 39.5% (8B) but never reaches 2B levels. Fine-tuning reverses this completely (olmOCR-7B: 92.6%, ablations_00: 79.6%). This category measures a learned convention, not raw capability.

**table_tests saturates at 4B.** 80.5% at 4B, 80.0% at 8B - essentially flat. The Qwen3-VL architecture reaches near-olmOCR (83.5%) table understanding at 4B without any fine-tuning. Additional parameters don't help, suggesting the remaining gap requires task-specific training rather than scale.

**old_scans has delayed scaling.** Near-zero gain from 2B→4B (+0.6pp), then a large jump from 4B→8B (+6.6pp). Degraded document understanding appears to need a threshold amount of capacity before it kicks in. At 8B (45.2%), Q3-VL finally approaches Q2.5-VL-7B levels (44.9%).

**long_tiny_text also shows delayed scaling.** Only +3.2pp from 2B→4B, then +6.6pp from 4B→8B. At 59.3%, the 8B matches ablations_00's score exactly, achieving through scale what fine-tuning achieved through 5K training samples.

**Linear non-math average scaling.** Q3-VL gains exactly +3.7pp per step (2B→4B and 4B→8B), reaching 60.5%. Still below Q2.5-VL-7B (63.2%) despite having more params. ablations_00 (2B, 65.3%) beats Q3-VL-8B by 4.8pp, showing that 5K training samples on the smallest model outperforms 4x the params untrained.

**Qwen3-VL-2B has better baseline reading order.** 70.1% multi_column vs Q2.5-VL-3B's 62.9% despite fewer params. The newer architecture handles spatial/column layout better out of the box. This scales steadily to 78.6% at 8B.

### Qwen3.5 scaling (non-monotonic)

| Category | Q3.5 0.8B→2B→4B→9B | 0.8B→2B | 2B→4B | 4B→9B |
|----------|:-------------------:|:-------:|:-----:|:-----:|
| headers_footers | 64.6%→40.3%→33.2%→29.6% | -24.3pp | -7.1pp | -3.6pp |
| long_tiny_text | 34.2%→57.9%→36.0%→44.3% | +23.7pp | **-21.9pp** | +8.3pp |
| multi_column | 56.9%→73.9%→64.0%→67.4% | +17.0pp | **-9.9pp** | +3.4pp |
| old_scans | 30.8%→37.8%→36.3%→40.1% | +7.0pp | -1.5pp | +3.8pp |
| table_tests | 54.4%→61.2%→57.0%→55.8% | +6.8pp | -4.2pp | -1.2pp |
| **Non-math avg** | 48.2%→54.2%→45.3%→47.4% | +6.0pp | **-8.9pp** | +2.1pp |
| **Extraction Avg.** | 44.1%→57.7%→48.3%→51.9% | +13.6pp | **-9.4pp** | +3.6pp |

**Qwen3.5 scaling is non-monotonic.** Unlike Qwen3-VL's linear +3.7pp/step, Qwen3.5 peaks at 2B and then regresses. The 4B scores *below* the 0.8B on non-math average (45.3% vs 48.2%). The 9B partially recovers but never reaches 2B levels. This is the opposite of every other architecture family tested.

**The 2B→4B regression is category-wide.** long_tiny_text drops 21.9pp, multi_column drops 9.9pp, table_tests drops 4.2pp. Only old_scans shows a minor dip (-1.5pp). The regression hits extraction hardest, with Extraction Avg. dropping 9.4pp.

**Mixed attention is the likely culprit, but confounds exist.** Qwen3.5 uses linear + full attention layers for efficiency. At 2B (24 LM layers), the ratio of linear-to-full attention may be tolerable. At 4B (32 LM layers), more linear attention layers degrade fine-grained visual understanding. Critically, the 2B and 4B share the same ViT depth (24 layers), so the regression comes entirely from the LM side. However, Qwen3.5 also differs from Qwen3-VL in vocabulary size (248K vs 152K), context length (262K vs 32K), and likely pre-training data composition. The non-monotonic scaling could stem from any combination of these factors, not just mixed attention. The shared-ViT-depth observation isolates the LM side as the source, but within the LM, attention mechanism is not the only variable.

**headers_footers inverse scaling is most extreme in Qwen3.5.** 64.6% at 0.8B down to 29.6% at 9B, a 35pp drop across the family. The 0.8B actually matches Qwen2.5-VL-3B (65.5%) on this category. Larger models extract headers/footers more aggressively, and Qwen3.5's architecture amplifies this tendency.

### Cross-architecture comparison (Qwen3-VL vs Qwen3.5)

| Size | Q3-VL Non-math | Q3.5 Non-math | Delta | Q3-VL Extraction | Q3.5 Extraction | Delta |
|:----:|:--------------:|:-------------:|:-----:|:----------------:|:---------------:|:-----:|
| 2B | 53.1% | 54.2% | +1.1pp | 54.8% | 57.7% | +2.9pp |
| 4B | 56.8% | 45.3% | -11.5pp | 61.6% | 48.3% | -13.3pp |
| 8B/9B | 60.5% | 47.4% | -13.1pp | 65.8% | 51.9% | -13.9pp |

**Qwen3.5 is competitive only at 2B.** At that scale, it slightly outperforms Qwen3-VL on both metrics. But at 4B+, Qwen3-VL pulls ahead by 11-14pp. The efficiency-focused mixed attention architecture trades visual quality for speed, and the tradeoff worsens with scale.

## Extraction Avg. Analysis

The Extraction Avg. (w/o H/F) strips out the headers_footers convention noise and reveals the true extraction picture.

**Q3-VL-8B (65.8%) exactly ties Q2.5-VL-7B (65.8%)** on pure extraction. The 2.7pp gap in non-math average (60.5% vs 63.2%) comes entirely from headers_footers convention differences. The two architectures have equivalent raw document understanding at comparable scale.

**Nanonets (66.6%) beats every untrained model** including Q3-VL-8B and Q2.5-VL-7B. Its non-math average (57.5%) is misleading because it's dragged down by a design choice (verbatim extraction), not a deficiency.

**ablations_00 (61.8%) roughly matches Q3-VL-4B (61.6%)** on extraction. Most of its non-math average advantage (+8.5pp over 4B) comes from the headers_footers convention it learned. The actual extraction capability boost from 5K CE training is +7.0pp over base (54.8% to 61.8%).

**Extraction Avg. scaling is front-loaded for Q3-VL.** +6.8pp from 2B→4B vs +4.2pp from 4B→8B. This is the opposite of the non-math avg pattern (+3.7pp both steps), because headers_footers drags down the non-math average at 4B (37.5%) while extraction categories like table_tests jump sharply (+19.0pp at 4B).

**Qwen3.5-2B (57.7%) outperforms Q3-VL-2B (54.8%) on extraction** by 2.9pp, making it the strongest 2B-class model. But this advantage reverses at scale: Q3.5-9B (51.9%) falls 13.9pp behind Q3-VL-8B (65.8%). The non-monotonic scaling means Qwen3.5's extraction capability peaks early and then degrades.

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
| **Extraction Avg.** | 54.8% | 61.8% | 71.0% | +7.0pp | 9.2pp | 43.2% |

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

### Two components of the gap

The 10pp gap between ablations_00 and olmOCR has two distinct components:

1. **Convention gap (~3pp from headers_footers)** - headers_footers contributes 13.0pp/5 = 2.6pp to the non-math avg gap. Learnable from data, CE handles this well (72% recovery already)
2. **Extraction gap (~7pp from 4 content categories)** - the remaining 7.4pp across long_tiny_text, multi_column, old_scans, table_tests. Harder, this is where KD should help most

### KD targets by category

| Category | CE-only | Realistic KD target | Rationale |
|----------|:-------:|:-------------------:|-----------|
| old_scans | 39.2% | 42-44% | Strongest KD case. CE gave +1.2pp, soft labels should transfer degraded-image representations |
| table_tests | 72.2% | 76-78% | Structured output benefits from distribution-level guidance |
| long_tiny_text | 59.3% | 61-63% | Fine-grained text recognition, diminishing returns |
| multi_column | 76.4% | 77-78% | Already high, small margin |
| headers_footers | 79.6% | 82-85% | Convention mostly learned, slight refinement |
| **Non-math avg** | **65.3%** | **68-70%** | +3-5pp over CE-only |
| **Extraction Avg.** | **61.8%** | **64-66%** | +2-4pp over CE-only |

## Experimental Axes

Four axes define the comparison space:

| Axis | Values | Coverage |
|------|--------|----------|
| **1. Base VLM** | Qwen2.5-VL, Qwen3-VL, Qwen3.5 | Architecture generation |
| **2. Param count** | 0.8B, 2B, 3B, 4B, 7B, 8B, 9B | Model capacity |
| **3. Dataset size** | 0, 5K, full (~260K) | Data scaling |
| **4. Training method** | None, CE (SFT), CE + KD | Distillation signal |

### Current data points

| Experiment | Base | Params | Data | Method | Non-math Avg | Extraction Avg. |
|------------|------|:------:|:----:|--------|:------------:|:---------------:|
| Qwen2.5-VL-3B (base) | Qwen2.5-VL | 3B | 0 | None | 56.1% | 53.8% |
| Nanonets-3B | Qwen2.5-VL | 3B | ? | CE | 57.5%\* | **66.6%** |
| Qwen2.5-VL-7B (base) | Qwen2.5-VL | 7B | 0 | None | 63.2% | 65.8% |
| olmOCR-7B | Qwen2.5-VL | 7B | full | CE + RL | 75.3% | **71.0%** |
| Qwen3-VL-2B (base) | Qwen3-VL | 2B | 0 | None | 53.1% | 54.8% |
| ablations_00 | Qwen3-VL | 2B | 5K | CE | 65.3% | 61.8% |
| Qwen3-VL-4B (base) | Qwen3-VL | 4B | 0 | None | 56.8% | 61.6% |
| Qwen3-VL-8B (base) | Qwen3-VL | 8B | 0 | None | 60.5% | 65.8% |
| Qwen3.5-0.8B (base) | Qwen3.5 | 0.8B | 0 | None | 48.2% | 44.1% |
| Qwen3.5-2B (base) | Qwen3.5 | 2B | 0 | None | 54.2% | 57.7% |
| Qwen3.5-4B (base) | Qwen3.5 | 4B | 0 | None | 45.3% | 48.3% |
| Qwen3.5-9B (base) | Qwen3.5 | 9B | 0 | None | 47.4% | 51.9% |

\* Nanonets non-math average is deflated by headers_footers (21.3%) due to different design goal (verbatim extraction).

### What the baselines isolate

- **Base VLM effect** (axis 1): Qwen3-VL-2B (53.1%) vs Qwen2.5-VL-3B (56.1%). The older 3B beats the newer 2B by 3.0pp — architecture upgrade doesn't fully compensate for fewer params. Similarly, Qwen3-VL-4B (56.8%) barely edges Qwen2.5-VL-3B (56.1%) despite 33% more params.
- **Param count effect** (axis 2): Qwen2.5-VL-3B (56.1%) vs Qwen2.5-VL-7B (63.2%). 3B→7B gains +7.1pp from scale alone. Within Qwen3-VL: 2B (53.1%) → 4B (56.8%) → 8B (60.5%), linear +3.7pp per step.
- **Training effect** (axis 4): Qwen2.5-VL-7B base (63.2%) vs olmOCR-7B (75.3%). OCR-specific training adds +12.1pp on top of the base.
- **Nanonets decomposition**: Qwen2.5-VL-3B base (56.1%) vs Nanonets-3B (57.5%). Only +1.4pp, but deflated by headers_footers design choice.
- **CE fine-tuning vs scale**: ablations_00 (2B, 5K CE, 65.3%) beats Qwen3-VL-8B base (60.5%) by +4.8pp and Qwen2.5-VL-7B base (63.2%) by +2.1pp. 5K samples of task-specific training on the smallest model outperforms untrained models with 4x the params. At Q3-VL's linear scaling rate (+3.7pp per doubling), matching 65.3% untrained would need 12.2pp/3.7pp = 3.3 doublings from 2B, or ~20B parameters. **5K training samples on 2B replaces roughly 10x the parameters.**
- **Architecture matters more than scale (Qwen3.5)**: Qwen3.5-9B (47.4%) scores below Qwen3-VL-2B (53.1%) by 5.7pp despite 4.5x more params. Mixed attention architecture degrades visual understanding enough that scaling from 2B to 9B within Qwen3.5 barely recovers. The efficiency-for-quality tradeoff in Qwen3.5's attention design is not viable for fine-grained OCR tasks.
- **Qwen3.5-2B is the sweet spot**: Best Qwen3.5 model (54.2% non-math, 57.7% extraction), slightly outperforms Qwen3-VL-2B (+1.1pp, +2.9pp). At 2B scale, the mixed attention tradeoff is acceptable. Beyond 2B, it becomes counterproductive.

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

6. **headers_footers distorts non-math average**: Excluding it, Q3-VL-8B (65.8%) exactly ties Q2.5-VL-7B (65.8%) on pure extraction. Nanonets (66.6%) beats every untrained model. ablations_00's advantage over Q3-VL-4B shrinks from +8.5pp to +0.2pp. See extraction-only comparison above.

7. **Parameter efficiency**: 5K training samples on Q3-VL-2B replaces ~10x the parameters. At Q3-VL's +3.7pp/doubling scaling rate, matching ablations_00 (65.3%) without training would require ~20B params.

8. **Category difficulty tiers**: multi_column and table_tests are approachable (78-84% at scale), long_tiny_text is medium difficulty (59-70%), old_scans is hard (45-48% even for the best models).

9. **Qwen3.5 non-monotonic scaling**: The 2B model is the best Qwen3.5 performer. Going from 2B to 4B causes a broad regression (-8.9pp non-math, -9.4pp extraction), with long_tiny_text dropping 21.9pp. The shared ViT depth between 2B and 4B (both 24 layers) and mixed attention architecture are likely causes.

10. **Attention architecture constrains visual scaling**: Qwen3.5's mixed attention (linear + full) trades quality for efficiency. This works at 2B where it's competitive with Qwen3-VL, but fails at 4B+ where Qwen3-VL pulls ahead by 11-14pp. For OCR/document tasks, full attention architectures scale better.

## Caveats

**Single-run results.** All scores are from single evaluation runs with no variance estimation. The non-monotonic scaling claims (Qwen3.5 4B regression) and close comparisons (Q3-VL-8B tying Q2.5-VL-7B at 65.8%) should be interpreted with caution. A 1-2pp difference may not be meaningful without repeated runs.

**Parameter efficiency extrapolation.** The "5K training samples replaces ~10x parameters" claim (section "What the baselines isolate") assumes Q3-VL's +3.7pp/doubling scaling rate continues linearly to 20B. In practice, scaling curves flatten at larger sizes, so the true parameter equivalent is likely lower than 10x. The directional conclusion holds (task-specific data is far more efficient than raw scale), but the 10x number is an upper-bound estimate.

**Cross-architecture comparisons are confounded.** Qwen3-VL vs Qwen3.5 differ in attention mechanism, vocabulary, context length, ViT architecture (DeepStack vs none), and pre-training data. Attributing performance differences to any single factor requires controlled experiments. The analysis identifies likely contributors but cannot isolate causation.

**Math test format mismatch.** The 0% math scores reflect a systematic format mismatch (LaTeX vs Unicode), not model capability. Models may have strong mathematical OCR ability that this benchmark cannot measure. Any future math evaluation needs format-normalized scoring.

## References

- [olmOCR GitHub](https://github.com/allenai/olmocr) - toolkit, scoring code, benchmark definitions
- [olmOCR paper](https://arxiv.org/abs/2502.18443) - "olmOCR: Unlocking Trillions of Tokens in PDFs with Vision Language Models"
- [olmOCR 2 paper](https://arxiv.org/abs/2510.19817) - "olmOCR 2: Unit Test Rewards for Document OCR"
- [olmOCR-Bench dataset](https://huggingface.co/datasets/allenai/olmOCR-bench) - 7,019 tests across 1,403 PDF pages
- [olmocr-mix training data](https://huggingface.co/datasets/allenai/olmOCR-mix-0225) - ~260K PDF pages with GPT-4o-generated labels

## Next Steps

**Priority 1: KD on Qwen3-VL-2B.** Run KD-trained model (distillation loss from olmOCR-7B teacher) and compare against ablations_00 (CE-only). Target the 10pp non-math avg gap. old_scans is the strongest KD case (12% CE recovery, 8.9pp remaining). This is the core experiment that the entire gap analysis is designed to inform.

**Priority 2: Data scaling on Qwen3-VL-2B.** CE-only at 25K and 50K samples to establish diminishing returns. The 5K→full curve shape determines whether more data or better signal (KD) is the higher-leverage axis. If 25K CE approaches KD-at-5K, the KD benefit may be marginal.

**Priority 3: Qwen3.5-2B fine-tuning.** The 2B model slightly outperforms Q3-VL-2B on extraction (+2.9pp) before any training. A CE-only 5K run on Q3.5-2B would reveal whether fine-tuning preserves or erases this advantage. If it holds, Q3.5-2B may be a better student than Q3-VL-2B for the final KD experiment. Low cost to test since the infrastructure is already in place.

**Deferred: KD + more data interaction.** Wait for Priority 1 and 2 results to determine if the interaction experiment (KD at 25K+) is worth the compute.
