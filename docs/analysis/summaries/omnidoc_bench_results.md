# OmniDocBench Results

Evaluation on [OmniDocBench](https://github.com/opendatalab/OmniDocBench): 1,352 document pages spanning PPTs, academic papers, textbooks, financial reports, newspapers, and more. Scores formatting quality across four dimensions.

## Scoring Metrics

| Bench Metric | Method | GRPO Reward | Range |
|-------------|--------|-------------|-------|
| **text** | Normalized edit distance on text blocks | `ocr_reward` (edit similarity) | 0-1 |
| **table** | Cell-level F1 across markdown tables | `table_cell_f1_reward` | 0-1 |
| **formula** | Character Distance Metric on LaTeX formulas | `formula_cdm_reward` | 0-1 |
| **reading_order** | LCS ratio of text block ordering | `reading_order_reward` | 0-1 |
| **overall** | Average of all four metrics | Sum of rewards (approx) | 0-1 |

Note: `format_reward` (GRPO reward #2) checks for generic markdown structure (headings, lists, bold) and has no direct bench counterpart. Table/formula/reading_order scores are null for pages that lack those elements.

## Results: ablations_00 (SFT Baseline)

Model: Qwen3-VL-2B + LoRA, CE-only on 5K olmocr-mix samples.

| Metric | Score | Count | Notes |
|--------|:-----:|:-----:|-------|
| **text** | 0.680 | 1,352 | Edit similarity on text blocks |
| **table** | 0.048 | 351 | Cell-level F1 (very low) |
| **formula** | 0.017 | 562 | CDM on LaTeX (near zero) |
| **reading_order** | 0.660 | 1,249 | Block ordering correctness |
| **overall** | 0.351 | 1,352 | Average of four metrics |

### Interpretation

**Text extraction (0.680)** is the strongest metric, consistent with the olmOCR-Bench results where ablations_00 achieves 67.1% overall. The SFT training on olmocr-mix teaches general text extraction well.

**Reading order (0.660)** is also reasonable. Qwen3-VL has good spatial understanding from pre-training (70.1% multi_column on olmOCR-Bench). The 0.66 score here confirms document-level reading order is partially learned.

**Table formatting (0.048)** is nearly zero. The model extracts table content but not in correct markdown table format (pipes, alignment). The olmocr-mix training data has table content but the SFT model hasn't learned precise cell-level formatting. On olmOCR-Bench, ablations_00 gets 72.2% on table_tests, but that benchmark checks content presence, not formatting structure. OmniDocBench's cell-level F1 is far more demanding.

**Formula formatting (0.017)** is essentially zero. Similar story: the model can sometimes extract formula text, but the LaTeX output doesn't match the reference structure. The CDM metric compares character-level formula similarity, which requires precise LaTeX formatting.

### Mapping to GRPO Training Rewards

The GRPO run (job 4932797/4935862) trains with 5 reward functions targeting these exact metrics. At epoch 0.82:

| Dimension | Bench (SFT) | GRPO Reward (ep 0.82) | Comparable? |
|-----------|:-----------:|:---------------------:|-------------|
| text / edit_sim | 0.680 | 0.705 | Roughly yes (same edit distance metric) |
| table / table_f1 | 0.048 | 0.418 | No - reward scores per-completion, bench scores per-page |
| formula / formula_cdm | 0.017 | 0.303 | No - reward on individual formulas, bench averages across page |
| reading_order | 0.660 | 0.695 | Roughly yes (same LCS approach) |
| format (no bench equiv) | n/a | 0.438 | n/a |

**Why the reward values look higher than bench scores:** The reward functions score individual completions during GRPO training (short prompt-completion pairs), while the bench scores full document pages (much longer, more complex). The bench also averages across pages that may lack tables/formulas (scoring null, counted as zero in the aggregate). The GRPO rewards only fire on completions that contain the relevant content.

## Baseline Context

From olmOCR-Bench results, ablations_00's position:

| Metric | Base Q3-VL-2B | ablations_00 (SFT) | olmOCR-7B | CE Recovery |
|--------|:------------:|:------------------:|:---------:|:-----------:|
| olmOCR Overall | 56.1% | 67.1% | 77.3% | 52% |
| olmOCR Extraction Avg | 54.8% | 61.8% | 71.0% | 43% |

OmniDocBench measures a different, harder dimension (formatting fidelity vs content presence). The near-zero table and formula scores show the SFT model has a large formatting gap that GRPO is explicitly targeting.

## Results: omnidoc_grpo (GRPO)

Model: Qwen3-VL-2B + SFT (ablations_00) + GRPO LoRA on OmniDocBench training data. 152 steps, 5 reward functions (ocr, format, table_cell_f1, formula_cdm, reading_order).

| Metric | Score | Count | Notes |
|--------|:-----:|:-----:|-------|
| **text** | 0.662 | 1,352 | Edit similarity on text blocks |
| **table** | 0.036 | 351 | Cell-level F1 |
| **formula** | 0.017 | 562 | CDM on LaTeX |
| **reading_order** | 0.639 | 1,249 | Block ordering correctness |
| **overall** | 0.338 | 1,352 | Average of four metrics |

### Comparison: SFT vs GRPO

| Metric | SFT (ablations_00) | GRPO (omnidoc_grpo) | Delta |
|--------|:------------------:|:-------------------:|:-----:|
| **text** | 0.680 | 0.662 | -0.018 |
| **table** | 0.048 | 0.036 | -0.012 |
| **formula** | 0.017 | 0.017 | 0.000 |
| **reading_order** | 0.660 | 0.639 | -0.021 |
| **overall** | 0.351 | 0.338 | -0.013 |

### Interpretation

GRPO regressed slightly across all metrics. Despite reward signals showing gains during training (table reward 0 to 0.418, formula to 0.303), these did not transfer to full-page document evaluation.

**Reward-bench disconnect.** The GRPO rewards score individual completions (short prompt-completion pairs with a single table or formula). The bench scores full document pages containing mixed content. The model learned to format isolated elements better in the reward's narrow context, but this didn't generalize to end-to-end page conversion.

**Text and reading order regressed (-1.8%, -2.1%).** These were the strongest SFT metrics. GRPO may have traded general text fidelity for formatting patterns that the rewards incentivized but the bench doesn't capture (e.g., format_reward encourages markdown headers/bold which can alter text content).

**Table and formula unchanged or worse.** The near-zero baseline left little room for the reward signal to find learnable gradients. With table F1 at 0.048 and formula CDM at 0.017, most completions score near zero, giving sparse reward signal for GRPO to learn from.
