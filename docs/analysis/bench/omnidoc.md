# OmniDocBench Results

Evaluation on [OmniDocBench](https://github.com/opendatalab/OmniDocBench): 1,352 document pages spanning PPTs, academic papers, textbooks, financial reports, newspapers, and more. Scores formatting quality across four dimensions.

## Methodology

| Metric | Method | Range |
|--------|--------|:-----:|
| **text** | Normalized edit distance on text blocks | 0-1 |
| **table** | Cell-level F1 across markdown tables | 0-1 |
| **formula** | Character Distance Metric on LaTeX formulas | 0-1 |
| **reading_order** | LCS ratio of text block ordering | 0-1 |
| **overall** | Average of text, table, formula | 0-1 |

Table/formula/reading_order scores are null for pages that lack those elements.

**Script**: `scripts/benchmarks/omnidoc_bench.py`

## Results

| Metric | ocr_sft | fmt_sft | Delta |
|--------|:-------:|:-------:|:-----:|
| **text** | 67.0 | 69.4 | +2.4 |
| **table** | 2.0 | 55.3 | +53.3 |
| **formula** | 2.1 | 29.5 | +27.4 |
| **reading_order** | 92.3 | 93.0 | +0.7 |
| **overall** | 23.7 | 51.4 | +27.7 |

Overall = avg(text, table, formula). Counts: text (1,352 pages), table (351), formula (562), reading_order (1,253).

## Models

| Short name | Full model | Backbone | Base | Training |
|------------|-----------|----------|------|----------|
| ocr_sft | ablations_00 (CE-only) | Qwen3-VL-2B | Qwen3-VL-2B | SFT on 5K olmocr-mix |
| fmt_sft | fmt_sft | Qwen3-VL-2B | ocr_sft | SFT (CE) on formatting-rich synthetic pages |

## Key Observations

**Formatting SFT unlocks latent capability.** The backbone already has the visual understanding to recognize tables, formulas, and formatting in images. ocr_sft's near-zero table/formula scores aren't a perception failure - the olmocr-mix ground truth (GPT-4o labels) strips formatting, so the model learned to output plain text. fmt_sft's synthetic training data provides the output format signal (markdown tables, LaTeX, inline formatting), and the gains are immediate: table 2.0% to 55.3%, formula 2.1% to 29.5%, formatting F1 0.0 to 0.766.

**Text extraction doesn't regress (+2.4pp).** Since the visual understanding was already in place, adding formatting patterns on top preserves (and slightly improves) text quality.

**Reading order is already strong (92.3%) and stable.** Both models achieve >92%, confirming Qwen3-VL-2B's spatial understanding handles document reading order well out of the box.

**Overall nearly doubles (+27.7pp).** The ocr_sft model's near-zero table/formula scores dragged its overall to 23.7%. Formatting training lifts both, pushing overall to 51.4%.

## Pending

- **fmt_grpo**: GRPO with formatting rewards, OmniDocBench scores pending (3 shards planned)
- **omnidoc_grpo**: Needs re-scoring with current code
- **Base model baselines**: No OmniDocBench runs for base Qwen3-VL-2B or other base models yet
