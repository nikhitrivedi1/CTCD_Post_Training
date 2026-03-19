# Formatting Bench Results (2026-03-18)

250-page formatting preservation benchmark. Updated scoring: nested formatting support, deterministic matching, hallucination detection. All Qwen3.5 models trained and evaluated with `enable_thinking=False`.

## Results

### Qwen3.5

| Model | Bold P | Bold R | Bold F1 | Italic P | Italic R | Italic F1 | Strike P | Strike R | Strike F1 | Overall F1 | Halluc % |
|-------|:------:|:------:|:-------:|:--------:|:--------:|:---------:|:--------:|:--------:|:---------:|:----------:|:--------:|
| pt_qwen35-2b (base) | 42.2 | 20.0 | 23.6 | 24.7 | 15.3 | 16.5 | 0.0 | 0.0 | 0.0 | 30.1 | 1.45 |
| ocr_sft-lora_qwen35-2b | 0.4 | 0.0 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 0.00 |
| fmt_sft-lora_qwen35-2b | 67.0 | 41.5 | 47.0 | 45.4 | 37.8 | 37.2 | 0.5 | 0.1 | 0.2 | 56.5 | 0.49 |
| fmt_grpo-lora_qwen35-2b | 45.4 | 21.9 | 25.8 | 24.5 | 15.4 | 16.7 | 0.0 | 0.0 | 0.0 | 31.6 | 1.47 |
| | | | | | | | | | | | |
| pt_qwen35-4b (base) | 60.0 | 51.7 | 51.5 | 38.2 | 26.1 | 27.6 | 0.3 | 0.1 | 0.2 | 59.0 | 0.18 |
| | | | | | | | | | | | |
| pt_qwen35-9b (base) | 45.3 | 30.0 | 31.5 | 21.2 | 10.0 | 12.3 | 1.3 | 0.3 | 0.5 | 31.4 | 5.02 |
| ocr_sft-lora_qwen35-9b | 1.3 | 0.7 | 0.7 | 0.2 | 0.1 | 0.1 | 0.4 | 0.1 | 0.2 | 0.8 | 0.00 |
| fmt_sft-lora_qwen35-9b | 79.1 | 66.6 | 69.3 | 55.7 | 35.2 | 39.3 | 52.3 | 37.7 | 40.9 | 73.1 | 0.16 |
| fmt_grpo-lora_qwen35-9b | 57.8 | 40.9 | 42.7 | 25.6 | 12.3 | 15.1 | 4.0 | 1.3 | 1.9 | 42.9 | 0.16 |

### Qwen3-VL

| Model | Bold P | Bold R | Bold F1 | Italic P | Italic R | Italic F1 | Strike P | Strike R | Strike F1 | Overall F1 | Halluc % |
|-------|:------:|:------:|:-------:|:--------:|:--------:|:---------:|:--------:|:--------:|:---------:|:----------:|:--------:|
| pt_qwen3vl-2b (base) | 5.6 | 4.7 | 4.7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 5.9 | 2.56 |
| ocr_sft-lora_qwen3vl-2b | 0.3 | 0.2 | 0.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.5 | 0.00 |
| fmt_sft-lora_qwen3vl-2b | 9.4 | 7.1 | 7.3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 8.2 | 1.67 |
| fmt_grpo-lora_qwen3vl-2b | 7.7 | 5.9 | 6.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 7.0 | 2.84 |
| | | | | | | | | | | | |
| pt_qwen3vl-4b (base) | 26.9 | 14.5 | 16.0 | 30.4 | 21.6 | 21.8 | 0.0 | 0.0 | 0.0 | 35.5 | 0.29 |
| | | | | | | | | | | | |
| pt_qwen3vl-8b (base) | 42.6 | 23.5 | 26.0 | 18.7 | 12.4 | 12.9 | 0.0 | 0.0 | 0.0 | 33.5 | 0.00 |
| ocr_sft-lora_qwen3vl-8b | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |
| fmt_sft-lora_qwen3vl-8b | 37.6 | 27.7 | 27.9 | 14.5 | 12.4 | 11.7 | 0.5 | 0.1 | 0.2 | 36.0 | 0.18 |
| fmt_grpo-lora_qwen3vl-8b | 39.4 | 23.6 | 25.7 | 17.1 | 9.4 | 10.8 | 0.0 | 0.0 | 0.0 | 31.5 | 0.00 |

### Overall Ranking

| Rank | Model | Overall F1 | Halluc % |
|:----:|-------|:----------:|:--------:|
| 1 | fmt_sft-lora_qwen35-9b | 73.1 | 0.16 |
| 2 | pt_qwen35-4b (base) | 59.0 | 0.18 |
| 3 | fmt_sft-lora_qwen35-2b | 56.5 | 0.49 |
| 4 | fmt_grpo-lora_qwen35-9b | 42.9 | 0.16 |
| 5 | fmt_sft-lora_qwen3vl-8b | 36.0 | 0.18 |
| 6 | pt_qwen3vl-4b (base) | 35.5 | 0.29 |
| 7 | pt_qwen3vl-8b (base) | 33.5 | 0.00 |
| 8 | fmt_grpo-lora_qwen35-2b | 31.6 | 1.47 |
| 9 | fmt_grpo-lora_qwen3vl-8b | 31.5 | 0.00 |
| 10 | pt_qwen35-9b (base) | 31.4 | 5.02 |
| 11 | pt_qwen35-2b (base) | 30.1 | 1.45 |
| 12 | fmt_sft-lora_qwen3vl-2b | 8.2 | 1.67 |
| 13 | fmt_grpo-lora_qwen3vl-2b | 7.0 | 2.84 |
| 14 | pt_qwen3vl-2b (base) | 5.9 | 2.56 |
| 15 | ocr_sft-lora_qwen35-9b | 0.8 | 0.00 |
| 16 | ocr_sft-lora_qwen3vl-2b | 0.5 | 0.00 |
| 17 | ocr_sft-lora_qwen35-2b | 0.1 | 0.00 |
| 18 | ocr_sft-lora_qwen3vl-8b | 0.0 | 0.00 |

## Span Counts

GT totals: bold=2,068, italic=1,213, strike=756 (total=4,037)

| Model | Bold | Italic | Strike | Total | Matched | Over-fmt | Halluc |
|-------|:----:|:------:|:------:|:-----:|:-------:|:--------:|:------:|
| fmt_sft-lora_qwen35-9b | 1,551 | 536 | 455 | 2,489 | 2,228 | 257 | 4 |
| pt_qwen35-4b | 1,615 | 579 | 6 | 2,196 | 1,850 | 342 | 4 |
| fmt_sft-lora_qwen35-2b | 1,028 | 806 | 2 | 1,832 | 1,545 | 278 | 9 |
| fmt_grpo-lora_qwen35-9b | 1,041 | 202 | 21 | 1,261 | 1,140 | 119 | 2 |
| pt_qwen3vl-4b | 419 | 629 | 0 | 1,048 | 860 | 185 | 3 |
| fmt_sft-lora_qwen3vl-8b | 813 | 309 | 1 | 1,123 | 916 | 205 | 2 |
| fmt_grpo-lora_qwen3vl-8b | 744 | 224 | 0 | 968 | 822 | 146 | 0 |
| pt_qwen3vl-8b | 682 | 285 | 0 | 967 | 829 | 138 | 0 |
| pt_qwen35-9b | 803 | 150 | 5 | 956 | 816 | 92 | 48 |
| fmt_grpo-lora_qwen35-2b | 561 | 324 | 0 | 884 | 748 | 123 | 13 |
| pt_qwen35-2b | 509 | 321 | 0 | 829 | 699 | 118 | 12 |
| fmt_grpo-lora_qwen3vl-2b | 317 | 0 | 0 | 317 | 152 | 156 | 9 |
| fmt_sft-lora_qwen3vl-2b | 240 | 0 | 0 | 240 | 182 | 54 | 4 |
| pt_qwen3vl-2b | 234 | 0 | 0 | 234 | 126 | 102 | 6 |
| ocr_sft-lora_qwen35-9b | 16 | 2 | 1 | 19 | 17 | 2 | 0 |
| ocr_sft-lora_qwen3vl-2b | 8 | 0 | 0 | 8 | 8 | 0 | 0 |
| ocr_sft-lora_qwen35-2b | 1 | 0 | 0 | 1 | 1 | 0 | 0 |
| ocr_sft-lora_qwen3vl-8b | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Failure Mode Analysis (fmt_sft-lora_qwen35-9b)

Detailed per-component error analysis across all 250 pages for the best model. Full output at `docs/analysis/data/failure_mode_analysis.txt`.

### Error Summary (1,254 total errors)

| Failure Mode | Count | % | Description |
|--------------|:-----:|:-:|-------------|
| Over-formatting | 590 | 47 | Model applies formatting to text that shouldn't be formatted |
| Table formatting dropped | 578 | 46 | Formatted spans inside markdown table rows lost |
| Merge | 51 | 4 | Adjacent GT spans merged into one longer pred span |
| Text missing entirely | 28 | 2 | Formatted text not present in output at all |
| Type swap | 4 | <1 | Correct text, wrong format type |
| Hallucinated | 3 | <1 | Text not in the GT document at all |

### 1. Over-formatting (590 spans)

The model formats real text that shouldn't have any markers. Bold is the worst offender.

| Type | Count |
|------|:-----:|
| Bold | 284 |
| Italic | 161 |
| Strikethrough | 145 |

Top over-formatted texts (model has learned a "looks important" bias):

| Text | Count | Types Applied |
|------|:-----:|---------------|
| emphasis | 10 | bold, italic, strike |
| initial estimate | 7 | bold, italic |
| Algorithm | 7 | bold |
| draft version | 6 | bold |
| previously reported | 6 | bold, italic |
| in vivo | 6 | italic, strike |
| significant | 6 | bold, strike |
| de novo | 6 | italic |
| prior methodology | 6 | bold, italic |
| Total | 6 | bold |
| per se | 5 | bold, italic, strike |
| Table | 5 | bold |
| Theorem | 5 | bold |
| critical | 5 | bold |
| essential | 5 | bold |

Pattern: Latin phrases (in vivo, de novo, per se) get italic/strike. "Important-sounding" words (significant, critical, essential, Algorithm) get bold. The model has internalized formatting conventions from pre-training rather than faithfully reproducing what's in the image.

### 2. Table Formatting Drops (578 spans)

The model systematically fails to preserve formatting inside markdown table cells. This is the biggest recall problem.

| Type | Count |
|------|:-----:|
| Bold | 398 |
| Italic | 135 |
| Strikethrough | 45 |

Examples of dropped bold table headers: "Duration", "Pass/Fail", "Test Case", "Metric", "Q1". The model generates the table structure and text correctly but strips the formatting markers inside `| ... |` cells.

### 3. Merge Errors (51 spans)

The model merges adjacent GT spans into one longer formatted run.

Examples:
- GT: `~~through synapses (~~` + `~~previously reported~~` → Pred: `~~through synapses (previously reported).~~`
- GT: `**chemical elements by**` + `**The periodic**` + `**and electron configuration.**` → Pred: one merged bold span

Strikethrough and italic are most affected. The model tends to extend formatting boundaries rather than matching the exact GT span boundaries.

### 4. Text Missing Entirely (28 spans)

GT formatted text not found anywhere in the prediction output. Strikethrough dominates.

| Type | Count |
|------|:-----:|
| Strikethrough | 17 |
| Italic | 9 |
| Bold | 2 |

The model sometimes drops strikethrough content entirely, likely because strikethrough text represents "deleted" content that the model learns to omit.

### 5. Type Swaps (4 spans)

Negligible. All 4 involve strikethrough GT rendered as bold or italic. Not a meaningful error mode.

### 6. Hallucinated Spans (3 spans)

Near-zero hallucination. The 3 cases:
- `id:` (bare label in a table page)
- A long legal clause not in GT
- `loose coupling: (observed behavior)` (technical term with parenthetical)

### Actionable Findings

1. **Over-formatting and table drops account for 93% of all errors.** These are the two problems to solve.

2. **Table-cell formatting is a structural failure.** 578 missed spans inside table rows. The model generates correct table structure but strips formatting markers inside cells. Adding table-rich documents with preserved inline formatting to the SFT dataset should directly address this.

3. **Over-formatting is a learned bias, not a random error.** The model systematically formats "important-sounding" words and Latin phrases. Training examples with these words appearing *without* formatting would teach the model to reproduce what it sees rather than what it "expects."

4. **Strikethrough is fragile.** It dominates both text-missing (17/28) and type-swap (4/4) errors. The model sometimes drops strikethrough content entirely or converts it to bold/italic. More strikethrough-heavy training examples needed.

5. **GRPO reward redesign.** A v2 reward that penalizes false-positive formatting (addressing the 590 over-formatted spans) while specifically rewarding table-interior formatting recall could target both top failure modes.

## Key Takeaways

1. **fmt_sft-lora_qwen35-9b is the clear winner**: 73.1% overall F1 with 0.16% hallucination. Only model with meaningful strikethrough detection (40.9% F1). Strong across all formatting types.

2. **Qwen3.5-4B pretrained is surprisingly strong**: 59.0% overall F1 out of the box, beating all fine-tuned 2B models and most 8B/9B fine-tunes. The 4B sweet spot for formatting may be a pre-training data artifact.

3. **Qwen3.5 >> Qwen3-VL for formatting**: At every size and training stage, Q35 models outperform their Q3VL counterparts. The VL architecture's vision encoder doesn't help with inline text formatting, and the smaller language model backbone hurts.

4. **GRPO regresses from SFT across the board**: fmt_grpo underperforms fmt_sft for every model (Q35-9B: 42.9 vs 73.1, Q35-2B: 31.6 vs 56.5, Q3VL-8B: 31.5 vs 36.0). GRPO training with the current reward configuration erodes formatting capability rather than improving it.

5. **OCR SFT kills formatting universally**: All four OCR-only models score near 0%. Confirmed across both model families and sizes. OCR training data contains no formatting markers, so the model learns to suppress them.

6. **Qwen3-VL-2B is too small for formatting**: Both base (5.9%) and fine-tuned (7-8%) Q3VL-2B models produce near-zero italic/strikethrough spans. The model lacks capacity for anything beyond basic bold detection.

7. **Hallucination correlates with model quality, not size**: pt_qwen35-9b has the worst hallucination rate (5.02%) despite being the largest Q35 model. Fine-tuning dramatically reduces hallucination (0.16% for fmt_sft-lora_qwen35-9b).

8. **Pretrained scaling is non-monotonic for Q35**: 4B (59.0%) >> 9B (31.4%) > 2B (30.1%). The 9B pretrained model is barely above 2B. This suggests the 9B model's pre-training distribution differs in formatting behavior, possibly due to different data mixtures or the 9B model being more conservative with formatting markers.
