# Formatting Bench Results

Inline formatting preservation benchmark: measures how well OCR models reproduce **bold**, *italic*, and ~~strikethrough~~ markup in generated markdown.

## Methodology

**Dataset**: 250 synthetic pages with inline formatting, sourced from OmniDocBench document pages containing bold/italic/strikethrough spans in their ground truth markdown.

**Scoring**: Fuzzy set matching between predicted and ground truth formatting spans per type.

1. Extract spans from markdown using regex (`**bold**`, `*italic*`, `~~strike~~`)
2. Match predicted spans to GT spans using Levenshtein similarity (threshold 0.5)
3. Compute precision (matched/predicted), recall (matched/GT), F1 per type
4. **Overall** = P/R/F1 across all span types combined (not average of per-type)

Pages without a given formatting type in GT are excluded from that type's score.

**Script**: `scripts/benchmarks/formatting_bench.py`

## Results

| Model | | Bold | Italic | Strike | Overall |
|-------|---|:----:|:------:|:------:|:-------:|
| ocr_sft | Precision | 0.000 | 0.000 | 0.000 | 0.000 |
| | Recall | 0.000 | 0.000 | 0.000 | 0.000 |
| | **F1** | **0.000** | **0.000** | **0.000** | **0.000** |
| fmt_sft | Precision | 0.724 | 0.503 | 0.320 | 0.837 |
| | Recall | 0.721 | 0.461 | 0.188 | 0.721 |
| | **F1** | **0.698** | **0.445** | **0.218** | **0.766** |
| fmt_grpo | Precision | 0.561 | 0.355 | 0.000 | 0.726 |
| | Recall | 0.628 | 0.275 | 0.000 | 0.575 |
| | **F1** | **0.563** | **0.283** | **0.000** | **0.623** |

### Span Counts

| Model | | Bold | Italic | Strike |
|-------|---|:----:|:------:|:------:|
| Ground truth | spans | 2,070 | 951 | 756 |
| ocr_sft | predicted | 0 | 0 | 0 |
| fmt_sft | predicted | 2,048 | 826 | 233 |
| fmt_grpo | predicted | 2,107 | 577 | 0 |

## Models

| Short name | Full model | Backbone | Base | Training |
|------------|-----------|----------|------|----------|
| ocr_sft | ablations_00 (CE-only) | Qwen3-VL-2B | Qwen3-VL-2B | SFT on 5K olmocr-mix (no formatting in GT) |
| fmt_sft | fmt_sft | Qwen3-VL-2B | ocr_sft | SFT (CE) on formatting-rich synthetic pages |
| fmt_grpo | fmt_grpo | Qwen3-VL-2B | fmt_sft | GRPO with inline_formatting_reward |

## Key Observations

**ocr_sft produces zero formatting.** The olmocr-mix training data has minimal inline formatting in its ground truth (GPT-4o labels strip formatting). The model learns to output plain text.

**Bold is well-balanced (P=0.724, R=0.721, F1=0.698).** Precision and recall are nearly equal, and span count is close (2,048 predicted vs 2,070 GT). Bold is the most visually salient type, making it the easiest to learn.

**Italic has a precision-recall gap (P=0.503, R=0.461, F1=0.445).** 826 predicted vs 951 GT spans. Recall is lower, meaning the model misses some italic spans, and the ones it does predict don't always match GT content.

**Strikethrough is weakest (P=0.320, R=0.188, F1=0.218).** Only 233 predicted vs 756 GT (31% recall by count). Low recall is the bottleneck - the model simply doesn't detect most strikethrough spans. Rarest type in training data and hardest to detect visually.

**Overall precision (0.837) is high.** When the model does predict formatting, it's usually correct. The gap to overall recall (0.721) comes from missed spans, primarily strikethrough.

**Overall F1 (0.766) exceeds any per-type score** because overall pools all span types. Bold dominates the pool (55% of GT spans) and has the highest F1, pulling overall up.

**Recall is the bottleneck.** Overall precision (0.837) is already high - the model rarely hallucinates formatting. The gap is in recall (0.721), driven by missed strikethrough spans (R=0.188).

**GRPO regressed across every metric.** fmt_grpo (Overall F1=0.623) drops -0.143 from fmt_sft (0.766). Bold precision fell from 0.724 to 0.561, italic from 0.503 to 0.355. Strikethrough collapsed entirely (0 predicted spans vs 233 from fmt_sft). The formatting reward signal pushed the model to produce more bold spans (2,107 vs 2,048) but made them less accurate, while italic and strikethrough generation atrophied. The reward function may be too coarse to guide fine-grained formatting reproduction.

## Pending

- **Base model (Qwen3-VL-2B-Instruct)**: Expected to produce some formatting from instruction tuning but less than fmt_sft
