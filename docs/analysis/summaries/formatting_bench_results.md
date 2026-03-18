# Formatting Bench Results (2026-03-17)

250-page formatting preservation benchmark. Updated scoring: nested formatting support, deterministic matching, hallucination detection.

## Results

### Qwen3.5-9B Pipeline

| Model | Bold P | Bold R | Bold F1 | Italic P | Italic R | Italic F1 | Strike P | Strike R | Strike F1 | Overall F1 | Halluc % |
|-------|:------:|:------:|:-------:|:--------:|:--------:|:---------:|:--------:|:--------:|:---------:|:----------:|:--------:|
| pt_qwen35-9b (base) | 45.3 | 30.0 | 31.5 | 21.2 | 10.0 | 12.3 | 1.3 | 0.3 | 0.5 | 31.4 | 5.02 |
| ocr_lora_qwen35-9b | 1.3 | 0.6 | 0.7 | 0.0 | 0.0 | 0.0 | 0.4 | 0.1 | 0.2 | 0.7 | 0.00 |
| fmt_lora_qwen35-9b | 79.1 | 67.1 | 69.7 | 55.8 | 35.9 | 40.0 | 53.3 | 38.2 | 41.5 | 73.6 | 0.12 |
| fmt_grpo_qwen35-9b* | 45.2 | 29.2 | 30.7 | 20.0 | 9.2 | 11.3 | 1.7 | 0.4 | 0.6 | 30.8 | 13.24 |

*`fmt_grpo_qwen35-9b` is contaminated by `</think>` leakage baked into weights (GRPO trained before `enable_thinking=False` fix). Needs re-training.

### Qwen3.5-2B Pipeline

| Model | Bold P | Bold R | Bold F1 | Italic P | Italic R | Italic F1 | Strike P | Strike R | Strike F1 | Overall F1 | Halluc % |
|-------|:------:|:------:|:-------:|:--------:|:--------:|:---------:|:--------:|:--------:|:---------:|:----------:|:--------:|
| pt_qwen35-2b (base) | 42.2 | 20.0 | 23.6 | 24.7 | 15.3 | 16.5 | 0.0 | 0.0 | 0.0 | 30.1 | 1.45 |
| fmt_lora_qwen35-2b | 69.2 | 46.2 | 51.6 | 47.8 | 38.5 | 39.5 | 0.0 | 0.0 | 0.0 | 58.2 | 1.26 |
| kd_fmt_2b_response-kl | 48.1 | 25.3 | 29.2 | 28.5 | 20.2 | 20.9 | 0.0 | 0.0 | 0.0 | 36.8 | 1.88 |

### Qwen3.5 Pretrained Scaling (2B, 9B)

| Model | Bold F1 | Italic F1 | Strike F1 | Overall F1 | Pred Spans | Halluc % |
|-------|:-------:|:---------:|:---------:|:----------:|:----------:|:--------:|
| pt_qwen35-2b | 23.6 | 16.5 | 0.0 | 30.1 | 830 | 1.45 |
| pt_qwen35-9b | 31.5 | 12.3 | 0.5 | 31.4 | 958 | 5.02 |

Scaling observations:
- Bold F1 improves with scale (24 -> 32)
- Italic: 2B is better (16.5 vs 12.3) - different formatting strategy
- Strikethrough: nearly absent at both scales
- Hallucination rate increases with model size (1.5% -> 5.0%)

## Span Counts

GT totals: bold=2,068, italic=1,213, strike=756 (total=4,037)

| Model | Bold | Italic | Strike | Total | Matched | Over-fmt | Halluc |
|-------|:----:|:------:|:------:|:-----:|:-------:|:--------:|:------:|
| pt_qwen35-2b | 509 | 321 | 0 | 830 | 699 | 118 | 12 |
| pt_qwen35-9b | 803 | 150 | 5 | 958 | 816 | 92 | 48 |
| ocr_lora_qwen35-9b | 14 | 0 | 1 | 15 | 14 | 1 | 0 |
| fmt_lora_qwen35-9b | 1,556 | 546 | 470 | 2,572 | 2,245 | 263 | 3 |
| fmt_lora_qwen35-2b | 1,124 | 873 | 0 | 1,997 | 1,639 | 328 | 25 |
| kd_fmt_2b_response-kl | 710 | 516 | 0 | 1,226 | 940 | 263 | 23 |

## Key Takeaways

1. **fmt_lora_qwen35-9b is strong**: 73.6% overall F1, best hallucination rate (0.12%), good balance across all three types. Strikethrough F1 (41.5) is much better than any 2B model.

2. **OCR LoRA kills formatting**: ocr_lora_qwen35-9b drops from 31.4% (pretrained) to 0.7%. OCR SFT actively suppresses formatting markers since its training data has none.

3. **2B SFT is decent but no strikethrough**: fmt_lora_qwen35-2b reaches 58.2% overall F1 but produces zero strikethrough spans. 2B may lack capacity for rare formatting types.

4. **KD underperforms baseline SFT**: kd_fmt_2b_response-kl (36.8%) is well below fmt_lora_qwen35-2b (58.2%). Response KL with alpha=0.5, T=2.0 hurts more than it helps at this scale.

5. **9B GRPO needs re-training** with thinking disabled. The contamination is in the model weights (GRPO trained with `enable_thinking=True`), not an inference artifact. 13.24% hallucination rate confirms `</think>` content learned during training.

6. **Hallucination scales with model size** in pretrained models (1.5% -> 5.0%). Fine-tuning dramatically reduces it (0.12% for fmt_lora_qwen35-9b).

## Next Steps

- Re-train `fmt_grpo_qwen35-9b` with thinking disabled (contamination is in weights, not inference)
- Investigate KD failure: try lower alpha, different temperature, or SFT on teacher outputs instead of KL
- Investigate why 2B produces zero strikethrough spans
