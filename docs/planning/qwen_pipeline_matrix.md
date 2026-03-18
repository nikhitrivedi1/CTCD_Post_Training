# Pipeline Matrix

Model families, naming conventions, and chain plans for OCR training pipelines.

## Model Families

### Qwen3.5 (native multimodal text)

Native vision encoder built in. No separate VL variant needed.

| Size | HuggingFace ID | Config dir slug |
|------|---------------|-----------------|
| 0.8B | `Qwen/Qwen3.5-0.8B` | `qwen35-08b` |
| 2B | `Qwen/Qwen3.5-2B` | `qwen35-2b` |
| 4B | `Qwen/Qwen3.5-4B` | `qwen35-4b` |
| 9B | `Qwen/Qwen3.5-9B` | `qwen35-9b` |

All use the instruct-tuned variant (not `-Base`). Reasoning: our SFT/RL pipeline uses `apply_chat_template` with `enable_thinking`, which requires the chat template baked into the instruct model. Base models have no chat template.

`-Base` variants (`Qwen3.5-2B-Base`, etc.) exist on HF for from-scratch fine-tuning but are not used here.

### Qwen3-VL (vision-language, -Instruct)

Separate VL architecture with DeepStack visual processing.

| Size | HuggingFace ID | Config dir slug |
|------|---------------|-----------------|
| 2B | `Qwen/Qwen3-VL-2B-Instruct` | `qwen3vl-2b` |
| 4B | `Qwen/Qwen3-VL-4B-Instruct` | `qwen3vl-4b` |
| 8B | `Qwen/Qwen3-VL-8B-Instruct` | `qwen3vl-8b` |

All use `-Instruct` suffix.

## Run Naming Convention

Format: `{stage}_{objective}-{param_method}_{model}`

| Component | Values | Description |
|-----------|--------|-------------|
| stage | `ocr`, `fmt` | Training stage (OCR grounding, formatting) |
| objective | `sft`, `grpo`, `kd` | Training objective |
| param_method | `lora`, `full` | Parameter method (LoRA adapter, full fine-tune) |
| model | `qwen35-2b`, `qwen3vl-4b`, etc. | Model slug |

### Training step examples

| Stage | Run name | Description |
|-------|----------|-------------|
| OCR grounding | `ocr_sft-lora_qwen35-2b` | LoRA SFT on olmocr-mix |
| Formatting SFT | `fmt_sft-lora_qwen35-2b` | LoRA SFT on formatting |
| Formatting GRPO | `fmt_grpo-lora_qwen35-2b` | LoRA GRPO on formatting-rl |
| Full FT baseline | `ocr_sft-full_qwen35-9b` | Full fine-tune baseline |
| KD | `fmt_kd-lora_qwen35-2b_response-kl` | KD with loss variant suffix |

### Bench step naming

Appended to the training run name:

| Type | Pattern | Example |
|------|---------|---------|
| Formatting bench | `{run}_bench-formatting` | `fmt_grpo-lora_qwen35-2b_bench-formatting` |
| olmOCR shard | `{run}_bench-olmocr_s{N}` | `fmt_grpo-lora_qwen35-2b_bench-olmocr_s0` |
| olmOCR score | `{run}_bench-olmocr_score` | `fmt_grpo-lora_qwen35-2b_bench-olmocr_score` |

## Datasets

| Stage | Dataset | Path | Split | Notes |
|-------|---------|------|-------|-------|
| OCR grounding | olmocr-mix | `data/ocr/olmocr-mix-full` | train | Stratified sampling, default 5k samples |
| Formatting SFT | formatting | `data/ocr/formatting` | train/eval | 3 epochs |
| Formatting GRPO | formatting-rl | `data/ocr/formatting-rl` | train | 1 epoch |
| Formatting bench | formatting | `data/ocr/formatting` | eval | Scoring only |

Size configs for olmocr-mix live in `configs/ocr/olmocr-mix/sizes/` (5k, 10k, 25k, 50k, full). Default is 5k.

## Pipeline Structure

Each model gets a 3-stage training pipeline + formatting bench.

```
OCR grounding (SFT on olmocr-mix)
  -> Formatting SFT (SFT on formatting)
    -> Formatting GRPO (RL on formatting-rl)
```

Bench after each stage: formatting bench (single job).

Total per chain: 3 training + 3 formatting bench = 6 steps.

## Config Structure

Each model needs configs in two directories:

```
configs/ocr/olmocr-mix/{model_slug}/
  base.yaml          # model name, batch/grad_accum, LR=2e-5
  00_sft.yaml         # inherits base.yaml

configs/ocr/formatting/{model_slug}/
  base.yaml          # model name, batch/grad_accum, LR=2e-6
  00_sft.yaml         # inherits base.yaml
  rl/
    base.yaml        # RL params (algorithm, generations, completion length)
    00_grpo.yaml      # reward weights, temperature, beta
```

### Batch size rules

Effective batch = `batch_size * gradient_accumulation_steps` = 32 for all models.

| Model size | batch_size | gradient_accumulation_steps |
|-----------|-----------|---------------------------|
| 0.8B, 2B, 4B | 2 | 16 |
| 8B, 9B | 1 | 32 |

### Learning rates

| Stage | LR |
|-------|----|
| OCR grounding (olmocr-mix) | 2e-5 |
| Formatting SFT | 2e-6 |

## Chain Files

One chain YAML per model: `chains/{model_slug}_ocr_pipeline.yaml`

| Chain file | Model |
|-----------|-------|
| `qwen35-08b_ocr_pipeline.yaml` | Qwen3.5-0.8B |
| `qwen35-2b_ocr_pipeline.yaml` | Qwen3.5-2B |
| `qwen35-4b_ocr_pipeline.yaml` | Qwen3.5-4B |
| `qwen35-9b_ocr_pipeline.yaml` | Qwen3.5-9B |
| `qwen3vl-2b_ocr_pipeline.yaml` | Qwen3-VL-2B |
| `qwen3vl-4b_ocr_pipeline.yaml` | Qwen3-VL-4B |
| `qwen3vl-8b_ocr_pipeline.yaml` | Qwen3-VL-8B |

All chains and configs are complete.

## chat_template_kwargs

Resolved automatically from `MODEL_PROFILES` in `src/inference/model_profiles.py`:

| Model family | Task | enable_thinking |
|-------------|------|-----------------|
| Qwen3.5 | ocr (SFT + formatting) | False |
| Qwen3.5 | * (wildcard, incl. GRPO) | True |
| Qwen3-VL | any | N/A (not supported) |

Qwen3-VL's chat template does not support `enable_thinking`. The `<think>`/`</think>` special tokens exist in the tokenizer but the Jinja template has no conditional logic for them. No `chat_template_kwargs` are passed for Q3VL models.

Config-level override via `model.chat_template_kwargs` takes precedence over profiles.
