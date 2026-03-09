# Inference Module

Unified text generation engine with optimal attention settings.

**Source:** `src/inference/engine.py` (~460 lines)

---

## Overview

`InferenceEngine` provides a consistent interface for text generation across model types (VLM and causal LM). It handles model loading, attention implementation selection, checkpoint loading (merged model or adapter), base model resolution, and batched generation.

Used by:
- `TranslationMetricsCallback` (mid-training BLEU/COMET evaluation)
- `scripts/eval/olmocr_bench.py` (olmOCR-Bench evaluation)
- `scripts/eval/omnidoc_bench.py` (OmniDocBench evaluation)

---

## Model Types

| Type | Input | Model Class | Examples |
|------|-------|-------------|---------|
| `vlm` | (PIL.Image, str) tuples | `AutoModelForImageTextToText` | Qwen2.5-VL, Qwen3-VL, olmOCR |
| `causal_lm` | str prompts | `AutoModelForCausalLM` | Llama, Mistral |

---

## Helper Functions

### `_resolve_base_model(checkpoint_path) -> (model_path, revision)`

Resolves the base model for adapter loading. Checks in order:
1. `base/` with full weights (custom merged model from `--adapter`)
2. `base/model_ref.json` (HF Hub reference with revision)
3. `adapter_config.json`'s `base_model_name_or_path` (legacy fallback)

For `checkpoint-N/` directories, looks for `base/` in the parent run directory.

Returns `(model_path, revision)` where revision is `None` for local paths.

### `_has_merged_model(path) -> bool`

Checks if a directory contains a merged/full model by looking for `model.safetensors` or `model.safetensors.index.json`.

---

## Factory Methods

### InferenceEngine.load()

```python
engine = InferenceEngine.load(
    model_id="allenai/olmOCR-2-7B-1025-FP8",  # HuggingFace model ID
    checkpoint_path=None,                       # OR path to checkpoint
    model_type="vlm",                           # "vlm" or "causal_lm"
    device="cuda"                               # Device string
)
```

**Attention implementation:**
- CUDA: uses SDPA (scaled dot product attention) for optimal performance
- Other devices: lets transformers choose

**Checkpoint loading** (two strategies tried in order):

1. **Merged model** (if `model.safetensors` exists at checkpoint path): loads directly as a complete model, no adapter resolution needed
2. **Adapter-only** (if `adapter_config.json` exists): resolves base model via `_resolve_base_model()`, loads base with revision pinning, loads adapter via `PeftModel.from_pretrained()`, merges via `merge_and_unload()`

Both strategies also load the processor/tokenizer with the same revision pin when loading from a base model reference, to prevent version mismatches between model weights and tokenizer.

**dtype selection:**
- VLM: bfloat16 on CUDA, float16 on MPS, float32 on CPU
- Causal LM: float16 on CUDA/MPS, float32 on CPU

### InferenceEngine.from_model()

```python
engine = InferenceEngine.from_model(
    model=trained_model,
    tokenizer=tokenizer,    # For causal LMs
    processor=processor,    # For VLMs
)
```

Wraps an already-loaded model for generation. Used by training callbacks where the model is already loaded by the trainer. Uses whatever attention implementation the model was loaded with (may be eager for attention distillation).

Auto-detects model type from which arg is provided (tokenizer -> causal_lm, processor -> vlm).

---

## Generation

### engine.generate(inputs, max_new_tokens=128, batch_size=1)

Unified generation interface that dispatches to type-specific methods.

**Input formats:**
- VLM: list of `(PIL.Image, str)` tuples
- Causal LM: list of prompt strings (already formatted with chat template by caller)
- Single inputs (tuple or string) are auto-wrapped in a list

**Returns:** list of generated text strings.

### VLM Generation (`_hf_generate_vlm`)

Processes one image at a time (no batching - variable image sizes make batching complex):

1. Format chat message with image placeholder
2. Apply chat template via processor
3. Process text + image -> `pixel_values`, `input_ids`, etc.
4. Generate with `do_sample=False` (greedy decoding)
5. Extract only newly generated tokens (skip input)
6. Decode with `skip_special_tokens=True`

### Causal LM Generation (`_hf_generate_causal`)

Batched generation for text-only models:

1. Tokenize batch with left padding (`padding_side="left"`)
2. Generate with `do_sample=False`
3. For each output, slice off input tokens and decode
4. Process in chunks of `batch_size`

---

## Properties

| Property | Description |
|----------|-------------|
| `model_name` | Short model name (e.g., "olmOCR-2-7B-1025-FP8") |
| `commit_hash` | HuggingFace commit hash or "local:{path}" for checkpoints |
| `model_type` | "vlm" or "causal_lm" |

---

## Usage Examples

```python
# Load from HuggingFace
engine = InferenceEngine.load(model_id="allenai/olmOCR-2-7B-1025-FP8", model_type="vlm")
results = engine.generate([(image, "Extract text from this document")], max_new_tokens=2048)

# Load from run root (merged model)
engine = InferenceEngine.load(checkpoint_path="outputs/my_run/", model_type="vlm")

# Load from checkpoint (adapter-only, resolves base from base/ dir)
engine = InferenceEngine.load(checkpoint_path="outputs/my_run/checkpoint-400", model_type="vlm")

# Wrap existing model (in training callback)
engine = InferenceEngine.from_model(model, tokenizer=tokenizer)
translations = engine.generate(prompts, max_new_tokens=128, batch_size=8)
```

---

*Source: `src/inference/engine.py`*
