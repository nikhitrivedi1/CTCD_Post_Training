# Models Module

Loading teacher and student model pairs for distillation, SFT, and RL.

**Source:** `src/models/loader.py` (~332 lines)

---

## Overview

The models module provides primitives for loading individual models (classifier, causal LM, VLM) and compositions for loading teacher-student pairs. It handles initialization modes, LoRA wrapping, attention implementation selection, and checkpoint loading.

---

## Primitives (Single Model Loaders)

### `_load_classifier(name, init, num_labels, device)`

Loads a sequence classification model (BERT, DistilBERT, etc.).

| Init Mode | Behavior |
|-----------|----------|
| `pretrained` | `AutoModelForSequenceClassification.from_pretrained(name, num_labels=N)` |
| `fresh` | Load config, create model from scratch with random weights |

**DistilBERT workaround:** `from_config()` doesn't work with `AutoModelForSequenceClassification` for DistilBERT (not registered in the auto mapping). Falls back to importing `DistilBertForSequenceClassification` directly.

### `_load_causal_lm(name, init, device, tokenizer, eager_attention=False)`

Loads a causal language model (Llama, Mistral, etc.).

- **dtype:** fp16 on CUDA, fp32 on CPU
- **device_map:** set to the device string on CUDA (e.g., `"cuda"`), `None` otherwise
- **Attention:** SDPA on CUDA by default, eager if attention distillation needed
- **Embedding resize:** if `len(tokenizer) > model.config.vocab_size`, resizes embeddings to match (for models with custom tokens added during tokenizer setup)

### `_load_vlm(name, init, device, eager_attention=False)`

Loads a Vision-Language Model (Qwen2.5-VL, Qwen3-VL).

- **dtype:** bfloat16 on CUDA, float32 on CPU
- **device_map:** `"auto"` on CUDA (may span multiple GPUs), `None` otherwise
- **Auto-detection:** uses `AutoModelForImageTextToText` which auto-detects the correct model class from config
- **Fresh init:** logs a warning and falls back to pretrained (VLMs require pre-trained vision encoders)

### `_apply_lora(model, lora_cfg)`

Wraps a model with LoRA adapters via PEFT.

```python
LoraConfig(
    r=lora_cfg["r"],          # Rank (e.g., 16)
    lora_alpha=lora_cfg["alpha"],  # Scaling factor
    target_modules=lora_cfg["target_modules"],  # e.g., [q_proj, k_proj, v_proj, ...]
    lora_dropout=lora_cfg["dropout"],
    bias="none",
    task_type="CAUSAL_LM",
)
```

Returns the PEFT-wrapped model. Used for causal LMs and VLMs where full fine-tuning is too expensive.

---

## Helper Functions

### `_detect_attn_implementation(eager_attention, device)`

Selects the best attention implementation:
- `eager_attention=True`: returns `"eager"` (needed for attention distillation to get attention weight matrices)
- CUDA device: returns `"sdpa"` (fused attention, fastest)
- Otherwise: returns `None` (let transformers choose)

### `_needs_attention_outputs(config)`

Checks if the config requires attention outputs for distillation. Scans:
- `distillation.loss_type` for "attention"
- `distillation.components[].type` for "attention" (composite loss)
- `distillation.stages[].strategy` for "attention" (staged loss)

If any attention-based loss is configured, forces eager attention for both teacher and student.

### `_log_params(role, name, model)`

Logs model parameter count. Shows trainable vs total params when they differ (e.g., with LoRA):
```
Student: distilbert-base-uncased (66.0M params)
Student: Qwen3-VL-2B-Instruct (2.1M trainable / 2247.0M total)
```

---

## Public API

### `load_model(name, task, device, *, init, tokenizer, num_labels, eager, lora_config)`

Load a single model, routing by task type. Tries local checkpoint first (merged or adapter-only), falls back to Hub.

**Task routing:**

| Task Type | Loader | LoRA |
|-----------|--------|------|
| Classification | `_load_classifier` | No |
| Translation | `_load_causal_lm` | Config-driven |
| OCR | `_load_vlm` | Config-driven |

Callers (forge.py, evaluate.py) handle composition: loading the model, then optionally loading a teacher.

---

## Config Schema

```yaml
model:
  name: distilbert-base-uncased     # Model ID
  init: pretrained                   # pretrained or fresh
  processor: Qwen/Qwen3-VL-2B-Instruct  # For VLMs (if different from model name)
  teacher:
    name: bert-base-uncased          # Teacher model ID
    freeze: true                     # Frozen or co-trained

training:
  use_lora: true                     # Enable LoRA wrapping
  lora:
    r: 16
    alpha: 16
    dropout: 0.05
    target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

---

*Source: `src/models/loader.py`*
