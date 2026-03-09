# Model Save Structure

Forge saves a unified output structure for all training methods (FT, KD, GRPO) and modes (LoRA, Full FT).

## Directory Layout

```
outputs/run_name/
├── config.json                   # Merged model config (HF format)
├── model.safetensors             # Merged model weights (base + adapter)
├── tokenizer.json                # Tokenizer files
├── tokenizer_config.json
├── adapter_config.json           # LoRA adapter config (LoRA runs only)
├── adapter_model.safetensors     # LoRA adapter weights, delta only (LoRA runs only)
├── config.yaml                   # Forge training config
├── run_metadata.json             # Run metadata (model names, task, timestamps)
├── training_state.json           # Completion status, final metrics
├── latest_checkpoint             # Text file containing path to most recent checkpoint
├── output.log                    # Training log
├── base/                         # Base model reference (always present)
│   └── model_ref.json            # HF Hub: {"model_name_or_path": "...", "revision": "..."}
│   # OR for custom bases (e.g., --adapter merged model):
│   ├── config.json               # Full model weights saved locally
│   └── model.safetensors
├── checkpoint-100/               # Intermediate checkpoints
│   ├── adapter_config.json       # LoRA: adapter only
│   └── adapter_model.safetensors
└── checkpoint-200/
    ├── adapter_config.json
    └── adapter_model.safetensors
```

## What Goes Where

| | Root model | Root adapter | `base/` | `checkpoint-N/` |
|---|---|---|---|---|
| **Any LoRA run** | merged (base + adapter) | adapter delta | base model | adapter only |
| **Any Full FT run** | full model | n/a | base model | full model |

## Key Principles

- **Root = final, ready to use.** Load the run dir directly as a HuggingFace model, no assembly required.
- **Root has both merged model AND adapter.** The merged model is a convenience copy for direct loading. The adapter is the actual trained delta for composability.
- **`base/` always exists.** For HF Hub models, contains a `model_ref.json` with the model path and revision (immutable reference). For custom bases (e.g., merged model from `--adapter`), contains the actual model weights.
- **`base/` + `checkpoint-N/` = any intermediate state.** Useful for checkpoint evaluation or rollback.
- **`base/` + root adapter = final state.** The root merged model is just the pre-merged version of this.
- **No distinction between adapter types.** FT, KD, and GRPO adapters are all just LoRA adapters. The `--adapter` flag works with any training method.
- **"Any LoRA run" means any method with LoRA enabled** (FT+LoRA, KD+LoRA, GRPO+LoRA, or chained via `--adapter`).

## Inference Loading

```python
# Simple: load merged model from root
model = AutoModelForCausalLM.from_pretrained("outputs/run_name/")

# Compose: load base + specific checkpoint adapter
model = AutoModelForCausalLM.from_pretrained("outputs/run_name/base/")
model = PeftModel.from_pretrained(model, "outputs/run_name/checkpoint-100/")

# Chain: use this run's adapter as starting point for another run
forge.py --config new_config.yaml --adapter outputs/run_name/
```

## `base/` Contents

| Base model source | `base/` contains |
|---|---|
| **HF Hub model** (e.g., `Qwen/Qwen2.5-VL-3B-Instruct`) | `model_ref.json` with `{"model_name_or_path": "...", "revision": "abc123"}` |
| **Custom model** (e.g., merged from `--adapter`) | Full model weights (config.json + model.safetensors) |

HF Hub model weights are immutable per revision, so a path + revision is a complete, reproducible reference. No need to duplicate gigabytes of weights locally.

## Implementation Notes

- HF Trainer's `save_model()` saves PEFT adapters only (adapter_config.json + adapter_model.safetensors) for LoRA runs. Forge post-processes to also save the merged model and base reference/weights.
- For Full FT runs, `save_model()` saves the complete model. Forge saves the base reference to `base/` for provenance.
- Checkpoints use HF Trainer's default behavior (adapter-only for LoRA, full model for Full FT).
