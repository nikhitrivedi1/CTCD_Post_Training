# Forge vs Unsloth

Forge and [Unsloth](https://github.com/unslothai/unsloth) solve different problems in the model compression pipeline.

| | **Forge** | **Unsloth** |
|---|---|---|
| **Goal** | Produce a smaller model that learns from a larger one | Make fine-tuning the same model faster |
| **Method** | Knowledge distillation (teacher-student training) | Optimized CUDA kernels + memory tricks |
| **Output** | A different, smaller architecture (e.g., BERT → DistilBERT) | Same architecture with adapted weights |
| **Model types** | Any HuggingFace pair (encoders, decoders) | Decoder-only LLMs (Llama, Mistral, Gemma) |
| **Research question** | *What knowledge transfers and how?* | *How fast can we fine-tune?* |
| **Distillation signals** | Feature (MSE), Attention (KL), Response (KL) | N/A — no teacher model involved |
| **Analysis tooling** | Workbench: CKA, probing, disagreement, attention entropy | N/A |
| **Training mode** | Full fine-tune or LoRA, frozen or co-trained teacher | LoRA/QLoRA with fused kernels |

**When to use Forge:**
- You need a structurally smaller model (fewer layers, smaller hidden dim) for deployment
- You want to study *which* distillation signals matter (FAR factorial design)
- You're compressing across architectures (e.g., Llama-3B teaching a custom 500M model)
- You need post-hoc analysis of what knowledge transferred

**When to use Unsloth:**
- You want to fine-tune a single LLM as fast as possible
- You're doing LoRA/QLoRA adaptation and want 2-5x speedups
- You don't need a smaller architecture, just adapted weights

**Using them together:** Unsloth can accelerate the student's training loop inside Forge. Load the student via `FastLanguageModel`, apply LoRA, and pass it to `DistillationTrainer` — you get Unsloth's speed optimizations with Forge's knowledge transfer signals. This is most useful for LLM-scale distillation (e.g., Llama-70B → Llama-3B).
