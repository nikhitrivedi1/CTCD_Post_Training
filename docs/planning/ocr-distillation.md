# OCR Distillation Plan

## Overview

Three-phase plan for OCR knowledge distillation:
1. **Baseline benching** - bench Qwen3.5 models (0.8B, 2B, 4B, 9B) on olmOCR-Bench
2. **KD from olmOCR-7B** - to Qwen3-VL-xB (x = 2, 4, 8) and Qwen3.5-xB (x = 0.8, 2)
3. **Frankenmodel** - Qwen3.5-2B ViT (24L) + Qwen3.5-0.8B LM (24L), or Qwen3-VL-2B ViT + Qwen3-0.6B LM

Eval on [olmOCR-Bench](https://huggingface.co/datasets/allenai/olmOCR-bench) (7,019 tests, 1,403 PDF pages). Training on [olmocr-mix](https://huggingface.co/datasets/allenai/olmOCR-mix-0225) (~260K pages, GPT-4o labels).

## Phase 1: KD olmOCR-7B to Qwen3-VL

### Setup

| | Teacher | Students |
|---|---|---|
| Model | olmOCR-2-7B-1025-FP8 | Qwen3-VL-2B, 4B, 8B |
| Base VLM | Qwen2.5-VL-7B | Qwen3-VL |
| Vocab size | ~152K | ~152K (+ a few extra special tokens) |
| Architecture | Qwen2.5-VL (no DeepStack) | Qwen3-VL (DeepStack) |

### Cross-architecture considerations

**Vocab alignment.** Qwen3-VL has 4 extra special tokens vs Qwen2.5-VL. For KL divergence on logits, slice both to the shared vocab size (min of the two). The codebase already handles this.

**No feature/attention matching.** Different architectures (Qwen2.5-VL vs Qwen3-VL, different hidden dims, different number of layers for some sizes) make intermediate representation matching impractical. Use response-level KD only (KL on output logits).

**Temperature.** Start with T=15 (same as translation KD config). Large output space (152K vocab) benefits from high temperature to soften the distribution.

**Alpha.** Start with alpha=0.2 (80% CE, 20% KD). Same reasoning as translation: high T already softens KL, so less alpha needed.

### Experiments

| Experiment | Student | Data | Method | Expected non-math avg | Rationale |
|------------|:-------:|:----:|--------|:---------------------:|-----------|
| KD-2B-5K | 2B | 5K | CE + KD | 68-70% | Does KD close the 10pp gap vs olmOCR? |
| KD-4B-5K | 4B | 5K | CE + KD | 70-72% | 4B base (56.8%) + training + KD |
| KD-8B-5K | 8B | 5K | CE + KD | 72-74% | 8B base (60.5%) + training + KD |
| CE-4B-5K | 4B | 5K | CE only | 68-70% | CE baseline for 4B to isolate KD effect |
| CE-8B-5K | 8B | 5K | CE only | 70-72% | CE baseline for 8B to isolate KD effect |

ablations_00 (2B, CE-only, 65.3%) is the existing CE baseline for 2B.

### Expected diminishing KD returns with student size

The teacher is 7B. As the student approaches teacher size, KD benefit should shrink:
- **2B student** (3.5x smaller): largest capacity gap, most room for KD to help
- **4B student** (1.75x smaller): moderate gap, KD still useful
- **8B student** (1.14x larger in params): student has MORE params than teacher. KD benefit may be minimal here, mostly providing task-specific signal rather than capacity transfer

The 8B experiment tests whether a larger student can still learn task conventions from a smaller but better-trained teacher.

### Config template

```yaml
model:
  name: Qwen/Qwen3-VL-2B-Instruct  # or 4B, 8B
  teacher:
    name: allenai/olmOCR-2-7B-1025-FP8
    freeze: true

training:
  lora:
    r: 16
    alpha: 16
    dropout: 0.05
    target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  learning_rate: 5e-5
  epochs: 3
  batch_size: 2
  gradient_accumulation_steps: 16
  warmup_steps: 50
  max_seq_length: 1024

distillation:
  alpha: 0.2
  temperature: 15.0

data:
  dataset: allenai/olmOCR-mix-0225
  max_train_samples: 5000
```

## Phase 2: Qwen3.5 - Natively Multimodal Small Models

### Game changer

[Qwen3.5 small models](https://huggingface.co/collections/Qwen/qwen3-5-small-67c40339bb04fcbb94c1eb56) (released 2026-03-02) are **natively multimodal VLMs** with built-in ViTs. This eliminates the need for Frankenmodel surgery at the 0.8B scale.

### Qwen3.5 small model lineup

| Model | LM hidden | LM layers | LM heads | LM KV | ViT hidden | ViT depth | ViT out | DeepStack | Vocab |
|-------|:---------:|:---------:|:--------:|:-----:|:----------:|:---------:|:-------:|:---------:|:-----:|
| Qwen3.5-0.8B | 1024 | 24 | 8 | 2 | 768 | 12 | 1024 | None | 248320 |
| Qwen3.5-2B | 2048 | 24 | 8 | 2 | 1024 | 24 | 2048 | None | 248320 |
| Qwen3.5-4B | 2560 | 32 | 16 | 4 | 1024 | 24 | 2560 | None | 248320 |
| Qwen3.5-9B | 4096 | 32 | 16 | 4 | 1152 | 27 | 4096 | None | 248320 |

Key differences from Qwen3-VL:
- **All sizes are multimodal** (even 0.8B has a ViT)
- **No DeepStack** (empty `deepstack_visual_indexes` across all sizes)
- **Larger vocab** (248K vs 152K) - different tokenizer from Qwen3-VL and olmOCR-7B
- **Mixed attention** (linear + full attention layers alternating) for efficiency
- **262K context** (vs 32K for Qwen3-VL)
- **GQA with fewer KV heads** (2 KV heads vs 8 in Qwen3-VL)

### Architecture comparison: Qwen3-VL vs Qwen3.5

| | Qwen3-VL-2B | Qwen3.5-0.8B | Qwen3.5-2B |
|---|:---:|:---:|:---:|
| LM hidden | 2048 | 1024 | 2048 |
| LM layers | 28 | 24 | 24 |
| LM attn heads | 16 (8 KV) | 8 (2 KV) | 8 (2 KV) |
| ViT hidden | 1024 | 768 | 1024 |
| ViT depth | 24 | 12 | 24 |
| ViT out→LM | 2048 | 1024 | 2048 |
| DeepStack | [5,11,17] | None | None |
| Vocab | 151936 | 248320 | 248320 |
| Context | 32K | 262K | 262K |

### Bench plan

Bench all 4 Qwen3.5 small models (0.8B, 2B, 4B, 9B) on olmOCR-Bench to establish baselines.

**Prerequisite**: `transformers` must support `Qwen3_5ForConditionalGeneration`. Check with:
```bash
python -c "from transformers import AutoModelForImageTextToText; m = AutoModelForImageTextToText.from_pretrained('Qwen/Qwen3.5-0.8B', device_map='cpu', torch_dtype='auto')"
```
If unsupported, upgrade: add latest transformers to `environment.yml`.

### KD considerations for Qwen3.5 students

**Vocab mismatch is the main challenge.** olmOCR-7B uses 152K vocab (Qwen2.5-VL tokenizer), Qwen3.5 uses 248K vocab (Qwen3.5 tokenizer). Options for logit-level KD:

1. **Vocab intersection slicing** - find shared token IDs, compute KL only on the intersection. Lossy, since many tokens map differently.
2. **Text-level KD** - teacher generates text, student trains on generated text as labels (no logit alignment needed). Simpler but loses soft-label information.
3. **Projection-based alignment** - learn a small linear map between vocab spaces. More complex, probably overkill.

Option 2 (text-level) is the pragmatic starting point. Option 1 worth trying if the tokenizers share a large enough intersection.

### Qwen3.5 experiments

| Experiment | Student | Data | Method | Rationale |
|------------|:-------:|:----:|--------|-----------|
| Q3.5-0.8B base | 0.8B | 0 | None | Baseline - does the built-in ViT handle OCR? |
| Q3.5-2B base | 2B | 0 | None | Baseline - compare vs Qwen3-VL-2B (53.1%) |
| CE-0.8B-5K | 0.8B | 5K | CE | How far does CE go on the smallest VLM? |
| CE-2B-5K | 2B | 5K | CE | CE baseline, compare vs ablations_00 (65.3%) |
| KD-0.8B-5K | 0.8B | 5K | CE+KD (text) | Text-level KD from olmOCR-7B |
| KD-2B-5K | 2B | 5K | CE+KD (text) | Text-level KD, compare vs Qwen3-VL KD |

## Phase 3: Frankenmodel

### Updated landscape

With Qwen3.5-0.8B being natively multimodal, the Frankenmodel is no longer necessary for getting a sub-2B VLM. But it's still interesting as a research direction for two reasons:

1. **Stronger ViT on a small LM**: Qwen3.5-0.8B has a 12-layer ViT (hidden=768). Swapping in the 24-layer ViT from Qwen3.5-2B (hidden=1024) could boost visual understanding while keeping the LM small.
2. **Cross-generation surgery**: Qwen3-VL ViT (with DeepStack) + Qwen3-0.6B LM is still a valid experiment for architecture research.

### Option A: Qwen3.5-2B ViT + Qwen3.5-0.8B LM (clean swap)

| Component | Qwen3.5-2B (donor ViT) | Qwen3.5-0.8B (target LM) | Match? |
|-----------|:----------------------:|:------------------------:|:------:|
| ViT out_hidden_size | 2048 | LM hidden=1024 | **Mismatch** |
| ViT hidden_size | 1024 | - | - |
| ViT depth | 24 | (native: 12) | Upgrade |
| LM layers | 24 | 24 | Match |
| Vocab | 248320 | 248320 | Match |
| DeepStack | None | None | Match |

**Surgery**: Only the standard merger output projection needs resizing (2048 → 1024). No DeepStack mergers to worry about. Same-family tokenizer, no vocab mismatch.

### Option B: Qwen3-VL-2B ViT + Qwen3-0.6B LM (cross-generation, original plan)

| Component | Qwen3-VL-2B (donor ViT) | Qwen3-0.6B (target LM) | Match? |
|-----------|:----------------------:|:----------------------:|:------:|
| LM hidden_size | 2048 | 1024 | Mismatch |
| LM layers | 28 | 28 | Match |
| LM attn heads | 16 (8 KV) | 16 (8 KV) | Match |
| Vocab | 151936 | 151936 | Match |
| DeepStack | [5,11,17] | - | 4 mergers to fix |

**Surgery**: 4 merger projections (3 DeepStack + 1 standard) need resizing. More complex but same-vocab, no tokenizer issues.

### Qwen3-VL DeepStack vision pipeline (Option B reference)

```
Image → ViT (24 layers, hidden=1024)
          |
          +- Layer 5  → deepstack_merger[0] -> residual add at LM decoder layer 0
          +- Layer 11 → deepstack_merger[1] -> residual add at LM decoder layer 1
          +- Layer 17 → deepstack_merger[2] -> residual add at LM decoder layer 2
          |
          +- Layer 24 → standard merger -----> replaces <image> placeholder tokens

Each merger: ViT_hidden * spatial_merge^2 -> out_hidden_size (= LM hidden_size)
For Qwen3-VL-2B: 1024 * 2^2 = 4096 -> 2048. Surgery: 4096 -> 1024.
```

### Option B surgery table

| Component | Current (for 2B LM) | New (for 0.6B LM) | Notes |
|-----------|:-------------------:|:-----------------:|-------|
| deepstack_merger[0] | 4096 → 2048 | 4096 → **1024** | Reinitialize final Linear |
| deepstack_merger[1] | 4096 → 2048 | 4096 → **1024** | Reinitialize final Linear |
| deepstack_merger[2] | 4096 → 2048 | 4096 → **1024** | Reinitialize final Linear |
| standard merger | 4096 → 2048 | 4096 → **1024** | Reinitialize final Linear |
| LM decoder | 2B (28L, d=2048) | **0.6B (28L, d=1024)** | Swap entire decoder |
| lm_head | (2048, 151936) | **(1024, 151936)** | Comes with Qwen3-0.6B |

### Assembly pseudocode (Option B)

```python
from transformers import Qwen3VLForConditionalGeneration, AutoModelForCausalLM
import torch.nn as nn

# Load donor VLM and target LM
vlm = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
lm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

# 1. Replace LM decoder weights
vlm.model.layers = lm.model.layers          # 28 layers, d=1024
vlm.model.embed_tokens = lm.model.embed_tokens  # (151936, 1024)
vlm.model.norm = lm.model.norm              # RMSNorm(1024)
vlm.lm_head = lm.lm_head                    # (1024, 151936)

# 2. Update config
vlm.config.hidden_size = 1024
vlm.config.intermediate_size = 3072
vlm.config.vision_config.out_hidden_size = 1024

# 3. Reinitialize merger output projections (4096 -> 1024)
vlm.visual.merger.linear_fc2 = nn.Linear(4096, 1024)
for ds_merger in vlm.visual.deepstack_merger_list:
    ds_merger.linear_fc2 = nn.Linear(4096, 1024)

# 4. Freeze ViT, train mergers + LoRA on LM
for p in vlm.visual.blocks.parameters():
    p.requires_grad = False

# 5. Save
vlm.save_pretrained("Qwen3-VL-0.6B-Franken")
```

### Training plan (both options)

**Stage 1: Alignment (merger only)**
- Freeze ViT + freeze LM, train only merger projection(s)
- 1 epoch, 5K samples, LR 1e-4 (higher for randomly initialized layers)

**Stage 2: Fine-tuning (merger + LoRA)**
- Freeze ViT, train merger + LoRA on LM decoder
- 3 epochs, 5K-50K samples, LR 5e-5

**Stage 3 (optional): KD from olmOCR-7B**
- Add KL loss from teacher. For Option B (same vocab), logit-level KD works directly.
  For Option A (Qwen3.5, different vocab), use text-level KD.

### Frankenmodel comparison

| Variant | ViT | LM | Merger surgery | Vocab match | DeepStack |
|---------|-----|:--:|:--------------:|:-----------:|:---------:|
| **Option A** | Qwen3.5-2B (24L, 1024) | Qwen3.5-0.8B (24L, 1024) | 1 projection | Yes (248K) | No |
| **Option B** | Qwen3-VL-2B (24L, 1024) | Qwen3-0.6B (28L, 1024) | 4 projections | Yes (152K) | Yes (3 mergers) |
| **Baseline** | Qwen3.5-0.8B native (12L, 768) | Qwen3.5-0.8B (24L, 1024) | None | Yes (248K) | No |

Option A is the recommended path: simpler surgery, same family, no DeepStack complexity.
The baseline (native Qwen3.5-0.8B) should be tried first, since no surgery is needed at all.

### Risks

1. **Qwen3.5-0.8B's ViT may be sufficient.** The 12-layer ViT might handle OCR well enough that swapping in a bigger ViT isn't worth the complexity. Bench first.
2. **Merger training instability.** Randomly initialized projections feeding a pretrained LM. Mitigated by the two-stage approach.
3. **Option B DeepStack complexity.** 4 merger projections to retrain, and the injection pattern was co-trained with a different LM.

### Ablations

| Variant | Description | Tests |
|---------|-------------|-------|
| Native 0.8B ViT vs 2B ViT | Qwen3.5-0.8B native vs Option A Frankenmodel | Is the bigger ViT worth it? |
| Option A vs Option B | Qwen3.5 family vs Qwen3-VL family | Which donor ViT is better? |
| No DeepStack (Option B) | Disable deepstack, standard merger only | Is DeepStack worth it for 0.6B? |
| Full ViT fine-tune | Unfreeze ViT in Stage 2 | Does ViT adaptation help? |

## Eval Plan

All models evaluated on olmOCR-Bench. Three metrics tracked:
- **Per-JSONL Average (non-math)** - 5 categories, primary metric
- **Extraction Avg. (w/o H/F)** - 4 categories, fairer extraction metric
- **Per-category breakdown** - for diagnosing where gains come from

### Comparison matrix

| Model | Family | Params | Method | Non-math Avg | Extraction Avg. |
|-------|--------|:------:|--------|:------------:|:---------------:|
| **Existing baselines** | | | | | |
| Qwen3-VL-2B base | Qwen3-VL | 2B | None | 53.1% | 54.8% |
| Qwen3-VL-4B base | Qwen3-VL | 4B | None | 56.8% | 61.6% |
| Qwen3-VL-8B base | Qwen3-VL | 8B | None | 60.5% | 65.8% |
| ablations_00 | Qwen3-VL | 2B | CE 5K | 65.3% | 61.8% |
| olmOCR-7B (ceiling) | Qwen2.5-VL | 7B | CE+RL full | 75.3% | 71.0% |
| **Qwen3.5 baselines (to bench)** | | | | | |
| Qwen3.5-0.8B base | Qwen3.5 | 0.8B | None | ? | ? |
| Qwen3.5-2B base | Qwen3.5 | 2B | None | ? | ? |
| Qwen3.5-4B base | Qwen3.5 | 4B | None | ? | ? |
| Qwen3.5-9B base | Qwen3.5 | 9B | None | ? | ? |
| **Phase 1: KD to Qwen3-VL** | | | | | |
| KD-2B-5K | Qwen3-VL | 2B | CE+KD 5K | 68-70% | 64-66% |
| CE-4B-5K | Qwen3-VL | 4B | CE 5K | 68-70% | 66-68% |
| KD-4B-5K | Qwen3-VL | 4B | CE+KD 5K | 70-72% | 68-70% |
| CE-8B-5K | Qwen3-VL | 8B | CE 5K | 70-72% | 68-70% |
| KD-8B-5K | Qwen3-VL | 8B | CE+KD 5K | 72-74% | 70-72% |
| **Phase 2: Qwen3.5 training** | | | | | |
| CE-Q3.5-0.8B-5K | Qwen3.5 | 0.8B | CE 5K | ? | ? |
| CE-Q3.5-2B-5K | Qwen3.5 | 2B | CE 5K | ? | ? |
| KD-Q3.5-0.8B-5K | Qwen3.5 | 0.8B | CE+KD (text) 5K | ? | ? |
| **Phase 3: Frankenmodel** | | | | | |
| Franken-A (Q3.5 2B ViT + 0.8B LM) | Qwen3.5 | ~0.8B | CE 5K | ? | ? |
| Franken-B (Q3-VL 2B ViT + Q3-0.6B LM) | Qwen3/0.6B | ~0.6B | CE 5K | ? | ? |

### Key questions the experiments answer

1. **Qwen3.5 vs Qwen3-VL baseline**: Does the newer architecture (mixed attention, early-fusion multimodal, 262K context) perform better at comparable sizes?
2. **Does KD help beyond CE?** Compare CE-only vs CE+KD at each student size.
3. **How does KD benefit scale with student size?** Compare KD delta across 2B, 4B, 8B.
4. **Can a larger student learn from a smaller teacher?** 8B student from 7B teacher.
5. **Is Qwen3.5-0.8B's native ViT (12L) sufficient?** Compare native 0.8B vs Franken-A (with 2B's 24L ViT).
6. **Text-level KD across tokenizers**: Does it work when logit-level KD is blocked by vocab mismatch?

## References

- [olmOCR-Bench results](../analysis/summaries/olmocr_bench_results.md) - current baseline scores
- [olmOCR GitHub](https://github.com/allenai/olmocr) - scoring code
- [DeepStack paper](https://arxiv.org/abs/2406.04334) - "Deeply Stacking Visual Tokens" (NeurIPS 2024)
- [Qwen3-VL technical report](https://arxiv.org/abs/2511.21631)
- [Qwen3-VL-2B config](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/main/config.json)
- [Qwen3-0.6B config](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json)
- [Qwen3.5 small models announcement](https://www.marktechpost.com/2026/03/02/alibaba-just-released-qwen-3-5-small-models-a-family-of-0-8b-to-9b-parameters-built-for-on-device-applications/)
- [Qwen3.5-0.8B config](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json)
- [Qwen3.5-2B config](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json)
- [Qwen3.5-4B config](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json)
- [Qwen3.5-9B config](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json)
- [Qwen3.5 GitHub](https://github.com/QwenLM/Qwen3.5)
