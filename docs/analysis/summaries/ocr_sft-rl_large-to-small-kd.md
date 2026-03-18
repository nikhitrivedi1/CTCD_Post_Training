# RL on Large Model, KD into Small: Plan (2026-03-12)

Training RL (GRPO) on a larger VL model, then distilling into Qwen3-VL-2B via KD. Motivated by two failed GRPO attempts on 2B and validated by DeepSeek R1's approach.

## Status (2026-03-17)

Phase 1 and Phase 2 are complete for Qwen3.5-9B:

| Run | Status | Notes |
|-----|--------|-------|
| `ocr_lora_qwen35-9b` (Phase 1, Stage 1) | Done | OCR SFT on 5K olmocr-mix |
| `fmt_lora_qwen35-9b` (Phase 1, Stage 2) | Done | Formatting SFT, 3 epochs |
| `fmt_grpo_qwen35-9b` (Phase 2) | Done | GRPO with edit_sim + fmt_pertype |
| Formatting bench (all 3 runs) | **Invalid** - pending re-run | `</think>` contamination (172-187/250 files); model_profiles fix added 2026-03-17 |

Re-runs of all three formatting bench jobs are queued. The GRPO reward regression pattern (see `fmt_grpo_regression.md`) needs to be resolved before Phase 2 results are trustworthy as a teacher signal - E1-E4 experiments will inform whether `fmt_grpo_9b` or `fmt_lora_9b` is the better Phase 3 teacher.

**Available baselines (clean):**

| Model | Overall F1 | Pred spans (bold/italic/strike) | Notes |
|-------|:----------:|:-------------------------------:|-------|
| `pt_qwen35-0.8b` | 0.258 | 903 / 149 / 0 | Base, no training |
| `pt_qwen35-2b` | 0.277 | 509 / 325 / 0 | Base, no training |
| `pt_qwen35-9b` | pending | - | Running (job 5140618) |

These establish the Qwen3.5 family baseline before any OCR/formatting training.

## Motivation

Two GRPO runs on Qwen3-VL-2B have failed to improve on fmt_sft:

| Run | Overall F1 | vs fmt_sft |
|-----|:----------:|:----------:|
| fmt_sft (baseline) | 0.766 | - |
| fmt_grpo (v1, pooled reward, beta=0.0) | 0.623 | -0.143 |
| fmt_grpo_v2 (v2, per-type reward, beta=0.1) | TBD (stalled) | - |

v1 suffered entropy collapse from pooled reward + no KL penalty. v2 fixed those issues but training stalled at step 270/594 with entropy already at 0.048, formatting reward barely moving (0.304 to 0.317). The 2B model may lack the capacity to explore diverse formatting strategies through RL.

### DeepSeek R1 Precedent

DeepSeek R1 validated the "RL large, distill small" pipeline for reasoning:

1. Cold-start SFT on DeepSeek-V3-Base (671B MoE)
2. GRPO with rule-based rewards (accuracy + format)
3. Rejection sampling from RL model to generate new SFT data
4. Distill into dense models (1.5B through 70B) via SFT on generated data

**Key finding**: Distilled 7B and 14B models outperformed their directly-RL-trained counterparts. Small models can't discover complex behaviors through RL exploration, but can learn them through imitation once a larger model has figured them out.

### Why This Should Transfer to Formatting

- Formatting requires maintaining diverse output patterns (bold, italic, strikethrough) simultaneously. Small models collapse to the dominant pattern during RL.
- Larger models have more representational headroom to maintain pattern diversity under reward pressure.
- The distillation step is cheap (just SFT on teacher outputs) and the student already has the visual perception needed (fmt_sft proves Qwen3-VL-2B can do formatting with the right training signal).

---

## Models

### Available VL Models

The RL teacher must be a vision-language model (needs to see document images). Qwen3-VL has dedicated VL models. Qwen3.5 is natively multimodal (no separate "-VL" line needed).

#### Qwen3-VL (Sep-Oct 2025)

| Model | Params | Weights (bf16) | Type |
|-------|:------:|:--------------:|:----:|
| Qwen3-VL-2B | 2.1B | 4GB | Dense |
| Qwen3-VL-4B | ~4B | ~8GB | Dense |
| Qwen3-VL-8B | ~8B | ~16GB | Dense |
| Qwen3-VL-32B | ~32B | ~64GB | Dense |
| Qwen3-VL-30B-A3B | 30B (3B active) | ~60GB | MoE |
| Qwen3-VL-235B-A22B | 235B (22B active) | ~470GB | MoE |

All have Instruct + Thinking variants.

#### Qwen3.5 (Feb-Mar 2026, natively multimodal)

All Qwen3.5 models are Image-Text-to-Text. No separate VL line.

| Model | Params | Weights (bf16) | Type |
|-------|:------:|:--------------:|:----:|
| Qwen3.5-0.8B | 0.9B | ~2GB | Dense |
| Qwen3.5-2B | 2B | ~4GB | Dense |
| Qwen3.5-4B | 5B | ~10GB | Dense |
| Qwen3.5-9B | 10B | ~20GB | Dense |
| Qwen3.5-27B | 28B | ~56GB | Dense |
| Qwen3.5-35B-A3B | 36B (3B active) | ~72GB | MoE |
| Qwen3.5-122B-A10B | 125B (10B active) | ~250GB | MoE |
| Qwen3.5-397B-A17B | 403B (17B active) | ~806GB | MoE |

### Teacher Candidates (for RL training, dense only)

| Model | Params | Weights | Why |
|-------|:------:|:-------:|-----|
| **Qwen3.5-9B** | 10B | ~20GB | Latest generation, native VL, strong benchmarks for size |
| **Qwen3-VL-8B** | ~8B | ~16GB | Same family as current 2B student |
| **Qwen3.5-27B** | 28B | ~56GB | Much more capacity, still fits 1-2 GPUs |
| **Qwen3-VL-32B** | ~32B | ~64GB | Largest single-family option with current student |

### Student Model

Primary target: **Qwen3.5-2B** (same family as the 9B teacher). Secondary: **Qwen3.5-0.8B** for a lighter deployment option.

Rationale for switching from Qwen3-VL-2B to Qwen3.5-2B:
- Same architecture family as the teacher - representations, tokenizer, and attention mechanism match, making KD transfer more direct
- Natively multimodal (no separate vision encoder bolt-on, unlike Qwen3-VL)
- Base model baselines are now available: pt_qwen35-0.8b (F1=0.258) and pt_qwen35-2b (F1=0.277). These are pre-training baselines, not pre-SFT - both families start near zero on formatting before OCR/fmt SFT
- Qwen3-VL-2B is still a valid comparison point (fmt_sft F1=0.766 remains the target to beat)

**Student pipeline (same two-stage structure as teacher):**

```
Qwen3.5-2B base
    |
    v  Stage 1: OCR SFT on olmocr-mix (grounding)
ocr_sft_qwen35-2b
    |
    v  Stage 2: Formatting SFT (baseline for KD comparison)
fmt_sft_qwen35-2b
    |
    v  Phase 4: KD from 9B teacher outputs
fmt_kd_qwen35-2b
```

Run the same pipeline on Qwen3.5-0.8B in parallel if GPU time allows - the 0.8B is fast enough that both students can be evaluated without significant overhead.

---

## Data

### Available Data on Cluster

#### Formatting (synthetic, generated)

Location: `data/ocr/formatting/`

| Category | Train | Eval | Content |
|----------|:-----:|:----:|---------|
| 00_text | 475 | 25 | General prose, mixed inline formatting |
| 01_tables | 475 | 25 | Table-heavy documents |
| 02_academic | 475 | 25 | Papers, citations, key terms |
| 03_financial | 475 | 25 | Reports, bold totals, italic footnotes, strikethrough |
| 04_legal | 475 | 25 | Section refs, definitions |
| 05_technical | 475 | 25 | API docs, deprecated items |
| 06_books | 475 | 25 | Chapters, blockquotes |
| 07_slides | 475 | 25 | Presentations (high bold density) |
| 08_news | 475 | 25 | Headlines, bylines |
| 09_forms | 475 | 25 | Forms, exams, labels |
| **Total** | **4,750** | **250** | |

Each sample = PNG image + markdown GT. Generated by `scripts/data/formatting_generator.py` (markdown -> HTML+CSS -> Playwright render -> PNG). Size: 2.0GB.

Formatting density varies by category: bold (0.2-0.6), italic (0.1-0.5), strikethrough (0.02-0.3). Technical and financial have the most strikethrough. Slides and news have the most bold. Academic and books have the most italic.

#### olmocr-mix-full (real documents)

Location: `data/ocr/olmocr-mix-full/`

| Category | Train | Eval | Size |
|----------|------:|-----:|:----:|
| 00_documents | 231,668 | 1,122 | 78GB |
| 01_books | 16,575 | 899 | 6.8GB |
| 02_loc_transcripts | 9,891 | 98 | 9.1GB |
| 03_national_archives | 9,828 | 169 | 9.9GB |
| **Total** | **267,962** | **2,288** | **104GB** |

Arrow format (HF datasets). GT is GPT-4o generated plain text (no formatting markup). Used for ocr_sft baseline.

#### OmniDocBench (real documents, with formatting)

Location: `data/ocr/omnidocbench/`

1,355 pages spanning PPTs, papers, textbooks, financial reports, newspapers. Images (JPG) + converted markdown GT from structured annotations. 3.0GB.

The formatting bench (250 pages) is the subset of OmniDocBench pages that contain inline formatting in their GT.

#### olmocr-bench (evaluation only)

Location: `data/olmocr-bench/`

5,997 unit tests across 1,403 PDFs. Not training data.

### Stratified Subsets

All training phases use the same tiered approach. Subsets are stratified by category to maintain formatting type diversity.

| Tier | Samples | Composition | Purpose |
|------|:-------:|-------------|---------|
| **Smoke** | 50 | 5 per category | Validate pipeline runs end-to-end |
| **Small** | 250 | 25 per category | Quick iteration, initial quality signal |
| **Medium** | 1,000 | 100 per category | Confidence build, close to current 2B baseline |
| **Full** | 4,750 | 475 per category | Maximum available synthetic data |

Why start small:
- 9B SFT on 4,750 samples takes ~4-8 hours. On 250 samples, ~30-60 minutes.
- If prompt format is wrong, you waste hours instead of minutes.
- Small subset results tell you whether the setup works before committing to full training.

### GT Generation for olmocr-mix (Formatting-Aware)

olmocr-mix GT is GPT-4o generated plain text with no formatting markup. To use olmocr-mix images in the KD pipeline, we need formatting-aware GT. Several approaches, ranked by practicality:

#### Option 1: RL-Trained Teacher (Phase 2 output)

The primary plan. After Phase 2, the RL-trained teacher generates formatted markdown on olmocr-mix images. Quality depends on how well Phase 1-2 went. This is the GT used in Phase 3/4.

#### Option 2: Large Model Zero-Shot Inference

Use a larger model than the teacher purely for inference (no training). Since inference needs only weights + KV cache, much larger models fit on 1x H200.

**Model comparison for inference-only GT generation:**

| | Qwen3-VL-32B | Qwen3.5-27B | Qwen3.5-35B-A3B |
|---|---|---|---|
| Params | 32B dense | 28B dense | 36B total, 3B active (MoE) |
| BF16 memory | ~64GB | ~56GB | ~72GB |
| 4-bit quantized | ~16GB | ~14GB | ~18GB |
| Inference speed | Baseline | ~15% faster | ~10x faster (3B active) |
| Formatting quality | Strong | Best (newest arch + training) | Weak (only 3B active per token) |
| Family match | Qwen3 (same as student) | Qwen3.5 (newer, different tokenizer) | Qwen3.5 |

MoE models like 35B-A3B have only 3B active params per token, comparable to our 2B student, so they don't offer a quality advantage for structured output tasks despite loading 36B total params into memory.

**Recommendation: Qwen3.5-27B BF16 (~56GB on 1x H200).** Largest dense model from the latest generation, natively multimodal. At ~56GB, leaves 84GB headroom for KV cache and batching (BS 8-16). No quantization noise.

Qwen3-VL-32B BF16 (~64GB) is an alternative: 32B dense, same family as the student (Qwen3), also fits on 1x H200. Slightly older generation but more params. Worth testing both.

No reason to use either at 4-bit since we have plenty of H200 memory for BF16.

#### Option 3: fmt_sft (2B) with Filtering

Use the existing fmt_sft checkpoint to generate formatting-aware GT. Filter outputs using reward functions (edit_similarity > threshold, formatting F1 > threshold). Fast (~2h for 10K images on 1x H200) but limited by 2B capacity. Good for a quick baseline.

#### Option 4: Frontier API (GPT-4o, Claude)

Use an API model with a formatting-aware prompt. Highest quality ceiling but costs money ($2-5 per 1K images) and adds external dependency. Best used for a small validation set (500-1K images) to benchmark other approaches against.

#### GT Quality Evaluation Protocol

To compare GT from different sources, pick a shared evaluation set (e.g., 100 OmniDocBench images that have human-annotated formatting GT):

1. Generate outputs from each source (fmt_sft, Qwen3.5-27B, RL teacher, API)
2. Score all outputs with reward functions (edit_similarity, inline_formatting_pertype F1)
3. Manual inspection of 20-30 samples: check for hallucinated formatting, missed formatting, structural errors
4. Compare against OmniDocBench GT as reference

The source with the highest reward scores and fewest manual-inspection failures becomes the primary GT generator.

#### Recommended GT Strategy

1. Start with **Option 3** (fmt_sft) for a quick 10K-sample baseline with filtering
2. Try **Option 2** (Qwen3.5-27B BF16) for higher quality at moderate cost (GPU time only)
3. Use **Option 4** (API) on 500 images as quality reference to compare Options 2-3 against
4. Once Phase 2 completes, **Option 1** (RL teacher) becomes the primary path

Note: Both olmocr-mix GT (GPT-4o) and fmt_sft outputs are silver-standard (model-generated, not human-annotated). Any new GT we generate is also silver-standard. Quality is relative, not absolute.

### Scaling the Distillation Dataset

| Tier | Source | Samples | Purpose |
|------|--------|:-------:|---------|
| **Small** | Formatting train only | 4,750 | Validate KD pipeline works |
| **Medium** | Formatting + OmniDocBench | 6,105 | Add real-world document diversity |
| **Large** | Formatting + OmniDocBench + olmocr-mix 10K | 16,105 | Maximize diversity and volume |
| **XL** | All above + more olmocr-mix | 30,000+ | If more data clearly helps |

Start Small, benchmark, then scale.

### Generating More Synthetic Data

The formatting generator (`scripts/data/formatting_generator.py`) can produce unlimited synthetic data. Current dataset is 475 train per category (4,750 total). If more data helps:

| Scale | Per Category | Total | Generation Time |
|-------|:-----------:|:-----:|:---------------:|
| Current | 475 | 4,750 | ~2 hours |
| 2x | 1,000 | 10,000 | ~4 hours |
| 5x | 2,500 | 25,000 | ~10 hours |
| 10x | 5,000 | 50,000 | ~20 hours |

Generation runs on CPU (Playwright rendering). Can run on login node or a CPU job.

**When to generate more**: If the teacher SFT (Phase 1) or GRPO (Phase 2) plateaus and we suspect data diversity is the bottleneck rather than model capacity. Try the full 4,750 first before generating more.

---

## Compute

### Memory Estimates for GRPO with LoRA

GRPO memory budget per GPU (single-GPU case):

| Component | Formula | Notes |
|-----------|---------|-------|
| Model weights | 2 * P GB (bf16) | Frozen backbone |
| LoRA adapters | ~0.02 * P GB | r=16, 7 target modules |
| Optimizer (AdamW for LoRA) | ~0.06 * P GB | params + 2 moments |
| KV cache (4 gens, 1024 tok) | ~0.5-2 GB | Scales with layers/heads |
| Training activations | ~2-3 * P GB | With gradient checkpointing |
| Reference forward (beta>0) | ~1 * P GB | No grad, activation memory only |
| CUDA overhead | ~10-15 GB | Fragmentation, buffers |

Rule of thumb: total memory ~4-5x model weights + 15GB overhead.

SFT is cheaper than GRPO (no generation, no KV cache for multiple completions, no reference forward). SFT memory is roughly ~3x model weights + 10GB overhead.

### Feasibility Matrix: 1x H200 (140GB)

| Model | Weights (bf16) | GRPO Est. | SFT Est. | GRPO Verdict |
|-------|:--------------:|:---------:|:--------:|---------|
| Qwen3.5-4B | ~10GB | ~55GB | ~40GB | Easy |
| Qwen3-VL-8B | ~16GB | ~80GB | ~55GB | Comfortable |
| Qwen3.5-9B | ~20GB | ~95GB | ~70GB | Comfortable |
| Qwen3.5-27B | ~56GB | ~250GB | ~180GB | Does not fit |
| Qwen3-VL-32B | ~64GB | ~280GB | ~200GB | Does not fit |

### Feasibility Matrix: Multi-GPU with FSDP

FSDP (ZeRO-3) shards weights, gradients, and optimizer across GPUs. Activations remain per-GPU but shrink with smaller micro-batch. Communication overhead ~15-25%.

Estimated peak GRPO memory per GPU:

| Model | 2x H200 | 4x H200 | 8x H200 |
|-------|:-------:|:-------:|:-------:|
| 8-9B | ~55GB | ~35GB | ~25GB |
| 27B | ~140GB | ~75GB | ~45GB |
| 32B | ~160GB | ~90GB | ~55GB |
Usable per GPU: ~130GB (140GB minus OS/driver).

| Model | 1x | 2x | 4x | 8x |
|-------|:--:|:--:|:--:|:--:|
| 8-9B | Yes | Yes | Yes | Yes |
| 27B | No | Tight | Yes | Yes |
| 32B | No | No | Yes | Yes |
**"Yes" = fits with margin. "Tight" = fits with gradient checkpointing, minimal batch size. "No" = does not fit.**

### Explorer Cluster Partitions

| Partition | GPUs | Time Limit | Notes |
|-----------|:----:|:----------:|-------|
| gpu | 1 | 8h | Current setup for 2B training |
| multigpu | up to 8 | 24h | Required for all multi-GPU configs below |

### Training Time Estimates

**Baseline**: Qwen3-VL-2B on 1x H200 (`gpu` partition), GRPO on 250 formatting samples, 4 generations per prompt, 1024 max completion tokens. Takes ~16 hours (two 8h jobs with auto-resubmit).

**Scaling rules**:
- **Model size**: Time scales roughly linearly with parameter count
- **GPU count**: ~80% parallel efficiency per GPU added (communication overhead from FSDP/TP)
  - 2x GPUs = 1.6x speedup
  - 4x GPUs = 3.2x speedup
  - 8x GPUs = 6.4x speedup

Estimated total GRPO training time (hours), derived from 2B baseline (16h on 1x H200):

| Model | 1-GPU theoretical | 2x H200 (1.6x) | 4x H200 (3.2x) | 8x H200 (6.4x) |
|-------|:-----------------:|:---------------:|:---------------:|:---------------:|
| 2B (baseline) | 16h | 10h | 5h | 3h |
| 8-9B (~5x params) | 80h | 50h | 25h | 13h |
| 27B (~14x params) | 224h | 140h | 70h | 35h |
| 32B (~16x params) | 256h | 160h | 80h | 40h |
### SLURM Jobs (24h limit, auto-resubmit)

| Model | 2x H200 | 4x H200 | 8x H200 |
|-------|:-------:|:-------:|:-------:|
| 2B | 1 | 1 | 1 |
| 8-9B | 3 | 2 | 1 |
| 27B | 6 | 3 | 2 |
| 32B | - | 4 | 2 |
Multi-GPU setups default to 24h jobs. An 8h limit is also available and useful for faster queue turnaround or testing, at the cost of more resubmits and job init overhead.

### Recommended Configurations

#### Tier 1: Conservative (fastest iteration)

**Qwen3.5-9B on 2x H200, 24h limit**

- Memory: ~55GB per GPU, within 130GB usable
- Time: ~50h total, 3 jobs at 24h limit
- Why: Latest generation, natively multimodal, 5x current student, strong benchmarks for size class. Qwen3.5-9B matches or surpasses models 13x its size on reasoning benchmarks.
- Risk: Low. If RL works at all on a larger model, 9B is enough to demonstrate it.
- Alternative: Qwen3-VL-8B if staying within the Qwen3 family matters (same generation as student).

#### Tier 2: Moderate (more capacity, practical timeline)

**Qwen3.5-27B on 4x H200, 24h limit**

- Memory: ~75GB per GPU, comfortable
- Time: ~65-75h total, 3 jobs at 24h limit
- Why: 14x the student's capacity. 262K context window. Much more representational headroom to maintain formatting diversity under RL reward pressure.
- Risk: Moderate. Longer iteration cycle, but if 9B fails and 27B succeeds, it confirms the capacity hypothesis.

#### Tier 3: Ambitious (maximum teacher quality)

**Qwen3-VL-32B on 4x H200, 24h limit**

- Memory: ~90GB per GPU, fits with margin
- Time: ~80-90h total, 4 jobs at 24h limit
- Why: Largest Qwen3-VL dense model, same family as student (Qwen3), Thinking variant available for reasoning-enhanced generation.
- Risk: Long turnaround. Consider only if Tier 2 shows promising but insufficient results.

#### Recommendation

Start with **Tier 1 (Qwen3.5-9B on 2x H200)**. It's the newest model, natively multimodal, and fast enough to iterate. If RL succeeds, distill into 2B and bench. If it fails, try Tier 2 (27B) to test whether more capacity helps. If both fail, the problem is reward design, not model capacity.

---

## Pipeline

### Phase 1: Two-Stage LoRA SFT on Large Teacher

**Goal**: Teach the large model (Qwen3.5-9B) to extract text and produce formatted markdown from document images, mirroring the 2B pipeline.

Two-stage LoRA SFT, same approach as the 2B pipeline:
1. **ocr_sft**: 5K olmocr-mix samples for text extraction grounding
2. **fmt_sft**: Full 4,750 formatting samples for inline formatting

```
Qwen3.5-9B (natively multimodal)
    |
    v  Stage 1: LoRA SFT on 5K olmocr-mix (text extraction)
ocr_sft_9b
    |
    v  Stage 2: LoRA SFT on formatting data (4,750 samples)
fmt_sft_9b
```

- Hardware: 1x H200 (9B SFT fits easily, ~60-70GB peak)
- Time: Stage 1 ~2-4 hours (5K samples), Stage 2 ~4-8 hours (Full 4,750). Total ~8-12 hours across two 8h jobs.
- Config: Adapt existing SFT configs for 9B (adjust batch size, LR)
- Validate: Run formatting bench after Stage 2, confirm it matches or exceeds 2B's fmt_sft (F1 ~0.766)

For **27B**, same two-stage approach but needs 2x H200 with FSDP. Stage 1 ~6-10 hours, Stage 2 ~8-14 hours. Total ~15-25 hours across two 24h multigpu jobs.

**Stage 2 Progression**: Go straight to Full (4,750). Total fmt_sft time is only ~4-8 hours, so tiered progression adds overhead without saving meaningful time. Run Smoke (50) only if debugging config issues.

**Why two-stage?** The 2B pipeline proved that OCR grounding (Stage 1) before formatting (Stage 2) produces better results than formatting-only SFT. Even though Qwen3.5-9B has stronger pretrained OCR ability, the olmocr-mix stage provides domain-specific grounding on real document layouts that complements the synthetic formatting data. The cost is low (~2-4 hours) and the risk of skipping it isn't worth it.

### Phase 2: GRPO on Large Teacher

**Goal**: RL to improve the teacher's formatting quality beyond what SFT achieves.

```
fmt_sft_9b
    |
    v  GRPO with inline_formatting_pertype + edit_similarity
fmt_grpo_9b
```

- Hardware: 2x H200 (Tier 1 recommendation)
- Time: ~50 hours (3 jobs at 24h limit)
- Config: Adapt 01_grpo.yaml for 9B, may need to adjust:
  - `generation_batch_size`: reduce if OOM during generation
  - `gradient_accumulation_steps`: reduce (multi-GPU effectively increases batch)
  - `beta`: start at 0.1, adjust based on entropy trajectory
- Watch: Entropy (should stay >0.3), per-type formatting reward (all types should improve), span counts per type

**Progression** (same tiers as Phase 1):
1. **Smoke (50, max_steps=10)**: Verify GRPO runs with the large model. Rewards compute, gradients flow, checkpoints save.
2. **Small (250)**: First real RL run. Same size used for 2B GRPO. If 9B GRPO on 250 improves over 9B SFT, the approach works.
3. **Medium (1,000)**: More prompt diversity should help learn more generalizable formatting patterns.
4. **Full (4,750)**: Only if Medium plateaus and we suspect prompt diversity is the bottleneck.

**Why subset size matters more for RL**: In GRPO, each sample is a prompt that generates N completions scored by the reward function. The model learns from the variance across completions, not from memorizing GT. Too few prompts = overfit to specific layouts. Too many = noisier advantage estimates but better generalization.

With 4 generations per prompt and 1 epoch:
- 250 prompts = 1,000 total generations
- 1,000 prompts = 4,000 total generations
- 4,750 prompts = 19,000 total generations

**Data considerations**:
- **Category balance in prompts matters**: Bold-heavy categories (slides, news) are "easier." Stratified sampling ensures balanced exposure to italic/strikethrough.
- **Eval set as held-out reward check**: Use the 250 eval samples (25 per category). If training reward improves but eval reward doesn't, the model is overfitting to specific layouts.

### Phase 3: Generate Teacher Outputs

**Goal**: Use the RL-trained teacher (or a larger inference-only model) to generate high-quality formatted markdown for student distillation.

```
fmt_grpo_9b + training images     OR     Qwen3.5-27B / Qwen3-VL-32B + training images
    |                                        |
    v  Inference (greedy or rejection)        v  Zero-shot inference (BF16)
teacher_outputs/  (image -> markdown pairs)
```

- Hardware: 1x H200 (inference only)
- Time: ~1-2 hours for formatting data (4,750), ~2-3 hours for 10K olmocr-mix
- Multiple GT generation strategies available (see Data > GT Generation section above)
- Can generate for unlabeled olmocr-mix images to expand the training set beyond existing GT

**What to generate on:**

| Dataset | Samples | Why |
|---------|:-------:|-----|
| Formatting train | 4,750 | Core formatting data, replace synthetic GT with teacher GT |
| Formatting eval | 250 | Held-out quality check |
| OmniDocBench | 1,355 | Real documents with natural formatting patterns |
| olmocr-mix subset | 5,000-10,000 | Real documents, diverse layouts (no formatting GT exists) |

**Generation strategy:**
- **Temperature**: 0.0 (greedy) for deterministic outputs. Alternatively, generate 3-5 samples per image at temp=0.7 and pick the highest-reward one (rejection sampling, as in DeepSeek R1 Stage 3).
- **Filtering**: Discard outputs where the reward function scores below a threshold (e.g., formatting_pertype F1 < 0.5).
- **Augmentation via olmocr-mix**: The teacher can process real document images that have no formatting GT. This creates new training data that didn't exist before.

### Phase 4: Distill into 2B Student

**Goal**: SFT the 2B student on teacher-generated outputs.

```
Qwen3-VL-2B-Instruct (or fmt_sft as starting point)
    |
    v  SFT on teacher_outputs (CE loss)
fmt_kd_2b
```

- Hardware: 1x H200
- Time: ~2-4 hours (Small), ~8-12 hours (Large)
- Two starting points to try:
  - From base Qwen3-VL-2B (clean distillation, isolates the KD signal)
  - From fmt_sft checkpoint (progressive training, may converge faster)
- Can optionally add soft-label KD (student matches teacher logits), but SFT on teacher outputs is simpler and worked for DeepSeek R1

Try both, benchmark both. If warm start wins, the teacher outputs are complementary to synthetic GT. If cold start wins, the teacher outputs are sufficient on their own.

**Mixing teacher + synthetic GT?** Train on a mix of teacher-generated outputs and original synthetic GT with a configurable ratio (e.g., 80% teacher, 20% synthetic). Only worth exploring after the basic KD pipeline is validated.

### Phase 5: Benchmark

Run all three benches on the distilled 2B model:

| Bench | What it measures | Expectation |
|-------|-----------------|-------------|
| Formatting (250 pages) | Bold/italic/strike F1 | Should exceed fmt_sft (0.766) |
| olmOCR | Text extraction accuracy | Should not regress from fmt_sft (66.3%) |
| OmniDocBench | Multi-dimension doc quality | Should match or exceed fmt_sft (51.4%) |

---

## Execution

### Parallelizable Work

The 5-phase pipeline reads sequentially, but several workstreams can run in parallel:

| Workstream | Dependencies | Can start |
|---|---|---|
| fmt_sft GT generation (Option 3) | None - uses existing 2B checkpoint | Immediately |
| Qwen3.5-27B / Qwen3-VL-32B inference GT generation (Option 2) | None - zero-shot, no training | Immediately |
| API GT generation for validation set (Option 4) | None | Immediately |
| Phase 1: Teacher two-stage SFT (ocr_sft + fmt_sft) | None - 9B SFT fits on 1x H200 (~60-70GB) | Immediately |
| Phase 2: Teacher GRPO | Phase 1 complete + multi-GPU SLURM setup | After Phase 1 + infra |
| Phase 4: Student KD | Any GT source (Phase 3 or Options 2-4) | After any GT is ready |

This means we can have distillation data ready (via Options 2-3) before the teacher pipeline even begins. If the KD from a zero-shot 27B/32B model already beats fmt_sft, the teacher RL pipeline may be unnecessary.

### Timeline

Assuming multi-GPU access is available:

| Week | Work |
|---|---|
| 1 | Phase 1: Two-stage teacher SFT on 1x H200 (ocr_sft 5K + fmt_sft Full 4,750, ~8-12h total). Parallel: fmt_sft GT gen (Option 3) on 10K olmocr-mix. Parallel: multi-GPU SLURM scripts + FSDP testing for Phase 2 |
| 2 | Qwen3.5-27B GT gen (Option 2). KD experiments with Option 2-3 GT on 1x H200 |
| 3-4 | Phase 2: Teacher GRPO on 2x H200 (Smoke -> Small). Parallel: more KD experiments if Option 2-3 GT shows promise |
| 5 | Phase 3: Teacher GT generation. Phase 4: KD with teacher GT. Phase 5: Benchmark all variants |

If GRPO fails on 9B (like it did on 2B), pivot to Tier 2 (27B) or try PPO before scaling up.

### Infrastructure Changes

#### Multi-GPU SLURM Support

Current `run_training.sh` requests `gpu:h200:1`. For multi-GPU:

```bash
#SBATCH --gres=gpu:h200:2          # or 4, 8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8           # more CPUs for data loading
#SBATCH --mem=128G                  # scale with GPU count
```

Training launch changes from `python forge.py` to:
```bash
torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE forge.py ...
# or
accelerate launch --num_processes=$SLURM_GPUS_ON_NODE forge.py ...
```

#### Forge Changes

- `forge.py`: Support for multi-GPU launch (Accelerate or torchrun)
- `src/models/loader.py`: FSDP wrapping for large models
- `src/training/grpo_trainer.py` (or TRL config): Distributed generation + training
- Config: Add `distributed:` section for FSDP/TP settings

#### New Configs

```
configs/ocr/
    qwen35-9b/                    # Tier 1 teacher
        base.yaml                 # 9B-specific settings (LR, batch size)
        00_ocr_sft.yaml           # Phase 1, Stage 1: OCR SFT on 5K olmocr-mix
    formatting/
        qwen35-9b/
            00_fmt_sft.yaml       # Phase 1, Stage 2: Formatting SFT
            rl/
                base.yaml
                00_grpo.yaml      # Phase 2: GRPO
        qwen3vl-2b/
            kd/
                00_teacher_9b.yaml  # Phase 4: KD from 9B teacher
```

### Overall Progression

```
Phase 1 (Two-Stage SFT)                Phase 2 (GRPO)          Phase 3 (Generate)      Phase 4 (KD)

Stage 1: ocr_sft 5K (~2-4h)
    |
Stage 2: fmt_sft Full (4,750)     -->  GRPO Smoke: 50 prompts, 10 steps  -->  (skip)              -->  (skip)
  (~4-8h)                                "does GRPO run on 9B?"

                                        GRPO Small: 250 prompts            -->  Generate: 4,750     -->  KD: 4,750
                                         "does RL improve over SFT?"            "teacher outputs"        "does student learn?"

                                        GRPO Full: 4,750 prompts           -->  Generate: 16K+      -->  KD: 16K+
                                         "maximum RL"                            "+ olmocr-mix"           "maximum KD"
```

Phase 1 runs once on Full data (cheap enough at ~8-12h total). Phase 2 onward uses tiered progression. Stop scaling data when returns diminish.

---

## Alternatives

### PPO as Alternative to GRPO

If GRPO fails on the larger teacher (entropy collapse, reward stagnation), PPO is a natural next step. PPO uses a learned critic to estimate expected reward, computing advantages as actual - expected rather than GRPO's group comparison. This is more sample-efficient (1 generation per prompt vs 4) but requires training a second network.

With LoRA, PPO is practical: attach two small adapters (policy + critic) to the same frozen backbone, swap between them. Memory overhead is ~100MB over GRPO (two adapter sets + value head + two optimizers). A separate design doc covers this in detail.

---

## Open Questions

1. **Does TRL's GRPOTrainer support multi-GPU out of the box?** Need to verify FSDP compatibility with LoRA + generation.
2. **How much training data is enough for the teacher?** 250 samples may be fine for 2B but a 9B model might need more to fully learn formatting patterns via SFT. Consider data augmentation.
3. **Should we expand the training set?** The teacher can generate outputs for unlabeled document images, effectively creating more distillation data than original GT labels.
4. **Soft-label KD vs hard-label SFT?** DeepSeek R1 used hard labels (SFT on generated text). Soft labels (matching logits) preserve more information but require running the teacher during student training, adding complexity.
5. **Qwen3.5-9B vs Qwen3-VL-8B as teacher?** Qwen3.5 is newer with better benchmarks and native multimodality. Qwen3-VL-8B shares the Qwen3 family with the current student. Start with Qwen3.5-9B, fall back to Qwen3-VL-8B if there are compatibility issues.
