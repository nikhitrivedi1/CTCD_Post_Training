# Do Classification Findings Transfer? Distillation for Autoregressive Translation

Draft paper validating ablation and FAR findings on generation tasks.

---

## Abstract

Prior work established two key findings for classification distillation: (1) co-training outperforms frozen teachers, and (2) response-only distillation beats multi-signal approaches. We test whether these findings transfer to **autoregressive generation tasks**, specifically machine translation with decoder-only models (Llama 3B → 1B). [Results pending]

---

## 1. Introduction

### Motivation

Classification and generation distillation differ fundamentally:

| Aspect | Classification | Generation |
|--------|---------------|------------|
| **Output** | Fixed classes | Variable-length sequences |
| **Loss** | KL over N classes | KL over vocab × seq_len |
| **Architecture** | Encoder (BERT) | Decoder (Llama) |
| **Teacher signal** | Single distribution | Per-token distributions |

**Research question:** Do findings from classification transfer to generation?

### Hypotheses

Based on classification findings:

| Hypothesis | Classification Finding | Generation Prediction |
|------------|----------------------|----------------------|
| **H1** | Frozen SFT < CE-only | Should hold (overconfident translations) |
| **H2** | Co-train > Frozen | Should hold (adaptation matters) |
| **H3** | R-only > FAR | May differ (per-token vs per-sample) |

---

## 2. Background

### Translation Distillation Challenges

| Challenge | Impact |
|-----------|--------|
| **Large vocabulary** | 128K tokens × sequence length = memory intensive |
| **Sequence mismatch** | Teacher and student may generate different lengths |
| **Autoregressive** | Each token depends on previous, errors compound |
| **Dimension mismatch** | Llama-3B: 3072 hidden, Llama-1B: 2048 hidden |

### Key Differences from Classification

1. **Per-token soft labels:** Teacher provides distribution at each position
2. **Teacher forcing:** Student sees ground truth during training, not own predictions
3. **Exposure bias:** Training/inference mismatch for generation
4. **Evaluation:** BLEU, not accuracy

---

## 3. Experimental Setup

### Models

| Role | Model | Params | Hidden | Layers | Heads |
|------|-------|--------|--------|--------|-------|
| Teacher | meta-llama/Llama-3.2-3B | 3B | 3072 | 28 | 24 |
| Student | meta-llama/Llama-3.2-1B | 1B | 2048 | 16 | 8 |

### Dataset

**Europarl (English → German)**
- 2M parallel sentences
- Parliamentary/political domain
- Clean, professional translations

### Layer Mapping (for Feature Distillation)

```
Teacher (28 layers):  [2] [4] [5] [7] [9] [10] [12] [14] [16] [17] [19] [21] [23] [24] [26] [28]
                       ↓   ↓   ↓   ↓   ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓
Student (16 layers):  [1] [2] [3] [4] [5]  [6]  [7]  [8]  [9] [10] [11] [12] [13] [14] [15] [16]
```

### Projection

Linear projection: 2048 → 3072 for feature alignment.

### Ablation Configurations

| Config | Teacher | Frozen | Alpha | Description |
|--------|---------|--------|-------|-------------|
| 00 | None | - | 0.0 | CE only (no distillation) |
| 01 | Base | ✓ | 0.5 | Frozen base Llama |
| 02 | SFT | ✓ | 0.5 | Frozen fine-tuned Llama |
| 03 | Base | ✗ | 0.5 | Co-train base |
| 04 | SFT | ✗ | 0.5 | Co-train SFT |

### FAR Configurations

Same 8 combinations as classification study.

---

## 4. Results

### Ablation Results

*[Experiments pending]*

| Rank | Config | BLEU | Description |
|------|--------|------|-------------|
| ? | ? | ? | ? |

### FAR Results

*[Experiments pending]*

| Rank | Config | F | A | R | BLEU |
|------|--------|---|---|---|------|
| ? | ? | ? | ? | ? | ? |

---

## 5. Analysis

### Transfer of Findings

*[Pending results]*

| Finding | Classification | Translation | Transfers? |
|---------|---------------|-------------|------------|
| Frozen SFT hurts | ✓ | ? | ? |
| Co-train wins | ✓ | ? | ? |
| R-only > FAR | ✓ | ? | ? |

### Generation-Specific Observations

*[To be filled after experiments]*

---

## 6. Discussion

### Why Findings May Transfer

- Overconfidence affects per-token predictions similarly
- Co-training allows teacher adaptation to translation errors
- Feature/attention misalignment still causes gradient conflict

### Why Findings May Not Transfer

- Per-token distillation averages over many positions
- Decoder architecture may benefit differently from attention transfer
- Sequence-level evaluation (BLEU) may mask token-level effects

---

## 7. Conclusion

*[Pending results]*

---

## Experiment Status

### Configurations Ready

- [x] configs/translation/llama3p2-3b_llama3p2-1b/ablations/00-04
- [x] configs/translation/llama3p2-3b_llama3p2-1b/far/00-07

### To Run

- [ ] Ablation experiments (5 configs)
- [ ] FAR experiments (8 configs)
- [ ] Analysis and comparison with classification
