# Knowledge Distillation Tasks and Datasets

Catalog of tasks for systematically understanding knowledge distillation across different dimensions.

---

## Current Tasks

| Task | Type | Dataset | Size | Models |
|------|------|---------|------|--------|
| **MNLI** | 3-class classification | GLUE MNLI | 393k train | BERT → DistilBERT |
| **QQP** | Binary classification | GLUE QQP | 363k train | BERT → DistilBERT |
| **Translation** | Seq2seq generation | Europarl EN-FR | 46k examples | Llama-3B → Llama-1B |

### Task Details

**MNLI (Multi-Genre Natural Language Inference)**
- **Goal:** Classify premise-hypothesis pairs as entailment/neutral/contradiction
- **Why:** Tests reasoning transfer; multiple output classes reveal soft-label dynamics
- **Key findings:** Co-training outperforms frozen; overconfident-wrong teachers hurt

**QQP (Quora Question Pairs)**
- **Goal:** Detect duplicate questions (binary)
- **Why:** Class imbalance (~64% negative); tests robustness under skewed distributions

**Translation (EN → FR)**
- **Goal:** Translate English to French using causal LM
- **Why:** Sequence generation requires per-token distillation; tests if KL scales to large vocabularies

---

## Proposed Tasks

### Token-Level Classification

**NER (Named Entity Recognition)**
- Dataset: CoNLL-2003 (~14k train sentences)
- Task: Tag each token as PER/LOC/ORG/MISC/O
- Why: Forces fine-grained, per-token knowledge transfer
- Questions: Does feature distillation help more than response at token level?

### Reasoning Tasks

**BoolQ (Boolean Questions)**
- Dataset: ~10k yes/no reading comprehension questions
- Task: Answer yes/no given passage and question
- Questions: Does attention distillation affect comprehension?

**COPA (Choice of Plausible Alternatives)**
- Dataset: ~500 train examples (very small)
- Task: Select more plausible cause/effect
- Questions: Can distillation help when data is scarce?

**WinoGrande**
- Dataset: ~40k examples
- Task: Commonsense pronoun resolution
- Questions: Is commonsense knowledge better transferred via features or logits?

### Regression Tasks

**STS-B (Semantic Textual Similarity)**
- Dataset: GLUE STS-B (~5.7k train)
- Task: Predict similarity score (0-5 continuous)
- Questions: How does temperature affect regression distillation?

### Long-Context Tasks

**SQuAD (Extractive QA)**
- Dataset: SQuAD 2.0 (~130k train)
- Task: Extract answer span from passage
- Why: Span prediction requires precise token-level understanding

### Generation Tasks

**Summarization**
- Dataset: CNN/DailyMail (~287k) or XSum (~204k)
- Task: Generate abstractive summary
- Questions: Does distillation help faithfulness or fluency more?

---

## Experimental Dimensions

| Dimension | Current | Proposed Extensions |
|-----------|---------|---------------------|
| **Output granularity** | Sequence-level | Token-level (NER) |
| **Output type** | Classification, generation | Regression (STS-B), span (SQuAD) |
| **Reasoning depth** | Shallow inference | Causal (COPA), commonsense (WinoGrande) |
| **Data size** | Large (300k+) | Small (COPA ~500), medium (BoolQ ~10k) |
| **Sequence length** | 128-256 tokens | Longer (512+) for QA tasks |
| **Transfer** | Single-task | Cross-task zero-shot |

---

## Priority Ranking

1. **STS-B** - Regression exposes different distillation dynamics
2. **NER** - Token-level classification is fundamental gap
3. **BoolQ** - Reading comprehension with simple output
4. **COPA** - Low-data regime is underexplored
5. **SQuAD** - Span extraction combines multiple challenges
6. **Summarization** - Richer generation than translation

---

## Implementation Checklist

For each new task:
- [ ] Add dataset loader in `src/data/datasets.py`
- [ ] Create base config in `configs/tasks/{task}.yaml`
- [ ] Define appropriate metrics in evaluation
- [ ] Create L0 ablation configs (CE-only, frozen KD, co-train KD)
- [ ] Document expected baselines and teacher performance
