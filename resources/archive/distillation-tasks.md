# Knowledge Distillation Tasks & Datasets

This document catalogs tasks for systematically understanding knowledge distillation across different dimensions.

---

## Current Tasks

| Task | Type | Dataset | Size | Models |
|------|------|---------|------|--------|
| **MNLI** | 3-class classification | GLUE MNLI | 393k train | BERT → DistilBERT |
| **QQP** | Binary classification | GLUE QQP | 363k train | BERT → DistilBERT |
| **Translation** | Seq2seq generation | Europarl EN-FR | 46k examples | Llama-3B → Llama-1B |

### Task Details

#### MNLI (Multi-Genre Natural Language Inference)
- **Goal:** Classify premise-hypothesis pairs as entailment/neutral/contradiction
- **Why it matters:** Tests reasoning transfer; multiple output classes reveal soft-label dynamics
- **Key findings:** Co-training outperforms frozen distillation; overconfident-wrong teachers hurt

#### QQP (Quora Question Pairs)
- **Goal:** Detect duplicate questions (binary)
- **Why it matters:** Class imbalance (~64% negative); tests robustness of distillation under skewed distributions
- **Challenge:** Teacher confidence calibration matters more with imbalance

#### Translation (EN → FR)
- **Goal:** Translate English to French using causal LM
- **Why it matters:** Sequence generation requires per-token distillation; tests if KL on logits scales to large vocabularies
- **Architecture:** Uses prompt template with special end-of-translation token

---

## Proposed Tasks

### 1. Token-Level Classification

#### NER (Named Entity Recognition)
- **Dataset:** CoNLL-2003 (~14k train sentences)
- **Task:** Tag each token as PER/LOC/ORG/MISC/O
- **Why add it:**
  - Forces fine-grained, per-token knowledge transfer
  - Tests whether attention distillation helps alignment
  - Different from sequence-level classification dynamics
- **Distillation questions:**
  - Does feature distillation help more than response distillation at token level?
  - How does layer mapping affect entity boundary detection?

#### POS Tagging
- **Dataset:** Penn Treebank or Universal Dependencies
- **Task:** Assign part-of-speech tags to each token
- **Why add it:** Simpler than NER; good sanity check for token-level distillation

---

### 2. Reasoning Tasks

#### BoolQ (Boolean Questions)
- **Dataset:** ~10k yes/no reading comprehension questions
- **Task:** Answer yes/no given a passage and question
- **Why add it:**
  - Tests if distillation preserves reading comprehension ability
  - Binary output but requires passage understanding
- **Distillation questions:**
  - Does the student learn to attend to relevant passage spans?
  - How does attention distillation affect comprehension?

#### COPA (Choice of Plausible Alternatives)
- **Dataset:** ~500 train examples (very small)
- **Task:** Select the more plausible cause/effect given a premise
- **Why add it:**
  - Tests low-data distillation regime
  - Causal reasoning requires implicit world knowledge
- **Distillation questions:**
  - Can distillation help when data is scarce?
  - Does teacher knowledge compensate for limited examples?

#### WinoGrande
- **Dataset:** ~40k examples
- **Task:** Commonsense pronoun resolution
- **Why add it:**
  - Probes implicit/commonsense knowledge transfer
  - Requires understanding that may be encoded in hidden states
- **Distillation questions:**
  - Is commonsense knowledge better transferred via features or logits?

---

### 3. Regression Tasks

#### STS-B (Semantic Textual Similarity Benchmark)
- **Dataset:** GLUE STS-B (~5.7k train)
- **Task:** Predict similarity score (0-5 continuous)
- **Why add it:**
  - No classification head; tests soft target behavior in regression
  - MSE distillation on outputs vs. features
- **Distillation questions:**
  - How does temperature affect regression distillation?
  - Is feature distillation more important when output is continuous?

---

### 4. Transfer & Generalization

#### Zero-Shot Transfer Setup
- **Train:** Distill on MNLI
- **Evaluate:** RTE, WNLI, ANLI (without fine-tuning)
- **Why add it:**
  - Tests if distillation improves generalization vs. direct fine-tuning
  - Measures whether knowledge transfer is task-specific or general
- **Distillation questions:**
  - Does co-trained distillation generalize better than frozen?
  - Which distillation signal (R/F/A) helps transfer most?

---

### 5. Long-Context Tasks

#### QNLI (Question NLI)
- **Dataset:** GLUE QNLI (~105k train)
- **Task:** Given question + sentence, does sentence contain the answer?
- **Why add it:**
  - Longer sequences than MNLI/QQP
  - Tests attention distillation at scale

#### SQuAD (Extractive QA)
- **Dataset:** SQuAD 2.0 (~130k train)
- **Task:** Extract answer span from passage
- **Why add it:**
  - Span prediction requires precise token-level understanding
  - Tests if distillation helps localization

---

### 6. Generation Tasks

#### Summarization
- **Dataset:** CNN/DailyMail (~287k) or XSum (~204k)
- **Task:** Generate abstractive summary
- **Why add it:**
  - Different generation style than translation
  - Requires compression/abstraction vs. faithful translation
- **Distillation questions:**
  - Does distillation help faithfulness or fluency more?
  - How does sequence-level KL compare to token-level?

#### Code Generation
- **Dataset:** HumanEval subset or MBPP
- **Task:** Generate code from docstring
- **Why add it:**
  - Structured output with strict correctness requirements
  - Tests if distillation preserves logical consistency
- **Distillation questions:**
  - Can smaller models maintain code correctness via distillation?

---

## Experimental Dimensions

| Dimension | Current | Proposed Extensions |
|-----------|---------|---------------------|
| **Output granularity** | Sequence-level | Token-level (NER, POS) |
| **Output type** | Classification, generation | Regression (STS-B), span extraction (SQuAD) |
| **Reasoning depth** | Shallow inference | Causal (COPA), commonsense (WinoGrande) |
| **Data size** | Large (300k+) | Small (COPA ~500), medium (BoolQ ~10k) |
| **Sequence length** | 128-256 tokens | Longer (512+) for QA tasks |
| **Transfer** | Single-task | Cross-task zero-shot |

---

## Priority Ranking

For systematic understanding, implement in this order:

1. **STS-B** — Regression exposes different distillation dynamics
2. **NER** — Token-level classification is fundamental gap
3. **BoolQ** — Reading comprehension with simple output
4. **COPA** — Low-data regime is underexplored
5. **SQuAD** — Span extraction combines multiple challenges
6. **Summarization** — Richer generation than translation

---

## Implementation Checklist

For each new task:
- [ ] Add dataset loader in `src/data/datasets.py`
- [ ] Create base config in `configs/tasks/{task}.yaml`
- [ ] Define appropriate metrics in evaluation
- [ ] Create L0 ablation configs (CE-only, frozen KD, co-train KD)
- [ ] Document expected baselines and teacher performance
