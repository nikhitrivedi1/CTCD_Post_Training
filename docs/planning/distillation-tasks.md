# Knowledge Distillation Tasks and Datasets

Catalog of tasks for systematically understanding knowledge distillation across different dimensions.

## Quick Summary

**Current:** 3 tasks (MNLI, QQP, EN→FR Translation) with BERT/DistilBERT and Llama-3B/1B
**Proposed:** 43 additional Model-Pair_Dataset combinations across:
- 14 classification tasks (GLUE benchmark + domain-specific + RoBERTa variants)
- 16 translation tasks (10 language pairs + 3 datasets + 4 model architectures)
- 13 advanced tasks (token-level, QA, generation)

**Total Coverage:** 46 task configurations spanning 10+ languages, 4 task types, 8 model architectures

---

## Current Tasks (Implemented)

| Task | Type | Dataset | Size | Teacher → Student |
|------|------|---------|------|-------------------|
| **MNLI** | 3-class NLI | GLUE MNLI | 393k train | bert-base-uncased → distilbert-base-uncased |
| **QQP** | Binary classification | GLUE QQP | 363k train | bert-base-uncased → distilbert-base-uncased |
| **Translation (EN→FR)** | Seq2seq generation | Europarl en-fr | 46k examples | Llama-3.2-3B → Llama-3.2-1B |

---

## Complete Model-Pair_Dataset Matrix

### Classification: 16 Combinations

| # | Dataset | Teacher | Student | Size | Type | Status |
|---|---------|---------|---------|------|------|--------|
| 1 | GLUE MNLI | bert-base-uncased | distilbert-base-uncased | 393k | 3-class NLI | ✓ Implemented |
| 2 | GLUE QQP | bert-base-uncased | distilbert-base-uncased | 363k | Binary | ✓ Implemented |
| 3 | GLUE SST-2 | bert-base-uncased | distilbert-base-uncased | 67k | Binary sentiment | Proposed |
| 4 | GLUE CoLA | bert-base-uncased | distilbert-base-uncased | 8.5k | Binary grammar | Proposed |
| 5 | GLUE RTE | bert-base-uncased | distilbert-base-uncased | 2.5k | Binary entailment | Proposed |
| 6 | GLUE MRPC | bert-base-uncased | distilbert-base-uncased | 3.7k | Binary paraphrase | Proposed |
| 7 | GLUE STS-B | bert-base-uncased | distilbert-base-uncased | 5.7k | Regression | Proposed |
| 8 | GLUE QNLI | bert-base-uncased | distilbert-base-uncased | 105k | Binary QA | Proposed |
| 9 | GLUE WNLI | bert-base-uncased | distilbert-base-uncased | 635 | Binary coreference | Proposed |
| 10 | GLUE MNLI | roberta-base | distilroberta-base | 393k | 3-class NLI | Proposed |
| 11 | GLUE QQP | roberta-base | distilroberta-base | 363k | Binary | Proposed |
| 12 | GLUE SST-2 | roberta-large | roberta-base | 67k | Binary sentiment | Proposed |
| 13 | financial_phrasebank | bert-base-uncased | distilbert-base-uncased | 4.8k | Sentiment | Proposed |
| 14 | BC5CDR | biobert-v1.1 | distilbert-base-uncased | 5k abstracts | Biomedical NER | Proposed |
| 15 | AG News | bert-base-uncased | distilbert-base-uncased | 120k | 4-class topic | Proposed |
| 16 | IMDB | bert-base-uncased | distilbert-base-uncased | 25k | Binary sentiment | Proposed |

### Translation: 17 Combinations

| # | Source→Target | Dataset | Teacher | Student | Size | Status |
|---|---------------|---------|---------|---------|------|--------|
| 1 | EN → FR | Europarl | Llama-3.2-3B | Llama-3.2-1B | 46k | ✓ Implemented |
| 2 | EN → DE | Europarl | Llama-3.2-3B | Llama-3.2-1B | 46k | Proposed |
| 3 | EN → ES | Europarl | Llama-3.2-3B | Llama-3.2-1B | 46k | Proposed |
| 4 | EN → IT | Europarl | Llama-3.2-3B | Llama-3.2-1B | 46k | Proposed |
| 5 | FR → EN | Europarl | Llama-3.2-3B | Llama-3.2-1B | 46k | Proposed |
| 6 | DE → EN | Europarl | Llama-3.2-3B | Llama-3.2-1B | 46k | Proposed |
| 7 | EN → DE | WMT14 | Llama-3.2-3B | Llama-3.2-1B | 4.5M | Proposed |
| 8 | EN → FR | WMT14 | Llama-3.2-3B | Llama-3.2-1B | 40M | Proposed |
| 9 | EN → CS | WMT14 | Llama-3.2-3B | Llama-3.2-1B | 15M | Proposed |
| 10 | EN → RU | WMT14 | Llama-3.2-3B | Llama-3.2-1B | 2.1M | Proposed |
| 11 | EN → FI | Europarl | Llama-3.2-3B | Llama-3.2-1B | 46k | Proposed |
| 12 | EN → ET | OPUS | Llama-3.2-3B | Llama-3.2-1B | 20k | Proposed |
| 13 | EN → HI | OPUS | Llama-3.2-3B | Llama-3.2-1B | 50k | Proposed |
| 14 | EN → FR | Europarl | mT5-base | mT5-small | 46k | Proposed |
| 15 | EN → DE | Europarl | mBART-large | mBART-small | 46k | Proposed |
| 16 | EN → ES | Europarl | NLLB-3.3B | NLLB-600M | 46k | Proposed |
| 17 | EN → FR | WMT14 | NLLB-3.3B | NLLB-600M | 40M | Proposed |

**Total: 33 Model-Pair_Dataset combinations** (3 implemented + 30 proposed) across Classification and Translation

---

## Proposed Classification Tasks

### GLUE Benchmark Extensions

| Task | Type | Dataset | Size | Teacher → Student | Status |
|------|------|---------|------|-------------------|--------|
| **SST-2** | Binary sentiment | GLUE SST-2 | 67k train | bert-base → distilbert-base | Ready to implement |
| **CoLA** | Grammar acceptability | GLUE CoLA | 8.5k train | bert-base → distilbert-base | Ready to implement |
| **RTE** | Binary entailment | GLUE RTE | 2.5k train | bert-base → distilbert-base | Low-data regime |
| **MRPC** | Paraphrase detection | GLUE MRPC | 3.7k train | bert-base → distilbert-base | Low-data regime |
| **STS-B** | Regression (0-5) | GLUE STS-B | 5.7k train | bert-base → distilbert-base | Regression task |
| **QNLI** | Binary QA entailment | GLUE QNLI | 105k train | bert-base → distilbert-base | QA-style task |
| **WNLI** | Winograd coreference | GLUE WNLI | 635 train | bert-base → distilbert-base | Very low-data |

### RoBERTa Model Pairs

| Task | Dataset | Teacher → Student | Why |
|------|---------|-------------------|-----|
| **MNLI** | GLUE MNLI | roberta-base → distilroberta-base | Compare BERT vs RoBERTa distillation |
| **QQP** | GLUE QQP | roberta-base → distilroberta-base | Test robustness improvements |
| **SST-2** | GLUE SST-2 | roberta-large → roberta-base | Large→Base compression |

### Domain-Specific Classification

| Task | Dataset | Size | Teacher → Student | Domain |
|------|---------|------|-------------------|--------|
| **Financial Sentiment** | financial_phrasebank | 4.8k | bert-base → distilbert-base | Finance |
| **Scientific Papers** | scicite | 8.2k train | scibert-base → distilbert-base | Academia |
| **Biomedical NER** | BC5CDR | 5k abstracts | biobert-base → distilbert-base | Healthcare |

---

## Proposed Translation Tasks

### Language Pair Extensions (Europarl)

| Source → Target | Dataset | Size | Teacher → Student | Why |
|----------------|---------|------|-------------------|-----|
| **EN → DE** | Europarl en-de | ~46k | Llama-3.2-3B → Llama-3.2-1B | Germanic language pair |
| **EN → ES** | Europarl en-es | ~46k | Llama-3.2-3B → Llama-3.2-1B | Romance language pair |
| **EN → IT** | Europarl en-it | ~46k | Llama-3.2-3B → Llama-3.2-1B | Romance language |
| **FR → EN** | Europarl fr-en | ~46k | Llama-3.2-3B → Llama-3.2-1B | Reverse direction |
| **DE → EN** | Europarl de-en | ~46k | Llama-3.2-3B → Llama-3.2-1B | Reverse Germanic |

### WMT Translation Datasets

| Source → Target | Dataset | Size | Teacher → Student | Why |
|----------------|---------|------|-------------------|-----|
| **EN → DE** | WMT14 en-de | 4.5M pairs | Llama-3.2-3B → Llama-3.2-1B | Larger scale, news domain |
| **EN → FR** | WMT14 en-fr | 40M pairs | Llama-3.2-3B → Llama-3.2-1B | Very large scale |
| **EN → CS** | WMT14 en-cs | 15M pairs | Llama-3.2-3B → Llama-3.2-1B | Morphologically rich target |
| **EN → RU** | WMT14 en-ru | 2.1M pairs | Llama-3.2-3B → Llama-3.2-1B | Cyrillic script |

### Low-Resource Translation

| Source → Target | Dataset | Size | Teacher → Student | Why |
|----------------|---------|------|-------------------|-----|
| **EN → FI** | Europarl en-fi | ~46k | Llama-3.2-3B → Llama-3.2-1B | Agglutinative language |
| **EN → ET** | OPUS en-et | ~20k | Llama-3.2-3B → Llama-3.2-1B | Very low-resource |
| **EN → HI** | OPUS en-hi | ~50k | Llama-3.2-3B → Llama-3.2-1B | Indic language |

### Alternative Model Pairs for Translation

| Source → Target | Dataset | Teacher → Student | Why |
|----------------|---------|-------------------|-----|
| **EN → FR** | Europarl en-fr | mT5-base → mT5-small | Encoder-decoder architecture |
| **EN → DE** | Europarl en-de | mBART-large → mBART-small | Multilingual pretrained |
| **EN → ES** | Europarl en-es | NLLB-3.3B → NLLB-600M | Specialized translation models |

### Current Task Details

**MNLI (Multi-Genre Natural Language Inference)**
- **Goal:** Classify premise-hypothesis pairs as entailment/neutral/contradiction
- **Why:** Tests reasoning transfer; multiple output classes reveal soft-label dynamics
- **Key findings:** Co-training outperforms frozen; overconfident-wrong teachers hurt
- **Config:** `configs/classification/mnli/distilbert/bert-base/`

**QQP (Quora Question Pairs)**
- **Goal:** Detect duplicate questions (binary)
- **Why:** Class imbalance (~64% negative); tests robustness under skewed distributions
- **Config:** `configs/classification/qqp/distilbert/bert-base/`

**Translation (EN → FR)**
- **Goal:** Translate English to French using causal LM
- **Why:** Sequence generation requires per-token distillation; tests if KL scales to large vocabularies
- **Config:** `configs/translation/europarl/llama32-1b/llama32-3b/`

---

## Additional Task Types (Future Work)

### Token-Level Classification

| Task | Dataset | Size | Teacher → Student | Why |
|------|---------|------|-------------------|-----|
| **NER** | CoNLL-2003 | 14k sentences | bert-base → distilbert-base | Per-token knowledge transfer |
| **POS Tagging** | Penn Treebank | 39k sentences | bert-base → distilbert-base | Fine-grained syntax |
| **Chunking** | CoNLL-2000 | 8.9k sentences | bert-base → distilbert-base | Phrase-level understanding |

### Reading Comprehension & Reasoning

| Task | Dataset | Size | Teacher → Student | Why |
|------|---------|------|-------------------|-----|
| **SQuAD 2.0** | Stanford QA | 130k train | bert-large → bert-base | Extractive QA with unanswerable |
| **BoolQ** | Google | 10k examples | bert-large → distilbert-base | Yes/no comprehension |
| **COPA** | SuperGLUE | 500 train | bert-large → bert-base | Causal reasoning, low-data |
| **WinoGrande** | AI2 | 40k examples | roberta-large → roberta-base | Commonsense coreference |
| **HellaSwag** | AI2 | 40k examples | roberta-large → roberta-base | Commonsense inference |

### Generation Tasks

| Task | Dataset | Size | Teacher → Student | Why |
|------|---------|------|-------------------|-----|
| **Summarization** | CNN/DailyMail | 287k train | Llama-3.2-3B → Llama-3.2-1B | Abstractive generation |
| **Summarization** | XSum | 204k train | Llama-3.2-3B → Llama-3.2-1B | Extreme compression |
| **Dialogue** | DailyDialog | 13k conversations | Llama-3.2-3B → Llama-3.2-1B | Multi-turn coherence |
| **Code Generation** | CodeSearchNet | 2.3M functions | CodeLlama-7B → CodeLlama-3B | Programming tasks |

---

## Experimental Dimensions

| Dimension | Current (3 tasks) | Proposed Extensions |
|-----------|-------------------|---------------------|
| **Task types** | NLI, paraphrase, translation | Sentiment, grammar, regression, QA, summarization |
| **Output granularity** | Sequence-level | Token-level (NER, POS) |
| **Output type** | Classification, generation | Regression (STS-B), span (SQuAD), structured (NER) |
| **Model architectures** | BERT, Llama | RoBERTa, SciBERT, BioBERT, mT5, mBART |
| **Compression ratios** | 2:1 (BERT→DistilBERT, Llama-3B→1B) | 3:1 (RoBERTa-large→base), variable |
| **Languages** | English, EN→FR | 10+ languages (DE, ES, IT, CS, RU, FI, HI) |
| **Domains** | General, parliamentary | Finance, science, biomedical, news |
| **Data size** | Large (300k+) | Small (500), medium (5-10k), large (100k+), very large (1M+) |
| **Sequence length** | 128-256 tokens | Up to 512+ for QA and summarization |

### Coverage Summary

| Category | Implemented | Proposed | Total |
|----------|-------------|----------|-------|
| **Classification Tasks (BERT-based)** | 2 (MNLI, QQP) | 14 (SST-2, CoLA, RTE, MRPC, STS-B, QNLI, WNLI + RoBERTa variants + domain-specific + AG News, IMDB) | 16 |
| **Translation Tasks** | 1 (EN→FR) | 16 (5 Europarl pairs, 4 WMT pairs, 3 low-resource, 4 alternative models) | 17 |
| **Other Tasks** | 0 | 13 (NER, POS, SQuAD, BoolQ, COPA, WinoGrande, summarization, dialogue, code) | 13 |
| **Total Combinations** | 3 | 43 | 46 |

---

## Priority Ranking

### Near-Term (Easy Implementation)

1. **SST-2 (GLUE)** - Single-sentence classification, clean binary labels
2. **CoLA (GLUE)** - Grammar task, different from semantic tasks
3. **EN→DE Translation (Europarl)** - Germanic language pair comparison
4. **EN→ES Translation (Europarl)** - Romance language pair
5. **RoBERTa MNLI** - Architecture comparison with BERT

### Medium-Term (Moderate Effort)

6. **STS-B** - Regression exposes different distillation dynamics
7. **QNLI** - QA-style entailment, larger dataset
8. **RTE/MRPC** - Low-data regime experiments
9. **WMT14 EN→DE** - Large-scale translation
10. **RoBERTa-large → RoBERTa-base** - Larger compression ratio

### Long-Term (Complex Tasks)

11. **SQuAD 2.0** - Span extraction, extractive QA
12. **NER (CoNLL-2003)** - Token-level classification
13. **Summarization (CNN/DM or XSum)** - Abstractive generation
14. **BoolQ** - Reading comprehension
15. **WinoGrande** - Commonsense reasoning

---

## Implementation Guide

### Quick Wins (Minimal Code Changes)

**GLUE Tasks (SST-2, CoLA, RTE, MRPC, STS-B, QNLI, WNLI)**
- All use same infrastructure as MNLI/QQP
- Only need to add task name to `load_dataset("glue", task_name)`
- Metrics already available in HuggingFace Evaluate library

**Europarl Language Pairs**
- Already implemented! Just change `source_lang` and `target_lang` in config
- Supported: en-de, en-es, en-it, en-fr, de-en, fr-en, en-fi
- Uses existing `load_europarl()` function

**RoBERTa Variants**
- Change model names in config: `roberta-base` → `distilroberta-base`
- Same tokenizer/data pipeline as BERT

### Medium Effort (New Loaders)

**WMT Datasets**
- Add new loader similar to `load_europarl()`
- Dataset: `load_dataset("wmt14", "de-en")` or similar
- Same translation pipeline, different data source

**Domain-Specific Classification**
- Financial: `load_dataset("financial_phrasebank")`
- Scientific: `load_dataset("scicite")`
- Biomedical: requires loading BC5CDR with custom processing

### Implementation Checklist

For each new task:
- [ ] Add dataset loader to `src/data/datasets.py` (if needed)
- [ ] Add task branch to `load_dataset_for_task()` function
- [ ] Create base config in `configs/{task}/{teacher}_{student}/base.yaml`
- [ ] Define appropriate metrics (accuracy, F1, BLEU, ROUGE, etc.)
- [ ] Create L0 ablation configs:
  - [ ] `00_sft.yaml`
  - [ ] `01_kd_frozen-base.yaml`
  - [ ] `02_kd_frozen-sft.yaml`
  - [ ] `03_kd_cotrain-base.yaml`
  - [ ] `04_kd_cotrain-sft.yaml`
- [ ] Document expected baselines and teacher performance
- [ ] Run smoke test on small subset to verify pipeline
- [ ] Scale up to full training run

### Model Availability Check

| Teacher Model | Student Model | HuggingFace Hub | Status |
|---------------|---------------|-----------------|--------|
| bert-base-uncased | distilbert-base-uncased | ✓ | ✓ Implemented |
| roberta-base | distilroberta-base | ✓ | Ready |
| roberta-large | roberta-base | ✓ | Ready |
| allenai/scibert_scivocab_uncased | distilbert-base-uncased | ✓ | Ready |
| dmis-lab/biobert-v1.1 | distilbert-base-uncased | ✓ | Ready |
| meta-llama/Llama-3.2-3B | meta-llama/Llama-3.2-1B | ✓ | ✓ Implemented |
| google/mt5-base | google/mt5-small | ✓ | Needs implementation |
| facebook/mbart-large-50 | facebook/mbart-large-cc25 | ✓ | Needs implementation |
| facebook/nllb-200-3.3B | facebook/nllb-200-distilled-600M | ✓ | Needs implementation |
