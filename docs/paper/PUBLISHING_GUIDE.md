# Publishing Guide for Beginners

A beginner-friendly guide to academic publishing and our paper-ready content.

---

## Part 1: Publishing 101

### What is an Academic Paper?

An academic paper is a formal document that presents original research to the scientific community. It follows a standard structure (Abstract → Introduction → Methods → Results → Conclusion) and goes through **peer review** — other researchers evaluate it before publication.

### Conference vs Journal

| Aspect | Conference Paper | Journal Paper |
|--------|-----------------|---------------|
| **What it is** | Presented at an event (NeurIPS, ACL) | Published in a periodical (JMLR, TACL) |
| **Timeline** | 3-6 months (submit → publish) | 6-18 months |
| **Review** | 1 round, accept/reject | Multiple revision rounds |
| **Length** | 8-10 pages | 15-30+ pages |
| **Prestige** | Top conferences = very high in ML/NLP | Varies by journal |
| **Best for** | New findings, timely results | Comprehensive, polished work |

**For ML/NLP**: Conferences are the primary venue. Journals are secondary.

### Paper Formats by Length

| Format | Pages | What It Contains | Effort |
|--------|-------|------------------|--------|
| **Extended Abstract** | 2-4 | Core idea + preliminary results | Days |
| **Short Paper** | 4-6 | Complete study, focused scope | Weeks |
| **Full Paper** | 8-12 | Comprehensive study + analysis | Months |
| **Journal Article** | 15-30+ | Multiple studies, exhaustive | Months-Year |

### Conference Tiers (ML/NLP)

| Tier | Venues | Acceptance | Difficulty |
|------|--------|------------|------------|
| **Top-tier** | NeurIPS, ICML, ICLR, ACL, EMNLP | 20-25% | Very competitive |
| **Strong** | NAACL, EACL, COLING, AAAI | 25-30% | Competitive |
| **Workshop** | *WS at any conference | 40-60% | Accessible entry point |
| **Findings** | ACL Findings, EMNLP Findings | 30-35% | Good for solid work |

**Tip**: Workshops are the best entry point for first-time authors.

---

## Part 2: Types of Papers

### 1. Empirical Paper (Most Common)
> "We ran experiments and discovered X"

- **Requirements**: Baselines, ablations, multiple datasets, statistical tests
- **Example**: "We tested 5 distillation configs across 4 tasks and found frozen teachers fail"
- **Our fit**: Papers 1 and 2

### 2. Framework/Systems Paper
> "Here's a tool that enables researchers to do X"

- **Requirements**: Clean API, documentation, adoption potential
- **Example**: "A unified framework for reproducible KD research"
- **Our fit**: Paper 0 (better as appendix than standalone)

### 3. Survey Paper
> "We reviewed 200 papers and synthesized the field"

- **Requirements**: Comprehensive coverage, taxonomy, trends analysis
- **Example**: "A Survey of Knowledge Distillation Techniques 2015-2024"
- **Our fit**: Not applicable

### 4. Position/Opinion Paper
> "The field should move in direction X"

- **Requirements**: Compelling argument, vision, preliminary evidence
- **Example**: "Why We Should Rethink Multi-Signal Distillation"
- **Our fit**: Could reframe Paper 2 this way

### 5. Benchmark Paper
> "Here's a new dataset/task for the community"

- **Requirements**: Dataset, baselines, leaderboard, analysis
- **Example**: "GLUE: A Multi-Task Benchmark for NLU"
- **Our fit**: Not applicable

### 6. Negative Results Paper
> "Popular method X doesn't work, here's why"

- **Requirements**: Rigorous experiments, clear failure analysis
- **Example**: "Frozen Fine-Tuned Teachers Can Catastrophically Fail"
- **Our fit**: Paper 1 fits perfectly!

### 7. Replication Paper
> "We reproduced paper X and found..."

- **Venues**: ML Reproducibility Challenge, ReScience
- **Our fit**: Could replicate classic KD papers

---

## Part 3: Our Paper-Ready Content

### Overview

| Paper | Title | Type | Status | Best Venue |
|-------|-------|------|--------|------------|
| [Paper 0](paper-0-framework/draft.md) | Unified KD Framework | Systems | Draft | Workshop/Appendix |
| [Paper 1](paper-1-ablation/draft.md) | When Teachers Fail | Empirical/Negative | **Complete** | Workshop → Full |
| [Paper 2](paper-2-far/draft.md) | Simplicity Wins (R-only) | Ablation | Complete | Workshop → Full |
| [Paper 3](paper-3-translation/draft.md) | Translation Transfer | Validation | Pending | Needs experiments |

### Paper 0: A Unified Framework for KD Research

**Status**: Draft
**Contribution**: Config-driven framework for reproducible experiments

**Key points**:
- YAML inheritance for experiment configs
- FAR factorial design (8 combinations)
- Supports classification (BERT) and generation (Llama)

**Best use**: Appendix to main papers, or systems workshop

### Paper 1: When Teachers Fail (The Overconfident-Wrong Teacher Problem)

**Status**: Complete
**Contribution**: Discovery that frozen fine-tuned teachers can catastrophically fail

**Key finding**: Frozen SFT teacher drops QQP F1 by **34%** (90% → 56%)

| Task | Frozen SFT Effect | Co-train Recovery |
|------|-------------------|-------------------|
| MNLI | -1.8% | +2.9% |
| QQP | **-34.5%** | +34.6% |
| SST-2 | 0% | — |
| CoLA | -3.1% | (makes worse) |

**Key insight**: *A teacher's willingness to update matters more than its starting accuracy.*

**Publication-ready?**: Yes — striking result, clear story

### Paper 2: Simplicity Wins (R-only > FAR)

**Status**: Complete
**Contribution**: Response-only distillation beats multi-signal approaches

**Key finding**: Adding Feature (F) or Attention (A) signals hurts performance

| Config | Accuracy | vs R-only |
|--------|----------|-----------|
| R-only | **81.1%** | — |
| F+R | 80.5% | -0.6% |
| A+R | 80.6% | -0.5% |
| F+A+R | 80.2% | -0.9% |

**Key insight**: *More signals ≠ better. Soft labels contain the signal; hidden states add noise.*

**Publication-ready?**: Yes — clear ablation, practical recommendation

### Paper 3: Does It Transfer to Generation?

**Status**: Experiments pending
**Contribution**: Validate classification findings on Llama translation

**Hypotheses to test**:
1. Frozen SFT hurts translation too
2. Co-training helps
3. R-only beats FAR for generation

**Publication-ready?**: No — needs experiments

---

## Part 4: Publication Strategy

### Recommended Path

```
Week 1-2: Submit Paper 1 to workshop
    ↓
Week 3-4: Run Paper 3 experiments
    ↓
Month 2: Combine Papers 1+2+3 into full paper
    ↓
Month 3: Submit to NAACL/EACL/EMNLP
```

### Option A: Workshop Paper (Fastest)

**Target**: NeurIPS Workshop on Efficient ML, ICML Workshop on Knowledge Transfer

**What to submit**: Paper 1 alone (4 pages)
- Striking result (-34% F1)
- Clear narrative
- Practical takeaway

### Option B: Short Paper

**Target**: NAACL, EACL, ACL Findings

**What to submit**: Papers 1 + 2 combined (6 pages)
- Problem: Frozen teachers fail
- Solution: Use R-only distillation
- Contribution: Practical guidelines

### Option C: Full Paper (Best Long-term Value)

**Target**: ACL, EMNLP, NAACL

**What to submit**: Papers 1 + 2 + 3 (8 pages)
- Comprehensive study across tasks
- Classification + generation
- Strong empirical contribution

---

## Part 5: Writing Tips for Beginners

### The 5-Sentence Abstract

1. **Context**: "Knowledge distillation transfers knowledge from large to small models."
2. **Problem**: "Standard practice freezes fine-tuned teachers, but..."
3. **Approach**: "We present a systematic ablation across 4 tasks..."
4. **Result**: "We find frozen teachers can drop F1 by 34%..."
5. **Implication**: "Practitioners should use co-training for safety."

### Common Mistakes to Avoid

| Mistake | Fix |
|---------|-----|
| Burying the lede | Put your main finding in sentence 1-2 |
| Vague claims | Use specific numbers ("34% drop", not "significant decrease") |
| No baselines | Always compare to CE-only (no teacher) |
| Overclaiming | "We show X" not "We prove X" |
| Missing related work | Cite Hinton 2015, FitNets, TinyBERT |

### The "So What?" Test

Every paper must answer:
1. **What did you do?** — Ablation study across 4 tasks
2. **What did you find?** — Frozen SFT teachers fail catastrophically
3. **Why does it matter?** — Saves practitioners from 34% performance drops
4. **What should readers do differently?** — Use co-training, not frozen teachers

---

## Part 6: Quick Reference

### Venue Deadlines
- Check [aideadlin.es](https://aideadlin.es) for ML conference deadlines
- Check [ACL Rolling Review](https://aclrollingreview.org) for NLP venues

### Key References to Cite
- Hinton et al. (2015) — Original KD paper
- Sanh et al. (2019) — DistilBERT
- Romero et al. (2015) — FitNets (feature distillation)
- Zagoruyko & Komodakis (2017) — Attention transfer
