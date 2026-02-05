# Framework Direction: Path to Distillation Mastery

Building hands-on expertise to reach the top 1% in knowledge distillation.

---

## Competency Matrix

| Level | Knowledge | Skills | Output |
|-------|-----------|--------|--------|
| **Awareness** | "KD makes models smaller" | Copy-paste from tutorials | None |
| **Basic** | Soft labels, temperature | Run existing code | Blog post |
| **Applied** | Loss functions, architectures | Implement from paper | Internal tool |
| **Production** | Trade-offs, failure modes | Debug and optimize | Deployed system |
| **Expert** | Why things work/fail | Design new approaches | Published paper |
| **Top 1%** | Frontier + intuition | Advance the field | Influential work |

---

## The Mastery Path

### Phase 1: Foundation (Current)

**Goals:**
- ✓ Understand standard KD setup (Hinton et al.)
- ✓ Implement basic logit distillation
- ✓ Run controlled experiments
- ✓ Question assumptions (frozen vs co-train, SFT vs base)
- ✓ Discover non-obvious findings (overconfident teacher problem)

**Key Insight:** "Frozen SFT teachers can hurt. Overconfidence on wrong predictions damages student learning more than noise helps."

### Phase 2: Breadth (Next)

**Goals:**
- Implement multiple distillation strategies
- Test across diverse tasks and architectures
- Understand when each approach works/fails

**Experiments:**
- Feature distillation (hidden state matching)
- Attention transfer
- Translation task (Llama 3B → 1B)
- Fresh vs pretrained student init
- Temperature ablation (T=1,2,4,8)
- Alpha sweep (0.0, 0.3, 0.5, 0.7, 1.0)

### Phase 3: Depth (Future)

**Goals:**
- Deep dive into failure modes
- Theoretical understanding of when KD helps
- Develop novel improvements

**Research Directions:**
- Confidence calibration for teachers
- Adaptive alpha based on teacher uncertainty
- Curriculum distillation (easy → hard)
- Multi-teacher ensembles

### Phase 4: Frontier (Long-term)

**Goals:**
- Contribute novel methods to the field
- Influence how others think about distillation

---

## Experiment Depth Levels

### L0: Ablation Studies
- **Purpose:** Isolate what matters
- **Method:** Toggle one variable at a time
- **Status:** ✓ MNLI/QQP 5-config ablation complete

### L1: KD Strategy Exploration
- **Purpose:** Explore different distillation approaches
- **Options:**
  - Response-based (current): Soft label matching (KL)
  - Feature-based: Hidden state matching
  - Attention transfer
- **Status:** Planning

### L2: Hyperparameter Optimization
- **Purpose:** Fine-tune for best performance
- **Key Parameters:** Temperature, Alpha, Layer mapping, Learning rate

---

## L1 Strategy Combinations

| ID | Config | R (KL) | F (MSE) | A (Attn) | Description |
|----|--------|:------:|:-------:|:--------:|-------------|
| 00 | Baseline | - | - | - | CE-only, no teacher |
| 01 | R | ✓ | - | - | Response-based (current KL) |
| 05 | F | - | ✓ | - | Feature/hidden states only |
| 06 | A | - | - | ✓ | Attention transfer only |
| 07 | R+F | ✓ | ✓ | - | DistilBERT-style hybrid |
| 08 | R+A | ✓ | - | ✓ | Logits + attention |
| 10 | R+F+A | ✓ | ✓ | ✓ | Full combination |

---

## Skills Roadmap

| Skill | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|-------|---------|---------|---------|---------|
| **Loss Functions** | KL, CE | MSE, Combined | Custom losses | Novel losses |
| **Architectures** | Encoder (BERT) | Decoder (Llama) | Enc-Dec, Diffusion | Any |
| **Strategies** | Logit | Feature, Attention | Relational | Novel |
| **Tasks** | Classification | Generation | Multi-task | Any |
| **Scale** | Single GPU | Multi-GPU | Distributed | Production |

---

## Success Metrics

| Phase | Technical Marker | Knowledge Marker | Output Marker |
|-------|------------------|------------------|---------------|
| **1** | Run 5-config ablation | Explain frozen teacher failure | Course report |
| **2** | Implement 3+ strategies | Predict which strategy fits task | Workshop submission |
| **3** | Design novel experiment | Explain theoretical basis | Accepted paper |
| **4** | Advance state-of-art | Recognized expert | Invited talks |

---

## Reading List by Phase

**Phase 1 (Foundation):**
- Hinton et al., 2015 - "Distilling the Knowledge in a Neural Network"
- Sanh et al., 2019 - "DistilBERT"

**Phase 2 (Breadth):**
- Romero et al., 2015 - "FitNets: Hints for Thin Deep Nets"
- Zagoruyko and Komodakis, 2017 - "Attention Transfer"
- Park et al., 2019 - "Relational Knowledge Distillation"

**Phase 3 (Depth):**
- Cho and Hariharan, 2019 - "On the Efficacy of Knowledge Distillation"
- Mirzadeh et al., 2020 - "Improved Knowledge Distillation via Teacher Assistant"

**Phase 4 (Frontier):**
- Salimans and Ho, 2022 - "Progressive Distillation for Fast Sampling"
- Song et al., 2023 - "Consistency Models"

---

## Learning Principles

| Principle | Implementation |
|-----------|----------------|
| **Learn by doing** | Every concept → experiment |
| **Question assumptions** | Ablate everything, trust data over intuition |
| **Document insights** | Write it down or lose it |
| **Build incrementally** | Each experiment builds on previous |
| **Fail fast** | Quick experiments, rapid iteration |
| **Go deep** | Understand why, not just what |
