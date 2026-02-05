# Paper Writing Tips & Tricks

Practical advice for writing clear, compelling research papers.

---

## Well-Written Papers to Study

### Exemplary Writing

| Paper | Why It's Well-Written |
|-------|----------------------|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) | Title is a bold claim that's memorable. Clear structure. Introduced "Transformer" naming that stuck. |
| [BERT](https://arxiv.org/abs/1810.04805) (Devlin et al., 2018) | Simple, descriptive title. Each contribution clearly enumerated. Ablations are thorough. |
| [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) (Hinton et al., 2015) | "Dark knowledge" metaphor makes the concept memorable. Conversational tone accessible to newcomers. |
| [DistilBERT](https://arxiv.org/abs/1910.01108) (Sanh et al., 2019) | Concise (5 pages). Every sentence earns its place. Practical focus. |
| [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (GPT-3, Brown et al., 2020) | Title frames the contribution perfectly. Extensive experiments with clear takeaways. |
| [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (Kaplan et al., 2020) | Quantifies everything. Clear empirical laws. Figures tell the story. |
| [FlashAttention](https://arxiv.org/abs/2205.14135) (Dao et al., 2022) | Problem stated in first paragraph. IO-awareness concept explained simply. Impressive speedups quantified precisely. |

### What Makes Them Good

**Attention Is All You Need**
- Title: Makes a strong claim, invites curiosity
- Structure: Architecture → Training → Results → Ablations
- Naming: "Transformer", "self-attention" became standard vocabulary
- Key line: *"We propose a new simple network architecture, the Transformer, based solely on attention mechanisms"* — simple, direct

**Distilling the Knowledge (Hinton)**
- Introduces "dark knowledge" — a metaphor that stuck
- Temperature parameter explained with intuition (soft targets)
- Conversational: *"An obvious way to transfer the knowledge..."*
- Accessible to readers without deep background

**DistilBERT**
- 5 pages total — no filler
- Clear recipe: *"We applied knowledge distillation during the pre-training phase"*
- Honest about trade-offs: *"40% smaller, 60% faster, retains 97% of performance"*
- Reproducible: All hyperparameters listed

**Scaling Laws (Kaplan)**
- Leads with the finding: *"Performance depends strongly on scale"*
- Power laws with precise exponents (quantified)
- Figures are self-contained — you can understand the paper from figures alone
- Predictive: Tells you what will happen at larger scale

**FlashAttention**
- Problem in paragraph 1: *"Transformers are slow and memory-hungry on long sequences"*
- Solution in paragraph 2: IO-aware attention
- Benchmarks are comprehensive and fair
- Open source code linked

### Anti-Patterns to Avoid

| Paper Type | Problem | Fix |
|------------|---------|-----|
| "We achieve SOTA" | No insight into why | Explain the mechanism |
| "We propose X, Y, and Z" | Too many contributions dilute impact | Focus on one key idea |
| "Related work" that's 2 pages | Reader loses the thread | Keep it tight, cite inline |
| Figures require 5 minutes to parse | Reader skips them | One message per figure |
| "Future work" that's vague | Feels like filler | Be specific or cut it |

---

## 1. Find Real-Life Parallels

Technical concepts become memorable when mapped to everyday experiences. Readers grasp intuition before formulas.

### Examples from Our Work

| Technical Concept | Real-Life Parallel | Why It Works |
|-------------------|-------------------|--------------|
| **Frozen vs Co-training** | "Strong Beliefs, Loosely Held" | Captures that adaptability beats certainty |
| **Overconfident Teacher** | "GPS that's 92% right but says 'turn into the lake'" | Everyone knows GPS mistakes |
| **Co-training Self-Correction** | "Adaptive Domain Expert" who updates beliefs | Intuitive that learning experts are better |
| **Binary vs Multi-class Risk** | "Partial credit vs all-or-nothing exam" | Students understand grading |
| **Knowledge Transfer Efficiency** | "How much of your tutor's knowledge did you absorb?" | Learning from teachers is universal |

### How to Find Good Parallels

1. **Start with the mechanism** - What's actually happening?
2. **Think of everyday scenarios** - Where does this happen in life?
3. **Test for accuracy** - Does the parallel capture the key insight without misleading?
4. **Keep it simple** - If you need to explain the parallel, it's too complex

### Red Flags

- Parallel requires domain knowledge (physics, economics)
- Parallel has obvious failure modes that don't apply
- Parallel is clever but not illuminating
- Parallel distracts from the main point

---

## 2. Lead with the Counterintuitive

Papers get attention when they challenge assumptions.

### Strong Openers

| Weak | Strong |
|------|--------|
| "We study knowledge distillation" | "Knowledge distillation can catastrophically fail" |
| "We compare frozen and co-training" | "A 91% accurate teacher destroys student performance" |
| "We analyze different loss functions" | "More signals make distillation worse, not better" |

### The Pattern

```
Common belief: X is true
Our finding: X fails when Y
Implication: Do Z instead
```

Example:
```
Common belief: Better teachers produce better students
Our finding: Overconfident teachers produce worse students than no teacher
Implication: Teacher adaptability > teacher accuracy
```

---

## 3. Quantify Everything

Vague claims are forgettable. Numbers stick.

### Examples

| Vague | Quantified |
|-------|------------|
| "The teacher is often wrong" | "The teacher is wrong 8.4% of the time" |
| "High confidence when wrong" | "82% average confidence on wrong predictions" |
| "Performance drops significantly" | "F1 drops from 90.4% to 55.9% (-34.5 points)" |
| "Large dataset" | "393k examples" |

### The Magic Numbers

Find the one or two numbers that capture your main finding:
- **82%** - confidence when wrong (the problem)
- **-34.5%** - F1 drop (the consequence)
- **8.4%** - wrong predictions (seems small, but catastrophic)

These become your "headline stats" that readers remember.

---

## 4. Tables Over Prose

Comparisons belong in tables. Prose buries the signal.

### Before (Prose)

> "We found that frozen SFT achieved 55.9% F1, while co-train SFT achieved 90.5% F1, and the CE-only baseline achieved 90.4% F1. This shows that frozen SFT performs much worse than both alternatives."

### After (Table)

| Config | F1 Score | Effect |
|--------|----------|--------|
| CE-only | 90.4% | Baseline |
| Co-train SFT | 90.5% | ✓ +0.1% |
| Frozen SFT | **55.9%** | ✗ **-34.5%** |

The table makes the catastrophic failure immediately visible.

---

## 5. One Idea Per Section

Each section should have exactly one main point. If you're making multiple points, split into subsections.

### Structure Template

```markdown
## Section: [Main Claim]

[1-2 sentence setup]

[Evidence: table, figure, or key numbers]

[Interpretation: what this means]

[Implication: what to do about it]
```

### Example

```markdown
## Frozen Teachers Can Catastrophically Fail

On QQP, a frozen fine-tuned teacher causes student performance to collapse.

| Config | F1 Score |
|--------|----------|
| CE-only | 90.4% |
| Frozen SFT | 55.9% |

The 34-point drop occurs because the teacher is 82% confident on wrong predictions.
When frozen, these confident mistakes create irreconcilable gradient conflict.

**Recommendation:** Always use co-training for binary classification tasks.
```

---

## 6. Name Your Contributions

Findings need names to be memorable and citable.

### Examples

| Unnamed | Named |
|---------|-------|
| "The problem where teachers are confident when wrong" | "The Overconfident Teacher Problem" |
| "When frozen teachers hurt performance" | "Catastrophic Teacher Failure" |
| "The finding that R-only beats combinations" | "The Simplicity Principle" |
| "Measuring how much knowledge transferred" | "Knowledge Transfer Efficiency" |

### Naming Guidelines

- Short (2-4 words)
- Descriptive (captures the essence)
- Not jargon-heavy
- Googleable (unique enough to find)

---

## 7. The Intuition Sandwich

Technical content should be wrapped in intuition.

### Pattern

```
1. Intuition: "Think of it like..."
2. Technical: Formulas, algorithms, details
3. Implication: "This means that..."
```

### Example

```markdown
**Intuition:** Think of the teacher as a GPS that's right 92% of the time.
When it's wrong, it confidently says "turn left into the lake."

**Technical:** The loss function combines CE and KL:
L = (1-α) × CE(student, labels) + α × KL(student, teacher)

When teacher confidence is high but wrong, the KL gradient
points opposite to the CE gradient, creating conflict.

**Implication:** A student following this GPS will get stuck—
half the signal says "go forward," half says "turn into the lake."
```

---

## 8. Ablation Tables Are Your Friends

Show what matters by removing it.

### Standard Ablation Format

| Component | Included | Performance | Δ from Full |
|-----------|----------|-------------|-------------|
| Full (R+F+A) | ✓ ✓ ✓ | 82.1% | - |
| R only | ✓ - - | **82.8%** | +0.7% |
| F only | - ✓ - | 81.2% | -0.9% |
| A only | - - ✓ | 80.9% | -1.2% |
| None (CE) | - - - | 81.5% | -0.6% |

This table immediately shows R-only wins.

---

## 9. Failure Modes Are Features

Don't hide failures—they're often the most interesting findings.

### What to Include

- When does your method fail?
- Why does it fail?
- Can you predict failure?
- What should practitioners do instead?

### Example

```markdown
### When KD Fails

| Condition | Effect | Recommendation |
|-----------|--------|----------------|
| Dataset < 10k | KD adds noise | Skip KD |
| Binary + imbalanced | Catastrophic if frozen | Co-train only |
| Task saturated (>90%) | No room to improve | Skip KD |
```

This is more useful than "our method works great."

---

## 10. The Decision Flowchart

End with actionable guidance.

### Example

```
Should you use knowledge distillation?

Dataset size < 10k?
  └─ YES → Skip KD (use CE-only)
  └─ NO ↓

Binary + imbalanced?
  └─ YES → Use co-training ONLY
  └─ NO ↓

Task saturated (>90%)?
  └─ YES → Skip KD (marginal gains)
  └─ NO → Use KD with co-training
```

Readers can apply your findings immediately.

---

## Quick Checklist

Before submitting, verify:

- [ ] Lead with counterintuitive finding
- [ ] All claims are quantified
- [ ] Comparisons are in tables
- [ ] Each section has one main point
- [ ] Key findings are named
- [ ] Technical content has intuition wrappers
- [ ] Real-life parallels for hard concepts
- [ ] Failure modes are documented
- [ ] Ends with actionable guidance
