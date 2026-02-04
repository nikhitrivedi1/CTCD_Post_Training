# Paper Drafts

Three focused papers exploring different aspects of knowledge distillation.

---

## Papers

| Paper | Title | Focus | Status |
|-------|-------|-------|--------|
| [Paper 1](paper-1-ablation.md) | When Teachers Fail | Frozen vs Co-training, Overconfident Teacher Paradox | **Complete** (MNLI + QQP) |
| [Paper 2](paper-2-far.md) | Simplicity Wins | R-only > FAR, Multi-signal gradient conflict | **Complete** (MNLI) |
| [Paper 3](paper-3-translation.md) | Do Findings Transfer? | Validate on generation (Llama 3B → 1B) | **Configs ready** |

---

## Key Findings Summary

### Paper 1: Ablation Study
- **Frozen SFT teacher < CE-only** — Overconfident wrong predictions hurt
- **Co-training > Frozen** — Adaptive teachers beat stubborn experts
- **Pattern holds across MNLI and QQP**

### Paper 2: FAR Study
- **R-only wins** — Classic KL distillation beats all multi-signal combinations
- **Signal ranking: R >> F > A**
- **Adding F or A to R hurts performance** — Gradient conflict

### Paper 3: Translation (Pending)
- Tests if classification findings transfer to autoregressive generation
- Llama 3B → 1B on Europarl (en-de)

---

## Narrative Arc

```
Paper 1: "What we thought about distillation is wrong"
    └── Expert frozen teachers can hurt
    └── Co-training provides robustness

Paper 2: "And the fix is simpler than we thought"
    └── R-only beats complex multi-signal approaches
    └── More signals ≠ better transfer

Paper 3: "Does this generalize?"
    └── Validate on fundamentally different task (generation)
    └── Different architecture (decoder-only)
```

---

## Option to Combine

These papers can be combined into a single comprehensive paper:

**Title:** "Rethinking Knowledge Distillation: Less is More"

**Structure:**
1. Introduction
2. Background
3. Study 1: Frozen vs Co-training (Paper 1)
4. Study 2: R vs FAR (Paper 2)
5. Study 3: Transfer to Generation (Paper 3)
6. Discussion
7. Conclusion
