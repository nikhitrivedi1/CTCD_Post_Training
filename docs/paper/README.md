# Papers

Research papers on knowledge distillation findings.

---

## Paper Series

| Paper | Title | Focus | Status |
|-------|-------|-------|--------|
| [Paper 0](paper-0-framework/draft.md) | A Unified Framework | Config-driven distillation research framework | **Draft** |
| [Paper 1](paper-1-ablation/draft.md) | When Teachers Fail | Frozen vs Co-training, Overconfident Teacher Paradox | **Complete** (MNLI + QQP) |
| [Paper 2](paper-2-far/draft.md) | Simplicity Wins | R-only > FAR, Multi-signal gradient conflict | **Complete** (MNLI) |
| [Paper 3](paper-3-translation/draft.md) | Do Findings Transfer? | Validate on generation (Llama 3B → 1B) | **In Progress** |

---

## Narrative Arc

```
Paper 0: "Here's how to do distillation research right"
    └── Unified framework for reproducibility
    └── FAR factorial design methodology

Paper 1: "What we thought about distillation is wrong"
    └── Expert frozen teachers can hurt
    └── Co-training provides robustness

Paper 2: "And the fix is simpler than we thought"
    └── R-only beats complex multi-signal approaches
    └── More signals ≠ better transfer

Paper 3: "Does this generalize?"
    └── Validate on fundamentally different task (generation)
    └── Different architecture (decoder-only LLMs)
```

---

## Key Findings

### Paper 0: Framework
- Config-driven experiment management
- FAR factorial design for systematic ablation
- Automatic CE-only baseline detection
- Multi-task support (classification + translation)

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
- Llama 3B → 1B on Europarl (en-fr)
- Uses LoRA for parameter-efficient distillation

---

## Combination Option

These papers can be combined into a single comprehensive paper:

**Title:** "Rethinking Knowledge Distillation: Less is More"

**Structure:**
1. Introduction
2. Framework & Methodology (Paper 0)
3. Study 1: Frozen vs Co-training (Paper 1)
4. Study 2: R vs FAR (Paper 2)
5. Study 3: Transfer to Generation (Paper 3)
6. Discussion
7. Conclusion

---

## Directory Structure

```
paper/
├── README.md                    # This file
├── paper-0-framework/
│   ├── draft.md                 # Framework paper draft
│   └── figures/                 # Architecture diagrams
├── paper-1-ablation/
│   ├── draft.md                 # Ablation study draft
│   └── figures/                 # Result plots
├── paper-2-far/
│   ├── draft.md                 # FAR study draft
│   └── figures/                 # Factorial analysis plots
└── paper-3-translation/
    ├── draft.md                 # Translation study draft
    └── figures/                 # BLEU/COMET plots
```

---

## Analysis Data

See [../analysis/](../analysis/) for:
- W&B data export scripts
- Jupyter analysis notebooks
- Summary tables and plots
