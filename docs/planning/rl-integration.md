# RL Integration: Post-Training with TRL

A planning doc for adding reinforcement learning as a first-class training mode in Forge.

---

## Background

Forge currently supports knowledge distillation (KD) via `DistillationTrainer`. This doc covers expanding the codebase to support RL-based post-training — both standalone and (eventually) combined with KD.

**Training modes the framework should support:**

| Mode | Description | Status |
|------|-------------|--------|
| **Pure KD** | FAR distillation via DistillationTrainer | Existing |
| **Pure RL** | DPO/GRPO standalone, no teacher | Planned |
| **KD+RL hybrid** | Distillation signals + preference optimization | Future |
| **KD → RL pipeline** | Sequential: distill first, then align | Future |

**Library:** [TRL](https://huggingface.co/docs/trl/) (Transformer Reinforcement Learning) from HuggingFace. Provides `DPOTrainer`, `GRPOTrainer`, `PPOTrainer` — all extend the same HF `Trainer` base class we use.

---

## Part 1: RL Algorithms

### DPO (Direct Preference Optimization)

- **Type:** Offline (no generation during training)
- **Data:** Preference pairs — (prompt, chosen, rejected)
- **Models in memory:** Policy + reference (2)
- **Used by:** Llama 2, Llama 3

**Loss function:**

```
L = -log σ( β * (reward_chosen - reward_rejected) )
```

Where `reward(y|x) = log π_policy(y|x) - log π_ref(y|x)`.

This is logistic regression on log-probability differences. The model learns to assign higher probability to "chosen" relative to "rejected," while the reference model (frozen copy of initial policy) prevents the policy from drifting too far.

- **β** controls sensitivity — higher β requires sharper distinction between chosen/rejected
- **Reference model** prevents reward hacking (model collapsing to degenerate outputs)

**DPO is contrastive learning** — the signal comes from comparing two outputs, not memorizing a single target.

### GRPO (Group Relative Policy Optimization)

- **Type:** Online (generates during training)
- **Data:** Prompts only — completions generated on-the-fly
- **Models in memory:** Policy + reference (2)
- **Used by:** DeepSeek-R1

**How it works:**
1. For each prompt, generate G completions
2. Score each completion with a reward function
3. Normalize rewards within the group: `advantage_i = (reward_i - mean) / std`
4. Update policy to increase probability of high-advantage completions

**Loss function:**

```
L = -mean[ min(ratio * A, clip(ratio, 1-ε, 1+ε) * A) ]
```

Where `ratio = π_policy / π_ref` and `A` is the group-relative advantage.

**Key property:** No reward model needed — any function that scores completions works. For translation, this can be BLEU or COMET scores.

### PPO (Proximal Policy Optimization)

- **Type:** Online (generates during training)
- **Data:** Prompts + learned reward model
- **Models in memory:** Policy + reference + reward model + value head (4)
- **Used by:** ChatGPT (original RLHF)

**How it works:** Classic actor-critic RL. Policy generates, reward model scores, value head estimates returns, policy updates with clipped objective.

**Downsides:** Most complex to tune. 4 models in memory is expensive. The reward model itself must be trained from preference data.

### Algorithm Comparison

| | DPO | GRPO | PPO |
|---|---|---|---|
| **Complexity** | Lowest | Medium | Highest |
| **Memory** | 2 models | 2 models | 4 models |
| **Generation** | None (offline) | Every step | Every step |
| **Reward** | Implicit | Any function | Learned model |
| **Stability** | Very stable | Moderate | Hardest to tune |
| **Best for** | Preference alignment | Verifiable tasks | Large-scale RLHF |

---

## Part 2: Signal Sources

The algorithms above describe *how* to optimize. The signal source describes *where the reward comes from*. These are orthogonal — any algorithm can use any signal.

### RLHF (Reinforcement Learning from Human Feedback)

- Human annotators rank model outputs
- Expensive, slow, gold standard
- Example: "Translation A reads more naturally than Translation B"

### RLAIF (Reinforcement Learning from AI Feedback)

- An AI model (GPT-4, Claude) judges outputs
- Cheaper and scalable, but introduces AI bias
- Example: Prompt Claude to judge translation quality, generate preference pairs

### RLVR (Reinforcement Learning from Verifiable Rewards)

- Programmatic, rule-based rewards — no humans or AI judges
- Examples: BLEU score, COMET score, math correctness, unit tests passing
- Cheapest and most reproducible

### Algorithm × Signal Matrix

Algorithms and signal sources are mix-and-match:

| | RLHF | RLAIF | RLVR |
|---|---|---|---|
| **DPO** | Llama 2/3 (human prefs → DPO) | Constitutional AI style (AI judge → DPO) | Score outputs, binarize → DPO (lossy) |
| **GRPO** | Human scores as reward (rare) | AI judge as reward function | **Natural fit** (BLEU/COMET as reward) |
| **PPO** | ChatGPT (human prefs → reward model → PPO) | AI judge → reward model → PPO | Verifiable reward → PPO |

**Natural pairings for translation:**
- **RLVR + GRPO**: Generate N translations, score with BLEU/COMET, learn from relative quality. Most natural.
- **RLAIF + DPO**: Use LLM judge to rank translations, train on preference pairs.

**Lossy pairing:**
- **RLVR + DPO**: Binarizes a continuous score (BLEU) into chosen/rejected. Works but throws away information.

---

## Part 3: DPO vs SFT

DPO is sometimes confused with standard supervised fine-tuning (SFT). The key difference:

**SFT (standard training):**
```
Input:  "Translate: The cat sat on the mat"
Signal: "Le chat était assis sur le tapis"  ← match this target
Loss:   Cross-entropy against ground truth tokens
```
The model sees the correct answer and learns to reproduce it. Absolute signal.

**DPO (contrastive):**
```
Input:    "Translate: The cat sat on the mat"
Chosen:   "Le chat était assis sur le tapis"      ← good
Rejected: "Le chat assis sur la mat"               ← bad
Loss:     Increase P(chosen) relative to P(rejected)
```
The model sees both good and bad outputs and learns what distinguishes them. Relative signal.

**Where DPO adds value over SFT:**
- When the reward signal captures things ground truth can't (fluency, safety, style)
- When preference data is from the model's own distribution (on-policy)
- When you want to prevent reward hacking via the reference model constraint

**Where SFT is sufficient:**
- When you have high-quality ground truth and just want the model to match it
- When the task metric is directly optimizable (e.g., classification accuracy)

---

## Part 4: Preference Dataset Construction

DPO requires datasets with columns: `prompt`, `chosen`, `rejected`.

### Option 1: Ground Truth as Chosen (Simplest)

Use existing Europarl ground truth translations as "chosen." Generate "rejected" from the student model.

```
prompt:   "Translate English to French: The parliament voted today."
chosen:   "Le parlement a voté aujourd'hui."                ← Europarl ground truth
rejected: "Le parlement voté aujourd'hui."                  ← student generation (pre-DPO)
```

- Works with existing data — just need one generation pass from the student
- Ground truth isn't from the same distribution as the model (slightly unprincipled for DPO)
- Simplest starting point

### Option 2: Teacher as Chosen, Student as Rejected

Generate from both teacher (3B) and student (1B) on the same prompts.

```
prompt:   "Translate English to French: The parliament voted today."
chosen:   "Le parlement a voté aujourd'hui."                ← teacher (3B) generation
rejected: "Le parlement voté aujourd'hui."                  ← student (1B) generation
```

- Both outputs from model distributions (more principled)
- Requires generation from both models (offline, run once)
- Can be done before or after KD (doesn't require KD first)

### Option 3: External Preference Datasets

Load pre-existing preference datasets from HuggingFace Hub. Available for general instruction-following but limited for translation-specific tasks.

---

## Part 5: Integration Architecture

Three-layer design:

```
┌─────────────────────────────────────────────┐
│  Config Layer (configs/)                     │
│  training.mode: kd | dpo | grpo             │
│  rl.method: dpo | grpo                      │
│  rl.beta, rl.loss_type, etc.                │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│  Algorithm Layer (src/training/rl.py)        │
│  Thin wrappers around TRL trainers          │
│  build_dpo_trainer() → DPOTrainer           │
│  build_grpo_trainer() → GRPOTrainer         │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│  Signal Layer                                │
│                                              │
│  Data (src/data/preference.py):              │
│    load_preference_dataset()  ← for DPO     │
│    load_prompt_dataset()      ← for GRPO    │
│                                              │
│  Rewards (src/rewards/):                     │
│    bleu_reward()      ← RLVR: translation   │
│    comet_reward()     ← RLVR: translation   │
│    llm_judge_reward() ← RLAIF: API call     │
└─────────────────────────────────────────────┘
```

**Entry point dispatch** in `forge.py`:

```python
mode = config["training"].get("mode", "kd")

if mode == "kd":
    # Existing DistillationTrainer path (unchanged)
    ...
elif mode == "dpo":
    # TRL DPOTrainer via src/training/rl.py
    ...
elif mode == "grpo":
    # TRL GRPOTrainer via src/training/rl.py
    ...
```

**Trainer hierarchy:**

```
              HF Trainer
             /    |     \
DistillationTrainer  DPOTrainer  GRPOTrainer
    (existing)        (TRL)        (TRL)
                        |
              [future: DPODistillationTrainer]
              [hybrid KD+DPO recipe]
```

No custom trainer subclass needed for pure RL — use TRL's trainers directly. Hybrid subclass comes later.

---

## Part 6: Future Recipes

These combine KD and RL and are deferred to later:

| Recipe | Description | Trainer |
|--------|-------------|---------|
| **kd_dpo** | KD + DPO simultaneously | `DPODistillationTrainer(DPOTrainer)` — adds KD loss to DPO |
| **kd_then_dpo** | KD first, then DPO refinement | Sequential: run KD, load checkpoint, run DPO |
| **grpo** | GRPO with BLEU/COMET reward | `GRPOTrainer` with custom reward functions |
| **FAR+DPO** | Which KD signals help when combined with DPO? | FAR factorial extended to hybrid training |

---

## References

- Rafailov et al., 2023 — "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (DPO)
- Shao et al., 2024 — "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" (GRPO)
- Schulman et al., 2017 — "Proximal Policy Optimization Algorithms" (PPO)
- Ouyang et al., 2022 — "Training language models to follow instructions with human feedback" (RLHF + PPO)
- Lee et al., 2023 — "RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback"
