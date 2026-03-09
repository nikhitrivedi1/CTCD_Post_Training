# Training: Loss Strategies

Pluggable distillation loss functions using the Strategy Pattern.

**Source:** `src/training/losses.py` (~906 lines)

---

## Overview

All loss strategies implement the `DistillationLossStrategy` abstract base class and are registered in `LOSS_REGISTRY` for config-driven creation. The naming convention is `{signal}_{method}` where signal is one of `response`, `feature`, or `attention`.

---

## Core Abstractions

### LossOutput

```python
@dataclass
class LossOutput:
    total_loss: torch.Tensor       # The combined loss to backpropagate
    components: Dict[str, Tensor]  # Breakdown for logging (ce_weighted, kl_raw, etc.)
```

Every strategy returns a `LossOutput`. The `components` dict is logged to W&B via the trainer's metric accumulator.

### DistillationLossStrategy (ABC)

```python
class DistillationLossStrategy(ABC):
    requires_hidden_states: bool = False  # Set True for Feature losses
    requires_attentions: bool = False     # Set True for Attention losses

    @abstractmethod
    def compute(self, student_logits, teacher_logits, ce_loss, labels=None,
                student_outputs=None, teacher_outputs=None, **kwargs) -> LossOutput

    @property
    @abstractmethod
    def name(self) -> str
```

The `requires_*` flags tell the trainer to pass `output_hidden_states=True` or `output_attentions=True` to the model forward pass. Only strategies that need intermediate representations set these.

### compute_kl_divergence (helper)

```python
def compute_kl_divergence(student, teacher, temperature, input_is_probs=False, reverse=False)
```

Core KL computation used by multiple strategies. Key details:
- **Computes in fp32** to avoid underflow with high temperatures (fp16 `softmax(logits/T)` can underflow when T is large)
- **T^2 scaling** maintains gradient magnitude across different temperatures (Hinton et al. 2015)
- **Forward KL** (`reverse=False`): `KL(teacher || student)` - mean-seeking, student covers all teacher modes
- **Reverse KL** (`reverse=True`): `KL(student || teacher)` - mode-seeking, student picks dominant modes (MiniLLM, Gu et al. 2024)
- **Probability inputs** (`input_is_probs=True`): for attention weights, converts probs to logits first via `log(probs + eps)`

---

## Response-Based Losses

These operate on output logits only. No hidden states or attentions needed.

### KLDistillationLoss

```
Loss = (1 - alpha) * CE + alpha * T^2 * KL(softmax(s/T) || softmax(t/T))
```

**Name:** `response_kl` (or `response_reverse-kl`)

The classic Hinton et al. (2015) "dark knowledge" distillation loss. The teacher's soft probability distribution transfers inter-class relationships that hard labels cannot capture (e.g., "this 7 looks a bit like a 1").

Parameters:
- `temperature` (float, default 2.0): Higher T makes distributions softer, revealing more dark knowledge
- `alpha` (float, default 0.5): Balance between CE (ground truth) and KL (teacher)
- `reverse` (bool, default False): Use reverse KL for mode-seeking behavior

Logged components: `ce_weighted`, `kl_weighted`, `ce_raw`, `kl_raw`, `alpha`

### MSEDistillationLoss

```
Loss = (1 - alpha) * CE + alpha * MSE(student_logits, teacher_logits)
```

**Name:** `response_mse`

Direct L2 distance on raw logits. Simpler than KL, no temperature needed. Optional normalization (z-score) prevents magnitude differences from dominating.

Parameters:
- `alpha` (float, default 0.5)
- `normalize` (bool, default True): Z-score normalize logits before MSE

Logged components: `ce_weighted`, `mse_weighted`, `ce_raw`, `mse_raw`, `alpha`

### CosineDistillationLoss

```
Loss = (1 - alpha) * CE + alpha * (1 - cosine_sim(student, teacher))
```

**Name:** `response_cosine`

Focuses on the direction of output vectors, ignoring magnitude. Useful when teacher and student have different logit scales.

Parameters:
- `alpha` (float, default 0.5)

### SoftTargetDistillationLoss

```
Loss = (1 - alpha) * CE_hard + alpha * T^2 * CE(student/T, softmax(teacher/T))
```

**Name:** `response_ce`

Cross-entropy with soft labels (teacher probabilities) instead of KL divergence. Mathematically related to KL but can behave differently in practice because CE_soft = KL + H(teacher), and the entropy term H(teacher) provides a constant offset that doesn't affect gradients.

Parameters:
- `temperature` (float, default 2.0)
- `alpha` (float, default 0.5)

Logged components: `ce_hard_weighted`, `ce_soft_weighted`, `ce_hard_raw`, `ce_soft_raw`, `alpha`

---

## Feature-Based Losses

These operate on hidden states from intermediate layers. Require `output_hidden_states=True`.

### FeatureLossStrategy

```
Loss = (1 - alpha) * CE + alpha * mean(MSE(proj(student_hidden_i), teacher_hidden_j))
```

**Name:** `feature_mse`

FitNets-style hidden state matching (Romero et al. 2015). For each mapped layer pair (teacher_j, student_i):
1. Get hidden states: `(batch, seq_len, hidden_dim)`
2. Project student if dimensions differ via `projector(student_hidden)`
3. Align dtypes (projector may output fp32 while teacher is fp16)
4. Optionally normalize both (z-score) before comparison
5. Compute MSE loss

The losses across all layer pairs are averaged.

Parameters:
- `alpha` (float, default 0.5)
- `layer_mapping` (list of tuples): passed at init or via kwargs
- `projector` (nn.Module): passed at init or via kwargs
- `normalize` (bool, default True)

Logged components: `ce_weighted`, `feature_weighted`, `ce_raw`, `feature_raw`, `num_layers_matched`, `alpha`

### CosineFeatureLoss

```
Loss = (1 - alpha) * CE + alpha * mean(1 - cosine_sim(student_hidden_i, teacher_hidden_j))
```

**Name:** `feature_cosine`

DistilBERT-style feature matching (Sanh et al. 2019). Uses cosine similarity instead of MSE, focusing on direction rather than magnitude of hidden state vectors.

Finding: DistilBERT reports +1.46 GLUE points when cosine feature loss is included alongside response KL.

Parameters:
- `alpha` (float, default 0.5)
- `layer_mapping`, `projector` (same as FeatureLossStrategy)

Logged components: `ce_weighted`, `feature_cosine_weighted`, `ce_raw`, `feature_cosine_raw`, `num_layers_matched`, `alpha`

---

## Attention-Based Losses

These operate on attention weight matrices. Require `output_attentions=True`, which forces eager attention (SDPA doesn't expose weights).

### AttentionLossStrategy

```
Loss = (1 - alpha) * CE + alpha * mean(KL(student_attn_i || teacher_attn_j))
```

**Name:** `attention_kl`

Transfers attention patterns from teacher to student (Zagoruyko & Komodakis 2017). For each mapped layer pair:
1. Get attention weights: `(batch, num_heads, seq_len, seq_len)`
2. Handle different head counts by averaging both to single head
3. Flatten to `(N, seq_len)` for KL computation
4. Use `compute_kl_divergence` with `input_is_probs=True` (attention weights are already probabilities)

Parameters:
- `alpha` (float, default 0.5)
- `layer_mapping` (list of tuples)
- `temperature` (float, default 1.0)

**Important:** Attention distillation requires `attn_implementation="eager"` for both teacher and student. SDPA fuses attention internally and never materializes the attention weight matrix.

---

## Composite Strategies

### CombinedDistillationLoss

Combines multiple strategies with configurable weights. Used for FAR factorial experiments (8 combinations of Feature/Attention/Response).

```python
CombinedDistillationLoss(
    strategies=[FeatureLossStrategy(...), AttentionLossStrategy(...), KLDistillationLoss(...)],
    weights=[1.0, 0.5, 1.0],
    alpha=0.5
)
```

**Name:** `combined_{strategy1}_{strategy2}_...`

How it works:
1. CE contribution: `(1 - alpha) * ce_loss` (computed once)
2. For each sub-strategy, compute its output and extract the distillation component: `strategy_distill = output.total_loss - (1 - strategy.alpha) * ce_loss`
3. Weighted sum of distillation components
4. Total: `ce_weighted + alpha * weighted_distill_sum`

Weights are normalized to sum to 1. Aggregates `requires_hidden_states` and `requires_attentions` flags from all sub-strategies.

### StagedDistillationLoss

Switches strategies over training progress. Enables curriculum-style distillation.

```yaml
# Config example
stages:
  - strategy: feature_mse
    until_progress: 0.5    # First 50% of training
  - strategy: response_kl
    until_progress: 1.0    # Remaining 50%
```

**Name:** `staged_{strategy1}_then_{strategy2}`

The trainer passes `training_progress` (global_step / max_steps) via kwargs. `get_current_strategy(progress)` selects the active strategy based on `until_progress` thresholds.

Each stage can override `alpha` and `temperature`:
```yaml
stages:
  - strategy: feature_mse
    alpha: 0.7
    until_progress: 0.5
  - strategy: response_kl
    alpha: 0.3
    temperature: 4.0
    until_progress: 1.0
```

Logged components include `active_stage` (0-indexed) and `training_progress` for monitoring.

**Motivation:** DistilBERT found that Feature loss helps during pre-training (representation alignment), but our experiments found that Response-only (KL) wins during fine-tuning. Staged distillation lets you combine both insights.

---

## Loss Registry

```python
LOSS_REGISTRY = {
    # Response-based
    "response_kl":         KLDistillationLoss,
    "response_reverse-kl": partial(KLDistillationLoss, reverse=True),
    "response_mse":        MSEDistillationLoss,
    "response_cosine":     CosineDistillationLoss,
    "response_ce":         SoftTargetDistillationLoss,

    # Feature-based
    "feature_mse":    FeatureLossStrategy,
    "feature_cosine": CosineFeatureLoss,

    # Attention-based
    "attention_kl": AttentionLossStrategy,

    # Meta
    "staged": StagedDistillationLoss,

    # Backward-compat aliases
    "kl_divergence":    KLDistillationLoss,
    "mse_logits":       MSEDistillationLoss,
    "cosine_embedding": CosineDistillationLoss,
    "soft_target":      SoftTargetDistillationLoss,
}
```

### Factory: `get_loss_strategy(name, **kwargs)`

Creates a strategy instance by name. Filters kwargs to only what each strategy's constructor accepts (different strategies take different params). This prevents errors like passing `temperature` to `MSEDistillationLoss` which doesn't use it.

```python
# Accepted params per strategy
KLDistillationLoss:       {temperature, alpha, reverse}
MSEDistillationLoss:      {alpha, normalize}
CosineDistillationLoss:   {alpha}
SoftTargetDistillationLoss: {temperature, alpha}
FeatureLossStrategy:      {alpha, layer_mapping, projector, normalize}
CosineFeatureLoss:        {alpha, layer_mapping, projector}
AttentionLossStrategy:    {alpha, layer_mapping, temperature}
StagedDistillationLoss:   {stages, alpha, temperature}
```

---

## Adding a New Loss Strategy

1. Subclass `DistillationLossStrategy`
2. Set `requires_hidden_states` / `requires_attentions` class attributes if needed
3. Implement `compute()` returning a `LossOutput`
4. Implement `name` property
5. Add to `LOSS_REGISTRY`
6. Add accepted params to `ACCEPTED_PARAMS` in `get_loss_strategy()`

---

*Source: `src/training/losses.py`*
