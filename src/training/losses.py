"""Loss functions for knowledge distillation using Strategy Pattern."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class LossOutput:
    """Container for loss computation results."""
    total_loss: torch.Tensor
    components: Dict[str, torch.Tensor]


class DistillationLossStrategy(ABC):
    """
    Abstract base class for distillation loss strategies.

    Implement this interface to create new loss functions that can be
    swapped in/out of the trainer without modifying the Trainer class.

    Subclasses should set class attributes:
    - requires_hidden_states: True if strategy needs model hidden states
    - requires_attentions: True if strategy needs attention weights
    """

    # Override in subclasses that need intermediate outputs
    requires_hidden_states: bool = False
    requires_attentions: bool = False

    @abstractmethod
    def compute(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        ce_loss: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        student_outputs: Optional[Any] = None,
        teacher_outputs: Optional[Any] = None,
        **kwargs,
    ) -> LossOutput:
        """
        Compute the distillation loss.

        Args:
            student_logits: Raw output logits from student model
            teacher_logits: Raw output logits from teacher model
            ce_loss: Pre-computed cross-entropy loss from student
            labels: Ground truth labels (optional, some strategies may use)
            student_outputs: Full model outputs (for hidden states/attention)
            teacher_outputs: Full model outputs (for hidden states/attention)
            **kwargs: Additional arguments (layer_mapper, projector, etc.)

        Returns:
            LossOutput with total loss and component breakdown
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this loss strategy."""
        pass


class KLDistillationLoss(DistillationLossStrategy):
    """
    Standard KL Divergence loss from Hinton et al. (2015).

    Loss = (1 - alpha) * CE + alpha * T^2 * KL(softmax(s/T) || softmax(t/T))

    This is the classic distillation loss that transfers "dark knowledge"
    from teacher to student by matching softened probability distributions.
    """

    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        self.temperature = temperature
        self.alpha = alpha

    @property
    def name(self) -> str:
        return "kl_divergence"

    def compute(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        ce_loss: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        student_outputs: Optional[Any] = None,
        teacher_outputs: Optional[Any] = None,
        **kwargs,
    ) -> LossOutput:
        # Always compute KL divergence for monitoring (even when alpha=0)
        kl_raw = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # Weighted contributions
        ce_weighted = (1 - self.alpha) * ce_loss
        kl_weighted = self.alpha * kl_raw

        # Combined loss
        total_loss = ce_weighted + kl_weighted

        return LossOutput(
            total_loss=total_loss,
            components={
                "ce_weighted": ce_weighted,
                "kl_weighted": kl_weighted,
                "alpha": torch.tensor(self.alpha),
                "ce_raw": ce_loss,
                "kl_raw": kl_raw,
            }
        )


class MSEDistillationLoss(DistillationLossStrategy):
    """
    Mean Squared Error loss between logits.

    Loss = (1 - alpha) * CE + alpha * MSE(student_logits, teacher_logits)

    Simpler than KL divergence, directly minimizes L2 distance between
    raw logits. Can be more stable in some cases.
    """

    def __init__(self, alpha: float = 0.5, normalize: bool = True, **kwargs):
        # kwargs captures unused parameters like temperature for compatibility
        self.alpha = alpha
        self.normalize = normalize

    @property
    def name(self) -> str:
        return "mse_logits"

    def compute(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        ce_loss: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        student_outputs: Optional[Any] = None,
        teacher_outputs: Optional[Any] = None,
        **kwargs,
    ) -> LossOutput:
        # Always compute MSE for monitoring (even when alpha=0)
        if self.normalize:
            student_norm = (student_logits - student_logits.mean()) / (student_logits.std() + 1e-8)
            teacher_norm = (teacher_logits - teacher_logits.mean()) / (teacher_logits.std() + 1e-8)
            mse_raw = F.mse_loss(student_norm, teacher_norm)
        else:
            mse_raw = F.mse_loss(student_logits, teacher_logits)

        # Weighted contributions
        ce_weighted = (1 - self.alpha) * ce_loss
        mse_weighted = self.alpha * mse_raw
        total_loss = ce_weighted + mse_weighted

        return LossOutput(
            total_loss=total_loss,
            components={
                "ce_weighted": ce_weighted,
                "mse_weighted": mse_weighted,
                "alpha": torch.tensor(self.alpha),
                "ce_raw": ce_loss,
                "mse_raw": mse_raw,
            }
        )


class CosineDistillationLoss(DistillationLossStrategy):
    """
    Cosine similarity loss between logits/representations.

    Loss = (1 - alpha) * CE + alpha * (1 - cosine_sim(student, teacher))

    Focuses on matching the direction of the output vectors rather than
    their magnitude. Useful when scale differences are problematic.
    """

    def __init__(self, alpha: float = 0.5, **kwargs):
        # kwargs captures unused parameters like temperature for compatibility
        self.alpha = alpha

    @property
    def name(self) -> str:
        return "cosine_embedding"

    def compute(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        ce_loss: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        student_outputs: Optional[Any] = None,
        teacher_outputs: Optional[Any] = None,
        **kwargs,
    ) -> LossOutput:
        # Always compute cosine for monitoring (even when alpha=0)
        cosine_sim = F.cosine_similarity(student_logits, teacher_logits, dim=-1)
        cosine_raw = (1 - cosine_sim).mean()

        # Weighted contributions
        ce_weighted = (1 - self.alpha) * ce_loss
        cosine_weighted = self.alpha * cosine_raw
        total_loss = ce_weighted + cosine_weighted

        return LossOutput(
            total_loss=total_loss,
            components={
                "ce_weighted": ce_weighted,
                "cosine_weighted": cosine_weighted,
                "alpha": torch.tensor(self.alpha),
                "ce_raw": ce_loss,
                "cosine_raw": cosine_raw,
            }
        )


class SoftTargetDistillationLoss(DistillationLossStrategy):
    """
    Soft target cross-entropy loss.

    Loss = (1 - alpha) * CE_hard + alpha * CE_soft

    Uses cross-entropy with soft labels (teacher probabilities) instead
    of KL divergence. Mathematically related but can behave differently.
    """

    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        self.temperature = temperature
        self.alpha = alpha

    @property
    def name(self) -> str:
        return "soft_target"

    def compute(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        ce_loss: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        student_outputs: Optional[Any] = None,
        teacher_outputs: Optional[Any] = None,
        **kwargs,
    ) -> LossOutput:
        # Always compute soft CE for monitoring (even when alpha=0)
        soft_labels = F.softmax(teacher_logits / self.temperature, dim=-1)
        ce_soft_raw = -(soft_labels * F.log_softmax(student_logits / self.temperature, dim=-1)).sum(dim=-1).mean()
        ce_soft_raw = ce_soft_raw * (self.temperature ** 2)

        # Weighted contributions
        ce_hard_weighted = (1 - self.alpha) * ce_loss
        ce_soft_weighted = self.alpha * ce_soft_raw
        total_loss = ce_hard_weighted + ce_soft_weighted

        return LossOutput(
            total_loss=total_loss,
            components={
                "ce_hard_weighted": ce_hard_weighted,
                "ce_soft_weighted": ce_soft_weighted,
                "alpha": torch.tensor(self.alpha),
                "ce_hard_raw": ce_loss,
                "ce_soft_raw": ce_soft_raw,
            }
        )


class CombinedDistillationLoss(DistillationLossStrategy):
    """
    Combines multiple loss strategies with configurable weights.

    Useful for experimentation: e.g., KL + MSE + Cosine together,
    or Response + Feature + Attention for RFA matrix experiments.
    """

    def __init__(self, strategies: list, weights: list, alpha: float = 0.5):
        assert len(strategies) == len(weights), "Must have same number of strategies and weights"
        self.strategies = strategies
        self.weights = weights
        self.alpha = alpha
        # Normalize weights
        total = sum(weights)
        self.weights = [w / total for w in weights]

        # Aggregate requirement flags from all sub-strategies
        self.requires_hidden_states = any(s.requires_hidden_states for s in strategies)
        self.requires_attentions = any(s.requires_attentions for s in strategies)

    @property
    def name(self) -> str:
        return "combined_" + "_".join(s.name for s in self.strategies)

    def compute(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        ce_loss: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        student_outputs: Optional[Any] = None,
        teacher_outputs: Optional[Any] = None,
        **kwargs,
    ) -> LossOutput:
        # Weighted CE contribution
        ce_weighted = (1 - self.alpha) * ce_loss
        all_components = {
            "ce_weighted": ce_weighted,
            # distill_weighted components added by loop
            "alpha": torch.tensor(self.alpha),
            "ce_raw": ce_loss,
            # distill_raw components added by loop
        }
        distill_loss = torch.tensor(0.0, device=student_logits.device)

        for strategy, weight in zip(self.strategies, self.weights):
            # Pass full outputs and kwargs to sub-strategies (needed for Feature/Attention)
            output = strategy.compute(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                ce_loss=ce_loss,
                labels=labels,
                student_outputs=student_outputs,
                teacher_outputs=teacher_outputs,
                **kwargs,
            )
            # Extract the distillation component (total - ce contribution)
            strategy_distill = output.total_loss - (1 - strategy.alpha) * ce_loss
            distill_loss = distill_loss + weight * strategy_distill

            # Collect components with prefixed names (exclude ce_raw/ce_weighted/alpha)
            for k, v in output.components.items():
                if k not in ("ce_raw", "ce_weighted", "alpha"):
                    all_components[f"{strategy.name}_{k}"] = v

        total_loss = ce_weighted + self.alpha * distill_loss

        return LossOutput(
            total_loss=total_loss,
            components=all_components
        )


class FeatureLossStrategy(DistillationLossStrategy):
    """
    Feature-based distillation loss (MSE on hidden states).

    Loss = (1 - alpha) * CE + alpha * mean(MSE(proj(student_hidden), teacher_hidden))

    Transfers knowledge by matching intermediate representations between
    teacher and student. Requires layer mapping to handle different depths.

    References:
    - Romero et al. (2015): "FitNets: Hints for Thin Deep Nets"
    """

    requires_hidden_states = True

    def __init__(
        self,
        alpha: float = 0.5,
        layer_mapping: Optional[List[Tuple[int, int]]] = None,
        projector: Optional[Any] = None,
        normalize: bool = True,
        **kwargs,
    ):
        """
        Initialize feature distillation loss.

        Args:
            alpha: Weight for distillation loss vs CE loss
            layer_mapping: List of (teacher_idx, student_idx) pairs
            projector: Optional projection module for dimension alignment
            normalize: Whether to normalize hidden states before comparison
            **kwargs: Unused, for compatibility
        """
        self.alpha = alpha
        self.layer_mapping = layer_mapping or []
        self.projector = projector
        self.normalize = normalize

    @property
    def name(self) -> str:
        return "feature_mse"

    def compute(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        ce_loss: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        student_outputs: Optional[Any] = None,
        teacher_outputs: Optional[Any] = None,
        **kwargs,
    ) -> LossOutput:
        # Get layer mapping from kwargs if not set in init
        layer_mapping = kwargs.get("layer_mapping", self.layer_mapping)
        projector = kwargs.get("projector", self.projector)

        # Extract hidden states
        student_hidden = getattr(student_outputs, "hidden_states", None)
        teacher_hidden = getattr(teacher_outputs, "hidden_states", None)

        if student_hidden is None or teacher_hidden is None:
            raise ValueError(
                "Feature loss requires hidden_states. "
                "Ensure output_hidden_states=True in model config."
            )

        # Compute feature MSE for each mapped layer pair
        feature_losses = []
        for teacher_idx, student_idx in layer_mapping:
            t_hidden = teacher_hidden[teacher_idx]  # (batch, seq, hidden)
            s_hidden = student_hidden[student_idx]

            # Project student hidden if dimensions don't match
            if projector is not None:
                s_hidden = projector(s_hidden)

            # Normalize if requested
            if self.normalize:
                t_hidden = (t_hidden - t_hidden.mean()) / (t_hidden.std() + 1e-8)
                s_hidden = (s_hidden - s_hidden.mean()) / (s_hidden.std() + 1e-8)

            # MSE loss for this layer pair
            layer_loss = F.mse_loss(s_hidden, t_hidden)
            feature_losses.append(layer_loss)

        # Average across all layer pairs
        if feature_losses:
            feature_raw = torch.stack(feature_losses).mean()
        else:
            feature_raw = torch.tensor(0.0, device=student_logits.device)

        # Weighted contributions
        ce_weighted = (1 - self.alpha) * ce_loss
        feature_weighted = self.alpha * feature_raw
        total_loss = ce_weighted + feature_weighted

        return LossOutput(
            total_loss=total_loss,
            components={
                "ce_weighted": ce_weighted,
                "feature_weighted": feature_weighted,
                "alpha": torch.tensor(self.alpha),
                "ce_raw": ce_loss,
                "feature_raw": feature_raw,
                "num_layers_matched": torch.tensor(len(layer_mapping)),
            }
        )


class AttentionLossStrategy(DistillationLossStrategy):
    """
    Attention-based distillation loss (KL on attention distributions).

    Loss = (1 - alpha) * CE + alpha * mean(KL(student_attn || teacher_attn))

    Transfers knowledge by matching attention patterns between teacher
    and student. Helps student learn where to "look" in the input.

    References:
    - Zagoruyko & Komodakis (2017): "Paying More Attention to Attention"
    """

    requires_attentions = True

    def __init__(
        self,
        alpha: float = 0.5,
        layer_mapping: Optional[List[Tuple[int, int]]] = None,
        temperature: float = 1.0,
        **kwargs,
    ):
        """
        Initialize attention distillation loss.

        Args:
            alpha: Weight for distillation loss vs CE loss
            layer_mapping: List of (teacher_idx, student_idx) pairs
            temperature: Temperature for softening attention distributions
            **kwargs: Unused, for compatibility
        """
        self.alpha = alpha
        self.layer_mapping = layer_mapping or []
        self.temperature = temperature

    @property
    def name(self) -> str:
        return "attention_kl"

    def compute(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        ce_loss: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        student_outputs: Optional[Any] = None,
        teacher_outputs: Optional[Any] = None,
        **kwargs,
    ) -> LossOutput:
        # Get layer mapping from kwargs if not set in init
        layer_mapping = kwargs.get("layer_mapping", self.layer_mapping)

        # Extract attention weights
        student_attns = getattr(student_outputs, "attentions", None)
        teacher_attns = getattr(teacher_outputs, "attentions", None)

        if student_attns is None or teacher_attns is None:
            raise ValueError(
                "Attention loss requires attentions. "
                "Ensure output_attentions=True in model config."
            )

        # Compute attention KL for each mapped layer pair
        attention_losses = []
        for teacher_idx, student_idx in layer_mapping:
            t_attn = teacher_attns[teacher_idx]  # (batch, heads, seq, seq)
            s_attn = student_attns[student_idx]

            # Handle different number of attention heads by averaging
            if t_attn.shape[1] != s_attn.shape[1]:
                # Average teacher heads to match student head count
                # or expand student heads - for now, average both to single
                t_attn = t_attn.mean(dim=1, keepdim=True)
                s_attn = s_attn.mean(dim=1, keepdim=True)

            # Apply temperature and compute KL divergence
            # Attention weights are already normalized (softmax), but we can
            # re-soften them with temperature for better transfer
            t_attn_soft = F.softmax(t_attn / self.temperature, dim=-1)
            s_attn_log = F.log_softmax(s_attn / self.temperature, dim=-1)

            # KL divergence: KL(P || Q) where we want student to match teacher
            # Flatten for KL computation
            kl_loss = F.kl_div(
                s_attn_log.view(-1, s_attn_log.size(-1)),
                t_attn_soft.view(-1, t_attn_soft.size(-1)),
                reduction='batchmean'
            )
            attention_losses.append(kl_loss)

        # Average across all layer pairs
        if attention_losses:
            attention_raw = torch.stack(attention_losses).mean()
        else:
            attention_raw = torch.tensor(0.0, device=student_logits.device)

        # Weighted contributions
        ce_weighted = (1 - self.alpha) * ce_loss
        attention_weighted = self.alpha * attention_raw
        total_loss = ce_weighted + attention_weighted

        return LossOutput(
            total_loss=total_loss,
            components={
                "ce_weighted": ce_weighted,
                "attention_weighted": attention_weighted,
                "alpha": torch.tensor(self.alpha),
                "ce_raw": ce_loss,
                "attention_raw": attention_raw,
                "num_layers_matched": torch.tensor(len(layer_mapping)),
            }
        )


# Registry for easy lookup by name
LOSS_REGISTRY = {
    "kl_divergence": KLDistillationLoss,
    "mse_logits": MSEDistillationLoss,
    "cosine_embedding": CosineDistillationLoss,
    "soft_target": SoftTargetDistillationLoss,
    "feature_mse": FeatureLossStrategy,
    "attention_kl": AttentionLossStrategy,
}


def get_loss_strategy(name: str, **kwargs) -> DistillationLossStrategy:
    """
    Factory function to get a loss strategy by name.

    Args:
        name: Name of the loss strategy
        **kwargs: Arguments to pass to the strategy constructor

    Returns:
        Instantiated loss strategy

    Example:
        loss = get_loss_strategy("kl_divergence", temperature=2.0, alpha=0.5)
    """
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss strategy: {name}. Available: {list(LOSS_REGISTRY.keys())}")
    return LOSS_REGISTRY[name](**kwargs)
