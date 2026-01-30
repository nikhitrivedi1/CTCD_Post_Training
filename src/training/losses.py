"""Loss functions for knowledge distillation using Strategy Pattern."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

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
    """

    @abstractmethod
    def compute(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        ce_loss: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> LossOutput:
        """
        Compute the distillation loss.

        Args:
            student_logits: Raw output logits from student model
            teacher_logits: Raw output logits from teacher model
            ce_loss: Pre-computed cross-entropy loss from student
            labels: Ground truth labels (optional, some strategies may use)

        Returns:
            LossOutput with total loss and component breakdown
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this loss strategy."""
        pass


class KLDivergenceLoss(DistillationLossStrategy):
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
    ) -> LossOutput:
        # Compute KL divergence with temperature scaling
        kl_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # Combined loss
        total_loss = (1 - self.alpha) * ce_loss + self.alpha * kl_loss

        return LossOutput(
            total_loss=total_loss,
            components={
                "loss_ce": ce_loss,
                "loss_kl": kl_loss,
            }
        )


class MSELogitsLoss(DistillationLossStrategy):
    """
    Mean Squared Error loss between logits.

    Loss = (1 - alpha) * CE + alpha * MSE(student_logits, teacher_logits)

    Simpler than KL divergence, directly minimizes L2 distance between
    raw logits. Can be more stable in some cases.
    """

    def __init__(self, alpha: float = 0.5, normalize: bool = True):
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
    ) -> LossOutput:
        if self.normalize:
            # Normalize logits to have unit variance (helps with scale differences)
            student_norm = (student_logits - student_logits.mean()) / (student_logits.std() + 1e-8)
            teacher_norm = (teacher_logits - teacher_logits.mean()) / (teacher_logits.std() + 1e-8)
            mse_loss = F.mse_loss(student_norm, teacher_norm)
        else:
            mse_loss = F.mse_loss(student_logits, teacher_logits)

        total_loss = (1 - self.alpha) * ce_loss + self.alpha * mse_loss

        return LossOutput(
            total_loss=total_loss,
            components={
                "loss_ce": ce_loss,
                "loss_mse": mse_loss,
            }
        )


class CosineEmbeddingLoss(DistillationLossStrategy):
    """
    Cosine similarity loss between logits/representations.

    Loss = (1 - alpha) * CE + alpha * (1 - cosine_sim(student, teacher))

    Focuses on matching the direction of the output vectors rather than
    their magnitude. Useful when scale differences are problematic.
    """

    def __init__(self, alpha: float = 0.5):
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
    ) -> LossOutput:
        # Compute cosine similarity (1 = identical, -1 = opposite)
        cosine_sim = F.cosine_similarity(student_logits, teacher_logits, dim=-1)
        # Convert to loss (0 = identical, 2 = opposite)
        cosine_loss = (1 - cosine_sim).mean()

        total_loss = (1 - self.alpha) * ce_loss + self.alpha * cosine_loss

        return LossOutput(
            total_loss=total_loss,
            components={
                "loss_ce": ce_loss,
                "loss_cosine": cosine_loss,
            }
        )


class SoftTargetLoss(DistillationLossStrategy):
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
    ) -> LossOutput:
        # Soft labels from teacher
        soft_labels = F.softmax(teacher_logits / self.temperature, dim=-1)

        # Cross-entropy with soft labels
        soft_ce_loss = -(soft_labels * F.log_softmax(student_logits / self.temperature, dim=-1)).sum(dim=-1).mean()
        soft_ce_loss = soft_ce_loss * (self.temperature ** 2)

        total_loss = (1 - self.alpha) * ce_loss + self.alpha * soft_ce_loss

        return LossOutput(
            total_loss=total_loss,
            components={
                "loss_ce_hard": ce_loss,
                "loss_ce_soft": soft_ce_loss,
            }
        )


class CombinedLoss(DistillationLossStrategy):
    """
    Combines multiple loss strategies with configurable weights.

    Useful for experimentation: e.g., KL + MSE + Cosine together.
    """

    def __init__(self, strategies: list, weights: list, alpha: float = 0.5):
        assert len(strategies) == len(weights), "Must have same number of strategies and weights"
        self.strategies = strategies
        self.weights = weights
        self.alpha = alpha
        # Normalize weights
        total = sum(weights)
        self.weights = [w / total for w in weights]

    @property
    def name(self) -> str:
        return "combined_" + "_".join(s.name for s in self.strategies)

    def compute(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        ce_loss: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> LossOutput:
        all_components = {"loss_ce": ce_loss}
        distill_loss = torch.tensor(0.0, device=student_logits.device)

        for strategy, weight in zip(self.strategies, self.weights):
            output = strategy.compute(student_logits, teacher_logits, ce_loss, labels)
            # Extract the distillation component (total - ce contribution)
            strategy_distill = output.total_loss - (1 - strategy.alpha) * ce_loss
            distill_loss = distill_loss + weight * strategy_distill

            # Collect components with prefixed names
            for k, v in output.components.items():
                if k != "loss_ce":
                    all_components[f"{strategy.name}_{k}"] = v

        total_loss = (1 - self.alpha) * ce_loss + self.alpha * distill_loss

        return LossOutput(
            total_loss=total_loss,
            components=all_components
        )


# Registry for easy lookup by name
LOSS_REGISTRY = {
    "kl_divergence": KLDivergenceLoss,
    "mse_logits": MSELogitsLoss,
    "cosine_embedding": CosineEmbeddingLoss,
    "soft_target": SoftTargetLoss,
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
