"""Loss functions for knowledge distillation using Strategy Pattern."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def compute_kl_divergence(
    student: torch.Tensor,
    teacher: torch.Tensor,
    temperature: float,
    input_is_probs: bool = False,
) -> torch.Tensor:
    """
    Compute KL divergence with temperature scaling (Hinton 2015).

    KL(teacher || student) - student learns to match teacher.

    Args:
        student: Student logits or probabilities
        teacher: Teacher logits or probabilities
        temperature: Temperature for softening distributions
        input_is_probs: If True, inputs are probabilities (e.g., attention weights)
                       and need log conversion before temperature scaling

    Returns:
        KL divergence scaled by T² to maintain gradient magnitude
    """
    # Compute in fp32 to avoid underflow with high temperature (e.g., T=15)
    # fp16 softmax(logits/T) can underflow when T is large
    original_dtype = student.dtype
    student = student.float()
    teacher = teacher.float()

    if input_is_probs:
        # Convert probabilities to logits first
        eps = 1e-10
        student = torch.log(student + eps)
        teacher = torch.log(teacher + eps)

    student_log = F.log_softmax(student / temperature, dim=-1)
    teacher_soft = F.softmax(teacher / temperature, dim=-1)

    kl = F.kl_div(
        student_log,
        teacher_soft,
        reduction='batchmean'
    ) * (temperature ** 2)

    return kl.to(original_dtype)


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
        return "response_kl"

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
        kl_raw = compute_kl_divergence(
            student_logits, teacher_logits, self.temperature, input_is_probs=False
        )

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
        return "response_mse"

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
        return "response_cosine"

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
        return "response_ce"

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
    or Feature + Attention + Response for FAR matrix experiments.
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

            # Collect components (exclude ce_raw/ce_weighted/alpha - already tracked)
            for k, v in output.components.items():
                if k not in ("ce_raw", "ce_weighted", "alpha"):
                    all_components[k] = v

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

            # Ensure same dtype (projector may output fp32 while t_hidden is fp16)
            if s_hidden.dtype != t_hidden.dtype:
                t_hidden = t_hidden.to(s_hidden.dtype)

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


class CosineFeatureLoss(DistillationLossStrategy):
    """
    Feature-based distillation loss using cosine similarity (DistilBERT-style).

    Loss = (1 - alpha) * CE + alpha * mean(1 - cosine_sim(student_hidden, teacher_hidden))

    Unlike MSE, cosine similarity focuses on the direction of hidden state
    vectors rather than their magnitude. DistilBERT found this beneficial
    (+1.46 GLUE points when included).

    References:
    - Sanh et al. (2019): "DistilBERT, a distilled version of BERT"
    """

    requires_hidden_states = True

    def __init__(
        self,
        alpha: float = 0.5,
        layer_mapping: Optional[List[Tuple[int, int]]] = None,
        projector: Optional[Any] = None,
        **kwargs,
    ):
        self.alpha = alpha
        self.layer_mapping = layer_mapping or []
        self.projector = projector

    @property
    def name(self) -> str:
        return "feature_cosine"

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
        layer_mapping = kwargs.get("layer_mapping", self.layer_mapping)
        projector = kwargs.get("projector", self.projector)

        student_hidden = getattr(student_outputs, "hidden_states", None)
        teacher_hidden = getattr(teacher_outputs, "hidden_states", None)

        if student_hidden is None or teacher_hidden is None:
            raise ValueError(
                "Feature loss requires hidden_states. "
                "Ensure output_hidden_states=True in model config."
            )

        cosine_losses = []
        for teacher_idx, student_idx in layer_mapping:
            t_hidden = teacher_hidden[teacher_idx]
            s_hidden = student_hidden[student_idx]

            if projector is not None:
                s_hidden = projector(s_hidden)

            # Ensure same dtype (projector may output fp32 while t_hidden is fp16)
            if s_hidden.dtype != t_hidden.dtype:
                t_hidden = t_hidden.to(s_hidden.dtype)

            # Cosine similarity: 1 - cos(s, t) as loss (0 = identical direction)
            cos_sim = F.cosine_similarity(s_hidden, t_hidden, dim=-1)
            layer_loss = (1 - cos_sim).mean()
            cosine_losses.append(layer_loss)

        if cosine_losses:
            feature_raw = torch.stack(cosine_losses).mean()
        else:
            feature_raw = torch.tensor(0.0, device=student_logits.device)

        ce_weighted = (1 - self.alpha) * ce_loss
        feature_weighted = self.alpha * feature_raw
        total_loss = ce_weighted + feature_weighted

        return LossOutput(
            total_loss=total_loss,
            components={
                "ce_weighted": ce_weighted,
                "feature_cosine_weighted": feature_weighted,
                "alpha": torch.tensor(self.alpha),
                "ce_raw": ce_loss,
                "feature_cosine_raw": feature_raw,
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

            # Flatten attention matrices for KL computation: (batch, heads, seq, seq) -> (N, seq)
            s_flat = s_attn.view(-1, s_attn.size(-1))
            t_flat = t_attn.view(-1, t_attn.size(-1))

            # Compute KL divergence (attention weights are probabilities)
            kl_loss = compute_kl_divergence(
                s_flat, t_flat, self.temperature, input_is_probs=True
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


class StagedDistillationLoss(DistillationLossStrategy):
    """
    Staged distillation loss that switches strategies over training.

    Enables curriculum-style distillation: e.g., Feature loss early,
    Response loss late. Motivated by DistilBERT finding that F helps
    during pre-training but our finding that R-only wins during fine-tuning.

    Example config:
        stages:
          - strategy: feature_mse
            until_progress: 0.5  # First 50% of training
          - strategy: kl_divergence
            until_progress: 1.0  # Remainder
    """

    def __init__(
        self,
        stages: List[Dict[str, Any]],
        alpha: float = 0.5,
        temperature: float = 2.0,
        **kwargs,
    ):
        """
        Initialize staged distillation loss.

        Args:
            stages: List of stage configs, each with:
                - strategy: name of loss strategy (or instance)
                - until_progress: float 0-1, training progress threshold
            alpha: Default alpha for strategies that need it
            temperature: Default temperature for strategies that need it
            **kwargs: Additional args passed to sub-strategies
        """
        self.stages = []
        self.alpha = alpha
        self.temperature = temperature

        # Build strategy instances for each stage
        for stage_config in stages:
            strategy_name = stage_config["strategy"]
            until_progress = stage_config.get("until_progress", 1.0)

            # Get strategy-specific overrides or use defaults
            stage_alpha = stage_config.get("alpha", alpha)
            stage_temp = stage_config.get("temperature", temperature)

            # Create strategy instance
            if isinstance(strategy_name, str):
                strategy = get_loss_strategy(
                    strategy_name,
                    alpha=stage_alpha,
                    temperature=stage_temp,
                    **kwargs,
                )
            else:
                strategy = strategy_name  # Already an instance

            self.stages.append({
                "strategy": strategy,
                "until_progress": until_progress,
            })

        # Sort stages by progress threshold
        self.stages.sort(key=lambda s: s["until_progress"])

        # Aggregate requirement flags from all sub-strategies
        self.requires_hidden_states = any(
            s["strategy"].requires_hidden_states for s in self.stages
        )
        self.requires_attentions = any(
            s["strategy"].requires_attentions for s in self.stages
        )

    @property
    def name(self) -> str:
        stage_names = [s["strategy"].name for s in self.stages]
        return "staged_" + "_then_".join(stage_names)

    def get_current_strategy(self, progress: float) -> DistillationLossStrategy:
        """Get the active strategy for current training progress."""
        for stage in self.stages:
            if progress <= stage["until_progress"]:
                return stage["strategy"]
        # Default to last stage if progress exceeds all thresholds
        return self.stages[-1]["strategy"]

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
        # Get training progress from kwargs (passed by trainer)
        progress = kwargs.get("training_progress", 0.0)

        # Select active strategy based on progress
        active_strategy = self.get_current_strategy(progress)

        # Delegate to active strategy
        output = active_strategy.compute(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            ce_loss=ce_loss,
            labels=labels,
            student_outputs=student_outputs,
            teacher_outputs=teacher_outputs,
            **kwargs,
        )

        # Add stage info to components for logging
        output.components["active_stage"] = torch.tensor(
            self.stages.index(
                next(s for s in self.stages if s["strategy"] == active_strategy)
            )
        )
        output.components["training_progress"] = torch.tensor(progress)

        return output


# Registry for easy lookup by name
# Naming convention: {signal}_{method}
#   signal: response, feature, attention
#   method: kl, mse, cosine, ce
LOSS_REGISTRY = {
    # Response-based (output logits)
    "response_kl": KLDistillationLoss,
    "response_mse": MSEDistillationLoss,
    "response_cosine": CosineDistillationLoss,
    "response_ce": SoftTargetDistillationLoss,

    # Feature-based (hidden states)
    "feature_mse": FeatureLossStrategy,
    "feature_cosine": CosineFeatureLoss,

    # Attention-based (attention weights)
    "attention_kl": AttentionLossStrategy,

    # Meta/compound strategies
    "staged": StagedDistillationLoss,

    # Backward compatibility aliases (deprecated)
    "kl_divergence": KLDistillationLoss,
    "mse_logits": MSEDistillationLoss,
    "cosine_embedding": CosineDistillationLoss,
    "soft_target": SoftTargetDistillationLoss,
}


def get_loss_strategy(name: str, **kwargs) -> DistillationLossStrategy:
    """
    Factory function to get a loss strategy by name.

    Args:
        name: Name of the loss strategy (format: {signal}_{method})
        **kwargs: Arguments to pass to the strategy constructor

    Returns:
        Instantiated loss strategy

    Example:
        loss = get_loss_strategy("response_kl", temperature=2.0, alpha=0.5)
        loss = get_loss_strategy("feature_cosine", alpha=0.3)
    """
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss strategy: {name}. Available: {list(LOSS_REGISTRY.keys())}")
    return LOSS_REGISTRY[name](**kwargs)
