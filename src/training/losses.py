"""Loss functions for knowledge distillation using Strategy Pattern."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def _resolve_ce_kd_coefficients(
    ce_loss_coeff: Optional[float] = None,
    kd_loss_coeff: Optional[float] = None,
    *,
    alpha_h: Optional[float] = None,
    alpha_s: Optional[float] = None,
) -> Tuple[float, float]:
    """Resolve CE vs KD weights from explicit coefficients or caller-facing ``alpha_h`` / ``alpha_s``.

    Precedence: if either ``ce_loss_coeff`` or ``kd_loss_coeff`` is set, infer the pair from those;
    else if either ``alpha_h`` or ``alpha_s`` is set, infer the pair (same semantics: hard vs soft);
    else default to 0.5 / 0.5.

    Loss strategies store only ``ce_loss_coeff`` and ``kd_loss_coeff``; this helper is the single
    bridge from config/trainer naming.
    """
    if ce_loss_coeff is not None or kd_loss_coeff is not None:
        kd = float(kd_loss_coeff) if kd_loss_coeff is not None else 1.0 - float(ce_loss_coeff)
        ce = float(ce_loss_coeff) if ce_loss_coeff is not None else 1.0 - float(kd)
        return ce, kd
    if alpha_h is not None or alpha_s is not None:
        kd = float(alpha_s) if alpha_s is not None else 1.0 - float(alpha_h)
        ce = float(alpha_h) if alpha_h is not None else 1.0 - float(alpha_s)
        return ce, kd
    return 0.5, 0.5


def _merge_coeff_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """If kwargs contain alpha_h/alpha_s or ce/kd coeffs, normalize to ce_loss_coeff/kd_loss_coeff."""
    d = dict(kwargs)
    coeff_keys = ("alpha_h", "alpha_s", "ce_loss_coeff", "kd_loss_coeff")
    if not any(k in d for k in coeff_keys):
        return d
    ce = d.pop("ce_loss_coeff", None)
    kd = d.pop("kd_loss_coeff", None)
    ah = d.pop("alpha_h", None)
    ass = d.pop("alpha_s", None)
    ce2, kd2 = _resolve_ce_kd_coefficients(ce, kd, alpha_h=ah, alpha_s=ass)
    d["ce_loss_coeff"] = ce2
    d["kd_loss_coeff"] = kd2
    return d


def compute_kl_divergence(
    model: torch.Tensor,
    teacher: torch.Tensor,
    temperature: float,
    input_is_probs: bool = False,
    reverse: bool = False
) -> torch.Tensor:
    """
    Compute KL divergence with temperature scaling (Hinton 2015).

    When reverse=False (default): KL(teacher || model) - model learns to match teacher (Hinton et al. 2015).
    When reverse=True: KL(model || teacher) - reverse KL for mode-seeking behavior (Gu et al., MiniLLM, ICLR 2024).

    Args:
        model: Model logits or probabilities
        teacher: Teacher logits or probabilities
        temperature: Temperature for softening distributions
        input_is_probs: If True, inputs are probabilities (e.g., attention weights)
                       and need log conversion before temperature scaling
        reverse: If True, compute reverse KL(model || teacher) instead of KL(teacher || model).

    Returns:
        KL divergence scaled by T² to maintain gradient magnitude
    """
    # Compute in fp32 to avoid underflow with high temperature (e.g., T=15)
    # fp16 softmax(logits/T) can underflow when T is large
    original_dtype = model.dtype

    if reverse:
        input, target = teacher.float(), model.float()
    else:
        input, target = model.float(), teacher.float()

    if input_is_probs:
        # Convert probabilities to logits first
        eps = 1e-10
        input = torch.log(input + eps)
        target = torch.log(target + eps)

    input_log = F.log_softmax(input / temperature, dim=-1)
    target_soft = F.softmax(target / temperature, dim=-1)
    kl = F.kl_div(
        input_log,
        target_soft,
        reduction='batchmean'
    ) * (temperature ** 2)

    return kl.to(original_dtype)


def compute_entropy_of_output(logits: torch.Tensor) -> torch.Tensor:
    """Batch-mean Shannon entropy of the output distribution induced by logits.

    Interprets the last dimension of ``logits`` as classes, applies softmax, and computes
    H(p) = -E_p[log p] in nats, averaged over all leading dimensions (e.g., batch).
    """
    original_dtype = logits.dtype
    logits_f = logits.float()
    log_probs = F.log_softmax(logits_f, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    return entropy.to(original_dtype)


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

    Co-training (unfrozen teacher): the trainer uses a separate ``teacher_loss_strategy`` (β
    coefficients baked in) with swapped logits in :meth:`compute`; the student branch still
    calls this class's :meth:`compute` with the usual argument roles.
    """

    # Override in subclasses that need intermediate outputs
    requires_hidden_states: bool = False
    requires_attentions: bool = False

    @abstractmethod
    def compute(
        self,
        model_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        ce_loss: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        model_outputs: Optional[Any] = None,
        teacher_outputs: Optional[Any] = None,
        **kwargs,
    ) -> LossOutput:
        """
        Compute the distillation loss.

        Args:
            model_logits: Raw output logits from model
            teacher_logits: Raw output logits from teacher model
            ce_loss: Pre-computed cross-entropy loss from model
            labels: Ground truth labels (optional, some strategies may use)
            model_outputs: Full model outputs (for hidden states/attention)
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

    Loss = ce_loss_coeff * CE + kd_loss_coeff * T^2 * KL(softmax(s/T) || softmax(t/T))

    Use ``get_loss_strategy(..., alpha_h=..., alpha_s=...)`` so callers stay on student/teacher
    naming; internally only CE vs KD coefficients are stored.
    """

    def __init__(
        self,
        temperature: float = 2.0,
        ce_loss_coeff: float = 0.5,
        kd_loss_coeff: float = 0.5,
        reverse: bool = False,
    ):
        self.temperature = temperature
        self.reverse = reverse
        self.ce_loss_coeff = float(ce_loss_coeff)
        self.kd_loss_coeff = float(kd_loss_coeff)

    @property
    def name(self) -> str:
        return "response_reverse-kl" if self.reverse else "response_kl"

    def compute(
        self,
        model_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        ce_loss: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        model_outputs: Optional[Any] = None,
        teacher_outputs: Optional[Any] = None,
        **kwargs,
    ) -> LossOutput:
        # Always compute KL divergence for monitoring (even when kd_loss_coeff=0)
        kl_raw = compute_kl_divergence(
            model_logits, teacher_logits, self.temperature, input_is_probs=False, reverse=self.reverse
        )

        ce_weighted = self.ce_loss_coeff * ce_loss
        kl_weighted = self.kd_loss_coeff * kl_raw
        total_loss = ce_weighted + kl_weighted

        raw_key = "reverse-kl_raw" if self.reverse else "kl_raw"
        weighted_key = "reverse-kl_weighted" if self.reverse else "kl_weighted"

        return LossOutput(
            total_loss=total_loss,
            components={
                "ce_weighted": ce_weighted,
                weighted_key: kl_weighted,
                "ce_loss_coeff": torch.tensor(self.ce_loss_coeff),
                "kd_loss_coeff": torch.tensor(self.kd_loss_coeff),
                "ce_raw": ce_loss,
                raw_key: kl_raw,
            }
        )


class MSEDistillationLoss(DistillationLossStrategy):
    """
    Mean Squared Error loss between logits.

    Loss = ce_loss_coeff * CE + kd_loss_coeff * MSE(model_logits, teacher_logits)
    """

    def __init__(
        self,
        ce_loss_coeff: float = 0.5,
        kd_loss_coeff: float = 0.5,
        normalize: bool = True,
    ):
        self.normalize = normalize
        self.ce_loss_coeff = float(ce_loss_coeff)
        self.kd_loss_coeff = float(kd_loss_coeff)

    @property
    def name(self) -> str:
        return "response_mse"

    def compute(
        self,
        model_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        ce_loss: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        model_outputs: Optional[Any] = None,
        teacher_outputs: Optional[Any] = None,
        **kwargs,
    ) -> LossOutput:
        # Always compute MSE for monitoring (even when kd_loss_coeff=0)
        if self.normalize:
            model_norm = (model_logits - model_logits.mean()) / (model_logits.std() + 1e-8)
            teacher_norm = (teacher_logits - teacher_logits.mean()) / (teacher_logits.std() + 1e-8)
            mse_raw = F.mse_loss(model_norm, teacher_norm)
        else:
            mse_raw = F.mse_loss(model_logits, teacher_logits)

        ce_weighted = self.ce_loss_coeff * ce_loss
        mse_weighted = self.kd_loss_coeff * mse_raw
        total_loss = ce_weighted + mse_weighted

        return LossOutput(
            total_loss=total_loss,
            components={
                "ce_weighted": ce_weighted,
                "mse_weighted": mse_weighted,
                "ce_loss_coeff": torch.tensor(self.ce_loss_coeff),
                "kd_loss_coeff": torch.tensor(self.kd_loss_coeff),
                "ce_raw": ce_loss,
                "mse_raw": mse_raw,
            }
        )


class CosineDistillationLoss(DistillationLossStrategy):
    """
    Cosine similarity loss between logits/representations.

    Loss = ce_loss_coeff * CE + kd_loss_coeff * (1 - cosine_sim(model, teacher))
    """

    def __init__(
        self,
        ce_loss_coeff: float = 0.5,
        kd_loss_coeff: float = 0.5,
    ):
        self.ce_loss_coeff = float(ce_loss_coeff)
        self.kd_loss_coeff = float(kd_loss_coeff)

    @property
    def name(self) -> str:
        return "response_cosine"

    def compute(
        self,
        model_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        ce_loss: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        model_outputs: Optional[Any] = None,
        teacher_outputs: Optional[Any] = None,
        **kwargs,
    ) -> LossOutput:
        # Always compute cosine for monitoring (even when kd_loss_coeff=0)
        cosine_sim = F.cosine_similarity(model_logits, teacher_logits, dim=-1)
        cosine_raw = (1 - cosine_sim).mean()

        ce_weighted = self.ce_loss_coeff * ce_loss
        cosine_weighted = self.kd_loss_coeff * cosine_raw
        total_loss = ce_weighted + cosine_weighted

        return LossOutput(
            total_loss=total_loss,
            components={
                "ce_weighted": ce_weighted,
                "cosine_weighted": cosine_weighted,
                "ce_loss_coeff": torch.tensor(self.ce_loss_coeff),
                "kd_loss_coeff": torch.tensor(self.kd_loss_coeff),
                "ce_raw": ce_loss,
                "cosine_raw": cosine_raw,
            }
        )


class SoftTargetDistillationLoss(DistillationLossStrategy):
    """
    Soft target cross-entropy loss.

    Loss = ce_loss_coeff * CE_hard + kd_loss_coeff * CE_soft
    """

    def __init__(
        self,
        temperature: float = 2.0,
        ce_loss_coeff: float = 0.5,
        kd_loss_coeff: float = 0.5,
    ):
        self.temperature = temperature
        self.ce_loss_coeff = float(ce_loss_coeff)
        self.kd_loss_coeff = float(kd_loss_coeff)

    @property
    def name(self) -> str:
        return "response_ce"

    def compute(
        self,
        model_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        ce_loss: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        model_outputs: Optional[Any] = None,
        teacher_outputs: Optional[Any] = None,
        **kwargs,
    ) -> LossOutput:
        # Always compute soft CE for monitoring (even when kd_loss_coeff=0)
        soft_labels = F.softmax(teacher_logits / self.temperature, dim=-1)
        ce_soft_raw = -(soft_labels * F.log_softmax(model_logits / self.temperature, dim=-1)).sum(dim=-1).mean()
        ce_soft_raw = ce_soft_raw * (self.temperature ** 2)

        ce_hard_weighted = self.ce_loss_coeff * ce_loss
        ce_soft_weighted = self.kd_loss_coeff * ce_soft_raw
        total_loss = ce_hard_weighted + ce_soft_weighted

        return LossOutput(
            total_loss=total_loss,
            components={
                "ce_hard_weighted": ce_hard_weighted,
                "ce_soft_weighted": ce_soft_weighted,
                "ce_loss_coeff": torch.tensor(self.ce_loss_coeff),
                "kd_loss_coeff": torch.tensor(self.kd_loss_coeff),
                "ce_hard_raw": ce_loss,
                "ce_soft_raw": ce_soft_raw,
            }
        )


class CombinedDistillationLoss(DistillationLossStrategy):
    """
    Combines multiple loss strategies with configurable weights.

    Useful for experimentation: e.g., KL + MSE + Cosine together,
    or Feature + Attention + Response for FAR matrix experiments.

    Outer mix: ``ce_loss_coeff * CE + kd_loss_coeff * weighted_distill``. Pass
    ``ce_loss_coeff`` / ``kd_loss_coeff`` or ``alpha_h`` / ``alpha_s``; see ``_resolve_ce_kd_coefficients``.
    """

    def __init__(
        self,
        strategies: list,
        weights: list,
        ce_loss_coeff: Optional[float] = None,
        kd_loss_coeff: Optional[float] = None,
        alpha_h: Optional[float] = None,
        alpha_s: Optional[float] = None,
    ):
        assert len(strategies) == len(weights), "Must have same number of strategies and weights"
        self.strategies = strategies
        self.weights = weights
        self.ce_loss_coeff, self.kd_loss_coeff = _resolve_ce_kd_coefficients(
            ce_loss_coeff, kd_loss_coeff, alpha_h=alpha_h, alpha_s=alpha_s,
        )
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
        model_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        ce_loss: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        model_outputs: Optional[Any] = None,
        teacher_outputs: Optional[Any] = None,
        **kwargs,
    ) -> LossOutput:
        # Weighted CE contribution
        ce_weighted = self.ce_loss_coeff * ce_loss
        all_components = {
            "ce_weighted": ce_weighted,
            # distill_weighted components added by loop
            "ce_loss_coeff": torch.tensor(self.ce_loss_coeff),
            "kd_loss_coeff": torch.tensor(self.kd_loss_coeff),
            "ce_raw": ce_loss,
            # distill_raw components added by loop
        }
        distill_loss = torch.tensor(0.0, device=model_logits.device)

        skip_component_keys = frozenset({
            "ce_raw", "ce_weighted", "ce_loss_coeff", "kd_loss_coeff",
        })

        for strategy, weight in zip(self.strategies, self.weights):
            # Pass full outputs and kwargs to sub-strategies (needed for Feature/Attention)
            output = strategy.compute(
                model_logits=model_logits,
                teacher_logits=teacher_logits,
                ce_loss=ce_loss,
                labels=labels,
                model_outputs=model_outputs,
                teacher_outputs=teacher_outputs,
                **kwargs,
            )
            strategy_distill = output.total_loss - strategy.ce_loss_coeff * ce_loss
            distill_loss = distill_loss + weight * strategy_distill

            # Collect components (exclude shared CE / coeff keys - already tracked)
            for k, v in output.components.items():
                if k not in skip_component_keys:
                    all_components[k] = v

        total_loss = ce_weighted + self.kd_loss_coeff * distill_loss

        return LossOutput(
            total_loss=total_loss,
            components=all_components
        )


class FeatureLossStrategy(DistillationLossStrategy):
    """
    Feature-based distillation loss (MSE on hidden states).

    Loss = ce_loss_coeff * CE + kd_loss_coeff * mean(MSE(proj(model_hidden), teacher_hidden))

    Transfers knowledge by matching intermediate representations between
    teacher and model. Requires layer mapping to handle different depths.

    References:
    - Romero et al. (2015): "FitNets: Hints for Thin Deep Nets"
    """

    requires_hidden_states = True

    def __init__(
        self,
        ce_loss_coeff: float = 0.5,
        kd_loss_coeff: float = 0.5,
        layer_mapping: Optional[List[Tuple[int, int]]] = None,
        projector: Optional[Any] = None,
        normalize: bool = True,
    ):
        """
        Initialize feature distillation loss.

        Args:
            ce_loss_coeff: Weight on the hard-label (CE) term
            kd_loss_coeff: Weight on the feature matching (distillation) term
            layer_mapping: List of (teacher_idx, model_idx) pairs
            projector: Optional projection module for dimension alignment
            normalize: Whether to normalize hidden states before comparison
        """
        self.ce_loss_coeff = float(ce_loss_coeff)
        self.kd_loss_coeff = float(kd_loss_coeff)
        self.layer_mapping = layer_mapping or []
        self.projector = projector
        self.normalize = normalize

    @property
    def name(self) -> str:
        return "feature_mse"

    def compute(
        self,
        model_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        ce_loss: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        model_outputs: Optional[Any] = None,
        teacher_outputs: Optional[Any] = None,
        **kwargs,
    ) -> LossOutput:
        # Get layer mapping from kwargs if not set in init
        layer_mapping = kwargs.get("layer_mapping", self.layer_mapping)
        projector = kwargs.get("projector", self.projector)

        # Extract hidden states
        model_hidden = getattr(model_outputs, "hidden_states", None)
        teacher_hidden = getattr(teacher_outputs, "hidden_states", None)

        if model_hidden is None or teacher_hidden is None:
            raise ValueError(
                "Feature loss requires hidden_states. "
                "Ensure output_hidden_states=True in model config."
            )

        # Compute feature MSE for each mapped layer pair
        feature_losses = []
        for teacher_idx, model_idx in layer_mapping:
            t_hidden = teacher_hidden[teacher_idx]  # (batch, seq, hidden)
            m_hidden = model_hidden[model_idx]

            # Project model hidden if dimensions don't match
            if projector is not None:
                m_hidden = projector(m_hidden)

            # Ensure same dtype (projector may output fp32 while t_hidden is fp16)
            if m_hidden.dtype != t_hidden.dtype:
                t_hidden = t_hidden.to(m_hidden.dtype)

            # Normalize if requested
            if self.normalize:
                t_hidden = (t_hidden - t_hidden.mean()) / (t_hidden.std() + 1e-8)
                m_hidden = (m_hidden - m_hidden.mean()) / (m_hidden.std() + 1e-8)

            # MSE loss for this layer pair
            layer_loss = F.mse_loss(m_hidden, t_hidden)
            feature_losses.append(layer_loss)

        # Average across all layer pairs
        if feature_losses:
            feature_raw = torch.stack(feature_losses).mean()
        else:
            feature_raw = torch.tensor(0.0, device=model_logits.device)

        # Weighted contributions
        ce_weighted = self.ce_loss_coeff * ce_loss
        feature_weighted = self.kd_loss_coeff * feature_raw
        total_loss = ce_weighted + feature_weighted

        return LossOutput(
            total_loss=total_loss,
            components={
                "ce_weighted": ce_weighted,
                "feature_weighted": feature_weighted,
                "ce_loss_coeff": torch.tensor(self.ce_loss_coeff),
                "kd_loss_coeff": torch.tensor(self.kd_loss_coeff),
                "ce_raw": ce_loss,
                "feature_raw": feature_raw,
                "num_layers_matched": torch.tensor(len(layer_mapping)),
            }
        )


class CosineFeatureLoss(DistillationLossStrategy):
    """
    Feature-based distillation loss using cosine similarity (DistilBERT-style).

    Loss = ce_loss_coeff * CE + kd_loss_coeff * mean(1 - cosine_sim(model_hidden, teacher_hidden))

    Unlike MSE, cosine similarity focuses on the direction of hidden state
    vectors rather than their magnitude. DistilBERT found this beneficial
    (+1.46 GLUE points when included).

    References:
    - Sanh et al. (2019): "DistilBERT, a distilled version of BERT"
    """

    requires_hidden_states = True

    def __init__(
        self,
        ce_loss_coeff: float = 0.5,
        kd_loss_coeff: float = 0.5,
        layer_mapping: Optional[List[Tuple[int, int]]] = None,
        projector: Optional[Any] = None,
    ):
        self.ce_loss_coeff = float(ce_loss_coeff)
        self.kd_loss_coeff = float(kd_loss_coeff)
        self.layer_mapping = layer_mapping or []
        self.projector = projector

    @property
    def name(self) -> str:
        return "feature_cosine"

    def compute(
        self,
        model_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        ce_loss: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        model_outputs: Optional[Any] = None,
        teacher_outputs: Optional[Any] = None,
        **kwargs,
    ) -> LossOutput:
        layer_mapping = kwargs.get("layer_mapping", self.layer_mapping)
        projector = kwargs.get("projector", self.projector)

        model_hidden = getattr(model_outputs, "hidden_states", None)
        teacher_hidden = getattr(teacher_outputs, "hidden_states", None)

        if model_hidden is None or teacher_hidden is None:
            raise ValueError(
                "Feature loss requires hidden_states. "
                "Ensure output_hidden_states=True in model config."
            )

        cosine_losses = []
        for teacher_idx, model_idx in layer_mapping:
            t_hidden = teacher_hidden[teacher_idx]
            m_hidden = model_hidden[model_idx]

            if projector is not None:
                m_hidden = projector(m_hidden)

            # Ensure same dtype (projector may output fp32 while t_hidden is fp16)
            if m_hidden.dtype != t_hidden.dtype:
                t_hidden = t_hidden.to(m_hidden.dtype)

            # Cosine similarity: 1 - cos(s, t) as loss (0 = identical direction)
            cos_sim = F.cosine_similarity(m_hidden, t_hidden, dim=-1)
            layer_loss = (1 - cos_sim).mean()
            cosine_losses.append(layer_loss)

        if cosine_losses:
            feature_raw = torch.stack(cosine_losses).mean()
        else:
            feature_raw = torch.tensor(0.0, device=model_logits.device)

        ce_weighted = self.ce_loss_coeff * ce_loss
        feature_weighted = self.kd_loss_coeff * feature_raw
        total_loss = ce_weighted + feature_weighted

        return LossOutput(
            total_loss=total_loss,
            components={
                "ce_weighted": ce_weighted,
                "feature_cosine_weighted": feature_weighted,
                "ce_loss_coeff": torch.tensor(self.ce_loss_coeff),
                "kd_loss_coeff": torch.tensor(self.kd_loss_coeff),
                "ce_raw": ce_loss,
                "feature_cosine_raw": feature_raw,
                "num_layers_matched": torch.tensor(len(layer_mapping)),
            }
        )


class AttentionLossStrategy(DistillationLossStrategy):
    """
    Attention-based distillation loss (KL on attention distributions).

    Loss = ce_loss_coeff * CE + kd_loss_coeff * mean(KL(model_attn || teacher_attn))

    Transfers knowledge by matching attention patterns between teacher
    and model. Helps model learn where to "look" in the input.

    References:
    - Zagoruyko & Komodakis (2017): "Paying More Attention to Attention"
    """

    requires_attentions = True

    def __init__(
        self,
        ce_loss_coeff: float = 0.5,
        kd_loss_coeff: float = 0.5,
        layer_mapping: Optional[List[Tuple[int, int]]] = None,
        temperature: float = 1.0,
    ):
        """
        Initialize attention distillation loss.

        Args:
            ce_loss_coeff: Weight on the hard-label (CE) term
            kd_loss_coeff: Weight on the attention matching (distillation) term
            layer_mapping: List of (teacher_idx, model_idx) pairs
            temperature: Temperature for softening attention distributions
        """
        self.ce_loss_coeff = float(ce_loss_coeff)
        self.kd_loss_coeff = float(kd_loss_coeff)
        self.layer_mapping = layer_mapping or []
        self.temperature = temperature

    @property
    def name(self) -> str:
        return "attention_kl"

    def compute(
        self,
        model_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        ce_loss: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        model_outputs: Optional[Any] = None,
        teacher_outputs: Optional[Any] = None,
        **kwargs,
    ) -> LossOutput:
        # Get layer mapping from kwargs if not set in init
        layer_mapping = kwargs.get("layer_mapping", self.layer_mapping)

        # Extract attention weights
        model_attns = getattr(model_outputs, "attentions", None)
        teacher_attns = getattr(teacher_outputs, "attentions", None)

        if model_attns is None or teacher_attns is None:
            raise ValueError(
                "Attention loss requires attentions. "
                "Ensure output_attentions=True in model config."
            )

        # Compute attention KL for each mapped layer pair
        attention_losses = []
        for teacher_idx, model_idx in layer_mapping:
            t_attn = teacher_attns[teacher_idx]  # (batch, heads, seq, seq)
            m_attn = model_attns[model_idx]

            # Handle different number of attention heads by averaging
            if t_attn.shape[1] != m_attn.shape[1]:
                # Average teacher heads to match model head count
                # or expand model heads - for now, average both to single
                t_attn = t_attn.mean(dim=1, keepdim=True)
                m_attn = m_attn.mean(dim=1, keepdim=True)

            # Flatten attention matrices for KL computation: (batch, heads, seq, seq) -> (N, seq)
            m_flat = m_attn.view(-1, m_attn.size(-1))
            t_flat = t_attn.view(-1, t_attn.size(-1))

            # Compute KL divergence (attention weights are probabilities)
            kl_loss = compute_kl_divergence(
                m_flat, t_flat, self.temperature, input_is_probs=True
            )
            attention_losses.append(kl_loss)

        # Average across all layer pairs
        if attention_losses:
            attention_raw = torch.stack(attention_losses).mean()
        else:
            attention_raw = torch.tensor(0.0, device=model_logits.device)

        # Weighted contributions
        ce_weighted = self.ce_loss_coeff * ce_loss
        attention_weighted = self.kd_loss_coeff * attention_raw
        total_loss = ce_weighted + attention_weighted

        return LossOutput(
            total_loss=total_loss,
            components={
                "ce_weighted": ce_weighted,
                "attention_weighted": attention_weighted,
                "ce_loss_coeff": torch.tensor(self.ce_loss_coeff),
                "kd_loss_coeff": torch.tensor(self.kd_loss_coeff),
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
        temperature: float = 2.0,
        ce_loss_coeff: Optional[float] = None,
        kd_loss_coeff: Optional[float] = None,
        alpha_h: Optional[float] = None,
        alpha_s: Optional[float] = None,
    ):
        """
        Initialize staged distillation loss.

        Args:
            stages: List of stage configs, each with:
                - strategy: name of loss strategy (or instance)
                - until_progress: float 0-1, training progress threshold
                Optional per-stage overrides: ``alpha_h``, ``alpha_s`` (same meaning as top-level).
            temperature: Default temperature for strategies that need it
            ce_loss_coeff, kd_loss_coeff: Default hard/soft weights for stages
            alpha_h, alpha_s: Same coefficients under caller-facing names (from trainer/config)
        """
        self.stages = []
        self.ce_loss_coeff, self.kd_loss_coeff = _resolve_ce_kd_coefficients(
            ce_loss_coeff, kd_loss_coeff, alpha_h=alpha_h, alpha_s=alpha_s,
        )
        self.temperature = temperature

        # Build strategy instances for each stage
        for stage_config in stages:
            strategy_name = stage_config["strategy"]
            until_progress = stage_config.get("until_progress", 1.0)

            stage_temp = stage_config.get("temperature", temperature)
            if "alpha_h" in stage_config or "alpha_s" in stage_config:
                stage_ce, stage_kd = _resolve_ce_kd_coefficients(
                    alpha_h=stage_config.get("alpha_h"),
                    alpha_s=stage_config.get("alpha_s"),
                )
            else:
                stage_ce, stage_kd = self.ce_loss_coeff, self.kd_loss_coeff

            # Create strategy instance
            if isinstance(strategy_name, str):
                sn = strategy_name
                strategy = get_loss_strategy(
                    sn,
                    alpha_h=stage_ce,
                    alpha_s=stage_kd,
                    temperature=stage_temp,
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
        model_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        ce_loss: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        model_outputs: Optional[Any] = None,
        teacher_outputs: Optional[Any] = None,
        **kwargs,
    ) -> LossOutput:
        # Get training progress from kwargs (passed by trainer)
        progress = kwargs.get("training_progress", 0.0)

        # Select active strategy based on progress
        active_strategy = self.get_current_strategy(progress)

        # Delegate to active strategy
        output = active_strategy.compute(
            model_logits=model_logits,
            teacher_logits=teacher_logits,
            ce_loss=ce_loss,
            labels=labels,
            model_outputs=model_outputs,
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


def build_distill_topology_strategy(
    distill_config: Dict[str, Any],
    alpha_h: float,
    alpha_s: float,
    temperature: float = 2.0,
) -> DistillationLossStrategy:
    """Build student-side loss topology (single, combined, or staged) with α coefficients."""
    stages_config = distill_config.get("stages")
    if stages_config:
        return StagedDistillationLoss(
            stages=stages_config,
            alpha_h=alpha_h,
            alpha_s=alpha_s,
            temperature=temperature,
        )
    if loss_strategies_config := distill_config.get("loss_strategies"):
        strategies = []
        weights = []
        for strat_config in loss_strategies_config:
            strat_type = strat_config["type"]
            strat_weight = strat_config.get("weight", 1.0)
            if strat_type.startswith("feature_"):
                strategy = get_loss_strategy(
                    strat_type, alpha_h=alpha_h, alpha_s=alpha_s,
                )
            elif strat_type.startswith("attention_"):
                strategy = get_loss_strategy(
                    strat_type,
                    alpha_h=alpha_h,
                    alpha_s=alpha_s,
                    temperature=temperature,
                )
            else:
                strategy = get_loss_strategy(
                    strat_type,
                    temperature=temperature,
                    alpha_h=alpha_h,
                    alpha_s=alpha_s,
                )
            strategies.append(strategy)
            weights.append(strat_weight)
        return CombinedDistillationLoss(
            strategies=strategies,
            weights=weights,
            alpha_h=alpha_h,
            alpha_s=alpha_s,
        )
    loss_type = distill_config.get("loss_type", "kl_divergence")
    if loss_type.startswith("feature_"):
        return get_loss_strategy(loss_type, alpha_h=alpha_h, alpha_s=alpha_s)
    if loss_type.startswith("attention_"):
        return get_loss_strategy(
            loss_type,
            alpha_h=alpha_h,
            alpha_s=alpha_s,
            temperature=temperature,
        )
    return get_loss_strategy(
        loss_type,
        temperature=temperature,
        alpha_h=alpha_h,
        alpha_s=alpha_s,
    )


def infer_teacher_response_loss_name(distill_config: Dict[str, Any]) -> str:
    """Pick a response (logit-level) strategy for teacher CTCD v1 (peer = detached student logits)."""
    if distill_config.get("loss_strategies"):
        for item in distill_config["loss_strategies"]:
            t = item["type"]
            if not t.startswith("feature_") and not t.startswith("attention_"):
                return t
        return "kl_divergence"
    if distill_config.get("stages"):
        for st in distill_config["stages"]:
            sn = st["strategy"]
            if isinstance(sn, str) and not sn.startswith("feature_") and not sn.startswith("attention_"):
                return sn
        return "kl_divergence"
    return distill_config.get("loss_type", "kl_divergence")


def build_teacher_response_only_strategy(
    distill_config: Dict[str, Any],
    beta_h: float,
    beta_s: float,
    temperature: float = 2.0,
) -> DistillationLossStrategy:
    """Teacher co-training branch: one response strategy with β coefficients (no F/A peer in v1)."""
    name = infer_teacher_response_loss_name(distill_config)
    if name.startswith("feature_") or name.startswith("attention_"):
        name = "kl_divergence"
    return get_loss_strategy(
        name,
        alpha_h=beta_h,
        alpha_s=beta_s,
        temperature=temperature,
    )


def build_layer_mapping_and_projector(
    distill_config: Dict[str, Any],
    loss_strategy: DistillationLossStrategy,
    model: Any,
    teacher_model: Any,
    device: str,
) -> Tuple[Optional[List[Tuple[int, int]]], Optional[torch.nn.Module]]:
    """Layer mapping and optional projector from distill config (matches forge semantics)."""
    from .layer_mapping import get_layer_mapping
    from .projections import create_projection_if_needed

    layer_mapping = None
    layer_mapping_config = distill_config.get("layer_mapping", {})
    if layer_mapping_config or loss_strategy.requires_hidden_states or loss_strategy.requires_attentions:
        mapping_type = layer_mapping_config.get("type", "uniform")
        teacher_layers = getattr(teacher_model.config, "num_hidden_layers", 12)
        model_layers = getattr(model.config, "num_hidden_layers", 6)

        if mapping_type == "custom" and "mapping" in layer_mapping_config:
            layer_mapping = [
                (pair["teacher"], pair["model"])
                for pair in layer_mapping_config["mapping"]
            ]
        else:
            mapper = get_layer_mapping(mapping_type)
            if not mapper.validate(teacher_layers, model_layers):
                raise ValueError(
                    f"Invalid layer mapping '{mapping_type}' for "
                    f"teacher={teacher_layers}, model={model_layers} layers"
                )
            layer_mapping = mapper.map(teacher_layers, model_layers)

    projector = None
    projection_config = distill_config.get("projection", {})
    if loss_strategy.requires_hidden_states:
        teacher_dim = getattr(teacher_model.config, "hidden_size", 768)
        model_dim = getattr(model.config, "hidden_size", 768)

        if teacher_dim != model_dim or projection_config.get("force", False):
            proj_type = projection_config.get("type", "linear")
            proj_kwargs: Dict[str, Any] = {}
            if projection_config.get("hidden_dim") is not None:
                proj_kwargs["hidden_dim"] = projection_config["hidden_dim"]
            if "activation" in projection_config:
                proj_kwargs["activation"] = projection_config["activation"]
            projector = create_projection_if_needed(
                teacher_dim=teacher_dim,
                model_dim=model_dim,
                projection_type=proj_type,
                **proj_kwargs,
            )
            if projector is not None:
                projector = projector.to(device)

    return layer_mapping, projector


# Registry for easy lookup by name
# Naming convention: {signal}_{method}
#   signal: response, feature, attention
#   method: kl, mse, cosine, ce
LOSS_REGISTRY = {
    # Response-based (output logits)
    "response_kl": KLDistillationLoss,
    "response_reverse-kl": partial(KLDistillationLoss, reverse=True),
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
        **kwargs: Arguments to pass to the strategy constructor. You may pass
            ``alpha_h`` / ``alpha_s``; these are converted to ``ce_loss_coeff`` / ``kd_loss_coeff``.

    Returns:
        Instantiated loss strategy

    Example:
        loss = get_loss_strategy("response_kl", temperature=2.0, alpha_h=0.5, alpha_s=0.5)
        loss = get_loss_strategy("feature_cosine", alpha_s=0.3, alpha_h=0.7)
    """
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss strategy: {name}. Available: {list(LOSS_REGISTRY.keys())}")

    cls = LOSS_REGISTRY[name]
    kwargs = _merge_coeff_kwargs(dict(kwargs))

    # Filter kwargs to only what each strategy accepts.
    # Strategies that don't use temperature (MSE, cosine, feature) should not
    # receive it. Update this mapping when extending strategy constructors.
    ACCEPTED_PARAMS = {
        KLDistillationLoss: {"temperature", "ce_loss_coeff", "kd_loss_coeff", "reverse"},
        MSEDistillationLoss: {"ce_loss_coeff", "kd_loss_coeff", "normalize"},
        CosineDistillationLoss: {"ce_loss_coeff", "kd_loss_coeff"},
        SoftTargetDistillationLoss: {"temperature", "ce_loss_coeff", "kd_loss_coeff"},
        FeatureLossStrategy: {"ce_loss_coeff", "kd_loss_coeff", "layer_mapping", "projector", "normalize"},
        CosineFeatureLoss: {"ce_loss_coeff", "kd_loss_coeff", "layer_mapping", "projector"},
        AttentionLossStrategy: {"ce_loss_coeff", "kd_loss_coeff", "layer_mapping", "temperature"},
        StagedDistillationLoss: {"stages", "temperature", "ce_loss_coeff", "kd_loss_coeff"},
    }

    # For partials (e.g., reverse-kl), unwrap to find the base class
    lookup_cls = getattr(cls, "func", cls)
    accepted = ACCEPTED_PARAMS.get(lookup_cls)

    if accepted is not None:
        filtered = {k: v for k, v in kwargs.items() if k in accepted}
        return cls(**filtered)

    return cls(**kwargs)
